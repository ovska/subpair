from __future__ import annotations

import argparse
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from subpair.cache import CacheError, load_cache, write_cache
from subpair.cli import (
    _build_parser,
    _parse_geometry_config,
    _parse_indices,
    _parse_room_dimensions,
)
from subpair.engine import (
    ARRIVAL_REPAIRED_WINDOW_FACTOR,
    ARRIVAL_SLIP_TOLERANCE_MS,
    GateThresholds,
    SearchOptions,
    _apply_band_edge_population_status,
    _score_wide_spectrum,
    _resolve_arrival_delays,
    _baseline_objective_curve,
    _basin_width,
    _detrended_symmetry,
    _geometry_jitter,
    _improvement_localization,
    _redundancy_residual,
    _ripple_deviation_correlation,
    run_search,
)
from subpair.dsp import (
    EqOptions,
    EXCURSION_POWER_DB_PER_OCTAVE,
    AnalysisContext,
    _denoised_residual,
    _excess_gd_authority,
    _smooth_by_variable_octaves,
    db20,
    excess_gd_peak_ms,
    excess_gd_tail_ms,
    fit_eq_filters,
    excess_group_delay,
    gd_smoothing_octaves,
    log_frequency_grid,
    low_end_power_db,
    low_shelf_response,
    pair_diagnostics,
    peq_response,
    smoothed_dip_db,
    usable_output_score_db,
)
from subpair.html_report import (
    _magnitude_figure,
    _overview_excess_figure,
    _overview_figure,
    _peq_text,
    _ranking_table,
    _room_mode_traces,
    _selected_axis_ranges,
    build_report,
    room_mode_frequencies,
)


def _synthetic_ir(sample_rate: float, length: int, delay: int, modes: list[tuple[float, float]]) -> np.ndarray:
    result = np.zeros(length, dtype=np.float64)
    result[delay] = 1.0
    time = np.arange(length - delay) / sample_rate
    for frequency, amplitude in modes:
        result[delay:] += amplitude * np.sin(2 * np.pi * frequency * time) * np.exp(-time / 0.12)
    return result


def _permissive_gates() -> GateThresholds:
    """Keep legacy pipeline tests focused on scoring rather than gate fixtures."""

    return GateThresholds(
        redundancy_reject=0.0,
        redundancy_caution=0.0,
        ripple_correlation_reject=1.0,
        ripple_complementary=-1.0,
        physical_percentile_reject=100.0,
        cancellation_deficit_reject_db=-300.0,
        cancellation_deficit_caution_db=-300.0,
        comb_index_reject=1.0,
        comb_index_caution=1.0,
        notch_depth_reject_db=300.0,
        gain_asymmetry_caution_db=100.0,
        band_edge_excess_spread_reject_db=300.0,
        localization_fraction_reject=1.0,
        localization_min_mean_improvement_db=0.0,
        basin_tolerance_db=300.0,
    )


class PipelineTests(unittest.TestCase):
    def test_basin_width_uses_only_the_contiguous_interval_around_the_minimum(self):
        tau = np.arange(-2.0, 2.0001, 0.05)
        objective = np.full(tau.size, 2.0)
        objective[(tau >= -0.5) & (tau <= 0.5)] = 0.0
        # A disconnected equally good region must not inflate the basin.
        objective[(tau >= 1.5) & (tau <= 1.8)] = 0.0
        centre = int(np.argmin(np.abs(tau)))
        self.assertLessEqual(
            abs(_basin_width(objective, tau, centre, 0.3) - 1.0), 0.05 + 1e-12
        )

    def test_geometry_jitter_reaches_2d_over_c_for_opposite_subs(self):
        excursion_ms, conservative = _geometry_jitter(
            0,
            1,
            (0.0, 0.0, 0.0),
            {1: (-2.0, 0.0, 0.0), 2: (2.0, 0.0, 0.0)},
            0.25,
            343.0,
        )
        self.assertFalse(conservative)
        self.assertAlmostEqual(excursion_ms, 1000.0 * 0.5 / 343.0)
        same_direction_ms, conservative = _geometry_jitter(
            0,
            1,
            (0.0, 0.0, 0.0),
            {1: (2.0, 0.0, 0.0), 2: (4.0, 0.0, 0.0)},
            0.25,
            343.0,
        )
        self.assertFalse(conservative)
        self.assertAlmostEqual(same_direction_ms, 0.0)

    def test_geometry_config_accepts_one_based_sub_position_object(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "geometry.json"
            path.write_text(
                json.dumps(
                    {
                        "listening_position_m": [1, 2, 1],
                        "sub_positions_m": {"1": [0, 0, 0], "2": [4, 0, 0]},
                        "room_dimensions_m": [5, 4, 2.5],
                    }
                )
            )
            listener, subs, room = _parse_geometry_config(path)
        self.assertEqual(listener, (1.0, 2.0, 1.0))
        self.assertEqual(subs[2], (4.0, 0.0, 0.0))
        self.assertEqual(room, (5.0, 4.0, 2.5))

    @staticmethod
    def _timing_cache(cache: Path, specs: list[tuple[int, int | None]]) -> None:
        """Impulses with a direct arrival and an optional larger later lobe.

        ``specs`` is ``(direct_index, slipped_lobe_index)``. The reported arrival
        is taken from the impulse's own largest sample, exactly as REW derives
        it, so a later lobe that outgrows the direct arrival produces a genuine
        cycle-slipped pick rather than a hand-written wrong number.
        """

        sample_rate, length, start = 3000.0, 4096, -0.5
        rows = []
        for index, (direct, slipped) in enumerate(specs, start=1):
            ir = np.zeros(length, dtype=np.float64)
            time = np.arange(length - direct) / sample_rate
            ir[direct:] += np.sin(2 * np.pi * 60.0 * time) * np.exp(-time / 0.05)
            if slipped is not None:
                t2 = np.arange(length - slipped) / sample_rate
                ir[slipped:] += 2.5 * np.sin(2 * np.pi * 60.0 * t2) * np.exp(-t2 / 0.05)
            reported_ms = 1000.0 * (start + int(np.argmax(np.abs(ir))) / sample_rate)
            rows.append(
                {
                    "source_index": index,
                    "title": f"Position {index}",
                    "uuid": f"uuid-{index}",
                    "sample_rate": sample_rate,
                    "start_time_seconds": start,
                    "impulse": ir,
                    "metadata": {"arrival_delay_ms": reported_ms},
                }
            )
        write_cache(cache, rows, {"test": True})

    def test_clean_arrival_peaks_are_left_alone(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache"
            self._timing_cache(cache, [(300, None), (306, None), (312, None), (309, None)])
            measurements, _ = load_cache(cache)
            delays, repaired, unusable, warnings, diag = _resolve_arrival_delays(
                measurements, (35.0, 150.0), 343.0, None
            )
        self.assertEqual(repaired, set())
        self.assertEqual(unusable, set())
        self.assertEqual(warnings, [])
        reported = [row.metadata["arrival_delay_ms"] for row in measurements]
        for actual, expected in zip(delays, reported):
            self.assertAlmostEqual(actual, expected)
        # The lag between peak and onset is what makes a slip detectable, so it
        # must be near-constant when nothing has slipped.
        lags = [entry["peak_minus_onset_ms"] for entry in diag["measurements"]]
        self.assertLess(max(lags) - min(lags), ARRIVAL_SLIP_TOLERANCE_MS)

    def test_cycle_slipped_peak_is_repaired_from_its_own_onset(self):
        # Position 3's impulse has a later lobe 2.5x the direct arrival, so its
        # largest sample -- and therefore REW's reported delay -- lands a full
        # cycle late. Its leading edge does not move, which is what the repair
        # keys on.
        slip_samples = 50
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache"
            self._timing_cache(
                cache,
                [(300, None), (306, None), (312, 312 + slip_samples), (309, None)],
            )
            measurements, _ = load_cache(cache)
            delays, repaired, unusable, warnings, diag = _resolve_arrival_delays(
                measurements, (35.0, 150.0), 343.0, None
            )
        self.assertEqual(repaired, {2})
        self.assertEqual(unusable, set())
        reported = measurements[2].metadata["arrival_delay_ms"]
        expected_slip_ms = 1000.0 * slip_samples / measurements[0].sample_rate
        # The repair recovers most of the slip but not all of it: the onset it
        # rebuilds from creeps later as the slipped lobe grows relative to the
        # direct arrival (here 2.5x). That residual bias is why a repaired
        # pair's delay window is widened and its Gate C is advisory, rather
        # than the reconstruction being trusted outright.
        recovered = reported - delays[2]
        self.assertGreater(recovered, 0.5 * expected_slip_ms)
        self.assertLessEqual(recovered, expected_slip_ms + 1.0)
        # Repair keeps this position's own distance rather than flattening it
        # to the median arrival: it sits later than 300/306 as its onset says.
        self.assertGreater(delays[2], delays[0])
        self.assertEqual(delays[0], measurements[0].metadata["arrival_delay_ms"])
        self.assertTrue(any("peak pick slipped" in text for text in warnings))
        self.assertTrue(diag["measurements"][2]["repaired"])

    def test_arrival_spread_beyond_the_room_diagonal_is_discarded(self):
        # A common timing offset cancels in the differences that are used, so
        # the room bounds the spread between arrivals, not each absolute value.
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache"
            self._timing_cache(cache, [(300, None), (306, None), (2400, None)])
            measurements, _ = load_cache(cache)
            _delays, _repaired, unusable, warnings, _diag = _resolve_arrival_delays(
                measurements, (35.0, 150.0), 343.0, (5.0, 4.0, 2.5)
            )
        self.assertIn(2, unusable)
        self.assertTrue(any("room diagonal" in text for text in warnings))

    def test_search_constrains_robust_delay_to_measured_physical_window(self):
        sample_rate = 4000.0
        length = 4096
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache"
            rows = []
            for index, (sample_delay, arrival_ms) in enumerate(((100, 25.0), (108, 27.0)), start=1):
                rows.append(
                    {
                        "source_index": index,
                        "title": f"Position {index}",
                        "uuid": f"uuid-{index}",
                        "sample_rate": sample_rate,
                        "start_time_seconds": -0.025,
                        "impulse": _synthetic_ir(sample_rate, length, sample_delay, [(50, 0.2)]),
                        "metadata": {"arrival_delay_ms": arrival_ms},
                    }
                )
            write_cache(cache, rows, {"test": True})
            result = run_search(
                cache,
                cache / "results.json",
                SearchOptions(
                    delay_range_ms=(-3.0, 3.0, 0.25),
                    gain_range_db=(0.0, 0.0, 1.0),
                    eq_bands=0,
                    physical_delay_window_ms=0.5,
                    gate_thresholds=_permissive_gates(),
                ),
            )
        pair = result["pairs"][0]
        self.assertEqual(pair["physical_tau"], -2.0)
        self.assertLessEqual(abs(pair["tau_robust"] + 2.0), 0.5 + 1e-12)
        self.assertTrue(pair["pair_valid"])
        gate_c = pair["gates"]["gate_c_physical_percentile"]
        self.assertAlmostEqual(
            pair["physical_objective_gap_db"],
            gate_c["objective_at_physical_db"] - pair["f_tau_star"],
        )
        self.assertEqual(gate_c["gap_reference"], "f(tau_star)")

    def _outlier_cache(
        self,
        cache: Path,
        sample_rate: float,
        length: int,
        outlier_sample_delay: int,
    ) -> None:
        """Three positions where the third's arrival delay is a clear outlier.

        ``outlier_sample_delay`` separates the two things a flagged arrival can
        mean: metadata that is wrong while the impulse is fine (a delay close
        to the others), or an impulse genuinely captured against a different
        timing reference (a distant one).
        """

        rows = []
        for index, (sample_delay, arrival_ms) in enumerate(
            ((100, 10.0), (108, 11.0), (outlier_sample_delay, 40.0)), start=1
        ):
            rows.append(
                {
                    "source_index": index,
                    "title": f"Position {index}",
                    "uuid": f"uuid-{index}",
                    "sample_rate": sample_rate,
                    "start_time_seconds": -0.025,
                    "impulse": _synthetic_ir(
                        sample_rate, length, sample_delay, [(50, 0.2), (83, 0.15)]
                    ),
                    "metadata": {"arrival_delay_ms": arrival_ms},
                }
            )
        write_cache(cache, rows, {"test": True})

    def test_bad_arrival_metadata_is_repaired_rather_than_discarding_the_pair(self):
        # Metadata wrong, impulse fine -- exactly the cycle-slip case. The
        # arrival is rebuilt from the leading edge, the pair is scored, and the
        # timing gate is demoted to advisory so a poor REW pick can change how
        # exact the reported delay is but never which pairs are recommended.
        sample_rate = 4000.0
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache"
            self._outlier_cache(cache, sample_rate, 4096, 104)
            result = run_search(
                cache,
                cache / "results.json",
                SearchOptions(
                    delay_range_ms=(-6.0, 6.0, 0.25),
                    gain_range_db=(0.0, 0.0, 1.0),
                    eq_bands=0,
                    gate_thresholds=_permissive_gates(),
                ),
            )
        self.assertIn(3, result["measurement_arrival_delay_repaired"])
        reported = result["arrival_timing"]["measurements"][2]["reported_ms"]
        resolved = result["measurement_arrival_delays_ms"][2]
        self.assertLess(resolved, reported)
        affected = [row for row in result["pairs"] if 3 in (row["first"], row["second"])]
        self.assertEqual(len(affected), 2)
        for row in affected:
            gate_c = row["gates"]["gate_c_physical_percentile"]
            self.assertTrue(gate_c["arrival_repaired"])
            self.assertNotEqual(gate_c["status"], "reject")
            # Repaired timing still aims the search, on a widened window.
            self.assertIsNotNone(row["physical_tau"])
            self.assertTrue(row["robustness"]["physical_window_widened"])
            self.assertAlmostEqual(
                row["robustness"]["physical_window_ms"],
                1.5 * ARRIVAL_REPAIRED_WINDOW_FACTOR,
            )
            self.assertTrue(row["optimized"])
            self.assertIsNotNone(row["score_db"])

    def test_delay_pinned_to_the_scan_edge_is_rejected(self):
        # An unconstrained pair whose true optimum lies outside --delay-range
        # runs to the boundary. That is the limit of the search, not an
        # optimum, and every robustness figure around it is one-sided.
        sample_rate = 4000.0
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache"
            # The third impulse really is 10 ms away from the others.
            self._outlier_cache(cache, sample_rate, 4096, 140)
            result = run_search(
                cache,
                cache / "results.json",
                SearchOptions(
                    # Far too narrow to reach the third position's alignment.
                    delay_range_ms=(-0.5, 0.5, 0.25),
                    gain_range_db=(0.0, 0.0, 1.0),
                    eq_bands=0,
                    gate_thresholds=_permissive_gates(),
                ),
            )
        pinned = [
            row
            for row in result["pairs"]
            if row["optimized"] and abs(abs(row["delay_ms"]) - 0.5) <= 1e-9
        ]
        self.assertTrue(pinned, "expected at least one boundary-pinned delay")
        for row in pinned:
            gate_c = row["gates"]["gate_c_physical_percentile"]
            self.assertTrue(gate_c["delay_at_scan_boundary"])
            self.assertEqual(gate_c["status"], "reject")
            self.assertEqual(row["verdict"], "reject")
            self.assertIn("--delay-range", gate_c["detail"])

    def test_selected_report_graphs_share_data_bounds_and_cap_negative_excess_gd(self):
        rows = [
            (
                {
                    "first": 1,
                    "second": 2,
                    "rank": 1,
                    "eq_rank": 1,
                    "relative_score_db": 0.0,
                    "post_eq_relative_score_db": 0.0,
                    "first_name": "A",
                    "second_name": "B",
                },
                {
                    "frequencies": np.array([20.0, 40.0, 80.0]),
                    "solo_first_db": np.array([-8.0, -2.0, 1.0]),
                    "solo_second_db": np.array([-5.0, 0.0, 3.0]),
                    "sum_db": np.array([-4.0, 2.0, 5.0]),
                    "post_eq_db": np.array([-3.0, 1.0, 4.0]),
                    "excess_curve_ms": np.array([-35.0, -2.0, 6.0]),
                    "post_eq_excess_curve_ms": np.array([-30.0, -1.0, 5.0]),
                },
            ),
            (
                {
                    "first": 3,
                    "second": 4,
                    "rank": 2,
                    "eq_rank": 2,
                    "relative_score_db": -1.0,
                    "post_eq_relative_score_db": -1.0,
                    "first_name": "C",
                    "second_name": "D",
                },
                {
                    "frequencies": np.array([20.0, 40.0, 80.0]),
                    "solo_first_db": np.array([-60.0, -50.0, -40.0]),
                    "solo_second_db": np.array([20.0, 30.0, 40.0]),
                    "sum_db": np.array([-55.0, 0.0, 35.0]),
                    "post_eq_db": np.array([-50.0, 0.0, 30.0]),
                    "excess_curve_ms": np.array([-80.0, 0.0, 70.0]),
                    "post_eq_excess_curve_ms": np.array([-70.0, 0.0, 60.0]),
                },
            ),
        ]
        selected = {"1-2"}

        magnitude_range, excess_range = _selected_axis_ranges(
            rows, raw=False, selected_keys=selected
        )
        self.assertEqual(magnitude_range, (-8.0, 4.0))
        self.assertEqual(excess_range, (-20.0, 5.0))

        magnitude = _overview_figure(rows, "eq", selected)
        excess = _overview_excess_figure(rows, "eq", selected)
        self.assertEqual(tuple(magnitude.layout.yaxis.range), (-3.0, 4.0))
        self.assertEqual(tuple(excess.layout.yaxis.range), (-20.0, 5.0))
        self.assertEqual(excess.layout.yaxis.minallowed, -20.0)

    def test_eq_response_and_band_markers_share_the_magnitude_axis(self):
        pair = {"first_name": "A", "second_name": "B"}
        data = {
            "frequencies": np.array([20.0, 40.0, 80.0]),
            "solo_first_db": np.array([-8.0, -2.0, 1.0]),
            "solo_second_db": np.array([-5.0, 0.0, 3.0]),
            "sum_db": np.array([-4.0, 2.0, 5.0]),
            "post_eq_db": np.array([-3.0, 1.0, 4.0]),
            "post_eq_excess_curve_ms": np.array([-2.0, 0.0, 1.0]),
            "eq_target_db": np.array([-3.0, 1.0, 4.0]),
            "filters": [{"fc_hz": 40.0, "gain_db": -10.0, "q": 0.8}],
            "eq_shelf": {
                "active": True,
                "freq_hz": 30.0,
                "gain_db": 3.0,
                "slope": 1.0,
            },
        }
        rows = [({"first": 1, "second": 2}, data)]
        magnitude_range, _ = _selected_axis_ranges(
            rows, raw=False, selected_keys={"1-2"}
        )
        self.assertEqual(magnitude_range, (-10.0, 4.0))

        figure = _magnitude_figure(pair, data, y_range=magnitude_range)
        traces = {trace.name: trace for trace in figure.data}
        response = traces["Combined EQ response (all bands)"]
        markers = traces["EQ band settings"]
        self.assertIsNone(response.yaxis)
        self.assertIsNone(markers.yaxis)
        self.assertNotIn("yaxis2", figure.layout.to_plotly_json())
        self.assertEqual(tuple(figure.layout.yaxis.range), (-10.0, 4.0))
        self.assertEqual(tuple(markers.x), (40.0, 30.0))
        self.assertEqual(tuple(markers.y), (-10.0, 3.0))
        self.assertEqual(
            tuple(markers.customdata),
            ("PK band · Q 0.800", "LS band · slope 1.00"),
        )
        self.assertEqual(markers.marker.size, 7)
        self.assertEqual(response.legendgroup, markers.legendgroup)

    def test_report_result_limit_argument(self):
        parser = _build_parser()
        defaults = parser.parse_args(["report"])
        self.assertEqual(defaults.limit, 15)
        self.assertFalse(defaults.raw)
        self.assertEqual(parser.parse_args(["report", "--limit", "24"]).limit, 24)
        self.assertTrue(parser.parse_args(["report", "--raw"]).raw)

    def test_search_score_weight_arguments(self):
        parser = _build_parser()
        defaults = parser.parse_args(["search"])
        self.assertEqual(defaults.delay_range[2], 0.05)
        self.assertEqual(defaults.max_cut, 18.0)
        self.assertEqual(defaults.score_low_end_weight, 0.5)
        self.assertEqual(defaults.score_dip_weight, 1.0)
        self.assertEqual(defaults.gate_redundancy_reject, 0.5)
        self.assertEqual(defaults.gate_ripple_reject, 0.3)
        self.assertEqual(defaults.gate_physical_percentile, 75.0)
        self.assertEqual(defaults.gate_basin_tolerance, 0.5)
        overridden = parser.parse_args(
            [
                "search",
                "--max-cut",
                "24",
                "--score-low-end-weight",
                "0.75",
                "--score-dip-weight",
                "1.5",
                "--gate-redundancy-reject",
                "0.45",
                "--gate-band-edge-excess-spread",
                "2.5",
            ]
        )
        self.assertEqual(overridden.max_cut, 24.0)
        self.assertEqual(overridden.score_low_end_weight, 0.75)
        self.assertEqual(overridden.score_dip_weight, 1.5)
        self.assertEqual(overridden.gate_redundancy_reject, 0.45)
        self.assertEqual(overridden.gate_band_edge_excess_spread, 2.5)

    def test_delayed_scaled_copy_triggers_redundancy_and_has_expected_symmetry(self):
        sample_rate = 4000.0
        length = 4096
        source = _synthetic_ir(
            sample_rate,
            length,
            100,
            [(43.0, 0.20), (91.0, 0.20)],
        )
        delayed = np.zeros_like(source)
        delayed[8:] = 0.7 * source[:-8]
        rows = [
            {
                "source_index": index,
                "title": title,
                "uuid": str(index),
                "sample_rate": sample_rate,
                "start_time_seconds": -0.025,
                "impulse": impulse,
            }
            for index, (title, impulse) in enumerate(
                [("reference", source), ("delayed copy", delayed)],
                start=1,
            )
        ]
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            write_cache(cache, rows, {"test": True})
            measurements, _manifest = load_cache(cache)
            context = AnalysisContext(measurements, (35.0, 150.0), 64)
            delays = np.arange(-10.0, 10.0001, 0.05)
            residual, fitted_delay_ms, _scale = _redundancy_residual(
                context, 0, 1, delays
            )
            self.assertLess(residual, 0.01)
            self.assertAlmostEqual(fitted_delay_ms, 2.0, delta=0.05)
            self.assertGreater(_ripple_deviation_correlation(context, 0, 1), 0.99)

            objective, _polarity = _baseline_objective_curve(
                context, 0, 1, delays, 0.5, 1.0
            )
            symmetry = _detrended_symmetry(objective, delays, -2.0)
            self.assertGreater(symmetry["correlation"], 0.98)
            self.assertAlmostEqual(symmetry["axis_ms"], -2.0, delta=0.1)

    def test_stage_one_rejection_skips_the_expensive_optimizer(self):
        sample_rate = 4000.0
        length = 4096
        source = _synthetic_ir(sample_rate, length, 100, [(48.0, 0.2)])
        rows = [
            {
                "source_index": index,
                "title": f"copy {index}",
                "uuid": str(index),
                "sample_rate": sample_rate,
                "start_time_seconds": -0.025,
                "impulse": source.copy(),
            }
            for index in (1, 2)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            results_path = cache / "search-results.json"
            write_cache(cache, rows, {"test": True})
            with mock.patch("subpair.engine._best_configurations") as optimizer:
                result = run_search(
                    cache,
                    results_path,
                    SearchOptions(
                        band=(35.0, 150.0),
                        delay_range_ms=(-3.0, 3.0, 0.05),
                        gain_range_db=(-1.0, 1.0, 1.0),
                        ppo=48,
                        eq_bands=0,
                    ),
                )
            optimizer.assert_not_called()
            pair = result["pairs"][0]
            self.assertFalse(pair["optimized"])
            self.assertEqual(pair["verdict"], "reject")
            self.assertEqual(pair["gates"]["gate_a_redundancy"]["status"], "reject")
            self.assertEqual(pair["gates"]["gate_b_ripple_correlation"]["status"], "reject")
            self.assertEqual(result["optimized_pair_count"], 0)
            self.assertEqual(
                result["settings"]["gates"]["thresholds"]["redundancy_reject"],
                0.5,
            )

    def test_automatic_low_shelf_cli_flag_defaults_on_for_search_only(self):
        parser = _build_parser()
        defaults = parser.parse_args(["search"])
        self.assertEqual(defaults.low_shelf, "on")
        self.assertTrue(SearchOptions().low_shelf)
        self.assertTrue(EqOptions().low_shelf)
        self.assertEqual(
            parser.parse_args(["search", "--low-shelf", "off"]).low_shelf,
            "off",
        )
        with self.assertRaises(SystemExit):
            parser.parse_args(["search", "--low-shelf", "maybe"])
        for removed in ("--low-shelf-freq", "--low-shelf-gain", "--low-shelf-slope"):
            with self.assertRaises(SystemExit):
                parser.parse_args(["search", removed, "40"])
        for command in ("report", "verify"):
            with self.assertRaises(SystemExit):
                parser.parse_args([command, "--low-shelf", "off"])

    def test_low_shelf_response_reaches_target_gain_at_dc_and_zero_up_high(self):
        frequencies = np.array([1.0, 640.0, 1000.0])
        response_db = db20(low_shelf_response(frequencies, 4000.0, 40.0, 6.0, slope=1.0))
        self.assertAlmostEqual(float(response_db[0]), 6.0, places=2)
        self.assertLess(abs(float(response_db[-1])), 0.01)
        # A standard RBJ shelf crosses half its dB gain at fc itself.
        at_fc = db20(low_shelf_response(np.array([40.0]), 4000.0, 40.0, 6.0, slope=1.0))
        self.assertAlmostEqual(float(at_fc[0]), 3.0, places=2)

    def test_low_shelf_response_is_monotonic_between_its_asymptotes(self):
        dense = np.geomspace(1.0, 1000.0, 500)
        response_db = db20(low_shelf_response(dense, 4000.0, 40.0, 6.0, slope=1.0))
        self.assertTrue(np.all(np.diff(response_db) <= 1e-9))

    def test_automatic_low_shelf_fits_corner_and_boost_or_cut_as_an_eq_band(self):
        frequencies = log_frequency_grid(20.0, 200.0, 48)
        fitted_corners = []
        for source_corner in (35.0, 90.0):
            spectrum = low_shelf_response(
                frequencies, 4000.0, source_corner, 6.0
            )
            filters, response, metadata = fit_eq_filters(
                spectrum,
                frequencies,
                4000.0,
                48,
                np.zeros_like(frequencies),
                EqOptions(
                    target="flat",
                    correction_range=(20.0, 200.0),
                    max_boost_db=0.0,
                    max_cut_db=12.0,
                    max_filters=1,
                    low_shelf=True,
                ),
            )
            shelf = metadata["shelf"]
            self.assertEqual(filters, [])
            self.assertTrue(shelf["active"])
            self.assertEqual(metadata["filter_count"], 1)
            self.assertLess(shelf["gain_db"], 0.0)
            self.assertLess(
                float(np.std(db20(spectrum * response))),
                0.6 * float(np.std(db20(spectrum))),
            )
            fitted_corners.append(shelf["freq_hz"])
        self.assertGreater(fitted_corners[1], 1.5 * fitted_corners[0])

        rolled_off = low_shelf_response(frequencies, 4000.0, 55.0, -6.0)
        _, _, boosted_metadata = fit_eq_filters(
            rolled_off,
            frequencies,
            4000.0,
            48,
            np.zeros_like(frequencies),
            EqOptions(
                target="flat",
                correction_range=(20.0, 200.0),
                max_boost_db=6.0,
                max_cut_db=12.0,
                max_filters=2,
                low_shelf=True,
            ),
        )
        self.assertTrue(boosted_metadata["shelf"]["active"])
        self.assertGreater(boosted_metadata["shelf"]["gain_db"], 0.0)
        self.assertEqual(boosted_metadata["filter_count"], 2)

    def test_automatic_low_shelf_can_be_disabled(self):
        frequencies = log_frequency_grid(20.0, 200.0, 48)
        spectrum = low_shelf_response(frequencies, 4000.0, 50.0, 6.0)
        _, _, metadata = fit_eq_filters(
            spectrum,
            frequencies,
            4000.0,
            48,
            np.zeros_like(frequencies),
            EqOptions(
                target="flat",
                correction_range=(20.0, 200.0),
                max_filters=1,
                low_shelf=False,
            ),
        )
        self.assertFalse(metadata["shelf"]["active"])

    def test_automatic_low_shelf_is_rendered_as_an_eq_filter(self):
        text = _peq_text(
            [],
            {
                "active": True,
                "freq_hz": 42.0,
                "gain_db": -3.5,
                "slope": 1.0,
            },
            -4.46,
        )
        self.assertIn("Preamp -4.5 dB", text)
        self.assertIn("LS Fc 42.0 Hz  Gain -3.5 dB", text)
        self.assertIn("automatically fitted EQ band", text)
        self.assertNotIn("No filters fitted", text)

    def test_low_end_power_keeps_a_flat_response_at_its_level(self):
        frequencies = log_frequency_grid(20.0, 150.0, 48)
        trend = np.full_like(frequencies, 7.5)
        self.assertAlmostEqual(low_end_power_db(trend, frequencies), 7.5)

    def test_low_end_power_weights_lower_output_more_than_higher_output(self):
        frequencies = log_frequency_grid(20.0, 100.0, 48)
        flat = np.zeros_like(frequencies)
        low_boost = flat + np.where(frequencies <= 40.0, 3.0, 0.0)
        high_boost = flat + np.where(frequencies >= 60.0, 3.0, 0.0)
        self.assertGreater(
            low_end_power_db(low_boost, frequencies),
            low_end_power_db(high_boost, frequencies),
        )
        self.assertAlmostEqual(EXCURSION_POWER_DB_PER_OCTAVE, 12.0412, places=3)

    def test_low_end_power_is_continuous_in_response_level(self):
        frequencies = log_frequency_grid(20.0, 150.0, 48)
        rolloff = np.where(
            frequencies >= 45.0,
            0.0,
            -12.0 * np.log2(45.0 / frequencies),
        )
        louder = rolloff + 0.35
        self.assertAlmostEqual(
            low_end_power_db(louder, frequencies)
            - low_end_power_db(rolloff, frequencies),
            0.35,
            places=6,
        )

    def test_low_end_power_ignores_response_above_100_hz(self):
        frequencies = log_frequency_grid(20.0, 200.0, 48)
        baseline = np.zeros_like(frequencies)
        high_peak = np.where(frequencies > 100.0, 30.0, 0.0)
        self.assertAlmostEqual(
            low_end_power_db(high_peak, frequencies),
            low_end_power_db(baseline, frequencies),
            places=6,
        )

    def test_low_end_power_vectorizes_over_candidate_responses(self):
        frequencies = log_frequency_grid(20.0, 200.0, 48)
        responses = np.stack(
            [np.zeros_like(frequencies), np.full_like(frequencies, 6.0)]
        )
        scores = low_end_power_db(responses, frequencies)
        np.testing.assert_allclose(scores, [0.0, 6.0], atol=1e-9)

    def test_smoothed_dip_finds_a_narrow_dip_but_ignores_log_linear_rolloff(self):
        frequencies = log_frequency_grid(10.0, 400.0, 48)
        rolloff = -18.0 * np.log2(frequencies / 80.0)
        score_slice = slice(48, -48)
        self.assertLess(
            float(smoothed_dip_db(rolloff, 48, score_slice=score_slice)),
            1e-6,
        )
        notched = rolloff.copy()
        centre = int(np.argmin(np.abs(frequencies - 80.0)))
        notched[centre] -= 9.0
        self.assertGreater(
            float(smoothed_dip_db(notched, 48, score_slice=score_slice)),
            7.0,
        )

    def test_usable_output_score_blends_output_and_deducts_dip(self):
        self.assertAlmostEqual(
            usable_output_score_db(10.0, 6.0, 2.0),
            6.0,
        )
        self.assertAlmostEqual(
            usable_output_score_db(
                10.0, 6.0, 2.0, low_end_weight=0.75, dip_weight=0.5
            ),
            6.0,
        )

    def test_ranking_table_renders_relative_low_end_power(self):
        base = {
            "polarity": 1,
            "delay_ms": 0.0,
            "gain_db": 0.0,
            "headroom_db": -4.5,
            "dip_db": 1.0,
            "excess_gd_ms": 0.5,
            "raw_tail_ms": 10.0,
            "effective_tail_ms": 10.0,
            "relative_spl_db": 0.0,
        }
        pairs = [
            {**base, "first": 1, "second": 2, "rank": 1,
             "relative_score_db": 0.0,
             "relative_low_end_power_db": 1.25},
            {**base, "first": 3, "second": 4, "rank": 2,
             "relative_score_db": -2.0,
             "relative_low_end_power_db": -2.5},
        ]
        table = _ranking_table(pairs, "raw", "ranking-raw", {"1-2", "3-4"})
        self.assertIn("Low-end power (dB)", table)
        self.assertIn("Headroom (dB)", table)
        self.assertIn('data-value="-4.5"', table)
        self.assertIn(">-4.50</td>", table)
        self.assertIn('data-value="1.25"', table)
        self.assertIn(">+1.25</td>", table)
        self.assertIn('data-value="-2.5"', table)
        self.assertIn("Score (dB)", table)
        self.assertNotIn(">Rank</th>", table)
        self.assertIn("Residual dip (dB)", table)

    def test_peq_is_a_local_cut_not_broadband_attenuation(self):
        frequencies = np.asarray([25.0, 80.0, 150.0])
        response_db = db20(peq_response(frequencies, 4000.0, 80.0, 4.0, -6.0))
        self.assertAlmostEqual(response_db[1], -6.0, places=6)
        self.assertGreater(response_db[0], -1.0)
        self.assertGreater(response_db[2], -1.0)

    def test_flat_eq_obeys_range_boost_and_excess_gd_guard(self):
        frequencies = log_frequency_grid(25.0, 150.0, 48)
        magnitude_db = 6.0 * np.log2(frequencies / 55.0)
        spectrum = 10.0 ** (magnitude_db / 20.0)
        options = EqOptions(
            target="flat",
            correction_range=(30.0, 90.0),
            correction_slope_db_per_octave=48.0,
            max_boost_db=6.0,
            low_shelf=False,
        )
        filters, response, metadata = fit_eq_filters(
            spectrum,
            frequencies,
            4000.0,
            48,
            np.zeros_like(frequencies),
            options,
        )
        self.assertTrue(any(item["gain_db"] > 0.0 for item in filters))
        self.assertTrue(
            all(item["q"] <= 1.0 for item in filters if item["gain_db"] > 0.0)
        )
        self.assertTrue(all(30.0 <= item["fc_hz"] <= 90.0 for item in filters))
        self.assertLessEqual(float(np.max(db20(response))), 6.001)
        self.assertLess(metadata["eq_authority"][0], 1.0)
        self.assertLess(metadata["eq_authority"][-1], 0.1)

        one_cycle_excess_ms = 1000.0 / frequencies
        guarded_filters, _, guarded_metadata = fit_eq_filters(
            spectrum,
            frequencies,
            4000.0,
            48,
            one_cycle_excess_ms,
            options,
        )
        self.assertEqual(guarded_filters, [])
        self.assertLess(float(np.max(guarded_metadata["eq_authority"])), 0.02)

        disabled_filters, disabled_response, _ = fit_eq_filters(
            spectrum,
            frequencies,
            4000.0,
            48,
            np.zeros_like(frequencies),
            EqOptions(
                target="flat",
                correction_range=(30.0, 90.0),
                max_boost_db=6.0,
                max_filters=0,
            ),
        )
        self.assertEqual(disabled_filters, [])
        np.testing.assert_allclose(disabled_response, 1.0)
        with self.assertRaisesRegex(ValueError, "between 0 and 16"):
            EqOptions(max_filters=17)

    def test_dsp_eq_target_fits_the_same_flat_curve_as_flat(self):
        frequencies = log_frequency_grid(25.0, 150.0, 48)
        magnitude_db = 6.0 * np.log2(frequencies / 55.0)
        spectrum = 10.0 ** (magnitude_db / 20.0)
        common = dict(
            correction_range=(30.0, 90.0),
            correction_slope_db_per_octave=48.0,
            max_boost_db=6.0,
        )
        flat_filters, flat_response, flat_metadata = fit_eq_filters(
            spectrum,
            frequencies,
            4000.0,
            48,
            np.zeros_like(frequencies),
            EqOptions(target="flat", **common),
        )
        dsp_filters, dsp_response, dsp_metadata = fit_eq_filters(
            spectrum,
            frequencies,
            4000.0,
            48,
            np.zeros_like(frequencies),
            EqOptions(target="dsp", **common),
        )
        self.assertEqual(dsp_metadata["target_level_db"], flat_metadata["target_level_db"])
        np.testing.assert_allclose(
            dsp_metadata["nominal_target_db"], flat_metadata["nominal_target_db"]
        )
        self.assertEqual(len(dsp_filters), len(flat_filters))
        np.testing.assert_allclose(db20(dsp_response), db20(flat_response), atol=1e-9)

    def test_denoised_residual_suppresses_spikes_but_keeps_broad_dips(self):
        ppo = 48
        residual = np.zeros(200)
        spike_index = 100
        residual[spike_index] = -6.0
        plateau_start = 40
        plateau_end = plateau_start + int(round(ppo / 6))
        residual[plateau_start:plateau_end] = -6.0
        smoothed = _denoised_residual(residual, ppo)
        self.assertLess(abs(float(smoothed[spike_index])), 3.0)
        plateau_centre = (plateau_start + plateau_end) // 2
        self.assertGreater(abs(float(smoothed[plateau_centre])), 5.0)

    def test_eq_fitter_targets_broad_bump_not_isolated_noise_spike(self):
        frequencies = log_frequency_grid(25.0, 150.0, 48)
        magnitude_db = np.zeros_like(frequencies)
        spike_index = int(np.argmin(np.abs(frequencies - 130.0)))
        magnitude_db[spike_index] = 6.0
        bump_centre = int(np.argmin(np.abs(frequencies - 60.0)))
        half_width = max(4, int(round(48 / 12)))
        magnitude_db[bump_centre - half_width : bump_centre + half_width] = 4.0
        spectrum = 10.0 ** (magnitude_db / 20.0)
        options = EqOptions(
            target="flat",
            correction_range=(25.0, 150.0),
            correction_slope_db_per_octave=0.0,
            max_boost_db=0.0,
            max_filters=1,
        )
        filters, _, _ = fit_eq_filters(
            spectrum,
            frequencies,
            4000.0,
            48,
            np.zeros_like(frequencies),
            options,
        )
        self.assertEqual(len(filters), 1)
        self.assertLess(
            abs(filters[0]["fc_hz"] - frequencies[bump_centre]) / frequencies[bump_centre], 0.15
        )

    def test_excess_gd_authority_uses_a_broad_smooth_gate(self):
        frequencies = log_frequency_grid(25.0, 150.0, 48)
        excess_ms = np.zeros_like(frequencies)
        centre = int(np.argmin(np.abs(frequencies - 65.0)))
        # A deliberately narrow, one-cycle delay spike must be expanded into a
        # stable correction region rather than copied into the EQ target.
        excess_ms[centre - 1 : centre + 2] = 1000.0 / frequencies[
            centre - 1 : centre + 2
        ]
        authority = _excess_gd_authority(frequencies, excess_ms)
        self.assertLess(float(authority[centre]), 0.5)
        self.assertLess(float(np.max(np.abs(np.diff(authority)))), 0.12)
        half_octave = int(round(48 / 4))
        self.assertLess(float(authority[centre - half_octave]), 0.95)
        self.assertLess(float(authority[centre + half_octave]), 0.95)

    def test_excess_gd_authority_gates_narrow_and_wide_peaks_of_equal_height_alike(self):
        # A moving-average-based risk measure dilutes a peak in proportion
        # to how much narrower it is than the averaging window, so a
        # genuinely severe but narrow excess-GD spike could end up almost
        # entirely ignored while a wider bump of the very same peak height
        # was heavily gated. Authority should instead track true peak
        # height, not width.
        frequencies = log_frequency_grid(25.0, 150.0, 48)
        centre = int(np.argmin(np.abs(frequencies - 65.0)))
        cycles_height = 0.6

        def make(width_bins: int) -> np.ndarray:
            excess_ms = np.zeros_like(frequencies)
            half = max(0, width_bins // 2)
            lo, hi = centre - half, centre + half + 1
            excess_ms[lo:hi] = cycles_height * 1000.0 / frequencies[lo:hi]
            return excess_ms

        narrow_authority = float(np.min(_excess_gd_authority(frequencies, make(1))))
        wide_authority = float(np.min(_excess_gd_authority(frequencies, make(16))))
        self.assertLess(narrow_authority, 0.5)
        self.assertLess(wide_authority, 0.5)
        self.assertLess(abs(narrow_authority - wide_authority), 0.2)

    def test_excess_gd_score_is_limited_to_integration_range(self):
        sample_rate = 4000.0
        n_fft = 8192
        fft_frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        evaluation_frequencies = log_frequency_grid(25.0, 150.0, 48)
        # Flat magnitude makes the minimum-phase counterpart zero phase. Put
        # phase storage well above the requested 30--90 Hz correction range.
        phase = 2.5 * np.exp(
            -0.5 * (np.log2(np.maximum(fft_frequencies, 1e-6) / 125.0) / 0.08) ** 2
        )
        spectrum = np.exp(1j * phase)
        full_score, _ = excess_group_delay(
            spectrum, fft_frequencies, evaluation_frequencies
        )
        limited_score, _ = excess_group_delay(
            spectrum,
            fft_frequencies,
            evaluation_frequencies,
            integration_range=(30.0, 90.0),
        )
        self.assertGreater(full_score, 0.5)
        self.assertLess(limited_score, full_score * 0.01)

    def test_gd_smoothing_octaves_grows_toward_dc_and_vanishes_up_high(self):
        frequencies = np.array([15.0, 25.0, 40.0, 100.0, 150.0])
        sigma = gd_smoothing_octaves(frequencies, native_resolution_hz=1.0)
        self.assertTrue(np.all(np.diff(sigma) < 0.0))  # strictly decreasing with frequency
        self.assertGreater(float(sigma[0]), float(sigma[-1]) * 4.0)
        self.assertTrue(
            np.array_equal(
                gd_smoothing_octaves(frequencies, native_resolution_hz=0.0),
                np.zeros_like(frequencies),
            )
        )

    def test_smooth_by_variable_octaves_is_identity_at_zero_sigma(self):
        rng = np.random.default_rng(1)
        values = rng.standard_normal(200)
        smoothed = _smooth_by_variable_octaves(values, ppo=48, sigma_octaves=np.zeros_like(values))
        np.testing.assert_allclose(smoothed, values)

    def test_smooth_by_variable_octaves_matches_fixed_sigma_at_a_ladder_rung(self):
        rng = np.random.default_rng(2)
        values = rng.standard_normal(200)
        from scipy import ndimage

        expected = ndimage.gaussian_filter1d(values, sigma=1.0 * 48, mode="nearest", truncate=3.0)
        smoothed = _smooth_by_variable_octaves(values, ppo=48, sigma_octaves=np.full_like(values, 1.0))
        np.testing.assert_allclose(smoothed, expected)

    def test_excess_group_delay_native_resolution_smooths_subbass_noise_more_than_treble(self):
        sample_rate = 4000.0
        n_fft = 8192
        fft_frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        evaluation_frequencies = log_frequency_grid(15.0, 150.0, 48)
        rng = np.random.default_rng(0)
        # Zero-mean, per-native-bin phase noise: a stand-in for ordinary
        # measurement noise, with no genuine broadband GD feature at all.
        noise = 0.03 * rng.standard_normal(fft_frequencies.size)
        spectrum = np.exp(1j * noise)
        _, unsmoothed = excess_group_delay(spectrum, fft_frequencies, evaluation_frequencies)
        _, smoothed = excess_group_delay(
            spectrum,
            fft_frequencies,
            evaluation_frequencies,
            native_resolution_hz=1.0,
            ppo=48,
        )
        low_mask = evaluation_frequencies < 25.0
        high_mask = evaluation_frequencies > 90.0
        low_reduction = float(np.std(unsmoothed[low_mask]) / np.std(smoothed[low_mask]))
        high_reduction = float(np.std(unsmoothed[high_mask]) / np.std(smoothed[high_mask]))
        self.assertGreater(low_reduction, 3.0)
        self.assertGreater(low_reduction, high_reduction * 2.0)

    def test_excess_group_delay_native_resolution_preserves_a_genuine_broad_feature(self):
        sample_rate = 4000.0
        n_fft = 8192
        fft_frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        evaluation_frequencies = log_frequency_grid(15.0, 150.0, 48)
        # A real, roughly one-octave-wide phase-storage feature, well clear
        # of the band edge so edge-derivative artifacts don't confound it.
        genuine_phase = 2.5 * np.exp(
            -0.5 * (np.log2(np.maximum(fft_frequencies, 1e-6) / 40.0) / 0.4) ** 2
        )
        spectrum = np.exp(1j * genuine_phase)
        unsmoothed_score, unsmoothed_curve = excess_group_delay(
            spectrum, fft_frequencies, evaluation_frequencies
        )
        smoothed_score, smoothed_curve = excess_group_delay(
            spectrum,
            fft_frequencies,
            evaluation_frequencies,
            native_resolution_hz=1.0,
            ppo=48,
        )
        self.assertGreater(
            float(np.max(np.abs(smoothed_curve))), 0.5 * float(np.max(np.abs(unsmoothed_curve)))
        )
        self.assertGreater(smoothed_score, 0.7 * unsmoothed_score)

    def test_excess_gd_tail_ms_is_shape_neutral_for_equal_area(self):
        # A narrow, tall spike and a wider, shallower bump of the same area
        # (peak height x width) should score similarly: neither a naive
        # peak detector (which only sees the narrow one) nor a percentile
        # (which is blind to anything narrower than its own width cutoff,
        # however severe) achieves that.
        frequencies = log_frequency_grid(25.0, 150.0, 48)
        area = 40.0  # height_ms * width_bins, held constant

        def make(width_bins: int) -> np.ndarray:
            excess_ms = np.zeros_like(frequencies)
            excess_ms[:width_bins] = area / width_bins
            return excess_ms

        narrow_tail = excess_gd_tail_ms(make(2), frequencies)
        wide_tail = excess_gd_tail_ms(make(32), frequencies)
        self.assertGreater(narrow_tail, 0.0)
        self.assertGreater(wide_tail, 0.0)
        self.assertLess(abs(narrow_tail - wide_tail) / max(narrow_tail, wide_tail), 0.5)

    def test_excess_gd_tail_ms_matches_a_uniform_band(self):
        frequencies = log_frequency_grid(25.0, 150.0, 48)
        excess_ms = np.full_like(frequencies, 2.0)
        self.assertAlmostEqual(
            excess_gd_tail_ms(excess_ms, frequencies), 2.0, places=2
        )

    def test_excess_gd_tail_ms_respects_integration_range(self):
        frequencies = log_frequency_grid(25.0, 150.0, 48)
        excess_ms = np.zeros_like(frequencies)
        centre = int(np.argmin(np.abs(frequencies - 125.0)))
        half_width = max(1, int(round(frequencies.size * 0.06)))
        excess_ms[centre - half_width : centre + half_width] = 5.0
        full_tail = excess_gd_tail_ms(excess_ms, frequencies)
        limited_tail = excess_gd_tail_ms(
            excess_ms, frequencies, integration_range=(30.0, 90.0)
        )
        self.assertGreater(full_tail, 0.0)
        self.assertEqual(limited_tail, 0.0)

    def test_excess_gd_peak_ms_is_width_invariant_unlike_the_tail(self):
        # Same total area as the tail's shape-neutrality test, but the peak
        # metric should track peak *height*, not area: a narrow, tall spike
        # must score far higher than a wide, shallow bump of equal area,
        # which is exactly the opposite property from excess_gd_tail_ms.
        frequencies = log_frequency_grid(25.0, 150.0, 48)
        area = 40.0

        def make(width_bins: int) -> np.ndarray:
            excess_ms = np.zeros_like(frequencies)
            excess_ms[:width_bins] = area / width_bins
            return excess_ms

        narrow_peak = excess_gd_peak_ms(make(2), frequencies)
        wide_peak = excess_gd_peak_ms(make(32), frequencies)
        self.assertGreater(narrow_peak, 3.0 * wide_peak)

    def test_excess_gd_peak_ms_matches_equal_height_regardless_of_width(self):
        # Unlike area, peak height alone should not depend on feature width:
        # a narrow spike and a wide plateau of the same height score alike.
        frequencies = log_frequency_grid(25.0, 150.0, 48)

        def make(width_bins: int, height_ms: float) -> np.ndarray:
            excess_ms = np.zeros_like(frequencies)
            excess_ms[:width_bins] = height_ms
            return excess_ms

        narrow = excess_gd_peak_ms(make(2, 5.0), frequencies)
        wide = excess_gd_peak_ms(make(32, 5.0), frequencies)
        self.assertAlmostEqual(narrow, wide, delta=0.5)

    def test_excess_gd_peak_ms_matches_a_uniform_band(self):
        frequencies = log_frequency_grid(25.0, 150.0, 48)
        excess_ms = np.full_like(frequencies, 2.0)
        self.assertAlmostEqual(excess_gd_peak_ms(excess_ms, frequencies), 2.0, places=2)

    def test_excess_gd_peak_ms_respects_integration_range(self):
        frequencies = log_frequency_grid(25.0, 150.0, 48)
        excess_ms = np.zeros_like(frequencies)
        centre = int(np.argmin(np.abs(frequencies - 125.0)))
        excess_ms[centre] = 9.0
        full_peak = excess_gd_peak_ms(excess_ms, frequencies)
        limited_peak = excess_gd_peak_ms(
            excess_ms, frequencies, integration_range=(30.0, 90.0)
        )
        self.assertGreater(full_peak, 0.0)
        self.assertEqual(limited_peak, 0.0)

    def test_excess_gd_peak_ms_is_symmetric_in_sign(self):
        frequencies = log_frequency_grid(25.0, 150.0, 48)
        positive = np.zeros_like(frequencies)
        positive[10] = 6.0
        negative = -positive
        self.assertAlmostEqual(
            excess_gd_peak_ms(positive, frequencies),
            excess_gd_peak_ms(negative, frequencies),
        )

    def test_rejects_mismatched_lengths(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows = []
            for index, length in enumerate([100, 100, 101], start=1):
                rows.append(
                    {
                        "source_index": index,
                        "title": str(index),
                        "uuid": str(index),
                        "sample_rate": 48000,
                        "impulse": np.zeros(length),
                    }
                )
            with self.assertRaisesRegex(CacheError, "refusing to zero-pad"):
                write_cache(Path(temporary), rows, {})

    def test_write_cache_accepts_exactly_two_measurements(self):
        # With only two candidate positions there's exactly one possible
        # pair -- nothing to pick between -- but the pipeline should still
        # run rather than requiring a third, unrelated measurement just to
        # clear a count.
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            rows = [
                {
                    "source_index": index,
                    "title": f"sub {label}",
                    "uuid": str(index),
                    "sample_rate": 4000.0,
                    "impulse": _synthetic_ir(4000.0, 2048, 100, [(50, 0.2)]),
                }
                for index, label in enumerate(["L", "R"], start=1)
            ]
            write_cache(cache, rows, {"test": True})
            measurements, _manifest = load_cache(cache)
            self.assertEqual(len(measurements), 2)

    def test_write_cache_rejects_a_single_measurement(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows = [
                {
                    "source_index": 1,
                    "title": "sub L",
                    "uuid": "1",
                    "sample_rate": 4000.0,
                    "impulse": _synthetic_ir(4000.0, 2048, 100, [(50, 0.2)]),
                }
            ]
            with self.assertRaisesRegex(CacheError, "[Aa]t least 2"):
                write_cache(Path(temporary), rows, {})

    def test_fetch_count_and_indices_accept_exactly_two(self):
        parser = _build_parser()
        parsed = parser.parse_args(["fetch", "--count", "2"])
        self.assertEqual(parsed.count, 2)
        with self.assertRaises(SystemExit):
            parser.parse_args(["fetch", "--count", "1"])
        self.assertEqual(_parse_indices("3,5"), [3, 5])
        with self.assertRaises(ValueError):
            _parse_indices("3")

    def test_analysis_context_rejects_large_relative_start_time_offsets(self):
        from subpair.cache import CachedMeasurement
        from subpair.dsp import AnalysisContext

        sample_rate = 4000.0
        length = 4096
        rows = []
        for index, start in enumerate([0.0, 0.0, 2.5], start=1):
            rows.append(
                CachedMeasurement(
                    position=index,
                    source_index=index,
                    title=f"Position {index}",
                    uuid=f"uuid-{index}",
                    sample_rate=sample_rate,
                    start_time_seconds=start,
                    impulse=_synthetic_ir(sample_rate, length, 100, [(50, 0.2)]),
                    metadata={},
                    path=Path("unused"),
                )
            )
        with self.assertRaisesRegex(ValueError, "exceeds the safe zero-padded"):
            AnalysisContext(rows, (25.0, 150.0), 24)

    def test_synthetic_search_and_report(self):
        sample_rate = 4000.0
        length = 4096
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            rows = []
            definitions = [
                (100, [(42, 0.20), (75, 0.10)]),
                (106, [(48, 0.16), (92, 0.10)]),
                (112, [(58, 0.18), (110, 0.08)]),
                (118, [(68, 0.15), (125, 0.10)]),
            ]
            for index, (delay, modes) in enumerate(definitions, start=1):
                rows.append(
                    {
                        "source_index": index,
                        "title": f"Position {index}",
                        "uuid": f"uuid-{index}",
                        "sample_rate": sample_rate,
                        "start_time_seconds": -0.025,
                        "impulse": _synthetic_ir(sample_rate, length, delay, modes),
                    }
                )
            write_cache(cache, rows, {"test": True})
            results_path = cache / "search-results.json"
            result = run_search(
                cache,
                results_path,
                SearchOptions(
                    band=(25.0, 150.0),
                    delay_range_ms=(-2.0, 2.0, 1.0),
                    gain_range_db=(-1.0, 1.0, 1.0),
                    ppo=24,
                    gate_thresholds=_permissive_gates(),
                ),
            )
            self.assertEqual(len(result["pairs"]), 6)
            self.assertEqual([row["rank"] for row in result["pairs"]], list(range(1, 7)))
            self.assertEqual(
                sorted(row["eq_rank"] for row in result["pairs"]), list(range(1, 7))
            )
            # The documented sort key is verdict first, then score, so assert
            # exactly that rather than a flat score ordering: a gate that is not
            # threshold-driven (the delay-scan-boundary reject) cannot be made
            # permissive by _permissive_gates and legitimately reorders pairs.
            verdict_rank = {"accept": 0, "caution": 1, "reject": 2}
            raw_keys = [
                (verdict_rank[row["verdict"]], -row["score_db"])
                for row in result["pairs"]
            ]
            self.assertEqual(raw_keys, sorted(raw_keys))
            eq_keys = [
                (verdict_rank[row["verdict"]], -row["post_eq_score_db"])
                for row in sorted(result["pairs"], key=lambda row: row["eq_rank"])
            ]
            self.assertEqual(eq_keys, sorted(eq_keys))
            loaded = json.loads(results_path.read_text())
            self.assertEqual(loaded["settings"]["ranking"]["raw"][0], "verdict")
            self.assertEqual(loaded["settings"]["ranking"]["raw"][1], "score_db")
            self.assertEqual(loaded["settings"]["ranking"]["eq"][0], "verdict")
            self.assertEqual(
                loaded["settings"]["ranking"]["eq"][1], "post_eq_score_db"
            )
            self.assertEqual(
                loaded["settings"]["ranking"]["excess_gd_range_hz"],
                [25.0, 150.0],
            )
            self.assertEqual(loaded["settings"]["eq"]["max_filters"], 7)
            score_settings = loaded["settings"]["ranking"]["score"]
            self.assertEqual(score_settings["low_end_weight"], 0.5)
            self.assertEqual(score_settings["dip_weight"], 1.0)
            self.assertAlmostEqual(
                score_settings["dip_smoothing_octaves_fwhm"], 1.0 / 3.0
            )
            self.assertIn("headroom", loaded["settings"]["ranking"])
            low_end_settings = loaded["settings"]["ranking"]["low_end_power"]
            self.assertIn("fields", low_end_settings)
            self.assertEqual(low_end_settings["range_hz"][0], 25.0)
            self.assertLessEqual(low_end_settings["range_hz"][1], 100.0)
            self.assertAlmostEqual(
                low_end_settings["amplifier_power_weight_db_per_octave"],
                EXCURSION_POWER_DB_PER_OCTAVE,
            )
            self.assertIn("native_resolution", loaded["settings"])
            low_end_power_keys = (
                "low_end_power_db",
                "relative_low_end_power_db",
                "post_eq_low_end_power_db",
                "post_eq_relative_low_end_power_db",
            )
            for row in result["pairs"]:
                self.assertIn("dip_db", row)
                self.assertIn("post_eq_dip_db", row)
                self.assertIn("score_db", row)
                self.assertIn("post_eq_score_db", row)
                self.assertGreaterEqual(row["dip_db"], 0.0)
                self.assertGreaterEqual(row["post_eq_dip_db"], 0.0)
                self.assertAlmostEqual(
                    row["sound_power_db"],
                    0.5 * row["spl_db"] + 0.5 * row["low_end_power_db"],
                )
                self.assertAlmostEqual(
                    row["score_db"], row["sound_power_db"] - row["dip_db"]
                )
                self.assertAlmostEqual(
                    row["post_eq_sound_power_db"],
                    0.5 * row["post_eq_spl_db"]
                    + 0.5 * row["post_eq_low_end_power_db"],
                )
                self.assertAlmostEqual(
                    row["post_eq_score_db"],
                    row["post_eq_sound_power_db"] - row["post_eq_dip_db"],
                )
                self.assertNotIn("null_score_db", row)
                self.assertNotIn("post_eq_null_score_db", row)
                # --modal was not requested, so the CSD tail is always the
                # source and there is no dB figure to show for it.
                self.assertFalse(row["effective_tail_is_modal"])
                self.assertFalse(row["post_eq_effective_tail_is_modal"])
                self.assertIsNone(row["effective_tail_db"])
                self.assertIsNone(row["post_eq_effective_tail_db"])
                self.assertGreaterEqual(row["excess_gd_peak_ms"], 0.0)
                self.assertGreaterEqual(row["post_eq_excess_gd_peak_ms"], 0.0)
                self.assertGreaterEqual(row["delay_plateau_ms"], 0.0)
                self.assertGreaterEqual(row["gain_plateau_db"], 0.0)
                self.assertEqual(row["delay_ms"], row["tau_robust"])
                self.assertGreaterEqual(row["fragility"], -1e-12)
                self.assertGreaterEqual(row["basin_w03"], 0.0)
                self.assertGreaterEqual(row["basin_w05"], row["basin_w03"])
                self.assertEqual(
                    set(row["worst_case_penalty"]), {"0.5", "1.0", "1.5"}
                )
                # A penalty above the recommended delay, so never negative and
                # monotone in the half-width it is measured over.
                penalties = [
                    row["worst_case_penalty"][key] for key in ("0.5", "1.0", "1.5")
                ]
                self.assertGreaterEqual(min(penalties), 0.0)
                self.assertEqual(penalties, sorted(penalties))
                self.assertIn("objective_db", row["robustness"])
                self.assertTrue(row["robustness"]["geometry_conservative_bound"])
                self.assertAlmostEqual(
                    row["robustness"]["delta_tau_max_ms"], 1000.0 * 0.5 / 343.0
                )
                for key in low_end_power_keys:
                    self.assertIn(key, row)
                    self.assertTrue(np.isfinite(row[key]))
                self.assertAlmostEqual(
                    row["headroom_db"], -max(0.0, row["gain_db"])
                )
                self.assertLessEqual(
                    row["post_eq_headroom_db"], row["headroom_db"]
                )
                self.assertAlmostEqual(
                    row["relative_spl_db"],
                    row["spl_db"] - result["pairs"][0]["spl_db"],
                )
            self.assertAlmostEqual(
                result["pairs"][0]["relative_low_end_power_db"], 0.0
            )
            self.assertAlmostEqual(result["pairs"][0]["relative_score_db"], 0.0)
            eq_first = min(result["pairs"], key=lambda row: row["eq_rank"])
            self.assertAlmostEqual(eq_first["post_eq_relative_score_db"], 0.0)
            self.assertAlmostEqual(eq_first["post_eq_relative_low_end_power_db"], 0.0)
            for row in result["pairs"]:
                self.assertAlmostEqual(
                    row["post_eq_relative_spl_db"],
                    row["post_eq_spl_db"] - eq_first["post_eq_spl_db"],
                )

            # Final raw/post-EQ sums carry the serialized headroom gains.
            # The complete EQ transfer, including preamp, therefore cannot
            # exceed 0 dB anywhere on the analyzed grid.
            measurements, _ = load_cache(cache)
            context = AnalysisContext(measurements, (25.0, 150.0), 24)
            probe = max(result["pairs"], key=lambda row: row["gain_db"])
            diagnostic = pair_diagnostics(
                context,
                int(probe["first"]) - 1,
                int(probe["second"]) - 1,
                int(probe["polarity"]),
                float(probe["delay_ms"]),
                float(probe["gain_db"]),
                include_decay=True,
                eq_options=EqOptions(correction_range=(25.0, 150.0)),
            )
            self.assertAlmostEqual(diagnostic["headroom_db"], probe["headroom_db"])
            self.assertAlmostEqual(
                diagnostic["post_eq_headroom_db"], probe["post_eq_headroom_db"]
            )
            combined_eq_db = np.asarray(diagnostic["post_eq_db"]) - np.asarray(
                diagnostic["sum_db"]
            )
            self.assertLessEqual(float(np.max(combined_eq_db)), 1e-9)
            self.assertEqual(
                result["settings"]["eq"]["shelf"],
                {
                    "enabled": True,
                    "automatic": True,
                    "counts_toward_max_filters": True,
                    "note": result["settings"]["eq"]["shelf"]["note"],
                },
            )
            self.assertTrue(
                all(
                    row["eq_filter_count"]
                    == len(row["filters"]) + int(row["eq_shelf"]["active"])
                    <= result["settings"]["eq"]["max_filters"]
                    for row in result["pairs"]
                )
            )

            no_shelf_results_path = cache / "search-results-no-shelf.json"
            no_shelf_result = run_search(
                cache,
                no_shelf_results_path,
                SearchOptions(
                    band=(25.0, 150.0),
                    delay_range_ms=(-2.0, 2.0, 1.0),
                    gain_range_db=(-1.0, 1.0, 1.0),
                    ppo=24,
                    low_shelf=False,
                    gate_thresholds=_permissive_gates(),
                ),
            )
            self.assertFalse(no_shelf_result["settings"]["eq"]["shelf"]["enabled"])
            self.assertTrue(
                all(not row["eq_shelf"]["active"] for row in no_shelf_result["pairs"])
            )

            no_shelf_report = root / "report-no-shelf.html"
            shelf_report = root / "report-shelf.html"
            build_report(
                cache, no_shelf_results_path, no_shelf_report, top=2, limit=3
            )
            build_report(cache, results_path, shelf_report, top=2, limit=3)
            no_shelf_page = no_shelf_report.read_text()
            shelf_page = shelf_report.read_text()

            def ranking_tables(page: str) -> list[str]:
                return re.findall(
                    r'<table id="ranking-(?:raw|eq)".*?</table>', page, flags=re.DOTALL
                )

            self.assertEqual(len(ranking_tables(no_shelf_page)), 1)
            self.assertNotIn("LS Fc", no_shelf_page)
            self.assertIn("Automatic low shelf enabled", shelf_page)

            report = root / "report.html"
            build_report(cache, results_path, report, top=2, limit=3)
            first_render = report.read_bytes()
            build_report(cache, results_path, report, top=2, limit=3)
            self.assertEqual(first_render, report.read_bytes())
            page = first_render.decode()
            self.assertIn("plotly.js", page.lower())
            self.assertIn("id=\"ranking-eq\"", page)
            self.assertNotIn("id=\"ranking-raw\"", page)
            self.assertIn("id=\"selected-pairs-magnitude-eq\"", page)
            self.assertIn("id=\"selected-pairs-excess-eq\"", page)
            self.assertNotIn("id=\"selected-pairs-magnitude-raw\"", page)
            self.assertNotIn("setReportMode", page)
            self.assertIn("setOverviewView('magnitude')", page)
            self.assertIn("setOverviewView('excess')", page)
            self.assertIn("const excessGdLowerLimitMs=-20;", page)
            self.assertIn("function updateSharedYAxisRanges()", page)
            self.assertIn("low=Math.max(excessGdLowerLimitMs,low);", page)
            self.assertIn("data-pair-tabs", page)
            self.assertIn("Hotkeys 1–9", page)
            self.assertIn("aria-keyshortcuts", page)
            self.assertIn("document.addEventListener('keydown'", page)
            self.assertIn("activatePair(key);\n});", page)
            self.assertIn("showing up to 3 pairs", page)
            self.assertEqual(page.count('class="pair-select"'), 3)
            self.assertEqual(page.count(" checked aria-label"), 2)
            table_pair_keys = set(
                re.findall(
                    r'class="pair-select"[^>]*data-pair-key="([^"]+)"', page
                )
            )
            detail_pair_keys = set(
                re.findall(
                    r'class="pair-detail(?: is-inactive)?" data-pair-key="([^"]+)"',
                    page,
                )
            )
            self.assertEqual(detail_pair_keys, table_pair_keys)
            self.assertEqual(page.count("data-magnitude-min="), len(detail_pair_keys))
            self.assertEqual(page.count("data-excess-min="), len(detail_pair_keys))
            self.assertIn('"visible":"legendonly"', page)
            self.assertNotIn("Variable smoothed", page)
            self.assertNotIn("Nominal flat target", page)
            self.assertNotIn("1-oct trend", page)
            self.assertIn("peak ", page)
            self.assertIn("Combined EQ response (all bands)", page)
            self.assertIn('"shape":"spline"', page)
            self.assertIn("EQ authority", page)
            self.assertIn("background:hsla(", page)
            visible_eq_pairs = sorted(
                result["pairs"], key=lambda row: row["eq_rank"]
            )[:3]
            # Every sortable score, gate, and robustness metric is coloured.
            # Gate C, Gate I, and physical status are N/A in this fixture (no
            # arrival metadata), so those three cells per row stay grey.
            expected_colored_metrics = 19 * len(visible_eq_pairs)
            self.assertEqual(
                [
                    table.count("background:hsla(")
                    for table in ranking_tables(page)
                ],
                [expected_colored_metrics],
            )
            self.assertEqual(
                [
                    table.count('class="metric-cell is-empty"')
                    for table in ranking_tables(page)
                ],
                [3 * len(visible_eq_pairs)],
            )
            self.assertNotIn(".plotly-graph-div { width:100% !important; }", page)
            self.assertIn(".overview-panels,#pair-details { position:relative; }", page)
            self.assertIn(
                ".overview-panel.is-inactive,.pair-detail.is-inactive ", page
            )
            self.assertNotIn("Plotly.relayout(plot,{autosize:true})", page)
            self.assertNotIn("function revealPlots", page)
            self.assertNotIn("window.dispatchEvent(new Event('resize'))", page)
            self.assertIn("Low-end power", page)
            self.assertIn("usable-output score", page)
            self.assertIn("Residual dip (dB)", page)
            self.assertIn("Score (dB)", page)
            self.assertIn("Fragility (dB)", page)
            self.assertIn("Basin +0.3 (ms)", page)
            self.assertIn("Delay robustness (lower f is better)", page)
            self.assertIn("detrended mirror axis", page)
            self.assertIn("<summary>Score &amp; metric notes</summary>", page)
            self.assertIn("disqualifier, not a certificate", page)
            self.assertIn("Verdict pipeline.", page)
            self.assertIn("A — redundancy residual.", page)
            self.assertIn("I — improvement localisation.", page)
            self.assertIn("Delay-robustness basis.", page)
            self.assertIn("Robustness columns.", page)
            self.assertIn("Geometry and physical status.", page)
            self.assertIn("Robustness graph.", page)
            self.assertIn("Table colors.", page)
            self.assertIn("green with an inset outline is best and red is worst", page)
            self.assertIn("A residual and D deficit prefer higher values", page)
            # Hover help: every non-obvious column carries its own description,
            # rendered by a self-contained script rather than a title attribute
            # (a title has a browser-imposed delay and cannot be styled) or a
            # CSS pseudo-element (the table scrolls inside an overflow
            # container, which would clip it).
            headers = re.findall(r'<th data-key="([^"]+)"[^>]*>', page)
            documented = set(
                re.findall(r'<th data-key="([^"]+)"[^>]*data-help="[^"]+"', page)
            )
            self.assertTrue(headers)
            self.assertEqual(set(headers) - documented, {"pair"})
            for element in ("th", "td", "li"):
                self.assertEqual(
                    re.findall(rf"<{element}[^>]*\btitle=", page), []
                )
            self.assertIn(".hover-help {", page)          # styled inline
            self.assertIn("tip.className='hover-help'", page)  # built by the inline script
            self.assertIn("pointerover", page)
            # self-contained: nothing is fetched at view time
            self.assertEqual(re.findall(r"<script[^>]+\bsrc=", page), [])
            self.assertEqual(re.findall(r"<link[^>]+\bhref=", page), [])
            self.assertIn(
                'data-key="geometric_pass" data-type="number"', page
            )
            self.assertIn(
                'data-key="physical_status" data-type="number"', page
            )
            self.assertNotIn(">Rank</th>", page)
            self.assertIn("Fitted EQ filters", page)
            self.assertIn("Post-EQ excess GD", page)
            self.assertNotIn("Pre-EQ excess GD", page)
            self.assertIn("zero-referenced excess-GD overlay", page)
            for key in detail_pair_keys:
                self.assertLess(page.index(f'id="decay-{key}"'), page.index(f'id="excess-{key}"'))
            self.assertEqual(page.count('"staticPlot": true'), len(detail_pair_keys))
            self.assertEqual(page.count('"displayModeBar": false'), len(detail_pair_keys))
            self.assertNotIn("#f0abfc", page)
            self.assertGreater(report.stat().st_size, 1_000_000)

            raw_report = root / "report-raw.html"
            build_report(cache, results_path, raw_report, top=2, limit=3, raw=True)
            raw_page = raw_report.read_text()
            self.assertIn('id="ranking-raw"', raw_page)
            self.assertNotIn('id="ranking-eq"', raw_page)
            self.assertIn('id="selected-pairs-magnitude-raw"', raw_page)
            self.assertNotIn("Fitted EQ filters", raw_page)
            self.assertNotIn("Combined EQ response (all bands)", raw_page)
            self.assertNotIn("EQ authority", raw_page)
            self.assertIn("Raw CSD-style decay", raw_page)
            self.assertNotIn("Post-EQ CSD-style decay", raw_page)

    def test_search_and_report_with_exactly_two_measurements(self):
        # Exactly two candidate positions means exactly one possible pair --
        # nothing to rank it against -- but search/report must still run the
        # full pipeline on it rather than requiring a third position solely
        # to clear a minimum count.
        sample_rate = 4000.0
        length = 4096
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            rows = [
                {
                    "source_index": index,
                    "title": f"sub {label}",
                    "uuid": str(index),
                    "sample_rate": sample_rate,
                    "start_time_seconds": -0.025,
                    "impulse": _synthetic_ir(sample_rate, length, delay, [(50, 0.2)]),
                }
                for index, (label, delay) in enumerate(
                    [("L", 100), ("R", 106)], start=1
                )
            ]
            write_cache(cache, rows, {"test": True})
            results_path = cache / "search-results.json"
            result = run_search(
                cache,
                results_path,
                SearchOptions(
                    band=(25.0, 150.0),
                    delay_range_ms=(-2.0, 2.0, 1.0),
                    gain_range_db=(-1.0, 1.0, 1.0),
                    ppo=24,
                    gate_thresholds=_permissive_gates(),
                ),
            )
            self.assertEqual(len(result["pairs"]), 1)
            self.assertEqual(result["pairs"][0]["rank"], 1)
            self.assertEqual(result["pairs"][0]["eq_rank"], 1)
            self.assertAlmostEqual(result["pairs"][0]["relative_score_db"], 0.0)

            report_path = root / "report.html"
            build_report(cache, results_path, report_path, top=5, limit=15)
            self.assertIn("sub L", report_path.read_text())


class ScoreResolutionTests(unittest.TestCase):
    """The table must not present an order finer than the data supports."""

    @staticmethod
    def _cache(cache: Path) -> None:
        sample_rate, length = 4000.0, 4096
        definitions = [
            (100, [(42, 0.20), (75, 0.10)]),
            (106, [(48, 0.16), (92, 0.10)]),
            (112, [(58, 0.18), (110, 0.08)]),
            (118, [(68, 0.15), (125, 0.10)]),
        ]
        write_cache(
            cache,
            [
                {
                    "source_index": index,
                    "title": f"Position {index}",
                    "uuid": f"uuid-{index}",
                    "sample_rate": sample_rate,
                    "start_time_seconds": -0.025,
                    "impulse": _synthetic_ir(sample_rate, length, delay, modes),
                }
                for index, (delay, modes) in enumerate(definitions, start=1)
            ],
            {"test": True},
        )

    def _search(self, cache: Path, margin: float) -> dict:
        return run_search(
            cache,
            cache / f"results-{margin}.json",
            SearchOptions(
                band=(25.0, 150.0),
                delay_range_ms=(-2.0, 2.0, 0.25),
                gain_range_db=(-1.0, 1.0, 0.5),
                ppo=24,
                eq_bands=0,
                score_tie_margin_db=margin,
                gate_thresholds=_permissive_gates(),
            ),
        )

    def test_resolution_is_reported_and_widens_with_the_tie_margin(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache"
            self._cache(cache)
            tight = self._search(cache, 0.0)
            loose = self._search(cache, 0.75)
        by_pair = {(r["first"], r["second"]): r for r in tight["pairs"]}
        for row in tight["pairs"]:
            if not row["optimized"]:
                continue
            # Two non-negative components: grid quantisation plus this pair's
            # excess band-edge sensitivity over the population.
            self.assertIsNotNone(row["score_resolution_db"])
            self.assertGreaterEqual(row["score_resolution_db"], 0.0)
            self.assertGreaterEqual(row["score_quantisation_db"], 0.0)
            self.assertGreaterEqual(
                row["score_resolution_db"], row["score_quantisation_db"]
            )
        # The margin is a flat allowance added to every pair.
        for row in loose["pairs"]:
            if not row["optimized"]:
                continue
            key = (row["first"], row["second"])
            self.assertAlmostEqual(
                row["score_resolution_db"],
                by_pair[key]["score_resolution_db"] + 0.75,
            )
        # Widening the margin can only ever add ties, never remove them.
        def tied(result):
            return {
                (r["first"], r["second"])
                for r in result["pairs"]
                if r.get("score_ties_reference")
            }

        self.assertTrue(tied(tight) <= tied(loose))
        # Every flagged pair really is inside the resolution it was judged by.
        for row in loose["pairs"]:
            if row.get("score_ties_reference"):
                self.assertLessEqual(
                    abs(row["relative_score_db"]), row["score_resolution_db"] + 1e-9
                )

    def test_the_reference_pair_is_not_flagged_as_tied_with_itself(self):
        # It is trivially within its own resolution of itself, which says
        # nothing about whether the ordering below it is supported.
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache"
            self._cache(cache)
            result = self._search(cache, 5.0)
        optimized = [r for r in result["pairs"] if r["optimized"]]
        reference = max(optimized, key=lambda r: r["score_db"])
        self.assertEqual(reference["relative_score_db"], 0.0)
        self.assertIsNone(reference["score_ties_reference"])
        # Pairs that genuinely fall inside the widened resolution are flagged;
        # ones further away than 5 dB legitimately are not.
        for row in optimized:
            if row is reference:
                continue
            self.assertEqual(
                bool(row["score_ties_reference"]),
                abs(row["relative_score_db"]) <= row["score_resolution_db"] + 1e-9,
            )


class GateCorrectionTests(unittest.TestCase):
    """Each gate fires on the pair that is actually defective, not its opposite."""

    def _gate_h_set(self, spread_db: float) -> dict:
        return {"gate_h_band_edge_stability": {"status": "pass", "spread_db": spread_db}}

    def test_band_edge_gate_ignores_a_spread_every_pair_shares(self):
        # Shifting the evaluation band moves every pair by nearly the same
        # amount, because the subs roll off below the band. A raw-spread
        # threshold cutting through that cluster rejects pairs for a property
        # of the score function; the excess over the population must not.
        thresholds = GateThresholds(band_edge_excess_spread_reject_db=1.0)
        gate_sets = [self._gate_h_set(value) for value in (1.75, 1.82, 1.89, 2.04, 2.05, 2.06)]
        _apply_band_edge_population_status(gate_sets, thresholds)
        self.assertTrue(
            all(
                gate_set["gate_h_band_edge_stability"]["status"] == "pass"
                for gate_set in gate_sets
            )
        )
        self.assertAlmostEqual(
            gate_sets[0]["gate_h_band_edge_stability"]["population_median_spread_db"],
            1.965,
        )

    def test_band_edge_gate_still_rejects_a_genuine_outlier(self):
        thresholds = GateThresholds(band_edge_excess_spread_reject_db=1.0)
        gate_sets = [self._gate_h_set(value) for value in (1.8, 1.9, 2.0, 2.1, 9.0)]
        _apply_band_edge_population_status(gate_sets, thresholds)
        statuses = [g["gate_h_band_edge_stability"]["status"] for g in gate_sets]
        self.assertEqual(statuses, ["pass", "pass", "pass", "pass", "reject"])

    def test_band_edge_gate_abstains_with_a_single_scored_pair(self):
        thresholds = GateThresholds(band_edge_excess_spread_reject_db=0.0)
        gate_sets = [self._gate_h_set(50.0)]
        _apply_band_edge_population_status(gate_sets, thresholds)
        self.assertEqual(gate_sets[0]["gate_h_band_edge_stability"]["status"], "pass")
        self.assertEqual(gate_sets[0]["gate_h_band_edge_stability"]["excess_spread_db"], 0.0)

    @staticmethod
    def _two_position_context(second: np.ndarray, ppo: int = 48) -> AnalysisContext:
        sample_rate = 4000.0
        length = 4096
        source = _synthetic_ir(sample_rate, length, 100, [(43.0, 0.20), (91.0, 0.20)])
        rows = [
            {
                "source_index": index,
                "title": title,
                "uuid": str(index),
                "sample_rate": sample_rate,
                "start_time_seconds": -0.025,
                "impulse": impulse,
            }
            for index, (title, impulse) in enumerate(
                [("a", source), ("b", second(source))], start=1
            )
        ]
        temporary = tempfile.mkdtemp()
        cache = Path(temporary)
        write_cache(cache, rows, {"test": True})
        measurements, _manifest = load_cache(cache)
        return AnalysisContext(measurements, (35.0, 150.0), ppo)

    def test_physical_baseline_is_free_to_invert_polarity(self):
        # The Gate C baseline must not be pinned to normal polarity: a polarity
        # flip is free, exact and drift-immune, so scoring every
        # inverted-polarity pair in the configuration it explicitly rejects
        # produces a systematic penalty rather than a diagnostic one.
        def shift(source: np.ndarray) -> np.ndarray:
            out = np.zeros_like(source)
            out[13:] = 0.85 * source[:-13]
            return out

        context = self._two_position_context(shift)
        delays = np.arange(-5.0, 5.0001, 0.25)
        envelope, polarity = _baseline_objective_curve(context, 0, 1, delays, 0.5, 1.0)
        self.assertEqual(envelope.shape, delays.shape)
        self.assertEqual(set(np.unique(polarity)).difference({1, -1}), set())
        # Both polarities are the better choice somewhere in a 10 ms sweep, and
        # the envelope is never worse than normal polarity alone.
        self.assertTrue(np.any(polarity == -1))
        self.assertTrue(np.any(polarity == 1))
        phase = np.exp(
            -2j * np.pi * delays[:, None] * context.trend_frequencies[None, :] / 1000.0
        )
        normal = -np.asarray(
            _score_wide_spectrum(
                context,
                context.trend_spectra[0][None, :]
                + context.trend_spectra[1][None, :] * phase,
                0.5,
                1.0,
            ),
            dtype=np.float64,
        )
        self.assertTrue(np.all(envelope <= normal + 1e-9))

    def test_localization_reports_a_per_bin_mean_that_is_zero_without_improvement(self):
        # The concentration fraction is a ratio of two sums of positive ripple
        # improvement; once both collapse to noise it carries no information,
        # so the gate needs an absolute floor to test against.
        def shift(source: np.ndarray) -> np.ndarray:
            out = np.zeros_like(source)
            out[13:] = 0.85 * source[:-13]
            return out

        context = self._two_position_context(shift)
        physical = context.sum_on_trend_grid(0, 1, 1, 0.0, 0.0)
        identical = _improvement_localization(context, physical, physical, 0.5, 1.0)
        self.assertEqual(identical["mean_ripple_improvement_db"], 0.0)
        self.assertEqual(identical["positive_ripple_improvement_db_sum"], 0.0)

        changed = _improvement_localization(
            context, context.sum_on_trend_grid(0, 1, 1, 0.6, 0.0), physical, 0.5, 1.0
        )
        self.assertGreater(changed["mean_ripple_improvement_db"], 0.0)
        # A per-bin mean keeps the materiality floor independent of band width
        # and points-per-octave.
        self.assertAlmostEqual(
            changed["mean_ripple_improvement_db"] * context.frequencies.size,
            changed["positive_ripple_improvement_db_sum"],
            places=9,
        )


class RoomModeTests(unittest.TestCase):
    def test_hand_computed_single_axial_mode(self):
        # length=3.43 m -> first axial mode at c/(2L) = 343/6.86 = 50 Hz
        # exactly; width/height are short enough that their own first axial
        # modes (343 Hz) fall well outside the requested limit.
        modes = room_mode_frequencies((343.0, 50.0, 50.0), max_frequency_hz=55.0)
        self.assertEqual(len(modes), 1)
        self.assertAlmostEqual(modes[0]["frequency_hz"], 50.0, places=6)
        self.assertEqual(modes[0]["type"], "axial")
        self.assertEqual(modes[0]["indices"], (1, 0, 0))

    def test_classifies_axial_tangential_and_oblique_by_nonzero_index_count(self):
        # A cube with c/(2L) = 343/(2*1.715) = 100 Hz per axis: (1,0,0) is
        # axial, (1,1,0) is tangential, (1,1,1) is oblique.
        side_cm = 171.5
        modes = room_mode_frequencies((side_cm, side_cm, side_cm), max_frequency_hz=175.0)
        by_indices = {mode["indices"]: mode for mode in modes}
        self.assertEqual(by_indices[(1, 0, 0)]["type"], "axial")
        self.assertEqual(by_indices[(1, 1, 0)]["type"], "tangential")
        self.assertEqual(by_indices[(1, 1, 1)]["type"], "oblique")
        self.assertAlmostEqual(by_indices[(1, 0, 0)]["frequency_hz"], 100.0, places=3)
        self.assertAlmostEqual(
            by_indices[(1, 1, 0)]["frequency_hz"], 100.0 * 2**0.5, places=3
        )
        self.assertAlmostEqual(
            by_indices[(1, 1, 1)]["frequency_hz"], 100.0 * 3**0.5, places=3
        )

    def test_modes_are_sorted_and_respect_the_frequency_limit(self):
        modes = room_mode_frequencies((343.0, 400.0, 250.0), max_frequency_hz=120.0)
        frequencies = [mode["frequency_hz"] for mode in modes]
        self.assertEqual(frequencies, sorted(frequencies))
        self.assertTrue(all(f <= 120.0 for f in frequencies))
        self.assertTrue(modes)

    def test_max_order_caps_axis_indices_regardless_of_frequency_limit(self):
        # side_cm chosen so the first axial mode is exactly 100 Hz (as in the
        # classification test above); with a frequency limit generous enough
        # to admit a 4th-order axial mode (400 Hz), the default max_order=3
        # must still exclude it while keeping the 3rd-order one (300 Hz).
        side_cm = 171.5
        modes = room_mode_frequencies((side_cm, side_cm, side_cm), max_frequency_hz=500.0)
        indices = {mode["indices"] for mode in modes}
        self.assertIn((3, 0, 0), indices)
        self.assertNotIn((4, 0, 0), indices)
        self.assertTrue(all(max(idx) <= 3 for idx in indices))

    def test_max_order_can_be_raised_explicitly(self):
        side_cm = 171.5
        modes = room_mode_frequencies(
            (side_cm, side_cm, side_cm), max_frequency_hz=500.0, max_order=5
        )
        indices = {mode["indices"] for mode in modes}
        self.assertIn((4, 0, 0), indices)

    def test_rejects_non_positive_dimensions(self):
        with self.assertRaises(ValueError):
            room_mode_frequencies((0.0, 300.0, 250.0), max_frequency_hz=100.0)

    def test_room_mode_traces_group_one_trace_per_type(self):
        modes = [
            {"frequency_hz": 40.0, "type": "axial", "indices": (1, 0, 0)},
            {"frequency_hz": 60.0, "type": "axial", "indices": (2, 0, 0)},
            {"frequency_hz": 70.0, "type": "tangential", "indices": (1, 1, 0)},
        ]
        traces = _room_mode_traces(modes, (-10.0, 10.0), "vertical")
        names = {trace.name for trace in traces}
        self.assertEqual(names, {"Room mode: axial", "Room mode: tangential"})
        axial = next(trace for trace in traces if trace.name == "Room mode: axial")
        # Two axial modes -> two (x,x,None) segments -> 6 coordinates.
        self.assertEqual(len(axial.x), 6)
        self.assertEqual(list(axial.x[:3]), [40.0, 40.0, None])
        self.assertEqual(list(axial.y[:3]), [-10.0, 10.0, None])

    def test_room_mode_traces_hide_tangential_and_oblique_by_default(self):
        modes = [
            {"frequency_hz": 40.0, "type": "axial", "indices": (1, 0, 0)},
            {"frequency_hz": 70.0, "type": "tangential", "indices": (1, 1, 0)},
            {"frequency_hz": 90.0, "type": "oblique", "indices": (1, 1, 1)},
        ]
        traces = _room_mode_traces(modes, (-10.0, 10.0), "vertical")
        by_name = {trace.name: trace for trace in traces}
        self.assertEqual(by_name["Room mode: axial"].visible, True)
        self.assertEqual(by_name["Room mode: tangential"].visible, "legendonly")
        self.assertEqual(by_name["Room mode: oblique"].visible, "legendonly")

    def test_room_mode_traces_draws_horizontal_segments_for_csd(self):
        modes = [{"frequency_hz": 55.0, "type": "axial", "indices": (1, 0, 0)}]
        traces = _room_mode_traces(modes, (0.0, 500.0), "horizontal")
        self.assertEqual(len(traces), 1)
        self.assertEqual(list(traces[0].x[:3]), [0.0, 500.0, None])
        self.assertEqual(list(traces[0].y[:3]), [55.0, 55.0, None])

    def test_room_mode_traces_empty_for_no_modes_or_no_span(self):
        self.assertEqual(_room_mode_traces(None, (0.0, 10.0), "vertical"), [])
        self.assertEqual(_room_mode_traces([], (0.0, 10.0), "vertical"), [])
        modes = [{"frequency_hz": 40.0, "type": "axial", "indices": (1, 0, 0)}]
        self.assertEqual(_room_mode_traces(modes, None, "vertical"), [])

    def test_cli_parses_room_dimensions(self):
        self.assertEqual(_parse_room_dimensions("345x274x248"), (345.0, 274.0, 248.0))
        self.assertEqual(_parse_room_dimensions("345X274X248"), (345.0, 274.0, 248.0))
        for bad in ("345x274", "345x274x248x1", "axb x c", "0x274x248", "-1x274x248"):
            with self.assertRaises(argparse.ArgumentTypeError):
                _parse_room_dimensions(bad)

    def test_cli_room_argument_wired_into_the_report_parser(self):
        parser = _build_parser()
        parsed = parser.parse_args(["report", "--room", "345x274x248"])
        self.assertEqual(parsed.room, (345.0, 274.0, 248.0))
        self.assertIsNone(parser.parse_args(["report"]).room)
        with self.assertRaises(SystemExit):
            parser.parse_args(["report", "--room", "not-a-room"])

    def _write_small_cache(self, cache: Path) -> dict:
        sample_rate = 4000.0
        length = 4096
        rows = []
        definitions = [
            (100, [(42, 0.20), (75, 0.10)]),
            (106, [(48, 0.16), (92, 0.10)]),
            (112, [(58, 0.18), (110, 0.08)]),
        ]
        for index, (delay, modes) in enumerate(definitions, start=1):
            rows.append(
                {
                    "source_index": index,
                    "title": f"Position {index}",
                    "uuid": f"uuid-{index}",
                    "sample_rate": sample_rate,
                    "start_time_seconds": -0.025,
                    "impulse": _synthetic_ir(sample_rate, length, delay, modes),
                }
            )
        write_cache(cache, rows, {"test": True})
        results_path = cache / "search-results.json"
        return {
            "results_path": results_path,
            "result": run_search(
                cache,
                results_path,
                SearchOptions(
                    band=(25.0, 150.0),
                    delay_range_ms=(-2.0, 2.0, 1.0),
                    gain_range_db=(-1.0, 1.0, 1.0),
                    ppo=24,
                    gate_thresholds=_permissive_gates(),
                ),
            ),
        }

    def test_report_overlays_room_modes_when_requested(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            info = self._write_small_cache(cache)
            output = root / "report.html"
            build_report(
                cache,
                info["results_path"],
                output,
                top=2,
                limit=3,
                room_dimensions_cm=(343.0, 400.0, 250.0),
            )
            page = output.read_text()
            self.assertIn("Room mode: axial", page)
            self.assertIn("Room 343", page)
            # The decay figures must remain non-zoomable but stop being fully
            # static, since a static plot can't take legend clicks either.
            self.assertIn('"fixedrange":true', page)

    def test_report_omits_room_modes_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            info = self._write_small_cache(cache)
            output = root / "report.html"
            build_report(cache, info["results_path"], output, top=2, limit=3)
            page = output.read_text()
            self.assertNotIn("Room mode:", page)
            self.assertNotIn("theoretical rigid-box mode frequencies", page)


if __name__ == "__main__":
    unittest.main()
