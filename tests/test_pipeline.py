from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from subpair.cache import CacheError, write_cache
from subpair.cli import _build_parser
from subpair.engine import SearchOptions, run_search
from subpair.dsp import (
    EqOptions,
    LOW_END_EXTENSION_F3_THRESHOLD_DB,
    ShelfOptions,
    _denoised_residual,
    _excess_gd_authority,
    _isotonic_non_increasing,
    _monotonic_gd_baseline,
    _monotonic_gd_baseline_from_gradient,
    _smooth_by_variable_octaves,
    _two_sided_envelope_db,
    db20,
    excess_gd_peak_ms,
    excess_gd_tail_ms,
    fit_eq_filters,
    excess_group_delay,
    gd_smoothing_octaves,
    gd_weighted_null_score,
    log_frequency_grid,
    low_end_extension_hz,
    low_shelf_response,
    null_scores,
    peq_response,
)
from subpair.html_report import _ranking_table, build_report


def _synthetic_ir(sample_rate: float, length: int, delay: int, modes: list[tuple[float, float]]) -> np.ndarray:
    result = np.zeros(length, dtype=np.float64)
    result[delay] = 1.0
    time = np.arange(length - delay) / sample_rate
    for frequency, amplitude in modes:
        result[delay:] += amplitude * np.sin(2 * np.pi * frequency * time) * np.exp(-time / 0.12)
    return result


class PipelineTests(unittest.TestCase):
    def test_report_result_limit_argument(self):
        parser = _build_parser()
        defaults = parser.parse_args(["report"])
        self.assertEqual(defaults.limit, 15)
        self.assertFalse(defaults.raw)
        self.assertEqual(parser.parse_args(["report", "--limit", "24"]).limit, 24)
        self.assertTrue(parser.parse_args(["report", "--raw"]).raw)

    def test_search_max_cut_and_tie_tolerance_arguments(self):
        parser = _build_parser()
        defaults = parser.parse_args(["search"])
        self.assertEqual(defaults.max_cut, 18.0)
        self.assertEqual(defaults.tie_tolerance_db, 0.0)
        overridden = parser.parse_args(
            ["search", "--max-cut", "24", "--tie-tolerance-db", "0.5"]
        )
        self.assertEqual(overridden.max_cut, 24.0)
        self.assertEqual(overridden.tie_tolerance_db, 0.5)

    def test_search_gd_baseline_argument(self):
        parser = _build_parser()
        self.assertEqual(parser.parse_args(["search"]).gd_baseline, "flat")
        self.assertEqual(
            parser.parse_args(["search", "--gd-baseline", "monotonic"]).gd_baseline,
            "monotonic",
        )
        with self.assertRaises(SystemExit):
            parser.parse_args(["search", "--gd-baseline", "bogus"])

    def test_low_shelf_cli_arguments_on_search_only(self):
        # The shelf is a search-time EQ setting, like --max-boost/--eq-bands
        # - report/verify read whatever a given search-results.json already
        # has baked in and don't take their own --low-shelf-* overrides.
        parser = _build_parser()
        defaults = parser.parse_args(["search"])
        self.assertIsNone(defaults.low_shelf_freq)
        self.assertEqual(defaults.low_shelf_gain, 0.0)
        self.assertEqual(defaults.low_shelf_slope, 1.0)
        overridden = parser.parse_args(
            ["search", "--low-shelf-freq", "40", "--low-shelf-gain", "-4.5"]
        )
        self.assertEqual(overridden.low_shelf_freq, 40.0)
        self.assertEqual(overridden.low_shelf_gain, -4.5)
        with self.assertRaises(SystemExit):
            parser.parse_args(["search", "--low-shelf-gain", "20"])
        for command in ("report", "verify"):
            with self.assertRaises(SystemExit):
                parser.parse_args([command, "--low-shelf-freq", "40"])

    def test_shelf_options_requires_frequency_when_gain_is_nonzero(self):
        with self.assertRaisesRegex(ValueError, "requires a low-shelf frequency"):
            ShelfOptions(gain_db=3.0)
        ShelfOptions()  # inactive default is valid
        ShelfOptions(freq_hz=40.0, gain_db=0.0)  # frequency without gain is inert, not an error

    def test_shelf_options_inactive_response_is_flat(self):
        frequencies = log_frequency_grid(20.0, 150.0, 48)
        response = ShelfOptions().response(frequencies, 4000.0)
        np.testing.assert_allclose(response, 1.0)
        self.assertFalse(ShelfOptions().active)
        self.assertFalse(ShelfOptions(freq_hz=40.0).active)
        self.assertTrue(ShelfOptions(freq_hz=40.0, gain_db=3.0).active)

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

    def test_low_end_extension_hz_reports_band_edge_when_fully_extended(self):
        frequencies = log_frequency_grid(25.0, 150.0, 48)
        flat_trend = np.zeros_like(frequencies)
        self.assertAlmostEqual(low_end_extension_hz(flat_trend, frequencies), 25.0, places=3)

    def test_low_end_extension_hz_matches_a_known_rolloff_corner(self):
        frequencies = log_frequency_grid(25.0, 150.0, 48)
        corner = 60.0
        # Flat above the corner, -6 dB/octave below it: the -3 dB point is
        # exactly half an octave below the corner.
        trend = np.where(
            frequencies >= corner, 0.0, -6.0 * np.log2(corner / frequencies)
        )
        expected = corner * 2.0 ** -0.5
        self.assertLess(abs(low_end_extension_hz(trend, frequencies) - expected) / expected, 0.05)

    def test_low_end_extension_hz_ignores_an_isolated_recoverable_notch(self):
        # A notch is a placement defect the null score already measures on
        # its own terms; it must not by itself read as a loss of low-end
        # extension when the response is otherwise flat well past it.
        frequencies = log_frequency_grid(25.0, 150.0, 48)
        trend = np.zeros_like(frequencies)
        dip_index = int(np.argmin(np.abs(frequencies - 100.0)))
        trend[dip_index - 1 : dip_index + 2] = -5.0  # isolated notch, recovers to 0 on both sides
        self.assertAlmostEqual(low_end_extension_hz(trend, frequencies), 25.0, places=3)

    def test_low_end_extension_hz_flat_to_25hz_with_a_100hz_notch_is_not_100hz(self):
        # The motivating case: flat response down to the band edge with one
        # unrelated notch well above it must report full extension, not the
        # notch's own frequency.
        frequencies = log_frequency_grid(25.0, 150.0, 48)
        trend = np.zeros_like(frequencies)
        notch_index = int(np.argmin(np.abs(frequencies - 100.0)))
        trend[notch_index - 1 : notch_index + 2] = -5.0
        extension = low_end_extension_hz(trend, frequencies)
        self.assertLess(extension, 30.0)
        self.assertLess(extension, frequencies[notch_index] - 20.0)

    def test_low_end_extension_hz_still_finds_a_sustained_rolloff_past_a_notch(self):
        # A genuine, sustained rolloff below a notch must still be found:
        # the envelope must not blind the scan to a real corner just
        # because an unrelated notch sits above it.
        frequencies = log_frequency_grid(25.0, 150.0, 48)
        corner = 60.0
        trend = np.where(frequencies >= corner, 0.0, -6.0 * np.log2(corner / frequencies))
        notch_index = int(np.argmin(np.abs(frequencies - 120.0)))
        trend[notch_index - 1 : notch_index + 2] -= 5.0
        expected = corner * 2.0 ** -0.5
        extension = low_end_extension_hz(trend, frequencies)
        self.assertLess(abs(extension - expected) / expected, 0.05)

    def test_low_end_extension_hz_still_finds_a_permanent_rolloff(self):
        # Unlike a notch, a sustained drop that never recovers is the
        # genuine extension limit and must still be reported close to where
        # it happens, not smoothed away by the envelope.
        frequencies = log_frequency_grid(25.0, 150.0, 48)
        trend = np.where(frequencies < 140.0, -10.0, 0.0)
        extension = low_end_extension_hz(trend, frequencies)
        self.assertLess(abs(extension - 140.0) / 140.0, 0.05)

    def test_low_end_extension_hz_anchors_to_a_mid_band_peak_not_the_top_edge(self):
        # A two-subwoofer sum is routinely bandpass-shaped: it peaks well
        # below the top of the band and rolls off on both sides. Anchoring
        # to the top-of-band sample specifically (an earlier version of this
        # metric) would already be more than the 3 dB threshold below the
        # real peak here, collapsing the whole scan to "no extension"
        # (150.0) despite a perfectly good, well-extended low end. Anchoring
        # to the envelope's own peak must find the real corner instead.
        frequencies = log_frequency_grid(25.0, 150.0, 48)
        peak = 60.0
        # Rises 6 dB/octave below the peak, falls 3 dB/octave above it (a
        # gentler upper slope, like a natural crossover rolloff) - the top
        # edge (150 Hz) ends up ~4.4 dB below the peak, comfortably past the
        # default 3 dB threshold.
        trend = np.where(
            frequencies <= peak,
            -6.0 * np.log2(peak / frequencies),
            -3.0 * np.log2(frequencies / peak),
        )
        top_edge_db = float(trend[-1])
        peak_db = float(np.max(trend))
        self.assertLess(top_edge_db, peak_db - LOW_END_EXTENSION_F3_THRESHOLD_DB)
        expected = peak * 2.0 ** -0.5
        extension = low_end_extension_hz(trend, frequencies)
        self.assertLess(abs(extension - expected) / expected, 0.05)

    def test_ranking_table_renders_none_extension_as_an_empty_gray_cell(self):
        base = {
            "polarity": 1,
            "delay_ms": 0.0,
            "gain_db": 0.0,
            "null_score_db": 1.0,
            "excess_gd_ms": 0.5,
            "raw_tail_ms": 10.0,
            "relative_spl_db": 0.0,
        }
        pairs = [
            {**base, "first": 1, "second": 2, "rank": 1,
             "low_end_extension_f3_hz": 32.0, "low_end_extension_f6_hz": 28.0},
            {**base, "first": 3, "second": 4, "rank": 2,
             "low_end_extension_f3_hz": None, "low_end_extension_f6_hz": 40.0},
        ]
        table = _ranking_table(pairs, "raw", "ranking-raw", {"1-2", "3-4"})
        self.assertIn('data-value="32.0"', table)
        self.assertIn('data-value="Infinity"', table)
        self.assertIn('class="metric-cell is-empty" data-value="Infinity"><', table)
        # The real F6 value for the same (F3-less) row must still render and
        # be colour-scaled normally - a missing F3 must not blank out F6.
        self.assertIn('data-value="40.0"', table)
        self.assertNotIn("None", table)

    def test_two_sided_envelope_db_fills_in_a_narrow_dip_but_not_a_sustained_decline(self):
        frequencies = log_frequency_grid(25.0, 150.0, 48)
        notch = np.zeros_like(frequencies)
        notch_index = int(np.argmin(np.abs(frequencies - 100.0)))
        notch[notch_index - 1 : notch_index + 2] = -5.0
        envelope = _two_sided_envelope_db(notch)
        self.assertAlmostEqual(float(envelope[notch_index]), 0.0, places=6)

        decline = np.where(frequencies < 60.0, -10.0, 0.0)
        envelope = _two_sided_envelope_db(decline)
        np.testing.assert_allclose(envelope, decline)

    def test_banded_sort_key_is_identity_when_tolerance_is_zero(self):
        from subpair.engine import _banded_sort_key

        rows = [{"score": 1.234}, {"score": 1.235}, {"score": 5.0}]
        self.assertEqual(
            _banded_sort_key(rows, "score", 0.0), [1.234, 1.235, 5.0]
        )

    def test_banded_sort_key_groups_near_equal_scores(self):
        from subpair.engine import _banded_sort_key

        rows = [{"score": 1.0}, {"score": 1.05}, {"score": 1.3}]
        bands = _banded_sort_key(rows, "score", 0.2)
        self.assertEqual(bands[0], bands[1])
        self.assertNotEqual(bands[1], bands[2])

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

    def test_null_scores_detects_a_wide_shelf_dip(self):
        # A dip much wider than the ~1-octave trend window is largely
        # absorbed into that trend (the trend just follows it down), so the
        # narrow-only detector badly under-reports a real, sustained 10 dB
        # departure from the rest of the band as roughly half that.
        ppo = 48
        frequencies = log_frequency_grid(20.0, 300.0, ppo)
        magnitude_db = np.zeros_like(frequencies)
        dip_low, dip_high = 40.0, 40.0 * 2.0 ** 2.5  # 2.5-octave-wide dip
        mask = (frequencies >= dip_low) & (frequencies <= dip_high)
        magnitude_db[mask] = -10.0
        spectrum = 10.0 ** (magnitude_db / 20.0)
        score = float(null_scores(spectrum[None, :], frequencies, ppo)[0])
        self.assertGreater(score, 8.0)

    def test_null_scores_does_not_flag_a_smooth_monotonic_rolloff(self):
        # A plain rolloff has no "recovery" side, unlike a real bounded dip,
        # and must not be scored as though its whole range were a null.
        ppo = 48
        frequencies = log_frequency_grid(25.0, 150.0, ppo)
        magnitude_db = np.minimum(
            -6.0 * np.log2(55.0 / np.maximum(frequencies, 1.0)), 0.0
        )
        total_range_db = float(magnitude_db.max() - magnitude_db.min())
        spectrum = 10.0 ** (magnitude_db / 20.0)
        score = float(null_scores(spectrum[None, :], frequencies, ppo)[0])
        self.assertLess(score, 0.5 * total_range_db)

    def test_gd_weighted_null_score_inflates_dips_with_real_excess_gd(self):
        frequencies = log_frequency_grid(25.0, 150.0, 48)
        magnitude_db = np.zeros_like(frequencies)
        trend_db = np.zeros_like(frequencies)
        centre = int(np.argmin(np.abs(frequencies - 65.0)))
        magnitude_db[centre] = -6.0  # a 6 dB dip below the flat trend

        benign_gd = np.zeros_like(frequencies)
        severe_gd = np.zeros_like(frequencies)
        severe_gd[centre - 1 : centre + 2] = 1000.0 / frequencies[
            centre - 1 : centre + 2
        ]

        benign_score = gd_weighted_null_score(magnitude_db, trend_db, frequencies, benign_gd)
        severe_score = gd_weighted_null_score(magnitude_db, trend_db, frequencies, severe_gd)

        # No excess GD: the weighted score matches the plain magnitude dip.
        self.assertAlmostEqual(benign_score, 6.0, places=3)
        # Real excess GD at the same dip inflates its severity.
        self.assertGreater(severe_score, benign_score)
        self.assertGreater(severe_score, 6.0)

        # Excess GD with no accompanying magnitude dip is not scored as one.
        no_dip_magnitude = np.zeros_like(frequencies)
        self.assertEqual(
            gd_weighted_null_score(no_dip_magnitude, trend_db, frequencies, severe_gd),
            0.0,
        )

    def test_gd_weighted_null_score_only_penalises_non_minimum_phase_peaks(self):
        frequencies = log_frequency_grid(25.0, 150.0, 48)
        trend_db = np.zeros_like(frequencies)
        magnitude_db = np.zeros_like(frequencies)
        centre = int(np.argmin(np.abs(frequencies - 65.0)))
        magnitude_db[centre] = 6.0  # a 6 dB peak above the flat trend

        benign_gd = np.zeros_like(frequencies)
        severe_gd = np.zeros_like(frequencies)
        severe_gd[centre - 1 : centre + 2] = 1000.0 / frequencies[
            centre - 1 : centre + 2
        ]

        # A minimum-phase peak (no excess GD) is left alone entirely.
        benign_score = gd_weighted_null_score(magnitude_db, trend_db, frequencies, benign_gd)
        self.assertAlmostEqual(benign_score, 0.0, places=6)

        # The same peak, but with real excess GD (non-minimum-phase, a
        # resonance/ringing signature), is penalised.
        severe_score = gd_weighted_null_score(magnitude_db, trend_db, frequencies, severe_gd)
        self.assertGreater(severe_score, 0.0)

    def test_gd_weighted_null_score_dsp_target_lightly_penalises_min_phase_dips(self):
        frequencies = log_frequency_grid(25.0, 150.0, 48)
        trend_db = np.zeros_like(frequencies)
        dip_db = np.zeros_like(frequencies)
        centre = int(np.argmin(np.abs(frequencies - 65.0)))
        dip_db[centre] = -6.0  # a 6 dB minimum-phase dip

        peak_db = np.zeros_like(frequencies)
        peak_db[centre] = 6.0  # a 6 dB minimum-phase peak

        benign_gd = np.zeros_like(frequencies)
        severe_gd = np.zeros_like(frequencies)
        severe_gd[centre - 1 : centre + 2] = 1000.0 / frequencies[
            centre - 1 : centre + 2
        ]

        # A minimum-phase dip counts for its full depth normally, but only
        # lightly under the 'dsp' target (assumed fully fixable by DSP).
        normal_dip = gd_weighted_null_score(dip_db, trend_db, frequencies, benign_gd)
        dsp_dip = gd_weighted_null_score(
            dip_db, trend_db, frequencies, benign_gd, dsp_target=True
        )
        self.assertAlmostEqual(normal_dip, 6.0, places=3)
        self.assertLess(dsp_dip, normal_dip)
        self.assertGreater(dsp_dip, 0.0)

        # A non-minimum-phase dip (real excess GD) still scores up to
        # roughly the same severity in both targets: 'dsp' mode does not
        # excuse a genuinely unfixable cancellation.
        normal_severe = gd_weighted_null_score(dip_db, trend_db, frequencies, severe_gd)
        dsp_severe = gd_weighted_null_score(
            dip_db, trend_db, frequencies, severe_gd, dsp_target=True
        )
        self.assertLess(abs(normal_severe - dsp_severe) / normal_severe, 0.1)

        # Minimum-phase and non-minimum-phase peaks are unaffected by
        # dsp_target in either direction.
        self.assertEqual(
            gd_weighted_null_score(peak_db, trend_db, frequencies, benign_gd),
            gd_weighted_null_score(
                peak_db, trend_db, frequencies, benign_gd, dsp_target=True
            ),
        )
        self.assertEqual(
            gd_weighted_null_score(peak_db, trend_db, frequencies, severe_gd),
            gd_weighted_null_score(
                peak_db, trend_db, frequencies, severe_gd, dsp_target=True
            ),
        )

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
        full_score, _, _ = excess_group_delay(
            spectrum, fft_frequencies, evaluation_frequencies
        )
        limited_score, _, _ = excess_group_delay(
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
        _, unsmoothed, _ = excess_group_delay(spectrum, fft_frequencies, evaluation_frequencies)
        _, smoothed, _ = excess_group_delay(
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
        unsmoothed_score, unsmoothed_curve, _ = excess_group_delay(
            spectrum, fft_frequencies, evaluation_frequencies
        )
        smoothed_score, smoothed_curve, _ = excess_group_delay(
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

    def test_isotonic_non_increasing_reproduces_an_already_non_increasing_sequence(self):
        values = np.array([5.0, 5.0, 3.0, 3.0, 1.0])
        fit = _isotonic_non_increasing(values, np.ones_like(values))
        np.testing.assert_allclose(fit, values)

    def test_isotonic_non_increasing_pools_a_local_rise(self):
        values = np.array([1.0, 1.0, 5.0, 1.0, 1.0])
        fit = _isotonic_non_increasing(values, np.ones_like(values))
        self.assertTrue(np.all(np.diff(fit) <= 1e-9))
        self.assertLess(fit[2], values[2])
        self.assertGreater(fit[2], values[0])

    def test_monotonic_gd_baseline_is_symmetric_in_sign(self):
        group_delay = np.array([5.0, 4.0, 1.0, 3.0, 1.0])
        weights = np.ones_like(group_delay)
        positive_baseline = _monotonic_gd_baseline(group_delay, weights)
        negative_baseline = _monotonic_gd_baseline(-group_delay, weights)
        np.testing.assert_allclose(positive_baseline, -negative_baseline)

    def test_monotonic_gd_baseline_from_gradient_suppresses_a_lone_edge_outlier(self):
        # np.gradient(..., edge_order=2) can leave a single, sharply
        # elevated sample at the very first (lowest-frequency) point, which
        # an unfiltered PAVA fit adopts completely unfiltered since nothing
        # later in a mostly-flat array is large enough to force a merge.
        rest = 1.0 + 0.02 * np.sin(np.arange(59))  # mild, realistic ripple
        group_delay = np.concatenate([[6.0], rest])  # index 0 is the outlier
        weights = np.ones_like(group_delay)
        unfiltered = _monotonic_gd_baseline(group_delay, weights)
        denoised = _monotonic_gd_baseline_from_gradient(group_delay, weights, ppo=48)
        self.assertAlmostEqual(float(unfiltered[0]), 6.0, places=6)
        self.assertLess(float(denoised[0]), 2.0)
        # A run this short/mild elsewhere shouldn't be dragged around by the
        # denoise either - the tail should stay close to the raw ripple.
        self.assertLess(float(np.max(np.abs(denoised[10:] - rest[9:]))), 0.05)

    def test_monotonic_gd_baseline_from_gradient_still_finds_a_genuine_broad_rise(self):
        # A real, broad low-frequency rise on a realistic (log-spaced) grid
        # - not a single edge sample - must substantially survive the same
        # denoise. A full 1-octave median window (MONOTONIC_BASELINE_DENOISE_
        # OCTAVES) does measurably soften even a smooth, genuine C/f rise
        # (the same "preferentially preserves, not perfectly unaffected"
        # tradeoff as this module's native-resolution smoothing elsewhere),
        # but the shape and magnitude must stay recognizably close.
        frequencies = log_frequency_grid(25.0, 150.0, 48)
        group_delay = 5.0 / frequencies  # smooth, genuine C/f rise
        weights = np.ones_like(group_delay)
        unfiltered = _monotonic_gd_baseline(group_delay, weights)
        denoised = _monotonic_gd_baseline_from_gradient(group_delay, weights, ppo=48)
        self.assertLess(
            float(np.max(np.abs(denoised - unfiltered))) / float(np.max(unfiltered)), 0.2
        )

    def test_excess_group_delay_rejects_unknown_gd_baseline(self):
        fft_frequencies = np.fft.rfftfreq(8192, 1.0 / 4000.0)
        spectrum = np.ones_like(fft_frequencies, dtype=np.complex128)
        evaluation_frequencies = log_frequency_grid(25.0, 150.0, 48)
        with self.assertRaisesRegex(ValueError, "gd_baseline"):
            excess_group_delay(
                spectrum, fft_frequencies, evaluation_frequencies, gd_baseline="bogus"
            )

    def test_excess_group_delay_monotonic_baseline_absorbs_a_genuine_bass_rise(self):
        sample_rate = 4000.0
        n_fft = 8192
        fft_frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        evaluation_frequencies = log_frequency_grid(25.0, 150.0, 48)
        # Flat magnitude, phase = -2*pi*C*ln(f) => group delay = C/f: a smooth
        # rise toward the bottom of the band that gently declines with
        # frequency, the "normal room/port behaviour" case from the README.
        f_safe = np.maximum(fft_frequencies, 1.0)
        time_constant = 0.125
        phase = -2.0 * np.pi * time_constant * np.log(f_safe)
        spectrum = np.exp(1j * phase)
        flat_score, _, _ = excess_group_delay(
            spectrum, fft_frequencies, evaluation_frequencies, gd_baseline="flat"
        )
        monotonic_score, _, monotonic_baseline = excess_group_delay(
            spectrum, fft_frequencies, evaluation_frequencies, gd_baseline="monotonic"
        )
        self.assertGreater(flat_score, 0.5)
        self.assertLess(monotonic_score, 0.25 * flat_score)
        self.assertTrue(np.all(np.diff(np.abs(monotonic_baseline)) <= 1e-6))

    def test_excess_group_delay_monotonic_baseline_still_flags_a_higher_band_bump(self):
        sample_rate = 4000.0
        n_fft = 8192
        fft_frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        evaluation_frequencies = log_frequency_grid(25.0, 150.0, 48)
        # A genuine, roughly one-octave-wide non-minimum-phase bump centred
        # well above the bottom of the band, on top of near-zero phase
        # elsewhere - not explainable by any non-increasing fit starting
        # from ~0 at the bottom of the band.
        bump_phase = 2.5 * np.exp(
            -0.5 * (np.log2(np.maximum(fft_frequencies, 1e-6) / 100.0) / 0.15) ** 2
        )
        spectrum = np.exp(1j * bump_phase)
        flat_score, _, _ = excess_group_delay(
            spectrum, fft_frequencies, evaluation_frequencies, gd_baseline="flat"
        )
        monotonic_score, _, _ = excess_group_delay(
            spectrum, fft_frequencies, evaluation_frequencies, gd_baseline="monotonic"
        )
        self.assertGreater(monotonic_score, 0.5 * flat_score)

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

    def test_run_search_rejects_unknown_gd_baseline(self):
        with self.assertRaisesRegex(ValueError, "gd_baseline"):
            run_search(
                Path("/nonexistent-cache"),
                Path("/nonexistent-cache/out.json"),
                SearchOptions(gd_baseline="bogus"),
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
                ),
            )
            self.assertEqual(len(result["pairs"]), 6)
            self.assertEqual([row["rank"] for row in result["pairs"]], list(range(1, 7)))
            self.assertEqual(
                sorted(row["eq_rank"] for row in result["pairs"]), list(range(1, 7))
            )
            raw_keys = [
                (
                    row["null_score_db"],
                    row["excess_gd_ms"],
                    row["excess_gd_tail_ms"],
                    row["excess_gd_peak_ms"],
                    row["raw_tail_ms"],
                )
                for row in result["pairs"]
            ]
            self.assertEqual(raw_keys, sorted(raw_keys))
            eq_keys = [
                (
                    row["post_eq_null_score_db"],
                    row["post_eq_excess_gd_ms"],
                    row["post_eq_excess_gd_tail_ms"],
                    row["post_eq_excess_gd_peak_ms"],
                    row["post_eq_tail_ms"],
                )
                for row in sorted(result["pairs"], key=lambda row: row["eq_rank"])
            ]
            self.assertEqual(eq_keys, sorted(eq_keys))
            loaded = json.loads(results_path.read_text())
            self.assertEqual(
                loaded["settings"]["ranking"]["raw"][0], "null_score_db"
            )
            self.assertEqual(
                loaded["settings"]["ranking"]["eq"][0],
                "post_eq_null_score_db",
            )
            self.assertEqual(
                loaded["settings"]["ranking"]["excess_gd_range_hz"],
                [25.0, 150.0],
            )
            self.assertEqual(loaded["settings"]["eq"]["max_filters"], 7)
            self.assertEqual(
                loaded["settings"]["ranking"]["magnitude_basis"],
                "raw, unsmoothed",
            )
            self.assertEqual(loaded["settings"]["ranking"]["tie_tolerance_db"], 0.0)
            self.assertIn("low_end_extension", loaded["settings"]["ranking"])
            self.assertIn("native_resolution", loaded["settings"])
            # low_end_extension_f3_hz/f6_hz are diagnostic-only: they must
            # not appear in either ranking's declared sort-key field list.
            for key in ("low_end_extension_f3_hz", "low_end_extension_f6_hz"):
                self.assertNotIn(key, loaded["settings"]["ranking"]["raw"])
            for key in (
                "post_eq_low_end_extension_f3_hz",
                "post_eq_low_end_extension_f6_hz",
            ):
                self.assertNotIn(key, loaded["settings"]["ranking"]["eq"])
            self.assertIn("excess_gd_peak_ms", loaded["settings"]["ranking"]["raw"])
            self.assertIn(
                "post_eq_excess_gd_peak_ms", loaded["settings"]["ranking"]["eq"]
            )
            extension_keys = (
                "low_end_extension_f3_hz",
                "low_end_extension_f6_hz",
                "post_eq_low_end_extension_f3_hz",
                "post_eq_low_end_extension_f6_hz",
            )
            for row in result["pairs"]:
                self.assertIn("magnitude_only_null_score_db", row)
                self.assertIn("post_eq_magnitude_only_null_score_db", row)
                self.assertGreaterEqual(
                    row["null_score_db"], row["magnitude_only_null_score_db"] - 1e-9
                )
                self.assertGreaterEqual(row["excess_gd_peak_ms"], 0.0)
                self.assertGreaterEqual(row["post_eq_excess_gd_peak_ms"], 0.0)
                self.assertGreaterEqual(row["delay_plateau_ms"], 0.0)
                self.assertGreaterEqual(row["gain_plateau_db"], 0.0)
                for key in extension_keys:
                    self.assertIn(key, row)
                    if row[key] is not None:
                        self.assertGreaterEqual(row[key], 25.0)
                        self.assertLessEqual(row[key], 150.0)
            # The shelf is a search-time EQ setting, exactly like max_boost/
            # eq_bands: a different shelf means a different search, and its
            # effect is expected to show up in the post-EQ scores themselves,
            # not just as an unscored report-time overlay.
            shelf_results_path = cache / "search-results-shelf.json"
            shelf_result = run_search(
                cache,
                shelf_results_path,
                SearchOptions(
                    band=(25.0, 150.0),
                    delay_range_ms=(-2.0, 2.0, 1.0),
                    gain_range_db=(-1.0, 1.0, 1.0),
                    ppo=24,
                    low_shelf_freq_hz=40.0,
                    low_shelf_gain_db=4.0,
                ),
            )
            no_shelf_spl = {
                (row["first"], row["second"]): row["post_eq_spl_db"]
                for row in result["pairs"]
            }
            shelf_spl = {
                (row["first"], row["second"]): row["post_eq_spl_db"]
                for row in shelf_result["pairs"]
            }
            self.assertTrue(
                all(shelf_spl[key] > no_shelf_spl[key] + 0.2 for key in no_shelf_spl)
            )
            self.assertEqual(
                shelf_result["settings"]["eq"]["shelf"],
                {
                    "active": True,
                    "freq_hz": 40.0,
                    "gain_db": 4.0,
                    "slope": 1.0,
                    "note": shelf_result["settings"]["eq"]["shelf"]["note"],
                },
            )

            no_shelf_report = root / "report-no-shelf.html"
            shelf_report = root / "report-shelf.html"
            build_report(cache, results_path, no_shelf_report, top=2, limit=3)
            build_report(cache, shelf_results_path, shelf_report, top=2, limit=3)
            no_shelf_page = no_shelf_report.read_text()
            shelf_page = shelf_report.read_text()

            def ranking_tables(page: str) -> list[str]:
                return re.findall(
                    r'<table id="ranking-(?:raw|eq)".*?</table>', page, flags=re.DOTALL
                )

            self.assertEqual(len(ranking_tables(no_shelf_page)), 1)
            self.assertNotIn("LS Fc", no_shelf_page)
            self.assertIn("LS Fc 40.0 Hz  Gain +4.0 dB", shelf_page)
            self.assertIn("part of the score", shelf_page)

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
            self.assertIn('"visible":"legendonly"', page)
            self.assertNotIn("Variable smoothed", page)
            self.assertNotIn("Nominal flat target", page)
            self.assertNotIn("1-oct trend", page)
            self.assertIn("peak ", page)
            self.assertIn("Combined PEQ response (all bands)", page)
            self.assertIn('"shape":"spline"', page)
            self.assertIn("EQ authority", page)
            self.assertIn("background:hsla(", page)
            self.assertTrue(
                all(table.count("background:hsla(") == 18 for table in ranking_tables(page))
            )
            self.assertNotIn(".plotly-graph-div { width:100% !important; }", page)
            self.assertIn(".overview-panels,#pair-details { position:relative; }", page)
            self.assertIn(
                ".overview-panel.is-inactive,.pair-detail.is-inactive ", page
            )
            self.assertNotIn("Plotly.relayout(plot,{autosize:true})", page)
            self.assertNotIn("function revealPlots", page)
            self.assertNotIn("window.dispatchEvent(new Event('resize'))", page)
            self.assertIn("Extension", page)
            self.assertIn("not part of the ranking", page)
            self.assertIn("Fitted PEQ filters", page)
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
            self.assertNotIn("Fitted PEQ filters", raw_page)
            self.assertNotIn("Combined PEQ response (all bands)", raw_page)
            self.assertNotIn("EQ authority", raw_page)
            self.assertIn("Raw CSD-style decay", raw_page)
            self.assertNotIn("Post-EQ CSD-style decay", raw_page)

    def test_synthetic_search_and_report_with_monotonic_gd_baseline(self):
        sample_rate = 4000.0
        length = 4096
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            definitions = [
                (100, [(42, 0.20), (75, 0.10)]),
                (106, [(48, 0.16), (92, 0.10)]),
                (112, [(58, 0.18), (110, 0.08)]),
            ]
            rows = [
                {
                    "source_index": index,
                    "title": f"Position {index}",
                    "uuid": f"uuid-{index}",
                    "sample_rate": sample_rate,
                    "start_time_seconds": -0.025,
                    "impulse": _synthetic_ir(sample_rate, length, delay, modes),
                }
                for index, (delay, modes) in enumerate(definitions, start=1)
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
                    gd_baseline="monotonic",
                ),
            )
            self.assertEqual(len(result["pairs"]), 3)
            loaded = json.loads(results_path.read_text())
            self.assertEqual(loaded["settings"]["gd_baseline"]["mode"], "monotonic")
            for row in result["pairs"]:
                self.assertGreaterEqual(row["excess_gd_peak_ms"], 0.0)
                self.assertGreaterEqual(row["post_eq_excess_gd_peak_ms"], 0.0)
            report_path = root / "report.html"
            build_report(cache, results_path, report_path, top=2, limit=3)
            page = report_path.read_text()
            self.assertIn("monotonic GD baseline", page)


if __name__ == "__main__":
    unittest.main()
