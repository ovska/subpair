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
    _denoised_residual,
    _excess_gd_authority,
    db20,
    fit_eq_filters,
    excess_group_delay,
    gd_weighted_null_score,
    log_frequency_grid,
    peq_response,
)
from subpair.html_report import build_report


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
        self.assertEqual(parser.parse_args(["report"]).limit, 15)
        self.assertEqual(parser.parse_args(["report", "--limit", "24"]).limit, 24)

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
                (row["null_score_db"], row["excess_gd_ms"], row["raw_tail_ms"])
                for row in result["pairs"]
            ]
            self.assertEqual(raw_keys, sorted(raw_keys))
            eq_keys = [
                (
                    row["post_eq_null_score_db"],
                    row["post_eq_excess_gd_ms"],
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
            for row in result["pairs"]:
                self.assertIn("magnitude_only_null_score_db", row)
                self.assertIn("post_eq_magnitude_only_null_score_db", row)
                self.assertGreaterEqual(
                    row["null_score_db"], row["magnitude_only_null_score_db"] - 1e-9
                )
                self.assertGreaterEqual(row["delay_plateau_ms"], 0.0)
                self.assertGreaterEqual(row["gain_plateau_db"], 0.0)
            report = root / "report.html"
            build_report(cache, results_path, report, top=2, limit=3)
            first_render = report.read_bytes()
            build_report(cache, results_path, report, top=2, limit=3)
            self.assertEqual(first_render, report.read_bytes())
            page = first_render.decode()
            self.assertIn("plotly.js", page.lower())
            self.assertIn("id=\"ranking-raw\"", page)
            self.assertIn("id=\"ranking-eq\"", page)
            self.assertIn("id=\"selected-pairs-magnitude-raw\"", page)
            self.assertIn("id=\"selected-pairs-excess-raw\"", page)
            self.assertIn("id=\"selected-pairs-magnitude-eq\"", page)
            self.assertIn("id=\"selected-pairs-excess-eq\"", page)
            self.assertIn("setReportMode('raw')", page)
            self.assertIn("setReportMode('eq')", page)
            self.assertIn("setOverviewView('raw','magnitude')", page)
            self.assertIn("setOverviewView('eq','excess')", page)
            self.assertIn('data-pair-tabs="raw"', page)
            self.assertIn('data-pair-tabs="eq"', page)
            self.assertIn("Hotkeys 1–9", page)
            self.assertIn("aria-keyshortcuts", page)
            self.assertIn("document.addEventListener('keydown'", page)
            self.assertIn("showing up to 3 pairs per mode", page)
            self.assertEqual(page.count('class="pair-select"'), 6)
            self.assertEqual(page.count(" checked aria-label"), 4)
            table_pair_keys = set(
                re.findall(
                    r'class="pair-select"[^>]*data-pair-key="([^"]+)"', page
                )
            )
            detail_pair_keys = set(
                re.findall(r'class="pair-detail" data-pair-key="([^"]+)"', page)
            )
            self.assertEqual(detail_pair_keys, table_pair_keys)
            self.assertIn('"visible":"legendonly"', page)
            self.assertNotIn("Variable smoothed", page)
            self.assertNotIn("Nominal flat target", page)
            self.assertNotIn("1-oct trend", page)
            self.assertNotIn('"dash":', page)
            self.assertIn("Combined PEQ response (all bands)", page)
            self.assertIn('"shape":"spline"', page)
            self.assertIn("EQ authority", page)
            self.assertIn("background:hsla(", page)
            self.assertIn("Fitted PEQ filters", page)
            self.assertIn("Pre-EQ excess GD", page)
            self.assertIn("Post-EQ excess GD", page)
            self.assertIn("zero-referenced excess-GD overlay", page)
            self.assertEqual(page.count('"staticPlot": true'), len(detail_pair_keys))
            self.assertEqual(page.count('"displayModeBar": false'), len(detail_pair_keys))
            self.assertNotIn("#f0abfc", page)
            self.assertGreater(report.stat().st_size, 1_000_000)


if __name__ == "__main__":
    unittest.main()
