from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from subpair.cache import CacheError, write_cache
from subpair.engine import SearchOptions, run_search
from subpair.dsp import db20, peq_response
from subpair.html_report import build_report


def _synthetic_ir(sample_rate: float, length: int, delay: int, modes: list[tuple[float, float]]) -> np.ndarray:
    result = np.zeros(length, dtype=np.float64)
    result[delay] = 1.0
    time = np.arange(length - delay) / sample_rate
    for frequency, amplitude in modes:
        result[delay:] += amplitude * np.sin(2 * np.pi * frequency * time) * np.exp(-time / 0.12)
    return result


class PipelineTests(unittest.TestCase):
    def test_peq_is_a_local_cut_not_broadband_attenuation(self):
        frequencies = np.asarray([25.0, 80.0, 150.0])
        response_db = db20(peq_response(frequencies, 4000.0, 80.0, 4.0, -6.0))
        self.assertAlmostEqual(response_db[1], -6.0, places=6)
        self.assertGreater(response_db[0], -1.0)
        self.assertGreater(response_db[2], -1.0)

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
            keys = [
                (row["null_score_db"], row["excess_gd_ms"], row["post_eq_tail_ms"])
                for row in result["pairs"]
            ]
            self.assertEqual(keys, sorted(keys))
            loaded = json.loads(results_path.read_text())
            self.assertEqual(loaded["settings"]["ranking"][0], "null_score_db")
            report = root / "report.html"
            build_report(cache, results_path, report, top=2)
            first_render = report.read_bytes()
            build_report(cache, results_path, report, top=2)
            self.assertEqual(first_render, report.read_bytes())
            page = first_render.decode()
            self.assertIn("plotly.js", page.lower())
            self.assertIn("id=\"ranking\"", page)
            self.assertIn("id=\"top-pairs-overview\"", page)
            self.assertIn('"visible":"legendonly"', page)
            self.assertIn('"shape":"spline"', page)
            self.assertIn("background:hsla(", page)
            self.assertIn("Fitted PEQ cuts", page)
            self.assertGreater(report.stat().st_size, 1_000_000)


if __name__ == "__main__":
    unittest.main()
