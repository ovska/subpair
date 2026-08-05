"""Vectorised pair enumeration and deterministic result serialisation."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .cache import load_cache, write_json
from .dsp import AnalysisContext, EqOptions, inclusive_range, null_scores, pair_diagnostics


@dataclass(frozen=True)
class SearchOptions:
    band: tuple[float, float] = (25.0, 150.0)
    delay_range_ms: tuple[float, float, float] = (-10.0, 10.0, 0.1)
    gain_range_db: tuple[float, float, float] = (-3.0, 3.0, 0.5)
    ppo: int = 48
    eq_target: str = "trend"
    eq_range_hz: tuple[float, float] | None = None
    eq_range_slope_db_per_octave: float = 48.0
    max_boost_db: float = 0.0
    eq_bands: int = 7


def _best_configurations(
    context: AnalysisContext,
    first: int,
    second: int,
    delays: np.ndarray,
    gains: np.ndarray,
) -> list[tuple[int, float, float, float]]:
    polarities = np.asarray([1.0, -1.0])
    gain_linear = 10.0 ** (gains / 20.0)
    shifted = context.spectra[second][None, :] * np.exp(
        -2j * np.pi * delays[:, None] * context.frequencies[None, :] / 1000.0
    )
    candidates = (
        context.spectra[first][None, None, None, :]
        + polarities[:, None, None, None]
        * gain_linear[None, None, :, None]
        * shifted[None, :, None, :]
    )
    shape = candidates.shape[:-1]
    scores = null_scores(
        candidates.reshape((-1, candidates.shape[-1])), context.frequencies, context.ppo
    ).reshape(shape)
    minimum = float(np.min(scores))
    # Preserve exact primary-score ties so the expensive second and third
    # stages can resolve them without changing the lexicographic objective.
    flat_indices = np.flatnonzero(scores.reshape(-1) == minimum)
    result = []
    for flat_index in flat_indices:
        polarity_index, delay_index, gain_index = np.unravel_index(int(flat_index), shape)
        result.append(
            (
                int(polarities[polarity_index]),
                float(delays[delay_index]),
                float(gains[gain_index]),
                minimum,
            )
        )
    return result


def _best_configuration(
    context: AnalysisContext,
    first: int,
    second: int,
    delays: np.ndarray,
    gains: np.ndarray,
) -> tuple[int, float, float, float]:
    """Compatibility helper returning the first primary-score optimum."""
    return _best_configurations(context, first, second, delays, gains)[0]


def run_search(
    cache_dir: Path,
    output_path: Path,
    options: SearchOptions,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    measurements, manifest = load_cache(cache_dir)
    delays = inclusive_range(*options.delay_range_ms)
    gains = inclusive_range(*options.gain_range_db)
    context = AnalysisContext(measurements, options.band, options.ppo)
    eq_range = options.eq_range_hz or options.band
    if eq_range[0] < options.band[0] or eq_range[1] > options.band[1]:
        raise ValueError(
            f"EQ range {eq_range[0]:g}..{eq_range[1]:g} Hz must lie within "
            f"analysis band {options.band[0]:g}..{options.band[1]:g} Hz"
        )
    eq_options = EqOptions(
        target=options.eq_target,
        correction_range=eq_range,
        correction_slope_db_per_octave=options.eq_range_slope_db_per_octave,
        max_boost_db=options.max_boost_db,
        max_filters=options.eq_bands,
    )
    combinations = list(itertools.combinations(range(len(measurements)), 2))
    pairs: list[dict] = []
    for ordinal, (first, second) in enumerate(combinations, start=1):
        configurations = _best_configurations(
            context, first, second, delays, gains
        )
        finalists = []
        for polarity, delay_ms, gain_db, null_score in configurations:
            diagnostics = pair_diagnostics(
                context,
                first,
                second,
                polarity,
                delay_ms,
                gain_db,
                include_decay=False,
                eq_options=eq_options,
            )
            # Preserve the exhaustive score exactly; the diagnostic
            # recomputation is equivalent but may differ at the final bit.
            diagnostics["null_score_db"] = null_score
            finalists.append((polarity, delay_ms, gain_db, diagnostics))
        polarity, delay_ms, gain_db, diagnostics = min(
            finalists,
            key=lambda item: (
                item[3]["null_score_db"],
                item[3]["excess_gd_ms"],
                item[3]["raw_tail_ms"],
                0 if item[0] > 0 else 1,
                item[1],
                item[2],
            ),
        )
        pair = {
            "first": first + 1,
            "second": second + 1,
            "first_name": measurements[first].title,
            "second_name": measurements[second].title,
            "polarity": polarity,
            "delay_ms": delay_ms,
            "gain_db": gain_db,
            **diagnostics,
        }
        pairs.append(pair)
        if progress:
            progress(ordinal, len(combinations), f"{first + 1}+{second + 1}")

    pairs.sort(
        key=lambda row: (
            row["null_score_db"],
            row["excess_gd_ms"],
            row["raw_tail_ms"],
            row["first"],
            row["second"],
        )
    )
    reference_spl = pairs[0]["spl_db"]
    for rank, row in enumerate(pairs, start=1):
        row["rank"] = rank
        row["relative_spl_db"] = float(row["spl_db"] - reference_spl)
    eq_pairs = sorted(
        pairs,
        key=lambda row: (
            row["post_eq_null_score_db"],
            row["post_eq_excess_gd_ms"],
            row["post_eq_tail_ms"],
            row["first"],
            row["second"],
        ),
    )
    eq_reference_spl = eq_pairs[0]["post_eq_spl_db"]
    for eq_rank, row in enumerate(eq_pairs, start=1):
        row["eq_rank"] = eq_rank
        row["post_eq_relative_spl_db"] = float(
            row["post_eq_spl_db"] - eq_reference_spl
        )

    result = {
        "format_version": 4,
        "measurement_count": len(measurements),
        "sample_rate": measurements[0].sample_rate,
        "response_length": measurements[0].impulse.size,
        "settings": {
            "band_hz": list(options.band),
            "ppo": options.ppo,
            "delay_range_ms": list(options.delay_range_ms),
            "gain_range_db": list(options.gain_range_db),
            "eq": {
                "target": eq_options.target,
                "correction_range_hz": list(eq_range),
                "correction_slope_db_per_octave": eq_options.correction_slope_db_per_octave,
                "max_boost_db": eq_options.max_boost_db,
                "max_filters": eq_options.max_filters,
                "excess_gd_guard": (
                    "smoothed excess-GD peaks are expanded into broad gates; "
                    "authority falls as gated delay approaches 0.35 cycles"
                ),
            },
            "minimum_phase": {
                "method": "real cepstrum (discrete Hilbert equivalent)",
                "fft_zero_pad_factor": context.minphase_pad_factor,
                "log_magnitude_floor_db": -160.0,
                "bandwidth": "full cached response, DC to Nyquist",
                "common_delay": "energy-weighted median removed from excess GD",
            },
            "ranking": {
                "raw": ["null_score_db", "excess_gd_ms", "raw_tail_ms"],
                "eq": [
                    "post_eq_null_score_db",
                    "post_eq_excess_gd_ms",
                    "post_eq_tail_ms",
                ],
                "excess_gd_range_hz": list(eq_range),
                "magnitude_basis": "raw, unsmoothed",
            },
        },
        "cache_manifest_format": manifest.get("format_version"),
        "pairs": pairs,
    }
    write_json(output_path, result)
    return result
