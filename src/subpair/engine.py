"""Vectorised pair enumeration and deterministic result serialisation."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .cache import load_cache, write_json
from .dsp import (
    EXCESS_GD_TAIL_POWER,
    GD_BASELINE_MODES,
    LOW_END_EXTENSION_THRESHOLD_DB,
    MIN_RELIABLE_NATIVE_BINS,
    AnalysisContext,
    EqOptions,
    inclusive_range,
    null_scores,
    pair_diagnostics,
)


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
    max_cut_db: float = 18.0
    eq_bands: int = 7
    tie_tolerance_db: float = 0.0
    gd_baseline: str = "flat"


PLATEAU_TOLERANCE_DB = 0.5


def _plateau_width(scores_1d: np.ndarray, values: np.ndarray, index: int) -> float:
    """Width, in ``values`` units, of the near-optimal region around ``index``.

    Walks outward from the chosen index while the score stays within
    ``PLATEAU_TOLERANCE_DB`` of its value there. A wide plateau means the
    chosen delay/gain is robust to small real-world drift (quantization,
    temperature, cable length); a narrow one is a razor's-edge optimum.
    """
    threshold = scores_1d[index] + PLATEAU_TOLERANCE_DB
    left = index
    while left > 0 and scores_1d[left - 1] <= threshold:
        left -= 1
    right = index
    while right < scores_1d.size - 1 and scores_1d[right + 1] <= threshold:
        right += 1
    return float(values[right] - values[left])


def _best_configurations(
    context: AnalysisContext,
    first: int,
    second: int,
    delays: np.ndarray,
    gains: np.ndarray,
) -> list[tuple[int, float, float, float, float, float]]:
    polarities = np.asarray([1.0, -1.0])
    gain_linear = 10.0 ** (gains / 20.0)
    shifted = context.trend_spectra[second][None, :] * np.exp(
        -2j * np.pi * delays[:, None] * context.trend_frequencies[None, :] / 1000.0
    )
    candidates = (
        context.trend_spectra[first][None, None, None, :]
        + polarities[:, None, None, None]
        * gain_linear[None, None, :, None]
        * shifted[None, :, None, :]
    )
    shape = candidates.shape[:-1]
    scores = null_scores(
        candidates.reshape((-1, candidates.shape[-1])),
        context.trend_frequencies,
        context.ppo,
        score_slice=context.trend_slice,
    ).reshape(shape)
    minimum = float(np.min(scores))
    # Preserve exact primary-score ties so the expensive second and third
    # stages can resolve them without changing the lexicographic objective.
    flat_indices = np.flatnonzero(scores.reshape(-1) == minimum)
    result = []
    for flat_index in flat_indices:
        polarity_index, delay_index, gain_index = np.unravel_index(int(flat_index), shape)
        delay_plateau_ms = _plateau_width(
            scores[polarity_index, :, gain_index], delays, delay_index
        )
        gain_plateau_db = _plateau_width(
            scores[polarity_index, delay_index, :], gains, gain_index
        )
        result.append(
            (
                int(polarities[polarity_index]),
                float(delays[delay_index]),
                float(gains[gain_index]),
                minimum,
                delay_plateau_ms,
                gain_plateau_db,
            )
        )
    return result


def _banded_sort_key(rows: list[dict], primary: str, tolerance_db: float) -> list[float]:
    """Primary sort key, optionally binned into ``tolerance_db``-wide bands.

    With ``tolerance_db <= 0`` (the default) this is just the raw metric,
    giving byte-identical behaviour to strict lexicographic ranking. With a
    positive tolerance, primary-metric differences smaller than the
    tolerance are treated as ties so the secondary/tertiary metrics decide
    between pairs that are indistinguishable in practice.
    """
    if tolerance_db <= 0.0:
        return [float(row[primary]) for row in rows]
    minimum = min(float(row[primary]) for row in rows)
    return [
        math.floor((float(row[primary]) - minimum) / tolerance_db + 1e-9)
        for row in rows
    ]


def run_search(
    cache_dir: Path,
    output_path: Path,
    options: SearchOptions,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    if options.gd_baseline not in GD_BASELINE_MODES:
        raise ValueError(f"gd_baseline must be one of {GD_BASELINE_MODES}")
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
        max_cut_db=options.max_cut_db,
        max_filters=options.eq_bands,
    )
    combinations = list(itertools.combinations(range(len(measurements)), 2))
    pairs: list[dict] = []
    for ordinal, (first, second) in enumerate(combinations, start=1):
        configurations = _best_configurations(
            context, first, second, delays, gains
        )
        finalists = []
        for (
            polarity,
            delay_ms,
            gain_db,
            _magnitude_null_score,
            delay_plateau_ms,
            gain_plateau_db,
        ) in configurations:
            diagnostics = pair_diagnostics(
                context,
                first,
                second,
                polarity,
                delay_ms,
                gain_db,
                include_decay=False,
                eq_options=eq_options,
                gd_baseline=options.gd_baseline,
            )
            # null_score_db (GD-weighted severity, from pair_diagnostics) now
            # decides finalist ties, not the fast search's plain-magnitude
            # minimum; that value survives as magnitude_only_null_score_db.
            diagnostics["delay_plateau_ms"] = delay_plateau_ms
            diagnostics["gain_plateau_db"] = gain_plateau_db
            finalists.append((polarity, delay_ms, gain_db, diagnostics))
        polarity, delay_ms, gain_db, diagnostics = min(
            finalists,
            key=lambda item: (
                item[3]["null_score_db"],
                item[3]["excess_gd_ms"],
                item[3]["excess_gd_tail_ms"],
                item[3]["excess_gd_peak_ms"],
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

    raw_bands = _banded_sort_key(pairs, "null_score_db", options.tie_tolerance_db)
    raw_order = sorted(
        range(len(pairs)),
        key=lambda i: (
            raw_bands[i],
            pairs[i]["excess_gd_ms"],
            pairs[i]["excess_gd_tail_ms"],
            pairs[i]["excess_gd_peak_ms"],
            pairs[i]["raw_tail_ms"],
            pairs[i]["first"],
            pairs[i]["second"],
        ),
    )
    pairs = [pairs[i] for i in raw_order]
    reference_spl = pairs[0]["spl_db"]
    for rank, row in enumerate(pairs, start=1):
        row["rank"] = rank
        row["relative_spl_db"] = float(row["spl_db"] - reference_spl)
    eq_bands = _banded_sort_key(pairs, "post_eq_null_score_db", options.tie_tolerance_db)
    eq_order = sorted(
        range(len(pairs)),
        key=lambda i: (
            eq_bands[i],
            pairs[i]["post_eq_excess_gd_ms"],
            pairs[i]["post_eq_excess_gd_tail_ms"],
            pairs[i]["post_eq_excess_gd_peak_ms"],
            pairs[i]["post_eq_tail_ms"],
            pairs[i]["first"],
            pairs[i]["second"],
        ),
    )
    eq_pairs = [pairs[i] for i in eq_order]
    eq_reference_spl = eq_pairs[0]["post_eq_spl_db"]
    for eq_rank, row in enumerate(eq_pairs, start=1):
        row["eq_rank"] = eq_rank
        row["post_eq_relative_spl_db"] = float(
            row["post_eq_spl_db"] - eq_reference_spl
        )

    result = {
        "format_version": 9,
        "measurement_count": len(measurements),
        "sample_rate": measurements[0].sample_rate,
        "response_length": measurements[0].impulse.size,
        "settings": {
            "band_hz": list(options.band),
            "ppo": options.ppo,
            "delay_range_ms": list(options.delay_range_ms),
            "gain_range_db": list(options.gain_range_db),
            "gd_baseline": {
                "mode": options.gd_baseline,
                "note": (
                    "'flat' (default) removes a single constant - this "
                    "curve's weighted median - so any frequency-dependent "
                    "group delay at all is excess. 'monotonic' instead fits "
                    "a per-point baseline constrained to be non-increasing "
                    "in magnitude as frequency rises, over the full analysis "
                    "band; a genuine group-delay rise confined to the bottom "
                    "of the band is then treated as normal rather than "
                    "excess, while a bump anywhere the non-increasing "
                    "constraint cannot explain - regardless of width - still "
                    "counts in full. 'monotonic' is an explicit, opt-in "
                    "acoustic assumption, not a measurement-reliability "
                    "correction; it changes null_score_db, excess_gd_ms, "
                    "excess_gd_tail_ms, and excess_gd_peak_ms (and their "
                    "post_eq_ counterparts) versus the same search with "
                    "'flat'."
                ),
            },
            "eq": {
                "target": eq_options.target,
                "correction_range_hz": list(eq_range),
                "correction_slope_db_per_octave": eq_options.correction_slope_db_per_octave,
                "max_boost_db": eq_options.max_boost_db,
                "max_cut_db": eq_options.max_cut_db,
                "max_filters": eq_options.max_filters,
                "excess_gd_guard": (
                    "denoised excess-GD peaks are expanded into a gate at "
                    "least one-third octave wide via a maximum (not "
                    "averaging) filter, so a narrow severe spike is gated "
                    "the same as a wider bump of equal height; authority "
                    "falls as gated delay approaches 0.35 cycles"
                ),
                "dsp_target": (
                    "the 'dsp' target uses the same flat curve as 'flat', "
                    "but null_score_db/post_eq_null_score_db barely count a "
                    "minimum-phase dip at low excess GD (it is fully "
                    "correctable by any minimum-phase EQ); a non-minimum-"
                    "phase dip still scores up to the same maximum as other "
                    "targets, so ranking in 'dsp' mode favours flat excess "
                    "group delay over flat raw magnitude. Minimum-phase "
                    "peaks already score zero in every target."
                ),
            },
            "minimum_phase": {
                "method": "real cepstrum (discrete Hilbert equivalent)",
                "fft_zero_pad_factor": context.minphase_pad_factor,
                "log_magnitude_floor_db": -160.0,
                "bandwidth": "full cached response, DC to Nyquist",
                "common_delay": (
                    "energy-weighted median removed from excess GD"
                    if options.gd_baseline == "flat"
                    else "non-increasing monotonic baseline removed from "
                    "excess GD; see settings.gd_baseline"
                ),
            },
            "native_resolution": {
                "hz": context.native_resolution_hz,
                "min_reliable_native_bins": MIN_RELIABLE_NATIVE_BINS,
                "note": (
                    "the cache's unpadded capture length sets a native "
                    f"frequency resolution of {context.native_resolution_hz:g} Hz "
                    "(a resolution heuristic, not a hard reliable/unreliable "
                    "cutoff); zero-padding for minimum-phase/CSD work "
                    "interpolates that spectrum smoothly but adds no new "
                    "information. excess_gd_ms/excess_gd_tail_ms and their "
                    "post-EQ counterparts are progressively smoothed below "
                    f"roughly {MIN_RELIABLE_NATIVE_BINS:g}x this value per "
                    "octave (capped at 4 octaves of smoothing) so measurement "
                    "noise in the sub-bass, where a sweep this long resolves "
                    "only a few independent samples per octave, does not "
                    "read as excess group delay as easily; this "
                    "preferentially preserves broad, resolution-supported "
                    "low-frequency GD features over narrow noise, but a "
                    "genuine feature narrower than the smoothing applied at "
                    "its frequency is still attenuated, the same tradeoff "
                    "any smoothing-based denoiser makes"
                ),
            },
            "ranking": {
                "raw": [
                    "null_score_db",
                    "excess_gd_ms",
                    "excess_gd_tail_ms",
                    "excess_gd_peak_ms",
                    "raw_tail_ms",
                ],
                "eq": [
                    "post_eq_null_score_db",
                    "post_eq_excess_gd_ms",
                    "post_eq_excess_gd_tail_ms",
                    "post_eq_excess_gd_peak_ms",
                    "post_eq_tail_ms",
                ],
                "excess_gd_range_hz": list(eq_range),
                "magnitude_basis": "raw, unsmoothed",
                "tie_tolerance_db": options.tie_tolerance_db,
                "null_score_gd_weighting": (
                    "null_score_db/post_eq_null_score_db scale magnitude dips up, "
                    "and score magnitude peaks that only exist alongside it, "
                    "where they coincide with excess group delay (destructive-"
                    "interference nulls or non-minimum-phase resonance, not just "
                    "amplitude ripple or benign reinforcement); the plain "
                    "magnitude-only dip value survives as "
                    "magnitude_only_null_score_db/post_eq_magnitude_only_null_score_db. "
                    "The fast delay/gain/polarity search itself stays "
                    "magnitude-only for speed."
                ),
                "excess_gd_tail": (
                    "excess_gd_tail_ms/post_eq_excess_gd_tail_ms are |excess GD| "
                    f"integrated over log-frequency (power={EXCESS_GD_TAIL_POWER:g}) "
                    "across the same range as excess_gd_ms, unweighted by level "
                    "(unlike the energy-weighted mean) and by shape (a narrow "
                    "severe spike and a wider shallower bump of the same area "
                    "score the same), so a sum that is flat on magnitude but "
                    "smeary in phase somewhere quiet is still caught"
                ),
                "excess_gd_peak": (
                    "excess_gd_peak_ms/post_eq_excess_gd_peak_ms are the single "
                    "worst denoised |excess GD| sample across the same range, "
                    "width-invariant rather than area-based (unlike "
                    "excess_gd_tail_ms, a narrow severe spike and a wide bump of "
                    "the same area do not score the same here - only their peak "
                    "heights matter). It is a tie-break after excess_gd_tail_ms, "
                    "so it only ever separates placements whose tail already ties"
                ),
                "plateau_diagnostics": (
                    "delay_plateau_ms/gain_plateau_db report how far delay/gain "
                    f"can drift from the chosen value while the raw magnitude "
                    f"null score stays within {PLATEAU_TOLERANCE_DB:g} dB of its "
                    "optimum; wider is more robust to real-world drift"
                ),
                "low_end_extension": (
                    "low_end_extension_hz/post_eq_low_end_extension_hz are an "
                    "in-band, F3-style diagnostic: the lowest frequency the "
                    "broad trend's two-sided envelope holds up, scanning "
                    "down from the envelope's own peak (wherever in the "
                    "band it occurs, not necessarily the top edge - a "
                    "two-subwoofer sum is routinely bandpass-shaped), "
                    f"before permanently falling {LOW_END_EXTENSION_THRESHOLD_DB:g} "
                    "dB below that peak. This is self-referential by design "
                    "- each pair is scored against its own peak, not a "
                    "reference shared across pairs, since anchoring to a "
                    "shared level made any pair whose own peak fell more "
                    "than the threshold below it collapse to a 'no "
                    "extension' answer regardless of its actual low-end "
                    "shape; relative_spl_db/post_eq_relative_spl_db are the "
                    "cross-pair-comparable absolute-level metrics. The "
                    "envelope (not the raw trend) is used to find both the "
                    "peak and the corner so an isolated, recoverable notch "
                    "- already scored on its own terms by the null score - "
                    "cannot by itself read as a loss of extension; a "
                    "genuine sustained rolloff is still found normally. "
                    "Lower is more extended. This is diagnostic only - it "
                    "is not a raw or EQ'd ranking key - a placement's "
                    "null/excess-GD/tail severity always decides the "
                    "winner regardless of how extended it is"
                ),
            },
        },
        "cache_manifest_format": manifest.get("format_version"),
        "pairs": pairs,
    }
    write_json(output_path, result)
    return result
