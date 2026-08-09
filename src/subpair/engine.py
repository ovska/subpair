"""Vectorised pair enumeration and deterministic result serialisation."""

from __future__ import annotations

import itertools
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .cache import load_cache, write_json
from .dsp import (
    DEFAULT_SCORE_DIP_WEIGHT,
    DEFAULT_SCORE_LOW_END_WEIGHT,
    EPS,
    EXCURSION_POWER_DB_PER_OCTAVE,
    EXCESS_GD_TAIL_POWER,
    LOW_END_POWER_UPPER_HZ,
    MIN_RELIABLE_NATIVE_BINS,
    SCORE_DIP_SMOOTHING_OCTAVES,
    AnalysisContext,
    EqOptions,
    broad_trend_db,
    db20,
    inclusive_range,
    low_end_power_db,
    pair_diagnostics,
    smoothed_dip_db,
    usable_output_score_db,
)
from . import modal as modal_analysis
from .modal import ModalOptions, RoomModalSignature


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
    score_low_end_weight: float = DEFAULT_SCORE_LOW_END_WEIGHT
    score_dip_weight: float = DEFAULT_SCORE_DIP_WEIGHT
    low_shelf: bool = True
    modal: bool = False
    modal_tiebreak: bool = False

    def __post_init__(self) -> None:
        if self.modal_tiebreak and not self.modal:
            raise ValueError("modal_tiebreak requires modal analysis to be enabled")


PLATEAU_TOLERANCE_DB = 0.5
SHORTLIST_PER_OBJECTIVE = 8


def _plateau_width(scores_1d: np.ndarray, values: np.ndarray, index: int) -> float:
    """Width, in ``values`` units, of the near-optimal region around ``index``.

    Walks outward from the chosen index while the score stays within
    ``PLATEAU_TOLERANCE_DB`` of its value there. A wide plateau means the
    chosen delay/gain is robust to small real-world drift (quantization,
    temperature, cable length); a narrow one is a razor's-edge optimum.
    """
    threshold = scores_1d[index] - PLATEAU_TOLERANCE_DB
    left = index
    while left > 0 and scores_1d[left - 1] >= threshold:
        left -= 1
    right = index
    while right < scores_1d.size - 1 and scores_1d[right + 1] >= threshold:
        right += 1
    return float(values[right] - values[left])


def _best_configurations(
    context: AnalysisContext,
    first: int,
    second: int,
    delays: np.ndarray,
    gains: np.ndarray,
    score_low_end_weight: float,
    score_dip_weight: float,
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
    headroom_db = -np.maximum(gains, 0.0)
    normalized = candidates * 10.0 ** (headroom_db[None, None, :, None] / 20.0)
    magnitude_wide_db = db20(normalized)
    trend_db = broad_trend_db(magnitude_wide_db, context.ppo)[..., context.trend_slice]
    spl_db = 10.0 * np.log10(
        np.maximum(
            np.mean(np.abs(normalized[..., context.trend_slice]) ** 2, axis=-1),
            EPS,
        )
    )
    low_end_db = low_end_power_db(trend_db, context.frequencies)
    dip_db = smoothed_dip_db(
        magnitude_wide_db,
        context.ppo,
        score_slice=context.trend_slice,
    )
    scores = usable_output_score_db(
        spl_db,
        low_end_db,
        dip_db,
        score_low_end_weight,
        score_dip_weight,
    )
    shape = candidates.shape[:-1]
    flat_scores = np.asarray(scores).reshape(-1)
    flat_dips = np.asarray(dip_db).reshape(-1)
    count = min(SHORTLIST_PER_OBJECTIVE, flat_scores.size)
    # Full EQ fitting is too expensive for every point in the exhaustive grid.
    # Keep both the strongest raw usable-output candidates and the smoothest
    # candidates: cuts and their required preamp can make a slightly quieter,
    # smoother raw sum the best corrected result. Stable sorting plus sorted
    # indices keeps the shortlist deterministic.
    strongest = np.argsort(flat_scores, kind="stable")[-count:]
    smoothest = np.argsort(flat_dips, kind="stable")[:count]
    flat_indices = sorted(set(strongest.tolist() + smoothest.tolist()))
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
                float(scores[polarity_index, delay_index, gain_index]),
                delay_plateau_ms,
                gain_plateau_db,
            )
        )
    return result


_worker_state: dict[str, Any] = {}


def _init_pair_worker(
    context: AnalysisContext,
    delays: np.ndarray,
    gains: np.ndarray,
    eq_options: EqOptions,
    score_low_end_weight: float,
    score_dip_weight: float,
    modal_signature: RoomModalSignature | None,
    modal_options: ModalOptions | None,
) -> None:
    """Stash shared read-only search state once per worker process.

    Avoids re-pickling ``context`` (spectra for every cached measurement)
    on every one of the C(n,2) pair tasks; each worker process receives it
    exactly once, at pool startup.
    """
    _worker_state.update(
        context=context,
        delays=delays,
        gains=gains,
        eq_options=eq_options,
        score_low_end_weight=score_low_end_weight,
        score_dip_weight=score_dip_weight,
        modal_signature=modal_signature,
        modal_options=modal_options,
    )


def _pair_impulse(
    context: AnalysisContext, first: int, second: int, polarity: int, delay_ms: float, gain_db: float
) -> np.ndarray:
    full_sum, _ = context.sum_full(first, second, polarity, delay_ms, gain_db)
    n_fft = 2 * (full_sum.size - 1)
    return np.fft.irfft(full_sum, n=n_fft)


def _compute_pair(
    indices: tuple[int, int],
) -> tuple[int, int, int, float, float, dict]:
    first, second = indices
    context = _worker_state["context"]
    delays = _worker_state["delays"]
    gains = _worker_state["gains"]
    eq_options = _worker_state["eq_options"]
    score_low_end_weight = _worker_state["score_low_end_weight"]
    score_dip_weight = _worker_state["score_dip_weight"]
    modal_signature: RoomModalSignature | None = _worker_state["modal_signature"]
    modal_options: ModalOptions | None = _worker_state["modal_options"]

    configurations = _best_configurations(
        context, first, second, delays, gains, score_low_end_weight, score_dip_weight
    )
    finalists = []
    for (
        polarity,
        delay_ms,
        gain_db,
        _fast_score_db,
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
            include_trends=False,
            eq_options=eq_options,
            score_low_end_weight=score_low_end_weight,
            score_dip_weight=score_dip_weight,
        )
        diagnostics["delay_plateau_ms"] = delay_plateau_ms
        diagnostics["gain_plateau_db"] = gain_plateau_db
        finalists.append((polarity, delay_ms, gain_db, diagnostics))
    polarity, delay_ms, gain_db, diagnostics = max(
        finalists,
        key=lambda item: (
            item[3]["post_eq_score_db"],
            1 if item[0] > 0 else 0,
            -item[1],
            -item[2],
        ),
    )
    if modal_signature is not None and modal_options is not None:
        # Stage 2 (fixed-pole residue fit) is a single linear solve, so this
        # is cheap even though it runs once per pair -- see modal.py's module
        # docstring for why the two-stage design keeps it that way.
        impulse = _pair_impulse(context, first, second, polarity, delay_ms, gain_db)
        pair_modal = modal_analysis.compute_pair_modal_metrics(
            modal_signature, impulse, context.sample_rate, modal_options
        )
        pair_modal["robustness"] = modal_analysis.modal_robustness(
            modal_signature,
            lambda d, g: _pair_impulse(context, first, second, polarity, d, g),
            context.sample_rate,
            delay_ms,
            gain_db,
            modal_options,
        )
        diagnostics["modal"] = pair_modal
    return first, second, polarity, delay_ms, gain_db, diagnostics


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
    # With no EQ bands to fit, the correction range has nothing to scope: score,
    # excess GD, and every other diagnostic should see the full analysis band
    # rather than a possibly narrower --eq-range left over from another run.
    if options.eq_bands > 0:
        eq_range = options.eq_range_hz or options.band
        if eq_range[0] < options.band[0] or eq_range[1] > options.band[1]:
            raise ValueError(
                f"EQ range {eq_range[0]:g}..{eq_range[1]:g} Hz must lie within "
                f"analysis band {options.band[0]:g}..{options.band[1]:g} Hz"
            )
    else:
        eq_range = options.band
    eq_options = EqOptions(
        target=options.eq_target,
        correction_range=eq_range,
        correction_slope_db_per_octave=options.eq_range_slope_db_per_octave,
        max_boost_db=options.max_boost_db,
        max_cut_db=options.max_cut_db,
        max_filters=options.eq_bands,
        low_shelf=options.low_shelf,
    )
    modal_options: ModalOptions | None = None
    modal_signature: RoomModalSignature | None = None
    if options.modal:
        modal_options = ModalOptions()
        modal_signature = modal_analysis.estimate_room_poles(
            [(row.title, row.impulse) for row in measurements],
            context.sample_rate,
            modal_options,
        )
    combinations = list(itertools.combinations(range(len(measurements)), 2))
    worker_count = max(1, min((os.cpu_count() or 1) // 4, 8, len(combinations)))
    pairs: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_pair_worker,
        initargs=(
            context,
            delays,
            gains,
            eq_options,
            options.score_low_end_weight,
            options.score_dip_weight,
            modal_signature,
            modal_options,
        ),
    ) as executor:
        for ordinal, (first, second, polarity, delay_ms, gain_db, diagnostics) in enumerate(
            executor.map(_compute_pair, combinations), start=1
        ):
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

    def _modal_tiebreak_key(pair: dict) -> tuple[float, float]:
        """``(n_highQ, sum_modal_energy_db)``, both lower-is-better.

        Fewer audible high-Q modes, then less total stored modal energy --
        applied only when ``--modal-tiebreak`` is on, strictly after the
        primary usable-output score (see the module docstring in ``modal.py``
        and ``PLAN.md``: this must never outrank null suppression by default).
        A pair with no valid modal fit sorts last among ties rather than
        first, so a fit failure cannot look artificially good.
        """
        pair_modal = pair.get("modal")
        if not pair_modal or not pair_modal.get("valid"):
            return (math.inf, math.inf)
        energy_db = pair_modal.get("sum_modal_energy_db")
        return (
            float(pair_modal.get("n_high_q", 0)),
            float(energy_db) if energy_db is not None else -math.inf,
        )

    def _raw_sort_key(i: int) -> tuple:
        base = (-pairs[i]["score_db"],)
        tiebreak = _modal_tiebreak_key(pairs[i]) if options.modal_tiebreak else ()
        return base + tiebreak + (pairs[i]["first"], pairs[i]["second"])

    raw_order = sorted(range(len(pairs)), key=_raw_sort_key)
    pairs = [pairs[i] for i in raw_order]
    reference_spl = pairs[0]["spl_db"]
    reference_low_end_power = pairs[0]["low_end_power_db"]
    reference_score = pairs[0]["score_db"]
    for rank, row in enumerate(pairs, start=1):
        row["rank"] = rank
        row["relative_score_db"] = float(row["score_db"] - reference_score)
        row["relative_spl_db"] = float(row["spl_db"] - reference_spl)
        row["relative_low_end_power_db"] = float(
            row["low_end_power_db"] - reference_low_end_power
        )
    def _eq_sort_key(i: int) -> tuple:
        base = (-pairs[i]["post_eq_score_db"],)
        tiebreak = _modal_tiebreak_key(pairs[i]) if options.modal_tiebreak else ()
        return base + tiebreak + (pairs[i]["first"], pairs[i]["second"])

    eq_order = sorted(range(len(pairs)), key=_eq_sort_key)
    eq_pairs = [pairs[i] for i in eq_order]
    eq_reference_spl = eq_pairs[0]["post_eq_spl_db"]
    eq_reference_low_end_power = eq_pairs[0]["post_eq_low_end_power_db"]
    eq_reference_score = eq_pairs[0]["post_eq_score_db"]
    for eq_rank, row in enumerate(eq_pairs, start=1):
        row["eq_rank"] = eq_rank
        row["post_eq_relative_score_db"] = float(
            row["post_eq_score_db"] - eq_reference_score
        )
        row["post_eq_relative_spl_db"] = float(
            row["post_eq_spl_db"] - eq_reference_spl
        )
        row["post_eq_relative_low_end_power_db"] = float(
            row["post_eq_low_end_power_db"] - eq_reference_low_end_power
        )

    modal_signature_json = None
    if modal_signature is not None:
        modal_signature_json = {
            "valid": modal_signature.valid,
            "decimated_fs_hz": modal_signature.decimated_fs_hz,
            "window_seconds": modal_signature.window_seconds,
            "discard_fraction": modal_signature.discard_fraction,
            "warnings": list(modal_signature.warnings),
            "modes": [
                {
                    "frequency_hz": mode.frequency_hz,
                    "t60_s": modal_analysis.T60_LN_RATIO / mode.decay_rate_per_s,
                    "decay_rate_per_s": mode.decay_rate_per_s,
                    "measurement_persistence": mode.measurement_persistence,
                }
                for mode in modal_signature.modes
            ],
            # Per solo position: each of its own (pre-pooling) consensus poles
            # matched to a pooled-mode index, for the report's invariance
            # check (f_n/T60_n should agree across positions for a real mode).
            "per_measurement": [dict(entry) for entry in modal_signature.per_measurement],
        }

    low_end_power_mask = context.frequencies <= LOW_END_POWER_UPPER_HZ
    low_end_power_frequencies = (
        context.frequencies[low_end_power_mask]
        if np.any(low_end_power_mask)
        else context.frequencies
    )
    low_end_power_range_hz = [
        float(low_end_power_frequencies[0]),
        float(low_end_power_frequencies[-1]),
    ]

    result = {
        "format_version": 21,
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
                "max_cut_db": eq_options.max_cut_db,
                "max_filters": eq_options.max_filters,
                "shelf": {
                    "enabled": eq_options.low_shelf,
                    "automatic": True,
                    "counts_toward_max_filters": True,
                    "note": (
                        "when enabled, one automatic RBJ low shelf competes "
                        "with PK candidates against the same correction "
                        "objective; its corner and boost/cut are fitted per "
                        "pair, it obeys the same gain limits, and it consumes "
                        "one max_filters slot when selected"
                    ),
                },
                "excess_gd_guard": (
                    "denoised excess-GD peaks are expanded into a gate at "
                    "least one-third octave wide via a maximum (not "
                    "averaging) filter, so a narrow severe spike is gated "
                    "the same as a wider bump of equal height; authority "
                    "falls as gated delay approaches 0.35 cycles"
                ),
                "dsp_target": (
                    "the 'dsp' target uses the same flat curve as 'flat', "
                    "retained as a descriptive alias for workflows that will "
                    "apply the result in an external DSP. Scoring is now based "
                    "only on equal-drive output and the smoothed-response dip, "
                    "so 'dsp' has no target-specific scoring exception."
                ),
            },
            "modal": {
                "enabled": options.modal,
                "tiebreak": options.modal_tiebreak,
                "note": (
                    "parametric modal decomposition (matrix-pencil pole "
                    "estimation, jointly across every solo measurement) "
                    "reporting per-mode f_n/T60_n/Q_n/L_n/t_audible_n and "
                    "aggregate n_highQ/Q_max/sum_modal_energy_db per pair; "
                    "diagnostic-only and does not affect score_db/"
                    "post_eq_score_db. When tiebreak is enabled, "
                    "(n_highQ, sum_modal_energy_db) - both lower-is-better - "
                    "is inserted strictly after the primary usable-output "
                    "score, before the deterministic pair-index tie-break; "
                    "off by default, see modal_signature for the pooled room "
                    "pole set this was computed against"
                ),
                **(
                    {
                        "band_hz": list(modal_options.band),
                        "decimated_fs_hz": modal_options.decimated_fs_hz,
                        "window_seconds": modal_options.window_seconds,
                        "order_range": [modal_options.order_min, modal_options.order_max],
                        "order_step": modal_options.order_step,
                        "high_q_threshold": modal_options.high_q_threshold,
                        "level_gates_db": list(modal_options.level_gates_db),
                        "primary_gate_db": modal_options.primary_gate_db,
                        "audible_margin_db": modal_options.audible_margin_db,
                    }
                    if modal_options is not None
                    else {}
                ),
            },
            "minimum_phase": {
                "method": "real cepstrum (discrete Hilbert equivalent)",
                "fft_zero_pad_factor": context.minphase_pad_factor,
                "log_magnitude_floor_db": -160.0,
                "bandwidth": "full cached response, DC to Nyquist",
                "common_delay": "energy-weighted median removed from excess GD",
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
                "raw": ["score_db"],
                "eq": ["post_eq_score_db"],
                "excess_gd_range_hz": list(eq_range),
                "direction": "higher is better",
                "score": {
                    "fields": {
                        "raw": "score_db / relative_score_db",
                        "post_eq": "post_eq_score_db / post_eq_relative_score_db",
                    },
                    "formula": (
                        "(1 - low_end_weight) * full-band SPL + "
                        "low_end_weight * low-end power - dip_weight * "
                        "worst smoothed dip; absolute fields use the cache's "
                        "level reference and relative fields set the best pair "
                        "to 0 dB"
                    ),
                    "low_end_weight": options.score_low_end_weight,
                    "dip_weight": options.score_dip_weight,
                    "dip_smoothing_octaves_fwhm": SCORE_DIP_SMOOTHING_OCTAVES,
                    "dip_basis": (
                        "largest negative deviation from a one-third-octave "
                        "Gaussian-smoothed version of the same equal-drive "
                        "response; no two-sided null heuristic or group-delay "
                        "multiplier"
                    ),
                    "configuration_selection": (
                        f"the exhaustive raw pass shortlists up to "
                        f"{SHORTLIST_PER_OBJECTIVE} highest-score and "
                        f"{SHORTLIST_PER_OBJECTIVE} lowest-dip configurations "
                        "per pair; fitted post-EQ score selects the reported "
                        "polarity/delay/gain tuple"
                    ),
                },
                "headroom": {
                    "fields": {
                        "raw": "headroom_db",
                        "post_eq": "post_eq_headroom_db",
                    },
                    "basis": (
                        "negative global gain applied directly to every compared "
                        "response. Raw headroom removes positive relative pair "
                        "gain so the hottest driver is at 0 dB. Post-EQ headroom "
                        "also removes the fitted EQ response's largest in-band "
                        "boost. Magnitude comparisons, low-end power, Relative "
                        "SPL, and the reported combined EQ response all include "
                        "this gain"
                    ),
                },
                "excess_gd_tail": (
                    "excess_gd_tail_ms/post_eq_excess_gd_tail_ms are |excess GD| "
                    f"integrated over log-frequency (power={EXCESS_GD_TAIL_POWER:g}) "
                    "across the same range as excess_gd_ms, unweighted by level "
                    "(unlike the energy-weighted mean) and by shape (a narrow "
                    "severe spike and a wider shallower bump of the same area "
                    "score the same). This is a reported diagnostic and does "
                    "not change the usable-output score"
                ),
                "excess_gd_peak": (
                    "excess_gd_peak_ms/post_eq_excess_gd_peak_ms are the single "
                    "worst denoised |excess GD| sample across the same range, "
                    "width-invariant rather than area-based (unlike "
                    "excess_gd_tail_ms, a narrow severe spike and a wide bump of "
                    "the same area do not score the same here - only their peak "
                    "heights matter). This is a reported diagnostic and does "
                    "not change the usable-output score"
                ),
                "plateau_diagnostics": (
                    "delay_plateau_ms/gain_plateau_db report how far delay/gain "
                    f"can drift from the chosen value while the raw usable-output "
                    f"score stays within {PLATEAU_TOLERANCE_DB:g} dB of its "
                    "optimum; wider is more robust to real-world drift"
                ),
                "low_end_power": {
                    "fields": {
                        "raw": "low_end_power_db / relative_low_end_power_db",
                        "post_eq": (
                            "post_eq_low_end_power_db / "
                            "post_eq_relative_low_end_power_db"
                        ),
                    },
                    "range_hz": low_end_power_range_hz,
                    "amplifier_power_weight_db_per_octave": (
                        EXCURSION_POWER_DB_PER_OCTAVE
                    ),
                    "basis": (
                        "one-octave broad-trend pressure power, weighted by "
                        "the f^-4 amplifier/excursion cost and integrated over "
                        "log frequency. Pistonic pressure is proportional to "
                        "f^2 times cone displacement, so equal pressure one "
                        "octave lower needs 4x displacement and approximately "
                        "16x amplifier power (+12.04 dB). The input broad trend "
                        "already includes the shared raw or post-EQ headroom "
                        "gain described above, so this metric sees the same "
                        "equal-drive response as the magnitude and Relative SPL "
                        "comparisons. Exact watts/excursion require "
                        "driver impedance, motor, enclosure, and protection/DSP "
                        "data absent from REW impulse responses. Relative fields "
                        "reference the highest-scoring pair in the corresponding "
                        "raw or EQ'd comparison; higher is better. Low-end power "
                        "is one component of the usable-output score"
                    ),
                },
            },
        },
        "cache_manifest_format": manifest.get("format_version"),
        "modal_signature": modal_signature_json,
        "pairs": pairs,
    }
    write_json(output_path, result)
    return result
