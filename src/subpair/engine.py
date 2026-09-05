"""Vectorised pair enumeration and deterministic result serialisation."""

from __future__ import annotations

import itertools
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import ndimage

from .api import RewClient
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
    ARRIVAL_ONSET_THRESHOLD_DB,
    AnalysisContext,
    arrival_onset_index,
    EqOptions,
    broad_trend_db,
    db20,
    filters_response,
    inclusive_range,
    low_end_power_db,
    low_shelf_response,
    pair_diagnostics,
    smoothed_dip_db,
    usable_output_score_db,
)
from . import modal as modal_analysis
from .modal import ModalOptions, RoomModalSignature


@dataclass(frozen=True)
class GateThresholds:
    """All disqualifier thresholds, serialized verbatim with every search."""

    redundancy_reject: float = 0.50
    redundancy_caution: float = 0.60
    ripple_correlation_reject: float = 0.30
    ripple_complementary: float = -0.10
    physical_percentile_reject: float = 75.0
    cancellation_deficit_reject_db: float = -3.0
    cancellation_deficit_caution_db: float = -1.0
    comb_index_reject: float = 0.65
    comb_index_caution: float = 0.40
    notch_depth_reject_db: float = 8.0
    notch_max_width_octaves: float = 1.0 / 6.0
    gain_asymmetry_caution_db: float = 4.0
    band_edge_excess_spread_reject_db: float = 1.0
    localization_fraction_reject: float = 0.50
    localization_min_mean_improvement_db: float = 0.25
    basin_tolerance_db: float = 0.5

    def __post_init__(self) -> None:
        values = asdict(self)
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError("Gate thresholds must be finite")
        if not 0.0 <= self.redundancy_reject <= self.redundancy_caution <= 1.0:
            raise ValueError(
                "Redundancy thresholds must satisfy 0 <= reject <= caution <= 1"
            )
        if not -1.0 <= self.ripple_correlation_reject <= 1.0:
            raise ValueError("Ripple-correlation reject threshold must be between -1 and 1")
        if not -1.0 <= self.ripple_complementary <= self.ripple_correlation_reject:
            raise ValueError(
                "Ripple complementary threshold must not exceed the reject threshold"
            )
        if not 0.0 <= self.physical_percentile_reject <= 100.0:
            raise ValueError("Physical percentile threshold must be between 0 and 100")
        if self.cancellation_deficit_reject_db > self.cancellation_deficit_caution_db:
            raise ValueError("Cancellation reject threshold must not exceed caution")
        if not 0.0 <= self.comb_index_caution <= self.comb_index_reject <= 1.0:
            raise ValueError(
                "Comb thresholds must satisfy 0 <= caution <= reject <= 1"
            )
        if self.notch_depth_reject_db <= 0.0 or self.notch_max_width_octaves <= 0.0:
            raise ValueError("Notch depth and width thresholds must be positive")
        if self.gain_asymmetry_caution_db < 0.0:
            raise ValueError("Gain-asymmetry threshold must be non-negative")
        if self.band_edge_excess_spread_reject_db < 0.0:
            raise ValueError("Band-edge excess-spread threshold must be non-negative")
        if not 0.0 <= self.localization_fraction_reject <= 1.0:
            raise ValueError("Localization fraction threshold must be between 0 and 1")
        if self.localization_min_mean_improvement_db < 0.0:
            raise ValueError("Localization minimum improvement must be non-negative")
        if self.basin_tolerance_db <= 0.0:
            raise ValueError("Basin tolerance must be positive")


@dataclass(frozen=True)
class SearchOptions:
    band: tuple[float, float] = (25.0, 150.0)
    delay_range_ms: tuple[float, float, float] = (-10.0, 10.0, 0.05)
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
    listener_position_m: tuple[float, float, float] | None = None
    sub_positions_m: dict[int, tuple[float, float, float]] | None = None
    room_dimensions_m: tuple[float, float, float] | None = None
    listener_movement_m: float = 0.25
    speed_of_sound_m_per_s: float = 343.0
    physical_delay_window_ms: float = 1.5
    score_tie_margin_db: float = 0.0
    gate_thresholds: GateThresholds = field(default_factory=GateThresholds)

    def __post_init__(self) -> None:
        if self.modal_tiebreak and not self.modal:
            raise ValueError("modal_tiebreak requires modal analysis to be enabled")
        if self.delay_range_ms[2] <= 0.0:
            raise ValueError("delay grid step must be positive")
        if self.listener_movement_m <= 0.0:
            raise ValueError("listener movement must be positive")
        if self.speed_of_sound_m_per_s <= 0.0:
            raise ValueError("speed of sound must be positive")
        if self.physical_delay_window_ms <= 0.0:
            raise ValueError("physical delay window must be positive")
        if self.score_tie_margin_db < 0.0:
            raise ValueError("score tie margin must be non-negative")
        coordinates = []
        if self.listener_position_m is not None:
            coordinates.append(("listener position", self.listener_position_m))
        if self.sub_positions_m is not None:
            coordinates.extend(
                (f"sub position {position}", coordinate)
                for position, coordinate in self.sub_positions_m.items()
            )
        for label, coordinate in coordinates:
            if len(coordinate) != 3 or not all(math.isfinite(float(value)) for value in coordinate):
                raise ValueError(f"{label} must contain three finite metre coordinates")
        if self.room_dimensions_m is not None and (
            len(self.room_dimensions_m) != 3
            or not all(
                math.isfinite(float(value)) and float(value) > 0.0
                for value in self.room_dimensions_m
            )
        ):
            raise ValueError("room dimensions must contain three positive finite metre values")


PLATEAU_TOLERANCE_DB = 0.5
SHORTLIST_PER_OBJECTIVE = 8
MAX_DELAY_GRID_STEP_MS = 0.05
GAIN_JITTER_SIGMA_DB = 0.5


def _gaussian_expectation(values: np.ndarray, sigma_steps: float, axis: int = -1) -> np.ndarray:
    """Gaussian expectation on an existing regular grid, without re-scoring."""

    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(sigma_steps) or sigma_steps <= 1e-12:
        return array.copy()
    radius = max(1, int(math.ceil(4.0 * sigma_steps)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_steps) ** 2)
    kernel /= np.sum(kernel)
    moved = np.moveaxis(array, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    smoothed = np.empty_like(flat)
    for index, row in enumerate(flat):
        padded = np.pad(row, (radius, radius), mode="edge")
        smoothed[index] = np.convolve(padded, kernel, mode="valid")
    return np.moveaxis(smoothed.reshape(moved.shape), -1, axis)


def _robust_objective_surface(
    objective: np.ndarray,
    delay_step_ms: float,
    gain_step_db: float,
    sigma_tau_ms: float,
) -> np.ndarray:
    """Fold independent timing and gain jitter into an objective surface."""

    robust = _gaussian_expectation(objective, sigma_tau_ms / delay_step_ms, axis=1)
    # A +/-1 dB gain excursion is represented by roughly +/-2 sigma. Gain
    # jitter needs no separate output column, but belongs in the expectation.
    if gain_step_db > 0.0:
        robust = _gaussian_expectation(robust, GAIN_JITTER_SIGMA_DB / gain_step_db, axis=2)
    return robust


def _basin_width(
    objective_1d: np.ndarray,
    values: np.ndarray,
    index: int,
    tolerance_db: float,
) -> float:
    """Width of the contiguous below-threshold basin containing ``index``."""

    objective = np.asarray(objective_1d, dtype=np.float64)
    grid = np.asarray(values, dtype=np.float64)
    threshold = objective[index] + tolerance_db
    left = index
    while left > 0 and objective[left - 1] <= threshold:
        left -= 1
    right = index
    while right < objective.size - 1 and objective[right + 1] <= threshold:
        right += 1
    return float(grid[right] - grid[left])


def _worst_case(
    objective_1d: np.ndarray,
    values: np.ndarray,
    centre: float,
    half_width: float,
) -> float:
    """Maximum interpolated objective over a closed interval."""

    objective = np.asarray(objective_1d, dtype=np.float64)
    grid = np.asarray(values, dtype=np.float64)
    low = max(float(grid[0]), centre - half_width)
    high = min(float(grid[-1]), centre + half_width)
    interior = objective[(grid >= low) & (grid <= high)]
    samples = np.concatenate(
        [interior, np.asarray([np.interp(low, grid, objective), np.interp(high, grid, objective)])]
    )
    return float(np.max(samples))


def _local_minima_indices(values: np.ndarray) -> np.ndarray:
    """One deterministic index for each local-minimum plateau."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 1:
        return np.asarray([0], dtype=int)
    candidates = array <= np.minimum(
        np.r_[np.inf, array[:-1]], np.r_[array[1:], np.inf]
    )
    indices: list[int] = []
    start = 0
    while start < array.size:
        if not candidates[start]:
            start += 1
            continue
        end = start
        while end + 1 < array.size and candidates[end + 1] and array[end + 1] == array[start]:
            end += 1
        indices.append((start + end) // 2)
        start = end + 1
    return np.asarray(indices, dtype=int)


def _score_wide_spectrum(
    context: AnalysisContext,
    spectrum: np.ndarray,
    score_low_end_weight: float,
    score_dip_weight: float,
    band: tuple[float, float] | None = None,
) -> np.ndarray | float:
    """Apply the normal usable-output score to a margin-extended spectrum."""

    low_hz, high_hz = band or context.band
    indices = np.flatnonzero(
        (context.trend_frequencies >= low_hz)
        & (context.trend_frequencies <= high_hz)
    )
    if indices.size < 2:
        raise ValueError(f"Evaluation band {low_hz:g}..{high_hz:g} Hz is empty")
    score_slice = slice(int(indices[0]), int(indices[-1]) + 1)
    frequencies = context.trend_frequencies[score_slice]
    values = np.asarray(spectrum, dtype=np.complex128)
    magnitude_wide_db = db20(values)
    trend_db = broad_trend_db(magnitude_wide_db, context.ppo)[..., score_slice]
    spl_db = 10.0 * np.log10(
        np.maximum(np.mean(np.abs(values[..., score_slice]) ** 2, axis=-1), EPS)
    )
    low_end_db = low_end_power_db(trend_db, frequencies)
    dip_db = smoothed_dip_db(
        magnitude_wide_db,
        context.ppo,
        score_slice=score_slice,
    )
    return usable_output_score_db(
        spl_db,
        low_end_db,
        dip_db,
        score_low_end_weight,
        score_dip_weight,
    )


def _baseline_objective_curve(
    context: AnalysisContext,
    first: int,
    second: int,
    delays_ms: np.ndarray,
    score_low_end_weight: float,
    score_dip_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-gain delay landscape, free to pick either polarity, for Gate C.

    Returns ``(objective, polarity)``.  Fixing the baseline to normal polarity
    scores every inverted-polarity pair in the one configuration it explicitly
    rejects, and the resulting penalty is systematic rather than diagnostic.
    A polarity flip is free, exact and immune to drift, so it belongs in the
    "no tuning applied" reference; only delay and gain are the fragile,
    drift-sensitive tuning this gate exists to police.
    """

    phase = np.exp(
        -2j
        * np.pi
        * np.asarray(delays_ms, dtype=np.float64)[:, None]
        * context.trend_frequencies[None, :]
        / 1000.0
    )
    shifted = context.trend_spectra[second][None, :] * phase
    reference = context.trend_spectra[first][None, :]
    curves = np.vstack(
        [
            -np.asarray(
                _score_wide_spectrum(
                    context,
                    reference + polarity * shifted,
                    score_low_end_weight,
                    score_dip_weight,
                ),
                dtype=np.float64,
            )
            for polarity in (1.0, -1.0)
        ]
    )
    best = np.argmin(curves, axis=0)
    return curves[best, np.arange(curves.shape[1])], np.where(best == 0, 1, -1)


def _redundancy_residual(
    context: AnalysisContext,
    first: int,
    second: int,
    delays_ms: np.ndarray,
) -> tuple[float, float, complex]:
    """Best complex scaled/delayed-copy residual over the scoring band."""

    reference = context.spectra[first]
    target = context.spectra[second]
    denominator = float(np.vdot(reference, reference).real)
    target_norm = float(np.linalg.norm(target))
    if denominator <= EPS or target_norm <= EPS:
        return 1.0, 0.0, 0.0j
    best = (math.inf, 0.0, 0.0j)
    for delay_ms in np.asarray(delays_ms, dtype=np.float64):
        shifted = reference * np.exp(
            -2j * np.pi * context.frequencies * delay_ms / 1000.0
        )
        scale = np.vdot(shifted, target) / denominator
        residual = float(np.linalg.norm(target - scale * shifted) / target_norm)
        candidate = (residual, float(delay_ms), complex(scale))
        if candidate[:2] < best[:2]:
            best = candidate
    return best


def _ripple_deviation_correlation(
    context: AnalysisContext,
    first: int,
    second: int,
) -> float:
    """Correlation of each solo response after one-octave detrending."""

    magnitude_db = db20(context.trend_spectra[[first, second]])
    residual = (
        magnitude_db - broad_trend_db(magnitude_db, context.ppo)
    )[:, context.trend_slice]
    centred = residual - np.mean(residual, axis=1, keepdims=True)
    norms = np.linalg.norm(centred, axis=1)
    if np.any(norms <= EPS):
        return 0.0
    return float(np.dot(centred[0], centred[1]) / (norms[0] * norms[1]))


def _cancellation_deficit_db(
    context: AnalysisContext,
    first: int,
    second: int,
    polarity: int,
    delay_ms: float,
    gain_db: float,
) -> float:
    """Mean coherent level minus the matching incoherent power sum."""

    second_spectrum = (
        10.0 ** (gain_db / 20.0)
        * context.spectra[second]
        * np.exp(-2j * np.pi * context.frequencies * delay_ms / 1000.0)
    )
    coherent = context.spectra[first] + polarity * second_spectrum
    incoherent_db = 10.0 * np.log10(
        np.maximum(
            np.abs(context.spectra[first]) ** 2 + np.abs(second_spectrum) ** 2,
            EPS,
        )
    )
    return float(np.mean(db20(coherent) - incoherent_db))


def _detrended_ripple(
    context: AnalysisContext,
    spectrum_wide: np.ndarray,
) -> np.ndarray:
    magnitude_db = db20(np.asarray(spectrum_wide, dtype=np.complex128))
    return (
        magnitude_db - broad_trend_db(magnitude_db, context.ppo)
    )[context.trend_slice]


def _comb_signature(
    context: AnalysisContext,
    spectrum_wide: np.ndarray,
    delay_ms: float,
) -> dict[str, float | None]:
    """Autocorrelation at the linear-frequency comb spacing and harmonics."""

    absolute_delay_ms = abs(float(delay_ms))
    band_span_hz = float(context.frequencies[-1] - context.frequencies[0])
    if absolute_delay_ms <= 1e-9 or band_span_hz <= 0.0:
        return {"index": 0.0, "lag_hz": None, "fundamental_hz": None}
    fundamental_hz = 1000.0 / absolute_delay_ms
    if fundamental_hz > band_span_hz:
        return {
            "index": 0.0,
            "lag_hz": None,
            "fundamental_hz": fundamental_hz,
        }
    count = max(256, context.frequencies.size * 2)
    linear_frequencies = np.linspace(
        context.frequencies[0], context.frequencies[-1], count
    )
    ripple = np.interp(
        linear_frequencies,
        context.frequencies,
        _detrended_ripple(context, spectrum_wide),
    )
    ripple -= float(np.mean(ripple))
    step_hz = float(linear_frequencies[1] - linear_frequencies[0])
    best_index = 0.0
    best_lag_hz: float | None = None
    harmonic = 1
    while harmonic * fundamental_hz <= band_span_hz:
        target = harmonic * fundamental_hz
        centre = int(round(target / step_hz))
        for lag in range(max(1, centre - 1), min(count - 2, centre + 1) + 1):
            left = ripple[:-lag]
            right = ripple[lag:]
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            correlation = float(np.dot(left, right) / denominator) if denominator > EPS else 0.0
            if correlation > best_index:
                best_index = correlation
                best_lag_hz = lag * step_hz
        harmonic += 1
    return {
        "index": float(np.clip(best_index, 0.0, 1.0)),
        "lag_hz": best_lag_hz,
        "fundamental_hz": fundamental_hz,
    }


def _residual_notches(
    context: AnalysisContext,
    spectrum_wide: np.ndarray,
    depth_threshold_db: float,
    max_width_octaves: float,
) -> dict[str, Any]:
    """Find deep nulls whose -3 dB width is narrower than the configured limit."""

    ripple = _detrended_ripple(context, spectrum_wide)
    frequencies = context.frequencies
    offenders: list[dict[str, float]] = []
    for index in _local_minima_indices(ripple):
        depth_db = -float(ripple[index])
        if depth_db < depth_threshold_db:
            continue
        left = int(index)
        while left > 0 and ripple[left] <= -3.0:
            left -= 1
        right = int(index)
        while right + 1 < ripple.size and ripple[right] <= -3.0:
            right += 1
        # A band-edge crossing has unknown width outside the evaluated band,
        # so it cannot safely be called a narrow in-band notch.
        if left == 0 or right == ripple.size - 1:
            continue
        width_octaves = float(np.log2(frequencies[right] / frequencies[left]))
        if width_octaves <= max_width_octaves + 1e-12:
            offenders.append(
                {
                    "depth_db": depth_db,
                    "frequency_hz": float(frequencies[index]),
                    "width_octaves": width_octaves,
                }
            )
    offenders.sort(key=lambda item: (-item["depth_db"], item["frequency_hz"]))
    return {
        "count": len(offenders),
        "worst": offenders[0] if offenders else None,
        "offenders": offenders,
    }


def _band_edge_stability(
    context: AnalysisContext,
    spectrum_wide: np.ndarray,
    score_low_end_weight: float,
    score_dip_weight: float,
) -> dict[str, Any]:
    """Re-score after shifting the complete evaluation window by +/-1/6 octave."""

    factor = 2.0 ** (1.0 / 6.0)
    bands = {
        "down": (context.band[0] / factor, context.band[1] / factor),
        "nominal": context.band,
        "up": (context.band[0] * factor, context.band[1] * factor),
    }
    scores = {
        label: float(
            _score_wide_spectrum(
                context,
                spectrum_wide,
                score_low_end_weight,
                score_dip_weight,
                band,
            )
        )
        for label, band in bands.items()
    }
    return {
        "shift_octaves": 1.0 / 6.0,
        "scores_db": scores,
        "spread_db": float(max(scores.values()) - min(scores.values())),
    }


def _improvement_localization(
    context: AnalysisContext,
    selected_spectrum_wide: np.ndarray,
    physical_spectrum_wide: np.ndarray,
    score_low_end_weight: float,
    score_dip_weight: float,
) -> dict[str, float | None]:
    """Share of positive detrended-ripple improvement in one 1/6-octave region."""

    selected_db = db20(selected_spectrum_wide)
    physical_db = db20(physical_spectrum_wide)
    selected_score = float(
        _score_wide_spectrum(
            context,
            selected_spectrum_wide,
            score_low_end_weight,
            score_dip_weight,
        )
    )
    physical_score = float(
        _score_wide_spectrum(
            context,
            physical_spectrum_wide,
            score_low_end_weight,
            score_dip_weight,
        )
    )
    improvement_db = selected_score - physical_score
    selected_ripple = (
        selected_db - broad_trend_db(selected_db, context.ppo)
    )[context.trend_slice]
    physical_ripple = (
        physical_db - broad_trend_db(physical_db, context.ppo)
    )[context.trend_slice]
    positive_improvement = np.maximum(
        np.abs(physical_ripple) - np.abs(selected_ripple),
        0.0,
    )
    positive_total = float(np.sum(positive_improvement))
    # Per-bin mean, so the materiality test below is independent of band width
    # and points-per-octave.
    mean_improvement_db = positive_total / max(1, int(positive_improvement.size))
    if positive_total <= 1e-9:
        return {
            "fraction": 0.0,
            "frequency_hz": None,
            "contribution_db_sum": 0.0,
            "positive_ripple_improvement_db_sum": positive_total,
            "mean_ripple_improvement_db": mean_improvement_db,
            "score_improvement_db": improvement_db,
        }
    width_bins = max(1, int(round(context.ppo / 6.0)))
    window_sums = np.convolve(
        positive_improvement,
        np.ones(width_bins, dtype=np.float64),
        mode="valid",
    )
    best_start = int(np.argmax(window_sums))
    best_contribution = float(window_sums[best_start])
    band_frequencies = context.frequencies
    best_end = min(best_start + width_bins - 1, band_frequencies.size - 1)
    best_frequency = float(
        np.sqrt(band_frequencies[best_start] * band_frequencies[best_end])
    )
    return {
        "fraction": best_contribution / positive_total,
        "frequency_hz": best_frequency,
        "contribution_db_sum": best_contribution,
        "positive_ripple_improvement_db_sum": positive_total,
        "mean_ripple_improvement_db": mean_improvement_db,
        "score_improvement_db": improvement_db,
    }


def _detrended_symmetry(
    objective: np.ndarray,
    tau_grid_ms: np.ndarray,
    preferred_axis_ms: float | None = None,
) -> dict[str, float | None]:
    """Fit a mirror axis after removing the broad delay-envelope shape."""

    values = np.asarray(objective, dtype=np.float64)
    tau = np.asarray(tau_grid_ms, dtype=np.float64)
    if values.size < 9 or values.shape != tau.shape:
        return {"correlation": None, "axis_ms": None, "axis_offset_ms": None}
    broad = ndimage.gaussian_filter1d(
        values,
        sigma=max(2.0, values.size / 10.0),
        mode="nearest",
        truncate=3.0,
    )
    residual = values - broad
    step = float(np.median(np.diff(tau)))
    minimum_samples = max(7, values.size // 10)
    candidates: list[tuple[float, float]] = []
    for axis in tau:
        half_span = min(float(axis - tau[0]), float(tau[-1] - axis))
        sample_count = int(math.floor(half_span / step)) + 1
        if sample_count < minimum_samples:
            continue
        offsets = np.arange(sample_count, dtype=np.float64) * step
        left = np.interp(axis - offsets, tau, residual)
        right = np.interp(axis + offsets, tau, residual)
        left -= float(np.mean(left))
        right -= float(np.mean(right))
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        correlation = float(np.dot(left, right) / denominator) if denominator > EPS else 0.0
        candidates.append((correlation, float(axis)))
    if not candidates:
        return {"correlation": None, "axis_ms": None, "axis_offset_ms": None}
    best_correlation = max(correlation for correlation, _axis in candidates)
    if preferred_axis_ms is None:
        best_correlation, best_axis = max(
            candidates,
            key=lambda item: (item[0], -abs(item[1])),
        )
    else:
        # Periodic responses can have several almost indistinguishable mirror
        # axes.  Treat correlations within one percentage point as equivalent
        # and use the independently measured arrival difference to resolve that
        # visual ambiguity.  This does not affect any score or verdict.
        near_best = [
            item for item in candidates if item[0] >= best_correlation - 0.01
        ]
        best_correlation, best_axis = min(
            near_best,
            key=lambda item: (abs(item[1] - preferred_axis_ms), -item[0]),
        )
    return {
        "correlation": best_correlation,
        "axis_ms": best_axis,
        "axis_offset_ms": (
            best_axis - preferred_axis_ms if preferred_axis_ms is not None else None
        ),
    }


def _metadata_arrival_delay_ms(metadata: dict[str, Any]) -> float | None:
    """Read the normalized cached arrival delay, including older test caches."""

    direct_ms = metadata.get("arrival_delay_ms")
    direct_seconds = metadata.get("arrival_delay_seconds")
    if direct_ms is not None:
        value = float(direct_ms)
        return value if math.isfinite(value) else None
    if direct_seconds is not None:
        value = 1000.0 * float(direct_seconds)
        return value if math.isfinite(value) else None
    parsed = RewClient.measurement_arrival_delay_seconds(metadata)
    return None if parsed is None else 1000.0 * parsed


def _sweep_band_hz(
    measurements: list[Any], analysis_band: tuple[float, float], sample_rate: float
) -> tuple[float, float]:
    """Widest band the cache actually excited, for onset detection.

    Onset timing wants every octave the sweep really covered, not just the
    scored band: doubling the bandwidth halves the width of the leading edge.
    REW reports the sweep range per measurement, so read it rather than
    assuming one; fall back to an octave above the analysis band.
    """

    lows: list[float] = []
    highs: list[float] = []
    for row in measurements:
        summary = row.metadata.get("summary", {})
        low, high = summary.get("startFreq"), summary.get("endFreq")
        if low is None or high is None:
            continue
        low, high = float(low), float(high)
        if math.isfinite(low) and math.isfinite(high) and 0.0 < low < high:
            lows.append(low)
            highs.append(high)
    if not lows:
        return analysis_band[0], min(2.0 * analysis_band[1], 0.45 * sample_rate)
    # The intersection, so no measurement is asked for bandwidth it lacks.
    return max(lows), min(min(highs), 0.45 * sample_rate)


ARRIVAL_SLIP_TOLERANCE_MS = 1.5
# A repaired arrival is a reconstruction, not a measurement. The onset it is
# built from creeps later as the slipped lobe grows relative to the direct
# arrival, so the repaired value can sit a few milliseconds late. Widening the
# delay window for those pairs keeps the constraint doing its real job --
# excluding delays no room geometry could produce -- without letting a
# reconstruction dictate a narrow answer.
ARRIVAL_REPAIRED_WINDOW_FACTOR = 3.0


def _resolve_arrival_delays(
    measurements: list[Any],
    analysis_band: tuple[float, float],
    speed_of_sound_m_per_s: float,
    room_dimensions_m: tuple[float, float, float] | None,
) -> tuple[list[float | None], set[int], set[int], list[str], dict[str, Any]]:
    """Arrival delays with cycle-slipped peak picks detected and repaired.

    REW's arrival delay is the position of the largest sample in the impulse
    response.  Over a subwoofer's two or three octaves that impulse is a slow
    oscillatory blob, so at any position with strong modal reinforcement a
    later half-cycle can outgrow the direct arrival and the pick jumps a whole
    cycle -- tens of milliseconds, and silently.

    The leading edge does not jump.  So the lag between a measurement's peak
    and its own onset is near-constant across a cache (same loudspeaker, same
    room, one microphone), and a measurement whose lag departs from the
    population median has a slipped pick.  Repair rebuilds its peak from its
    own onset plus the median lag, which keeps that position's genuine
    distance rather than flattening it to the median arrival.

    Only differences between arrivals are ever used downstream, so a bias
    common to every onset cancels and is not corrected here.

    Returns ``(delays, repaired, unusable, warnings, diagnostics)``.
    """

    delays = [_metadata_arrival_delay_ms(row.metadata) for row in measurements]
    sample_rate = float(measurements[0].sample_rate)
    onset_band = _sweep_band_hz(measurements, analysis_band, sample_rate)
    warnings: list[str] = []
    onsets: list[float | None] = []
    for row in measurements:
        try:
            index = arrival_onset_index(row.impulse, sample_rate, onset_band)
        except ValueError:
            onsets.append(None)
            continue
        onsets.append(1000.0 * (row.start_time_seconds + index / sample_rate))

    lags = [
        peak - onset
        for peak, onset in zip(delays, onsets)
        if peak is not None and onset is not None
    ]
    median_lag = float(np.median(lags)) if lags else None

    repaired: set[int] = set()
    unusable: set[int] = set()
    resolved: list[float | None] = list(delays)
    details: list[dict[str, Any]] = []
    for index, row in enumerate(measurements):
        peak, onset = delays[index], onsets[index]
        entry: dict[str, Any] = {
            "position": row.position,
            "title": row.title,
            "reported_ms": peak,
            "onset_ms": onset,
            "peak_minus_onset_ms": (
                peak - onset if peak is not None and onset is not None else None
            ),
            "repaired": False,
        }
        if peak is None:
            warnings.append(
                f"Position {row.position} ({row.title}) has no parsed REW arrival delay; "
                "physical delay constraints are unavailable for its pairs. Re-fetch with "
                "a REW build that exposes loopback-referenced arrival metadata."
            )
            unusable.add(index)
            details.append(entry)
            continue
        if onset is not None and median_lag is not None:
            deviation = (peak - onset) - median_lag
            entry["lag_deviation_ms"] = deviation
            if abs(deviation) > ARRIVAL_SLIP_TOLERANCE_MS:
                corrected = onset + median_lag
                resolved[index] = corrected
                repaired.add(index)
                entry.update({"repaired": True, "resolved_ms": corrected})
                warnings.append(
                    f"Arrival delay repaired: position {row.position} ({row.title}) "
                    f"reported {peak:.3f} ms, but its peak sits {deviation:+.3f} ms "
                    f"further from its own onset than the other measurements "
                    f"({median_lag:.3f} ms) -- REW's peak pick slipped a cycle on a "
                    f"strongly resonant position. Using {corrected:.3f} ms, "
                    "reconstructed from this measurement's leading edge. Pair gates "
                    "that depend on physical timing are advisory for this position."
                )
        details.append(entry)

    # Whatever survives repair still has to be physically possible.
    room_diagonal = (
        math.sqrt(sum(dimension * dimension for dimension in room_dimensions_m))
        if room_dimensions_m is not None
        else None
    )
    if room_diagonal is not None:
        spread_limit_ms = 1000.0 * room_diagonal / speed_of_sound_m_per_s
        usable = [value for value in resolved if value is not None]
        reference = float(np.median(usable)) if usable else None
        for index, row in enumerate(measurements):
            value = resolved[index]
            if value is None or reference is None:
                continue
            # A common timing offset cancels in the differences that matter, so
            # test the spread between measurements rather than each absolute
            # delay: only the spread is bounded by the room.
            if abs(value - reference) > spread_limit_ms:
                unusable.add(index)
                warnings.append(
                    f"Arrival delay unusable: position {row.position} ({row.title}) "
                    f"sits {abs(value - reference):.3f} ms from the cache median, "
                    f"more than the {spread_limit_ms:.3f} ms a {room_diagonal:.2f} m "
                    "room diagonal allows. Physical timing is discarded for its pairs."
                )
    diagnostics = {
        "warnings": list(warnings),
        "onset_band_hz": list(onset_band),
        "onset_threshold_db": ARRIVAL_ONSET_THRESHOLD_DB,
        "median_peak_minus_onset_ms": median_lag,
        "slip_tolerance_ms": ARRIVAL_SLIP_TOLERANCE_MS,
        "measurements": details,
    }
    return resolved, repaired, unusable, warnings, diagnostics


def _geometry_jitter(
    first: int,
    second: int,
    listener_position_m: tuple[float, float, float] | None,
    sub_positions_m: dict[int, tuple[float, float, float]] | None,
    movement_m: float,
    speed_of_sound_m_per_s: float,
) -> tuple[float, bool]:
    """Return maximum differential delay excursion in ms and bound status."""

    if listener_position_m is not None and sub_positions_m is not None:
        a = sub_positions_m.get(first + 1)
        b = sub_positions_m.get(second + 1)
        if a is not None and b is not None:
            listener = np.asarray(listener_position_m, dtype=np.float64)
            vector_a = np.asarray(a, dtype=np.float64) - listener
            vector_b = np.asarray(b, dtype=np.float64) - listener
            norm_a = float(np.linalg.norm(vector_a))
            norm_b = float(np.linalg.norm(vector_b))
            if norm_a > 0.0 and norm_b > 0.0:
                unit_difference = float(
                    np.linalg.norm(vector_a / norm_a - vector_b / norm_b)
                )
                return 1000.0 * movement_m * unit_difference / speed_of_sound_m_per_s, False
    return 1000.0 * 2.0 * movement_m / speed_of_sound_m_per_s, True


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
    sigma_tau_ms: float,
    physical_tau_ms: float | None,
    physical_window_ms: float,
    excursion_half_width_ms: float,
    basin_tolerance_db: float,
) -> list[tuple[int, float, float, float, float, float, dict[str, Any]]]:
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
    objective = -np.asarray(scores, dtype=np.float64)
    delay_step_ms = float(delays[1] - delays[0]) if delays.size > 1 else 1.0
    gain_step_db = float(gains[1] - gains[0]) if gains.size > 1 else 0.0
    robust_objective = _robust_objective_surface(
        objective, delay_step_ms, gain_step_db, sigma_tau_ms
    )
    eligible_delays = np.ones(delays.size, dtype=bool)
    physical_window_in_scan = True
    if physical_tau_ms is not None:
        eligible_delays = np.abs(delays - physical_tau_ms) <= physical_window_ms + 1e-12
        physical_window_in_scan = bool(np.any(eligible_delays))
        if not physical_window_in_scan:
            eligible_delays[:] = True

    # Select delay by the jitter-averaged objective for every polarity/gain.
    # Full EQ fitting remains limited to a deterministic two-objective shortlist.
    candidate_indices: list[tuple[int, int, int]] = []
    for polarity_index in range(polarities.size):
        for gain_index in range(gains.size):
            curve = robust_objective[polarity_index, :, gain_index]
            delay_index = int(np.argmin(np.where(eligible_delays, curve, np.inf)))
            candidate_indices.append((polarity_index, delay_index, gain_index))
    count = min(SHORTLIST_PER_OBJECTIVE, len(candidate_indices))
    robust_order = sorted(
        candidate_indices,
        key=lambda item: (float(robust_objective[item]), item),
    )[:count]
    smooth_order = sorted(
        candidate_indices,
        key=lambda item: (float(dip_db[item]), item),
    )[:count]
    selected_indices = sorted(set(robust_order + smooth_order))
    result = []
    for polarity_index, delay_index, gain_index in selected_indices:
        delay_plateau_ms = _plateau_width(
            scores[polarity_index, :, gain_index], delays, delay_index
        )
        gain_plateau_db = _plateau_width(
            scores[polarity_index, delay_index, :], gains, gain_index
        )
        # How far the score moves between adjacent points of the tool's own
        # delay/gain grid. Nothing finer than this is resolved by the search, so
        # it is a hard floor on how precisely two pairs can be ordered -- and it
        # is free here, since the whole grid is already in `scores`.
        delay_slice = slice(max(0, delay_index - 1), delay_index + 2)
        gain_slice = slice(max(0, gain_index - 1), gain_index + 2)
        neighbourhood = scores[polarity_index, delay_slice, gain_slice]
        score_quantisation_db = float(np.max(neighbourhood) - np.min(neighbourhood))
        raw_curve = objective[polarity_index, :, gain_index]
        robust_curve = robust_objective[polarity_index, :, gain_index]
        tau_star_index = int(np.argmin(raw_curve))
        tau_star_ms = float(delays[tau_star_index])
        tau_robust_ms = float(delays[delay_index])
        minima = _local_minima_indices(raw_curve)
        competing = int(
            np.count_nonzero(raw_curve[minima] <= raw_curve[tau_star_index] + 0.3 + 1e-12)
        )
        # A penalty above the objective at the *recommended* delay, not the raw
        # objective around tau_star. As an absolute f it mostly restated how
        # well the pair scores, so comparing pairs measured their quality
        # rather than their robustness; and centred on tau_star it described a
        # delay the pair does not propose, which on a lopsided landscape can
        # disagree sharply with the configuration actually being recommended.
        worst_case_penalty = {
            f"{dt:.1f}": float(
                _worst_case(raw_curve, delays, tau_robust_ms, dt)
                - raw_curve[delay_index]
            )
            for dt in (0.5, 1.0, 1.5)
        }
        basin_w03 = _basin_width(raw_curve, delays, tau_star_index, 0.3)
        basin_w05 = _basin_width(raw_curve, delays, tau_star_index, 0.5)
        # Fragility is an absolute question -- how many dB does this pair lose
        # when the listener moves -- so it is measured in dB, at the delay the
        # pair actually recommends.  A tolerance scaled to each pair's own
        # objective range instead normalises away exactly the information the
        # gate exists to test: a pair whose objective varies by 1.4 dB across
        # the whole physical window is maximally delay-insensitive, yet a
        # 5%-of-range tolerance shrinks to 0.07 dB and fails it, while a pair
        # spanning 17 dB gets a 12x looser tolerance and passes with twice the
        # real excursion penalty.
        basin_tolerance_ms = _basin_width(
            raw_curve, delays, delay_index, basin_tolerance_db
        )
        excursion_penalty_db = float(
            _worst_case(raw_curve, delays, tau_robust_ms, excursion_half_width_ms)
            - raw_curve[delay_index]
        )
        non_physical = (
            physical_tau_ms is not None
            and abs(tau_star_ms - physical_tau_ms) > physical_window_ms + 1e-12
        )
        robustness = {
            "tau_grid_ms": [float(value) for value in delays],
            "objective_db": [float(value) for value in raw_curve],
            "robust_objective_db": [float(value) for value in robust_curve],
            "tau_star_ms": tau_star_ms,
            "tau_robust_ms": tau_robust_ms,
            "f_tau_star_db": float(raw_curve[tau_star_index]),
            "f_robust_tau_robust_db": float(robust_curve[delay_index]),
            "fragility_db": float(robust_curve[tau_star_index] - raw_curve[tau_star_index]),
            "basin_w03_ms": basin_w03,
            "basin_w05_ms": basin_w05,
            "basin_tolerance_db": basin_tolerance_db,
            "basin_tolerance_ms": basin_tolerance_ms,
            "excursion_half_width_ms": excursion_half_width_ms,
            "excursion_penalty_db": excursion_penalty_db,
            "worst_case_penalty_db": worst_case_penalty,
            "score_quantisation_db": score_quantisation_db,
            "n_competing": competing,
            "physical_tau_ms": physical_tau_ms,
            "physical_window_ms": physical_window_ms,
            "physical_window_in_scan": physical_window_in_scan,
            "non_physical_solution": non_physical,
            "detrended_symmetry": _detrended_symmetry(
                raw_curve,
                delays,
                physical_tau_ms,
            ),
        }
        result.append(
            (
                int(polarities[polarity_index]),
                float(delays[delay_index]),
                float(gains[gain_index]),
                float(scores[polarity_index, delay_index, gain_index]),
                delay_plateau_ms,
                gain_plateau_db,
                robustness,
            )
        )
    return result


_GATE_ORDER = (
    "gate_a_redundancy",
    "gate_b_ripple_correlation",
    "gate_c_physical_percentile",
    "basin_geometry",
    "gate_d_cancellation_deficit",
    "gate_e_comb_signature",
    "gate_f_residual_notches",
    "gate_g_gain_asymmetry",
    "gate_h_band_edge_stability",
    "gate_i_improvement_localization",
)


def _not_run_gate(stage: str) -> dict[str, Any]:
    return {"status": "not_run", "stage": stage}


def _append_detail(gate: dict[str, Any], text: str) -> None:
    """Add a finding without erasing one an earlier stage already recorded."""

    existing = gate.get("detail")
    gate["detail"] = f"{existing}; {text}" if existing else text


def _preoptimization_gates(
    context: AnalysisContext,
    first: int,
    second: int,
    delays: np.ndarray,
    first_arrival_ms: float | None,
    second_arrival_ms: float | None,
    arrival_outliers: set[int],
    arrival_repaired: set[int],
    options: SearchOptions,
) -> dict[str, dict[str, Any]]:
    """Run A/B and, when they pass, the delay-only physical Gate C."""

    thresholds = options.gate_thresholds
    residual, fitted_delay_ms, fitted_scale = _redundancy_residual(
        context, first, second, delays
    )
    if residual < thresholds.redundancy_reject:
        redundancy_status = "reject"
    elif residual < thresholds.redundancy_caution:
        redundancy_status = "caution"
    else:
        redundancy_status = "pass"
    correlation = _ripple_deviation_correlation(context, first, second)
    correlation_status = (
        "reject"
        if correlation > thresholds.ripple_correlation_reject
        else "pass"
    )
    correlation_signal = (
        "reinforcing"
        if correlation > thresholds.ripple_correlation_reject
        else (
            "complementary"
            if correlation < thresholds.ripple_complementary
            else "neutral"
        )
    )
    gates: dict[str, dict[str, Any]] = {
        "gate_a_redundancy": {
            "status": redundancy_status,
            "residual": residual,
            "best_fit_delay_ms": fitted_delay_ms,
            "best_fit_scale_real": fitted_scale.real,
            "best_fit_scale_imag": fitted_scale.imag,
            "reject_below": thresholds.redundancy_reject,
            "caution_below": thresholds.redundancy_caution,
        },
        "gate_b_ripple_correlation": {
            "status": correlation_status,
            "correlation": correlation,
            "signal": correlation_signal,
            "reject_above": thresholds.ripple_correlation_reject,
            "complementary_below": thresholds.ripple_complementary,
        },
    }
    stage_one_reject = any(
        gate["status"] == "reject" for gate in gates.values()
    )
    # A flagged arrival delay is suspect timing metadata, not a condemned
    # position: it makes the physical constraint unusable in exactly the way
    # absent metadata does, and absent metadata is only a caution.  Treating a
    # known-bad reading more harshly than a missing one is backwards, so both
    # take the same path -- discard this pair's physical timing and say so.
    physical_unreliable = first in arrival_outliers or second in arrival_outliers
    # A repaired arrival is good enough to aim the delay search with, but not
    # good enough to disqualify a pair on: the user's point is that a poor REW
    # timing pick must not change which pairs are recommended, only how exact
    # the reported delay figure is.
    physical_repaired = first in arrival_repaired or second in arrival_repaired
    reported_tau_ms = (
        float(first_arrival_ms - second_arrival_ms)
        if first_arrival_ms is not None and second_arrival_ms is not None
        else None
    )
    physical_tau_ms = None if physical_unreliable else reported_tau_ms
    # Same reasoning as the Gate C baseline: score the physical alignment with
    # whichever polarity is better there, not with a hardcoded normal polarity
    # that guarantees a cancellation deficit for every inverted-polarity pair.
    physical_deficit_db = (
        max(
            _cancellation_deficit_db(
                context, first, second, polarity, physical_tau_ms, 0.0
            )
            for polarity in (1, -1)
        )
        if physical_tau_ms is not None
        else None
    )
    gates["gate_d_cancellation_deficit"] = {
        **_not_run_gate("post_optimisation"),
        "physical_deficit_db": physical_deficit_db,
        "chosen_deficit_db": None,
        "reject_below_db": thresholds.cancellation_deficit_reject_db,
        "caution_below_db": thresholds.cancellation_deficit_caution_db,
    }
    if stage_one_reject:
        gates["gate_c_physical_percentile"] = _not_run_gate(
            "skipped_after_stage_1_reject"
        )
    elif physical_unreliable:
        gates["gate_c_physical_percentile"] = {
            "status": "caution",
            "percentile": None,
            "physical_tau_ms": None,
            "reported_tau_ms": reported_tau_ms,
            "objective_gap_db": None,
            "reject_above_percentile": thresholds.physical_percentile_reject,
            "detail": (
                "arrival-delay outlier; physical timing discarded for this pair"
            ),
        }
    elif physical_tau_ms is None:
        gates["gate_c_physical_percentile"] = {
            "status": "caution",
            "percentile": None,
            "physical_tau_ms": None,
            "objective_gap_db": None,
            "reject_above_percentile": thresholds.physical_percentile_reject,
            "detail": "arrival metadata unavailable",
        }
    elif physical_tau_ms < delays[0] or physical_tau_ms > delays[-1]:
        gates["gate_c_physical_percentile"] = {
            "status": "reject",
            "percentile": None,
            "physical_tau_ms": physical_tau_ms,
            "objective_gap_db": None,
            "reject_above_percentile": thresholds.physical_percentile_reject,
            "detail": "physical delay lies outside the scan",
        }
    else:
        objective, baseline_polarity = _baseline_objective_curve(
            context,
            first,
            second,
            delays,
            options.score_low_end_weight,
            options.score_dip_weight,
        )
        physical_objective = float(np.interp(physical_tau_ms, delays, objective))
        percentile = float(
            100.0 * np.mean(objective <= physical_objective + 1e-12)
        )
        gates["gate_c_physical_percentile"] = {
            "status": (
                "reject"
                if percentile > thresholds.physical_percentile_reject
                else "pass"
            ),
            "percentile": percentile,
            "physical_tau_ms": physical_tau_ms,
            "objective_at_physical_db": physical_objective,
            "objective_gap_db": physical_objective - float(np.min(objective)),
            "reject_above_percentile": thresholds.physical_percentile_reject,
            "baseline": "best polarity, equal gain",
            "baseline_polarity": int(
                baseline_polarity[int(np.argmin(np.abs(delays - physical_tau_ms)))]
            ),
        }
    gates["gate_c_physical_percentile"]["arrival_repaired"] = physical_repaired
    if physical_repaired and gates["gate_c_physical_percentile"]["status"] == "reject":
        gates["gate_c_physical_percentile"]["status"] = "caution"
        _append_detail(
            gates["gate_c_physical_percentile"],
            "advisory only: this pair's physical timing was reconstructed from a "
            "slipped REW peak pick, so it aims the delay search but cannot "
            "disqualify the pair",
        )
    for name in _GATE_ORDER:
        gates.setdefault(name, _not_run_gate("post_optimisation"))
    return {name: gates[name] for name in _GATE_ORDER}


def _postoptimization_gates(
    context: AnalysisContext,
    first: int,
    second: int,
    pair: dict[str, Any],
    gates: dict[str, dict[str, Any]],
    delays: np.ndarray,
    options: SearchOptions,
) -> dict[str, dict[str, Any]]:
    """Complete the basin and D-I checks for one chosen configuration."""

    thresholds = options.gate_thresholds
    polarity = int(pair["polarity"])
    delay_ms = float(pair["delay_ms"])
    gain_db = float(pair["gain_db"])
    chosen_wide = context.sum_on_trend_grid(
        first, second, polarity, delay_ms, gain_db
    )
    physical_tau_ms = pair.get("physical_tau")
    physical_wide = None
    physical_baseline_polarity = None
    if physical_tau_ms is not None:
        # "No tuning applied" includes the free polarity switch (see
        # _baseline_objective_curve); otherwise this gate measures the polarity
        # decision rather than the delay/gain tuning it is meant to police.
        candidates = [
            (
                float(
                    _score_wide_spectrum(
                        context,
                        candidate,
                        options.score_low_end_weight,
                        options.score_dip_weight,
                    )
                ),
                candidate_polarity,
                candidate,
            )
            for candidate_polarity, candidate in (
                (
                    baseline_polarity,
                    context.sum_on_trend_grid(
                        first, second, baseline_polarity, float(physical_tau_ms), 0.0
                    ),
                )
                for baseline_polarity in (1, -1)
            )
        ]
        _best_score, physical_baseline_polarity, physical_wide = max(
            candidates, key=lambda item: (item[0], item[1])
        )

    # The percentile remains defined against the cheap normal-polarity,
    # equal-gain delay scan from Stage 2.  Once the optimiser has run, replace
    # the provisional gap-to-baseline-minimum with the requested absolute gap
    # to this pair's reported unconstrained optimum.
    physical_objective = gates["gate_c_physical_percentile"].get(
        "objective_at_physical_db"
    )
    if physical_objective is not None and pair.get("f_tau_star") is not None:
        gates["gate_c_physical_percentile"]["objective_gap_db"] = (
            float(physical_objective) - float(pair["f_tau_star"])
        )
        gates["gate_c_physical_percentile"]["gap_reference"] = "f(tau_star)"

    # A delay sitting exactly on the edge of the scan is not an optimum, it is
    # the furthest the search was allowed to go: the real minimum lies outside
    # the configured range, and every robustness figure around it is one-sided
    # (``_worst_case`` clips at the grid edge, so the excursion penalty can even
    # read 0 dB).  Nothing about such a configuration has been established.
    # Only pairs with no usable physical timing can reach this, since the rest
    # are constrained to a window well inside the scan.
    delay_at_boundary = bool(
        abs(delay_ms - float(delays[0])) <= 1e-9
        or abs(delay_ms - float(delays[-1])) <= 1e-9
    )
    gates["gate_c_physical_percentile"]["delay_at_scan_boundary"] = delay_at_boundary
    if delay_at_boundary:
        gates["gate_c_physical_percentile"]["status"] = "reject"
        _append_detail(
            gates["gate_c_physical_percentile"],
            "the selected delay is pinned to the edge of the "
            f"{float(delays[0]):g}..{float(delays[-1]):g} ms scan, so the real "
            "optimum lies outside it; widen --delay-range to evaluate this pair",
        )

    if pair.get("non_physical_solution"):
        # The recommended delay was already constrained into the physical
        # window, so the pair is not relying on the distant unconstrained
        # optimum.  Where that optimum happens to sit is worth surfacing, but
        # it is not grounds to veto a recommendation that never used it.
        if gates["gate_c_physical_percentile"].get("status") == "pass":
            gates["gate_c_physical_percentile"]["status"] = "caution"
        _append_detail(
            gates["gate_c_physical_percentile"],
            "raw optimum lies outside the physical-delay window; the "
            "recommended delay is inside it",
        )

    gates["basin_geometry"] = {
        "status": "pass" if pair["geometric_pass"] else "reject",
        "excursion_penalty_db": pair["excursion_penalty_db"],
        "excursion_half_width_ms": pair["excursion_half_width_ms"],
        "tolerance_db": thresholds.basin_tolerance_db,
        "basin_at_tolerance_ms": pair["basin_tolerance_ms"],
        "required_excursion_ms": pair["delta_tau_max"],
    }

    chosen_deficit = _cancellation_deficit_db(
        context,
        first,
        second,
        polarity,
        delay_ms,
        gain_db,
    )
    physical_deficit = gates["gate_d_cancellation_deficit"].get(
        "physical_deficit_db"
    )
    evaluated_deficits = [chosen_deficit]
    if physical_deficit is not None:
        evaluated_deficits.append(float(physical_deficit))
    worst_deficit = min(evaluated_deficits)
    cancellation_status = (
        "reject"
        if worst_deficit < thresholds.cancellation_deficit_reject_db
        else (
            "caution"
            if worst_deficit < thresholds.cancellation_deficit_caution_db
            else "pass"
        )
    )
    gates["gate_d_cancellation_deficit"] = {
        "status": cancellation_status,
        "physical_deficit_db": physical_deficit,
        "chosen_deficit_db": chosen_deficit,
        "worst_deficit_db": worst_deficit,
        "reject_below_db": thresholds.cancellation_deficit_reject_db,
        "caution_below_db": thresholds.cancellation_deficit_caution_db,
    }

    comb = _comb_signature(context, chosen_wide, delay_ms)
    comb_index = float(comb["index"] or 0.0)
    gates["gate_e_comb_signature"] = {
        "status": (
            "reject"
            if comb_index >= thresholds.comb_index_reject
            else (
                "caution"
                if comb_index >= thresholds.comb_index_caution
                else "pass"
            )
        ),
        "comb_index": comb_index,
        "peak_lag_hz": comb["lag_hz"],
        "fundamental_hz": comb["fundamental_hz"],
        "reject_at_or_above": thresholds.comb_index_reject,
        "caution_at_or_above": thresholds.comb_index_caution,
    }

    notches = _residual_notches(
        context,
        chosen_wide,
        thresholds.notch_depth_reject_db,
        thresholds.notch_max_width_octaves,
    )
    gates["gate_f_residual_notches"] = {
        "status": "reject" if notches["count"] else "pass",
        **notches,
        "depth_threshold_db": thresholds.notch_depth_reject_db,
        "max_width_octaves": thresholds.notch_max_width_octaves,
    }

    gain_offset_db = abs(gain_db)
    gains = inclusive_range(*options.gain_range_db)
    at_search_boundary = bool(
        abs(gain_db - float(gains[0])) <= 1e-9
        or abs(gain_db - float(gains[-1])) <= 1e-9
    )
    gates["gate_g_gain_asymmetry"] = {
        "status": (
            "caution"
            if gain_offset_db > thresholds.gain_asymmetry_caution_db
            else "pass"
        ),
        "gain_offset_db": gain_offset_db,
        "caution_above_db": thresholds.gain_asymmetry_caution_db,
        "headroom_adjustment_db": pair["headroom_db"],
        "gain_at_search_boundary": at_search_boundary,
        "achievable_by_global_attenuation": True,
        "hardware_headroom_known": False,
    }

    edge = _band_edge_stability(
        context,
        chosen_wide,
        options.score_low_end_weight,
        options.score_dip_weight,
    )
    # Provisional only: the excess over the population median decides this
    # gate, so its status is set once every pair has been scored (see
    # _apply_band_edge_population_status).
    gates["gate_h_band_edge_stability"] = {
        "status": "pass",
        **edge,
        "reject_above_excess_db": thresholds.band_edge_excess_spread_reject_db,
    }

    if physical_wide is None:
        gates["gate_i_improvement_localization"] = {
            "status": "caution",
            "fraction": None,
            "frequency_hz": None,
            "detail": "physical alignment unavailable",
            "reject_above_fraction": thresholds.localization_fraction_reject,
        }
    else:
        localization = _improvement_localization(
            context,
            chosen_wide,
            physical_wide,
            options.score_low_end_weight,
            options.score_dip_weight,
        )
        fraction = float(localization["fraction"] or 0.0)
        mean_improvement_db = float(localization["mean_ripple_improvement_db"])
        # The fraction is a ratio of two sums of positive ripple improvement.
        # When the tuned configuration is barely distinguishable from the
        # physical one -- the safest outcome a pair can have, since it relies
        # on no delay trick at all -- both sums collapse to noise and the
        # fraction stops meaning anything.  Judge how an improvement is
        # distributed only once there is an improvement worth distributing.
        immaterial = (
            mean_improvement_db
            < thresholds.localization_min_mean_improvement_db
        )
        gates["gate_i_improvement_localization"] = {
            "status": (
                "pass"
                if immaterial or fraction <= thresholds.localization_fraction_reject
                else "reject"
            ),
            **localization,
            "baseline_polarity": physical_baseline_polarity,
            "reject_above_fraction": thresholds.localization_fraction_reject,
            "min_mean_improvement_db": (
                thresholds.localization_min_mean_improvement_db
            ),
            **(
                {
                    "detail": (
                        "improvement over the physical alignment is immaterial; "
                        "its distribution carries no information"
                    )
                }
                if immaterial
                else {}
            ),
        }
    return {name: gates[name] for name in _GATE_ORDER}


def _score_resolution_db(pair: dict[str, Any], margin_db: float = 0.0) -> float | None:
    """Smallest score difference this pair can actually be ordered by.

    Two components, both derived from the pair's own data. The delay/gain grid
    quantises the search, so the score cannot be trusted finer than it moves
    between adjacent grid points. And Gate H's band shift measures how much the
    score depends on where the band edges are put: the part of that shared by
    every pair cancels out of an ordering, but the excess over the population
    is specific to this pair and is therefore an ordering uncertainty.

    This is a floor, not a full error budget -- microphone placement, level
    calibration and measurement noise all add to it and none of them are
    visible in a single cached sweep.
    """

    quantisation = pair.get("score_quantisation_db")
    if quantisation is None:
        return None
    gate = pair.get("gates", {}).get("gate_h_band_edge_stability", {})
    excess = gate.get("excess_spread_db")
    band_edge = max(0.0, float(excess)) if excess is not None else 0.0
    return float(quantisation) + band_edge + float(margin_db)


def _apply_band_edge_population_status(
    gate_sets: list[dict[str, dict[str, Any]]],
    thresholds: GateThresholds,
) -> None:
    """Set Gate H from each pair's band-edge sensitivity *relative to its peers*.

    Shifting the evaluation band moves every pair in the same direction by
    almost the same amount: the subs roll off below the band and low-end power
    weights that region at f^-4, so an up-shifted band always scores higher.
    That common-mode term belongs to the score function, not to any pair, and
    thresholding the raw spread rejects pairs for a property they all share --
    in practice it cuts through the middle of a tight cluster while the pair
    ordering is identical at every band position.  Subtracting the population
    median leaves only the part that is specific to this pair, which is what
    band-edge stability was supposed to mean.  With a single scored pair the
    excess is zero by construction: there is nothing to compare against, so the
    gate correctly abstains.
    """

    spreads = sorted(
        float(gate_set["gate_h_band_edge_stability"]["spread_db"])
        for gate_set in gate_sets
        if gate_set["gate_h_band_edge_stability"].get("spread_db") is not None
    )
    if not spreads:
        return
    median = float(np.median(np.asarray(spreads, dtype=np.float64)))
    for gate_set in gate_sets:
        gate = gate_set["gate_h_band_edge_stability"]
        if gate.get("spread_db") is None:
            continue
        excess = float(gate["spread_db"]) - median
        gate["population_median_spread_db"] = median
        gate["excess_spread_db"] = excess
        gate["status"] = (
            "reject"
            if excess > thresholds.band_edge_excess_spread_reject_db
            else "pass"
        )


def _verdict_and_reasons(
    gates: dict[str, dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    reasons = []
    for name in _GATE_ORDER:
        gate = gates[name]
        if gate.get("status") not in {"caution", "reject"}:
            continue
        measured = {
            key: value
            for key, value in gate.items()
            if key not in {"status", "stage", "offenders"}
        }
        reasons.append(
            {
                "gate": name,
                "status": gate["status"],
                "measured": measured,
            }
        )
    verdict = (
        "reject"
        if any(reason["status"] == "reject" for reason in reasons)
        else ("caution" if reasons else "accept")
    )
    return verdict, reasons


def _gate_summary_fields(gates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Stable top-level columns for tabular JSON/report consumers."""

    notch = gates["gate_f_residual_notches"].get("worst") or {}
    cancellation = gates["gate_d_cancellation_deficit"]
    return {
        "redundancy_residual": gates["gate_a_redundancy"].get("residual"),
        "ripple_correlation": gates["gate_b_ripple_correlation"].get("correlation"),
        "physical_percentile": gates["gate_c_physical_percentile"].get("percentile"),
        "arrival_repaired": gates["gate_c_physical_percentile"].get("arrival_repaired"),
        "physical_objective_gap_db": gates["gate_c_physical_percentile"].get(
            "objective_gap_db"
        ),
        "physical_cancellation_deficit_db": cancellation.get("physical_deficit_db"),
        "cancellation_deficit_db": cancellation.get("chosen_deficit_db"),
        "comb_index": gates["gate_e_comb_signature"].get("comb_index"),
        "notch_count": gates["gate_f_residual_notches"].get("count"),
        "worst_notch_depth_db": notch.get("depth_db"),
        "worst_notch_frequency_hz": notch.get("frequency_hz"),
        "gain_asymmetry_db": gates["gate_g_gain_asymmetry"].get("gain_offset_db"),
        "band_edge_spread_db": gates["gate_h_band_edge_stability"].get("spread_db"),
        "band_edge_excess_spread_db": gates["gate_h_band_edge_stability"].get(
            "excess_spread_db"
        ),
        "improvement_localization_fraction": gates[
            "gate_i_improvement_localization"
        ].get("fraction"),
        "improvement_mean_db": gates["gate_i_improvement_localization"].get(
            "mean_ripple_improvement_db"
        ),
        "improvement_localization_frequency_hz": gates[
            "gate_i_improvement_localization"
        ].get("frequency_hz"),
    }


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
    arrival_delays_ms: list[float | None],
    arrival_outliers: set[int],
    arrival_repaired: set[int],
    search_options: SearchOptions,
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
        arrival_delays_ms=arrival_delays_ms,
        arrival_outliers=arrival_outliers,
        arrival_repaired=arrival_repaired,
        search_options=search_options,
    )


def _pair_impulse(
    context: AnalysisContext, first: int, second: int, polarity: int, delay_ms: float, gain_db: float
) -> np.ndarray:
    full_sum, _ = context.sum_full(first, second, polarity, delay_ms, gain_db)
    n_fft = 2 * (full_sum.size - 1)
    return np.fft.irfft(full_sum, n=n_fft)


def _post_eq_pair_impulse(
    context: AnalysisContext,
    first: int,
    second: int,
    polarity: int,
    delay_ms: float,
    gain_db: float,
    filters: list[dict[str, float]],
    shelf: dict[str, Any],
    post_eq_headroom_db: float,
) -> np.ndarray:
    """The pair sum's impulse after applying its already-fitted EQ bank.

    Reuses ``pair_diagnostics``'s already-fitted ``filters``/``shelf`` rather
    than refitting, mirroring how the rest of the pipeline treats a selected
    EQ as fixed once chosen (e.g. the robustness sweep also holds it fixed
    while perturbing delay/gain).
    """
    full_sum, full_frequencies = context.sum_full(first, second, polarity, delay_ms, gain_db)
    n_fft = 2 * (full_sum.size - 1)
    eq_full = filters_response(full_frequencies, context.sample_rate, filters)
    if shelf.get("active"):
        eq_full = eq_full * low_shelf_response(
            full_frequencies,
            context.sample_rate,
            float(shelf["freq_hz"]),
            float(shelf["gain_db"]),
            float(shelf.get("slope", 1.0)),
        )
    post_eq_headroom_linear = 10.0 ** (post_eq_headroom_db / 20.0)
    post_full = full_sum * eq_full * post_eq_headroom_linear
    return np.fft.irfft(post_full, n=n_fft)


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
    arrival_delays_ms: list[float | None] = _worker_state["arrival_delays_ms"]
    arrival_outliers: set[int] = _worker_state["arrival_outliers"]
    arrival_repaired: set[int] = _worker_state["arrival_repaired"]
    search_options: SearchOptions = _worker_state["search_options"]

    first_arrival = arrival_delays_ms[first]
    second_arrival = arrival_delays_ms[second]
    # Delay is applied to sub 2, hence the physical compensation is t_A-t_B.
    # A flagged arrival makes that figure untrustworthy, so it is discarded
    # rather than used to constrain the delay search (see
    # _preoptimization_gates); the pair is then treated exactly like one whose
    # arrival metadata was never captured.
    measurement_outlier = first in arrival_outliers or second in arrival_outliers
    measurement_repaired = first in arrival_repaired or second in arrival_repaired
    physical_tau_ms = (
        float(first_arrival - second_arrival)
        if first_arrival is not None
        and second_arrival is not None
        and not measurement_outlier
        else None
    )
    delta_tau_max_ms, conservative_bound = _geometry_jitter(
        first,
        second,
        search_options.listener_position_m,
        search_options.sub_positions_m,
        search_options.listener_movement_m,
        search_options.speed_of_sound_m_per_s,
    )
    sigma_tau_ms = delta_tau_max_ms / 2.0
    physical_window_ms = search_options.physical_delay_window_ms * (
        ARRIVAL_REPAIRED_WINDOW_FACTOR if measurement_repaired else 1.0
    )

    configurations = _best_configurations(
        context,
        first,
        second,
        delays,
        gains,
        score_low_end_weight,
        score_dip_weight,
        sigma_tau_ms,
        physical_tau_ms,
        physical_window_ms,
        delta_tau_max_ms / 2.0,
        search_options.gate_thresholds.basin_tolerance_db,
    )
    finalists = []
    for (
        polarity,
        delay_ms,
        gain_db,
        _fast_score_db,
        delay_plateau_ms,
        gain_plateau_db,
        robustness,
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
        diagnostics["robustness"] = robustness
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
    robustness = diagnostics["robustness"]
    robustness.update(
        {
            "delta_tau_max_ms": delta_tau_max_ms,
            "sigma_tau_ms": sigma_tau_ms,
            "geometry_conservative_bound": conservative_bound,
            "excursion_penalty_ok": (
                float(robustness["excursion_penalty_db"])
                <= float(robustness["basin_tolerance_db"]) + 1e-12
            ),
            "arrival_delay_first_ms": first_arrival,
            "arrival_delay_second_ms": second_arrival,
            "measurement_delay_outlier": measurement_outlier,
            "arrival_delay_repaired": measurement_repaired,
            "physical_window_widened": measurement_repaired,
            "pair_valid": bool(robustness["physical_window_in_scan"]),
        }
    )
    # Stable top-level names keep the JSON convenient for tabular consumers;
    # the full curves and units remain grouped in ``robustness``.
    diagnostics.update(
        {
            "tau_star": robustness["tau_star_ms"],
            "tau_robust": robustness["tau_robust_ms"],
            "f_tau_star": robustness["f_tau_star_db"],
            "f_robust_tau_robust": robustness["f_robust_tau_robust_db"],
            "fragility": robustness["fragility_db"],
            "basin_w03": robustness["basin_w03_ms"],
            "basin_w05": robustness["basin_w05_ms"],
            "basin_tolerance_db": robustness["basin_tolerance_db"],
            "basin_tolerance_ms": robustness["basin_tolerance_ms"],
            "excursion_half_width_ms": robustness["excursion_half_width_ms"],
            "excursion_penalty_db": robustness["excursion_penalty_db"],
            "worst_case_penalty": robustness["worst_case_penalty_db"],
            "score_quantisation_db": robustness["score_quantisation_db"],
            "n_competing": robustness["n_competing"],
            "geometric_pass": robustness["excursion_penalty_ok"],
            "delta_tau_max": robustness["delta_tau_max_ms"],
            "sigma_tau": robustness["sigma_tau_ms"],
            "geometry_conservative_bound": robustness["geometry_conservative_bound"],
            "physical_tau": robustness["physical_tau_ms"],
            "physical_constraint_available": robustness["physical_tau_ms"] is not None,
            "non_physical_solution": robustness["non_physical_solution"],
            "pair_valid": robustness["pair_valid"],
        }
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

        post_eq_impulse = _post_eq_pair_impulse(
            context,
            first,
            second,
            polarity,
            delay_ms,
            gain_db,
            diagnostics["filters"],
            diagnostics["eq_shelf"],
            diagnostics["post_eq_headroom_db"],
        )
        diagnostics["post_eq_modal"] = modal_analysis.compute_pair_modal_metrics(
            modal_signature, post_eq_impulse, context.sample_rate, modal_options
        )

    # The ranking table's "Tail" column uses the modal-derived audible-ringing
    # time whenever this pair's own fit is valid -- a physically-grounded,
    # audibility-referenced measure (see modal.aggregate_modal_metrics'
    # docstring) -- and otherwise falls back to the CSD-based envelope decay
    # time, so the field is always populated regardless of --modal.
    raw_modal = diagnostics.get("modal") or {}
    post_modal = diagnostics.get("post_eq_modal") or {}
    raw_ringing_ms = raw_modal.get("ringing_ms") if raw_modal.get("valid") else None
    post_ringing_ms = post_modal.get("ringing_ms") if post_modal.get("valid") else None
    diagnostics["effective_tail_ms"] = (
        raw_ringing_ms if raw_ringing_ms is not None else diagnostics["raw_tail_ms"]
    )
    diagnostics["effective_tail_is_modal"] = raw_ringing_ms is not None
    diagnostics["post_eq_effective_tail_ms"] = (
        post_ringing_ms if post_ringing_ms is not None else diagnostics["post_eq_tail_ms"]
    )
    diagnostics["post_eq_effective_tail_is_modal"] = post_ringing_ms is not None
    # ringing_ms saturates at 0 the instant a mode's level drops below
    # audible_margin_db, which a well-controlled room can do for every pair --
    # a wall of identical 0 ms values that carries no ranking information.
    # effective_tail_db carries on below that floor: it's the loudest
    # surviving mode's level relative to direct sound (modal.
    # aggregate_modal_metrics' worst_mode_level_db), so pairs stay
    # distinguishable even when none of them cross the audibility margin. Only
    # populated when the source is modal; the CSD fallback has no dB analogue.
    diagnostics["effective_tail_db"] = (
        raw_modal.get("worst_mode_level_db") if diagnostics["effective_tail_is_modal"] else None
    )
    diagnostics["post_eq_effective_tail_db"] = (
        post_modal.get("worst_mode_level_db")
        if diagnostics["post_eq_effective_tail_is_modal"]
        else None
    )
    return first, second, polarity, delay_ms, gain_db, diagnostics


def run_search(
    cache_dir: Path,
    output_path: Path,
    options: SearchOptions,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    measurements, manifest = load_cache(cache_dir)
    requested_delay_step_ms = float(options.delay_range_ms[2])
    effective_delay_step_ms = min(requested_delay_step_ms, MAX_DELAY_GRID_STEP_MS)
    delays = inclusive_range(
        options.delay_range_ms[0], options.delay_range_ms[1], effective_delay_step_ms
    )
    gains = inclusive_range(*options.gain_range_db)
    context = AnalysisContext(measurements, options.band, options.ppo)
    (
        arrival_delays_ms,
        arrival_repaired,
        arrival_outliers,
        arrival_warnings,
        arrival_diagnostics,
    ) = _resolve_arrival_delays(
        measurements,
        options.band,
        options.speed_of_sound_m_per_s,
        options.room_dimensions_m,
    )
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
    combinations = list(itertools.combinations(range(len(measurements)), 2))
    pre_gates: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    survivors: list[tuple[int, int]] = []
    for first, second in combinations:
        gates = _preoptimization_gates(
            context,
            first,
            second,
            delays,
            arrival_delays_ms[first],
            arrival_delays_ms[second],
            arrival_outliers,
            arrival_repaired,
            options,
        )
        pre_gates[(first, second)] = gates
        if not any(
            gates[name]["status"] == "reject"
            for name in (
                "gate_a_redundancy",
                "gate_b_ripple_correlation",
                "gate_c_physical_percentile",
            )
        ):
            survivors.append((first, second))

    modal_options: ModalOptions | None = None
    modal_signature: RoomModalSignature | None = None
    if options.modal:
        modal_options = ModalOptions()
        modal_signature = modal_analysis.estimate_room_poles(
            [(row.title, row.impulse) for row in measurements],
            context.sample_rate,
            modal_options,
        )
    worker_count = max(1, min((os.cpu_count() or 1) // 4, 8, len(survivors)))
    optimized: dict[tuple[int, int], tuple[int, float, float, dict[str, Any]]] = {}
    if survivors:
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
                arrival_delays_ms,
                arrival_outliers,
                arrival_repaired,
                options,
            ),
        ) as executor:
            for first, second, polarity, delay_ms, gain_db, diagnostics in executor.map(
                _compute_pair, survivors
            ):
                optimized[(first, second)] = (
                    polarity,
                    delay_ms,
                    gain_db,
                    diagnostics,
                )

    pair_records: list[tuple[dict, dict[str, dict[str, Any]]]] = []
    for ordinal, (first, second) in enumerate(combinations, start=1):
        gates = pre_gates[(first, second)]
        result = optimized.get((first, second))
        if result is not None:
            polarity, delay_ms, gain_db, diagnostics = result
            pair = {
                "first": first + 1,
                "second": second + 1,
                "first_name": measurements[first].title,
                "second_name": measurements[second].title,
                "optimized": True,
                "polarity": polarity,
                "delay_ms": delay_ms,
                "gain_db": gain_db,
                **diagnostics,
            }
            gates = _postoptimization_gates(
                context,
                first,
                second,
                pair,
                gates,
                delays,
                options,
            )
        else:
            pair = {
                "first": first + 1,
                "second": second + 1,
                "first_name": measurements[first].title,
                "second_name": measurements[second].title,
                "optimized": False,
                "polarity": None,
                "delay_ms": None,
                "gain_db": None,
            }
        pair_records.append((pair, gates))
        if progress:
            progress(ordinal, len(combinations), f"{first + 1}+{second + 1}")

    # Gate H is a comparison against the other pairs, so it can only be decided
    # once every pair has been scored.
    _apply_band_edge_population_status(
        [gates for _pair, gates in pair_records], options.gate_thresholds
    )

    pairs: list[dict] = []
    for pair, gates in pair_records:
        verdict, reasons = _verdict_and_reasons(gates)
        pair.update(
            {
                "verdict": verdict,
                "reasons": reasons,
                "gates": gates,
                **_gate_summary_fields(gates),
            }
        )
        pairs.append(pair)

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
        pair = pairs[i]
        verdict_order = {"accept": 0, "caution": 1, "reject": 2}
        score = pair.get("score_db")
        base = (
            verdict_order[pair["verdict"]],
            0 if pair["optimized"] else 1,
            -float(score) if score is not None else math.inf,
        )
        tiebreak = _modal_tiebreak_key(pair) if options.modal_tiebreak else ()
        return base + tiebreak + (pair["first"], pair["second"])

    raw_order = sorted(range(len(pairs)), key=_raw_sort_key)
    pairs = [pairs[i] for i in raw_order]
    optimized_pairs = [row for row in pairs if row["optimized"]]
    raw_reference = (
        max(optimized_pairs, key=lambda row: row["score_db"])
        if optimized_pairs
        else None
    )
    reference_spl = raw_reference["spl_db"] if raw_reference is not None else None
    reference_low_end_power = (
        raw_reference["low_end_power_db"] if raw_reference is not None else None
    )
    reference_score = raw_reference["score_db"] if raw_reference is not None else None
    reference_resolution = (
        _score_resolution_db(raw_reference, options.score_tie_margin_db)
        if raw_reference is not None
        else None
    )
    for rank, row in enumerate(pairs, start=1):
        row["rank"] = rank
        row["score_resolution_db"] = _score_resolution_db(
            row, options.score_tie_margin_db
        )
        row["relative_score_db"] = (
            float(row["score_db"] - reference_score) if row["optimized"] else None
        )
        # Ordering two pairs is only meaningful beyond the coarser of their two
        # resolutions; inside it the table is presenting an order the data does
        # not support.
        own = row["score_resolution_db"]
        # The reference is trivially within its own resolution of itself, which
        # says nothing; the flag is about pairs the table orders *against* it.
        row["score_ties_reference"] = (
            bool(abs(row["relative_score_db"]) <= max(own, reference_resolution or 0.0))
            if row["optimized"] and own is not None and row is not raw_reference
            else None
        )
        row["relative_spl_db"] = (
            float(row["spl_db"] - reference_spl) if row["optimized"] else None
        )
        row["relative_low_end_power_db"] = (
            float(row["low_end_power_db"] - reference_low_end_power)
            if row["optimized"]
            else None
        )

    def _eq_sort_key(i: int) -> tuple:
        pair = pairs[i]
        verdict_order = {"accept": 0, "caution": 1, "reject": 2}
        score = pair.get("post_eq_score_db")
        base = (
            verdict_order[pair["verdict"]],
            0 if pair["optimized"] else 1,
            -float(score) if score is not None else math.inf,
        )
        tiebreak = _modal_tiebreak_key(pair) if options.modal_tiebreak else ()
        return base + tiebreak + (pair["first"], pair["second"])

    eq_order = sorted(range(len(pairs)), key=_eq_sort_key)
    eq_pairs = [pairs[i] for i in eq_order]
    eq_optimized_pairs = [row for row in eq_pairs if row["optimized"]]
    eq_reference = (
        max(eq_optimized_pairs, key=lambda row: row["post_eq_score_db"])
        if eq_optimized_pairs
        else None
    )
    eq_reference_spl = (
        eq_reference["post_eq_spl_db"] if eq_reference is not None else None
    )
    eq_reference_low_end_power = (
        eq_reference["post_eq_low_end_power_db"]
        if eq_reference is not None
        else None
    )
    eq_reference_score = (
        eq_reference["post_eq_score_db"] if eq_reference is not None else None
    )
    eq_reference_resolution = (
        _score_resolution_db(eq_reference, options.score_tie_margin_db)
        if eq_reference is not None
        else None
    )
    for eq_rank, row in enumerate(eq_pairs, start=1):
        row["eq_rank"] = eq_rank
        row["post_eq_relative_score_db"] = (
            float(row["post_eq_score_db"] - eq_reference_score)
            if row["optimized"]
            else None
        )
        # The EQ'd score is fitted per configuration rather than read off the
        # grid, so it has no quantisation figure of its own; the delay/gain grid
        # underneath it is the same one, so the raw resolution is reused.
        own = row["score_resolution_db"]
        row["post_eq_score_ties_reference"] = (
            bool(
                abs(row["post_eq_relative_score_db"])
                <= max(own, eq_reference_resolution or 0.0)
            )
            if row["optimized"] and own is not None and row is not eq_reference
            else None
        )
        row["post_eq_relative_spl_db"] = (
            float(row["post_eq_spl_db"] - eq_reference_spl)
            if row["optimized"]
            else None
        )
        row["post_eq_relative_low_end_power_db"] = (
            float(row["post_eq_low_end_power_db"] - eq_reference_low_end_power)
            if row["optimized"]
            else None
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
        "format_version": 28,
        "measurement_count": len(measurements),
        "optimized_pair_count": len(optimized_pairs),
        "sample_rate": measurements[0].sample_rate,
        "response_length": measurements[0].impulse.size,
        "settings": {
            "band_hz": list(options.band),
            "ppo": options.ppo,
            "delay_range_ms": list(options.delay_range_ms),
            "delay_grid_step_ms": effective_delay_step_ms,
            "gain_range_db": list(options.gain_range_db),
            "gates": {
                "thresholds": asdict(options.gate_thresholds),
                "order": list(_GATE_ORDER),
                "verdict_order": ["accept", "caution", "reject"],
                "note": (
                    "Every gate is a disqualifier, not a certificate. Stage 1 "
                    "redundancy/ripple rejects and Stage 2 physical-percentile "
                    "rejects skip the expensive optimiser. Passing all gates only "
                    "means no configured single-position failure mode was detected; "
                    "only multi-position measurements validate a listening area."
                ),
                "arrival_timing": (
                    "REW's arrival delay is the position of the largest sample in "
                    "the impulse response. Across a subwoofer's two or three "
                    "octaves that impulse is a slow oscillatory blob, so wherever a "
                    "room mode rings hard a later half-cycle can outgrow the direct "
                    "arrival and the pick jumps a whole cycle. The leading edge does "
                    "not jump, so a measurement whose peak-minus-onset lag departs "
                    "from the cache median by more than slip_tolerance_ms is "
                    "repaired: its peak is rebuilt from its own onset plus the "
                    "median lag, keeping that position's genuine distance instead of "
                    "flattening it to the median arrival. Only differences between "
                    "arrivals are used downstream, so a bias common to every onset "
                    "cancels and is left uncorrected. A repaired arrival still aims "
                    "the delay search, but Gate C is capped at caution for its pairs "
                    "-- a poor timing pick must not change which pairs are "
                    "recommended, only how exact the reported delay figure is. See "
                    "the top-level arrival_timing block for per-measurement values."
                ),
                "physical_baseline": (
                    "Gates C, D and I reference the pair at its measured arrival "
                    "alignment with equal gain and whichever polarity is better "
                    "there. Polarity is free, exact and drift-immune, so it is "
                    "part of 'no tuning applied'; delay and gain are the fragile "
                    "tuning these gates police. An arrival-delay outlier discards "
                    "that pair's physical timing entirely -- the same caution path "
                    "as absent arrival metadata, since a known-bad reading is not "
                    "more informative than a missing one."
                ),
                "delay_scan_boundary": (
                    "A selected delay pinned to the edge of --delay-range is "
                    "rejected under Gate C: it is the limit of the search, not "
                    "an optimum, and every robustness figure around it is "
                    "one-sided. Only pairs with no usable physical timing can "
                    "reach the edge."
                ),
                "basin_geometry": (
                    "excursion_penalty_db is the worst degradation of f over "
                    "+/-delta_tau_max/2 around the recommended delay; reject above "
                    "basin_tolerance_db. The tolerance is absolute in dB and "
                    "measured at the recommended delay: a tolerance scaled to each "
                    "pair's own objective range normalises away the very "
                    "delay-insensitivity the gate is testing for."
                ),
                "band_edge_stability": (
                    "Gate H rejects on excess_spread_db -- this pair's +/-1/6-octave "
                    "band-shift score spread minus the population median -- above "
                    "band_edge_excess_spread_reject_db. Shifting the band moves "
                    "every pair by nearly the same amount (the subs roll off below "
                    "the band and low-end power weights f^-4), so the raw spread is "
                    "a property of the score, not of a pair. With one scored pair "
                    "the excess is zero and the gate abstains."
                ),
                "improvement_localization": (
                    "Gate I rejects only when the improvement over the physical "
                    "baseline is material -- mean positive ripple improvement at "
                    "or above localization_min_mean_improvement_db per log-frequency "
                    "bin -- and more than localization_fraction_reject of it falls "
                    "in one 1/6-octave window. Below that floor the fraction is a "
                    "ratio of two noise-level sums and carries no information."
                ),
            },
            "robustness": {
                "objective": "f = -raw usable-output score (lower is better)",
                "listener_movement_m": options.listener_movement_m,
                "speed_of_sound_m_per_s": options.speed_of_sound_m_per_s,
                "physical_delay_window_ms": options.physical_delay_window_ms,
                "gain_jitter_sigma_db": GAIN_JITTER_SIGMA_DB,
                "geometry_configured": (
                    options.listener_position_m is not None
                    and options.sub_positions_m is not None
                ),
                "listener_position_m": options.listener_position_m,
                "sub_positions_m": options.sub_positions_m,
                "room_dimensions_m": options.room_dimensions_m,
                "note": (
                    "sigma_tau is half the pair-specific maximum differential "
                    "arrival-time excursion for the configured listener movement. "
                    "Without complete coordinates, 2d/c is used and each pair is "
                    "marked geometry_conservative_bound. geometric_pass is "
                    "excursion_penalty_db -- the worst degradation of f over "
                    "+/-delta_tau_max/2 around the recommended delay -- against the "
                    "absolute basin_tolerance_db; basin_w03/w05 remain as "
                    "fixed-threshold diagnostics around tau_star. A large excursion "
                    "penalty proves timing fragility, but a small one does not model "
                    "the magnitude-response change caused by moving through the "
                    "room's modal pressure field. Only multi-position measurements "
                    "can validate a listening area."
                ),
            },
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
                    "aggregate n_highQ/Q_max/sum_modal_energy_db/ringing_ms "
                    "per pair (both a raw 'modal' block and a post-EQ "
                    "'post_eq_modal' block, the latter against the "
                    "already-fitted, not re-derived, EQ bank); diagnostic-"
                    "only and does not affect score_db/post_eq_score_db. "
                    "When tiebreak is enabled, (n_highQ, sum_modal_energy_db) "
                    "- both lower-is-better - is inserted strictly after the "
                    "primary usable-output score, before the deterministic "
                    "pair-index tie-break; off by default. ringing_ms feeds "
                    "effective_tail_ms/post_eq_effective_tail_ms and "
                    "worst_mode_level_db feeds effective_tail_db/"
                    "post_eq_effective_tail_db (see 'effective_tail' below) "
                    "whenever a pair's own fit is valid. The 0 dB reference "
                    "for level_db/worst_mode_level_db is the RMS of the "
                    "band-limited direct arrival over a window spanning at "
                    "least one period of the lowest in-band mode (floored at "
                    "20 ms), not a single peak sample in a fixed 20 ms window "
                    "-- a peak sample in a window shorter than the mode's own "
                    "period measures the onset transient, not the mode. See "
                    "modal_signature for the pooled room pole set this was "
                    "computed against"
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
                "raw": ["verdict", "score_db"],
                "eq": ["verdict", "post_eq_score_db"],
                "excess_gd_range_hz": list(eq_range),
                "direction": (
                    "accept before caution before reject; higher score within verdict"
                ),
                "resolution": (
                    "score_resolution_db is the smallest score difference a pair "
                    "can be ordered by: how far the score moves between adjacent "
                    "points of the delay/gain grid, plus that pair's excess over "
                    "the population median in Gate H's band shift (the shared part "
                    "cancels out of an ordering; the excess does not). "
                    "score_ties_reference marks pairs within the coarser of their "
                    "own and the reference pair's resolution, where the table is "
                    "presenting an order the data does not support. It is a floor: "
                    "microphone placement, level calibration and measurement noise "
                    "all add to it and none are visible in a single cached sweep. "
                    "--score-tie-margin adds a flat allowance for exactly those, "
                    "and is included in every reported score_resolution_db."
                ),
                "score_tie_margin_db": options.score_tie_margin_db,
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
                        "the exhaustive raw pass evaluates the full delay grid, "
                        "Gaussian-averages f=-score over pair-specific timing jitter "
                        "and +/-1 dB gain jitter, then chooses a robust delay for each "
                        f"polarity/gain and shortlists up to {SHORTLIST_PER_OBJECTIVE} "
                        f"lowest-robust-f and {SHORTLIST_PER_OBJECTIVE} lowest-dip "
                        "configurations; fitted post-EQ score selects polarity/gain "
                        "from that shortlist without changing the robust delay"
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
                "effective_tail": (
                    "effective_tail_ms/post_eq_effective_tail_ms are this pair's "
                    "modal.ringing_ms/post_eq_modal.ringing_ms (worst-case time "
                    "for any detected mode to fall below the audibility margin "
                    "relative to direct sound) when that pair's own --modal fit "
                    "is valid, else the original CSD-based raw_tail_ms/"
                    "post_eq_tail_ms envelope decay time. "
                    "effective_tail_is_modal/post_eq_effective_tail_is_modal "
                    "record which source was used for that pair. Always "
                    "present regardless of --modal; a diagnostic, not a score "
                    "component. The report's/CLI's 'Tail' column shows "
                    "effective_tail_db/post_eq_effective_tail_db (this pair's "
                    "modal.worst_mode_level_db/post_eq_modal.worst_mode_level_db, "
                    "the loudest detected mode's level relative to direct sound, "
                    "in dB) instead of the ms value whenever the source is "
                    "modal: ringing_ms saturates at 0 for every mode below the "
                    "audibility margin, which a well-controlled room can do for "
                    "every pair, so the ms value alone can be uninformative; "
                    "the dB value keeps varying below that floor. Falls back to "
                    "the ms value, with no dB equivalent, when the source is "
                    "the CSD envelope decay time"
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
        "warnings": arrival_warnings,
        "measurement_arrival_delays_ms": arrival_delays_ms,
        "measurement_arrival_delay_outliers": sorted(index + 1 for index in arrival_outliers),
        "measurement_arrival_delay_repaired": sorted(index + 1 for index in arrival_repaired),
        "arrival_timing": arrival_diagnostics,
        "modal_signature": modal_signature_json,
        "pairs": pairs,
    }
    write_json(output_path, result)
    return result
