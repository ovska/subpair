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
    AnalysisContext,
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


def _arrival_outliers(
    measurements: list[Any],
    speed_of_sound_m_per_s: float,
    room_dimensions_m: tuple[float, float, float] | None,
) -> tuple[list[float | None], set[int], list[str]]:
    delays = [_metadata_arrival_delay_ms(row.metadata) for row in measurements]
    positive = [value for value in delays if value is not None and value > 0.0]
    median = float(np.median(positive)) if positive else None
    room_diagonal = (
        math.sqrt(sum(dimension * dimension for dimension in room_dimensions_m))
        if room_dimensions_m is not None
        else None
    )
    outliers: set[int] = set()
    warnings: list[str] = []
    for index, (row, delay_ms) in enumerate(zip(measurements, delays)):
        reasons = []
        if delay_ms is None:
            warnings.append(
                f"Position {row.position} ({row.title}) has no parsed REW arrival delay; "
                "physical delay constraints are unavailable for its pairs. Re-fetch with "
                "a REW build that exposes loopback-referenced arrival metadata."
            )
            continue
        if median is not None and delay_ms > 1.5 * median:
            reasons.append(f"more than 1.5x the median ({median:.3f} ms)")
        path_m = delay_ms * speed_of_sound_m_per_s / 1000.0
        if room_diagonal is not None and path_m > room_diagonal:
            reasons.append(
                f"{path_m:.2f} m path exceeds the {room_diagonal:.2f} m room diagonal"
            )
        if reasons:
            outliers.add(index)
            warnings.append(
                f"Arrival delay outlier: position {row.position} ({row.title}) reports "
                f"{delay_ms:.3f} ms / {path_m:.2f} m; " + "; ".join(reasons) + "."
            )
    return delays, outliers, warnings


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
        raw_curve = objective[polarity_index, :, gain_index]
        robust_curve = robust_objective[polarity_index, :, gain_index]
        tau_star_index = int(np.argmin(raw_curve))
        tau_star_ms = float(delays[tau_star_index])
        tau_robust_ms = float(delays[delay_index])
        minima = _local_minima_indices(raw_curve)
        competing = int(
            np.count_nonzero(raw_curve[minima] <= raw_curve[tau_star_index] + 0.3 + 1e-12)
        )
        worst_case = {
            f"{dt:.1f}": _worst_case(raw_curve, delays, tau_star_ms, dt)
            for dt in (0.5, 1.0, 1.5)
        }
        basin_w03 = _basin_width(raw_curve, delays, tau_star_index, 0.3)
        basin_w05 = _basin_width(raw_curve, delays, tau_star_index, 0.5)
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
            "worst_case_db": worst_case,
            "n_competing": competing,
            "physical_tau_ms": physical_tau_ms,
            "physical_window_ms": physical_window_ms,
            "physical_window_in_scan": physical_window_in_scan,
            "non_physical_solution": non_physical,
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
    search_options: SearchOptions = _worker_state["search_options"]

    first_arrival = arrival_delays_ms[first]
    second_arrival = arrival_delays_ms[second]
    # Delay is applied to sub 2, hence the physical compensation is t_A-t_B.
    physical_tau_ms = (
        float(first_arrival - second_arrival)
        if first_arrival is not None and second_arrival is not None
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
        search_options.physical_delay_window_ms,
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
    measurement_outlier = first in arrival_outliers or second in arrival_outliers
    robustness.update(
        {
            "delta_tau_max_ms": delta_tau_max_ms,
            "sigma_tau_ms": sigma_tau_ms,
            "geometry_conservative_bound": conservative_bound,
            "basin_covers_geometry": (
                float(robustness["basin_w03_ms"]) + 1e-12 >= delta_tau_max_ms
            ),
            "arrival_delay_first_ms": first_arrival,
            "arrival_delay_second_ms": second_arrival,
            "measurement_delay_outlier": measurement_outlier,
            "pair_valid": (
                not measurement_outlier and bool(robustness["physical_window_in_scan"])
            ),
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
            "worst_case": robustness["worst_case_db"],
            "n_competing": robustness["n_competing"],
            "geometric_pass": robustness["basin_covers_geometry"],
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
    arrival_delays_ms, arrival_outliers, arrival_warnings = _arrival_outliers(
        measurements,
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
            arrival_delays_ms,
            arrival_outliers,
            options,
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
        # Invalid arrival metadata and a raw optimum outside the physical
        # window are disqualifiers. Score remains the primary ordering among
        # physically credible pairs and its formula is unchanged.
        eligibility = (
            0 if pairs[i].get("pair_valid", True) else 1,
            1 if pairs[i].get("non_physical_solution", False) else 0,
        )
        base = eligibility + (-pairs[i]["score_db"],)
        tiebreak = _modal_tiebreak_key(pairs[i]) if options.modal_tiebreak else ()
        return base + tiebreak + (pairs[i]["first"], pairs[i]["second"])

    raw_order = sorted(range(len(pairs)), key=_raw_sort_key)
    pairs = [pairs[i] for i in raw_order]
    raw_reference = max(pairs, key=lambda row: row["score_db"])
    reference_spl = raw_reference["spl_db"]
    reference_low_end_power = raw_reference["low_end_power_db"]
    reference_score = raw_reference["score_db"]
    for rank, row in enumerate(pairs, start=1):
        row["rank"] = rank
        row["relative_score_db"] = float(row["score_db"] - reference_score)
        row["relative_spl_db"] = float(row["spl_db"] - reference_spl)
        row["relative_low_end_power_db"] = float(
            row["low_end_power_db"] - reference_low_end_power
        )
    def _eq_sort_key(i: int) -> tuple:
        eligibility = (
            0 if pairs[i].get("pair_valid", True) else 1,
            1 if pairs[i].get("non_physical_solution", False) else 0,
        )
        base = eligibility + (-pairs[i]["post_eq_score_db"],)
        tiebreak = _modal_tiebreak_key(pairs[i]) if options.modal_tiebreak else ()
        return base + tiebreak + (pairs[i]["first"], pairs[i]["second"])

    eq_order = sorted(range(len(pairs)), key=_eq_sort_key)
    eq_pairs = [pairs[i] for i in eq_order]
    eq_reference = max(eq_pairs, key=lambda row: row["post_eq_score_db"])
    eq_reference_spl = eq_reference["post_eq_spl_db"]
    eq_reference_low_end_power = eq_reference["post_eq_low_end_power_db"]
    eq_reference_score = eq_reference["post_eq_score_db"]
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
        "format_version": 24,
        "measurement_count": len(measurements),
        "sample_rate": measurements[0].sample_rate,
        "response_length": measurements[0].impulse.size,
        "settings": {
            "band_hz": list(options.band),
            "ppo": options.ppo,
            "delay_range_ms": list(options.delay_range_ms),
            "delay_grid_step_ms": effective_delay_step_ms,
            "gain_range_db": list(options.gain_range_db),
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
                    "marked geometry_conservative_bound. Basin width is only a "
                    "disqualifier: a narrow basin proves timing fragility, but a "
                    "wide basin does not model the magnitude-response change caused "
                    "by moving through the room's modal pressure field. Only "
                    "multi-position measurements can validate a listening area."
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
        "modal_signature": modal_signature_json,
        "pairs": pairs,
    }
    write_json(output_path, result)
    return result
