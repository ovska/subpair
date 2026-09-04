"""Numerical primitives used by the search, report, and verifier."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy import ndimage, optimize, signal
from scipy.fft import next_fast_len

from .cache import CachedMeasurement


EPS = np.finfo(np.float64).tiny


def db20(value: np.ndarray, floor_db: float = -300.0) -> np.ndarray:
    value = np.asarray(value)
    peak = float(np.max(np.abs(value))) if value.size else 1.0
    floor = max(peak * 10.0 ** (floor_db / 20.0), EPS)
    return 20.0 * np.log10(np.maximum(np.abs(value), floor))


def inclusive_range(low: float, high: float, step: float) -> np.ndarray:
    if not np.isfinite([low, high, step]).all() or step <= 0 or high < low:
        raise ValueError(f"Invalid range {low:g}..{high:g} step {step:g}")
    count = int(math.floor((high - low) / step + 1e-9)) + 1
    values = low + step * np.arange(count, dtype=np.float64)
    if values[-1] < high - step * 1e-7:
        values = np.append(values, high)
    values[-1] = min(values[-1], high)
    return values


def log_frequency_grid(low: float, high: float, ppo: int = 48) -> np.ndarray:
    if low <= 0 or high <= low or ppo < 1:
        raise ValueError(f"Invalid log grid band {low:g}..{high:g} Hz at {ppo} PPO")
    count = int(math.floor(math.log2(high / low) * ppo + 1e-10)) + 1
    grid = low * 2.0 ** (np.arange(count, dtype=np.float64) / ppo)
    if grid[-1] < high * (1.0 - 1e-10):
        grid = np.append(grid, high)
    return grid


def margin_frequencies(
    frequencies: np.ndarray, ppo: int, margin_bins: int
) -> tuple[np.ndarray, slice]:
    """Extend a log-frequency grid by ``margin_bins`` samples on each side.

    ``broad_trend_db`` smooths with ``mode="nearest"``, which replicates the
    edge sample rather than using real spectral content beyond the reported
    band; that biases the trend (and any score derived from it) close to the
    band boundaries. Evaluating the trend over this wider, real-data grid and
    cropping the result back with the returned slice removes that bias
    without changing what is actually reported or scored.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)
    below = frequencies[0] * 2.0 ** (-np.arange(margin_bins, 0, -1) / ppo)
    above = frequencies[-1] * 2.0 ** (np.arange(1, margin_bins + 1) / ppo)
    extended = np.concatenate([below, frequencies, above])
    return extended, slice(margin_bins, margin_bins + frequencies.size)


def _interp_complex(source_f: np.ndarray, source_h: np.ndarray, target_f: np.ndarray) -> np.ndarray:
    return np.interp(target_f, source_f, source_h.real) + 1j * np.interp(
        target_f, source_f, source_h.imag
    )


def broad_trend_db(values: np.ndarray, ppo: int) -> np.ndarray:
    sigma = ppo / 2.354820045  # one-octave FWHM
    return ndimage.gaussian_filter1d(
        np.asarray(values, dtype=np.float64),
        sigma=max(sigma, 0.01),
        axis=-1,
        mode="nearest",
        truncate=3.0,
    )


SCORE_DIP_SMOOTHING_OCTAVES = 1.0 / 3.0
DEFAULT_SCORE_LOW_END_WEIGHT = 0.5
DEFAULT_SCORE_DIP_WEIGHT = 1.0


def smoothed_dip_db(
    magnitude_db: np.ndarray,
    ppo: int,
    smoothing_octaves: float = SCORE_DIP_SMOOTHING_OCTAVES,
    score_slice: slice | None = None,
) -> np.ndarray | float:
    """Worst negative deviation from a fractional-octave smoothed response.

    Unlike the old null detector, this has no two-sided recovery heuristic and
    no group-delay multiplier. It answers the visually direct question: how
    far does the response fall below its local, smoothed shape? A one-third-
    octave FWHM follows broad roll-off while retaining narrow cancellation
    dips. Callers with real response data beyond the scored range should pass
    that margin and crop only the residual via ``score_slice``.
    """
    magnitude_db = np.asarray(magnitude_db, dtype=np.float64)
    if magnitude_db.ndim < 1 or magnitude_db.shape[-1] == 0:
        raise ValueError("Smoothed-dip magnitude must have a non-empty final axis")
    if ppo < 1 or not math.isfinite(smoothing_octaves) or smoothing_octaves <= 0.0:
        raise ValueError("Smoothed-dip resolution and width must be positive")
    sigma = ppo * smoothing_octaves / 2.354820045
    smoothed = ndimage.gaussian_filter1d(
        magnitude_db,
        sigma=max(sigma, 0.01),
        axis=-1,
        mode="nearest",
        truncate=3.0,
    )
    residual = np.maximum(0.0, smoothed - magnitude_db)
    if score_slice is not None:
        residual = residual[..., score_slice]
    result = np.max(residual, axis=-1)
    return float(result) if result.ndim == 0 else result


def usable_output_score_db(
    spl_db: np.ndarray | float,
    low_end_power: np.ndarray | float,
    dip_db: np.ndarray | float,
    low_end_weight: float = DEFAULT_SCORE_LOW_END_WEIGHT,
    dip_weight: float = DEFAULT_SCORE_DIP_WEIGHT,
) -> np.ndarray | float:
    """Equal-drive output minus the weighted local-dip penalty, in dB.

    ``low_end_weight`` interpolates in dB between ordinary full-band pressure
    power and excursion-weighted low-end power. ``dip_weight`` controls how
    many score dB are deducted per dB below the one-third-octave reference.
    The result shifts with the cache's arbitrary level reference, but score
    differences and ordering do not; the engine also serializes a best=0 dB
    relative score for presentation.
    """
    if not math.isfinite(low_end_weight) or not 0.0 <= low_end_weight <= 1.0:
        raise ValueError("Score low-end weight must be between 0 and 1")
    if not math.isfinite(dip_weight) or dip_weight < 0.0:
        raise ValueError("Score dip weight must be non-negative")
    spl = np.asarray(spl_db, dtype=np.float64)
    low_end = np.asarray(low_end_power, dtype=np.float64)
    dip = np.asarray(dip_db, dtype=np.float64)
    score = (1.0 - low_end_weight) * spl + low_end_weight * low_end - dip_weight * dip
    return float(score) if score.ndim == 0 else score


def _grid_ppo(frequencies: np.ndarray) -> float:
    """Points per octave implied by a log-frequency grid's actual spacing."""
    if frequencies.size < 2:
        return 48.0
    steps = np.diff(np.log2(np.asarray(frequencies, dtype=np.float64)))
    return max(1.0, 1.0 / float(np.median(steps)))


LOW_END_POWER_UPPER_HZ = 100.0
EXCURSION_PRESSURE_FREQUENCY_EXPONENT = 2.0
EXCURSION_POWER_FREQUENCY_EXPONENT = (
    2.0 * EXCURSION_PRESSURE_FREQUENCY_EXPONENT
)
EXCURSION_POWER_DB_PER_OCTAVE = float(
    10.0 * np.log10(2.0**EXCURSION_POWER_FREQUENCY_EXPONENT)
)


def low_end_power_db(
    trend_db: np.ndarray,
    frequencies: np.ndarray,
    upper_hz: float = LOW_END_POWER_UPPER_HZ,
) -> np.ndarray | float:
    """Excursion-cost-weighted mean low-frequency pressure power.

    In the pistonic region, acoustic pressure is proportional to ``f**2``
    times cone displacement. Producing the same pressure one octave lower
    therefore needs four times the displacement and, under the deliberately
    simple voltage-proportional-to-displacement model available without
    driver impedance/T-S data, sixteen times the amplifier power. The score
    weights pressure power by ``f**-4`` (+12.04 dB/octave toward lower
    frequencies) before integrating over log frequency. A normalized
    weighted mean keeps a perfectly flat curve's score equal to its level.

    Only the analyzed range through ``upper_hz`` contributes. If an analysis
    band lies wholly above that limit, its complete range is used rather than
    returning an undefined diagnostic. ``trend_db`` is expected to be the
    one-octave broad response, so narrow placement nulls do not dominate this
    extension-oriented metric.

    The caller must pass the final equal-drive response, including its global
    headroom gain. Keeping headroom in the response itself -- rather than as
    a special deduction inside this one metric -- ensures magnitude plots,
    Relative SPL, and low-end power all compare exactly the same signal.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)
    trend_db = np.asarray(trend_db, dtype=np.float64)
    if frequencies.ndim != 1 or trend_db.ndim < 1:
        raise ValueError("Low-end power frequencies must be one-dimensional")
    if trend_db.shape[-1] != frequencies.size or frequencies.size == 0:
        raise ValueError(
            "Low-end power frequencies must match the trend's non-empty final axis"
        )
    if np.any(frequencies <= 0.0) or np.any(np.diff(frequencies) <= 0.0):
        raise ValueError("Low-end power frequencies must be positive and increasing")
    if not np.all(np.isfinite(trend_db)):
        raise ValueError("Low-end power trend must contain only finite values")
    if not math.isfinite(upper_hz) or upper_hz <= 0.0:
        raise ValueError("Low-end power upper frequency must be positive and finite")

    mask = frequencies <= upper_hz
    if not np.any(mask):
        mask = np.ones(frequencies.shape, dtype=bool)
    used_frequencies = frequencies[mask]
    used_trend_db = trend_db[..., mask]
    if used_frequencies.size == 1:
        result = used_trend_db[..., 0]
        return float(result) if result.ndim == 0 else result

    # The reference frequency cancels between numerator and denominator; the
    # highest included frequency keeps the intermediate weights near unity.
    weights = (
        used_frequencies[-1] / used_frequencies
    ) ** EXCURSION_POWER_FREQUENCY_EXPONENT
    peak_db = np.max(used_trend_db, axis=-1, keepdims=True)
    relative_pressure_power = 10.0 ** ((used_trend_db - peak_db) / 10.0)
    log_frequencies = np.log(used_frequencies)
    weighted_power = np.trapezoid(
        relative_pressure_power * weights, x=log_frequencies, axis=-1
    ) / float(np.trapezoid(weights, x=log_frequencies))
    result = peak_db[..., 0] + 10.0 * np.log10(np.maximum(weighted_power, EPS))
    return float(result) if result.ndim == 0 else result


def peq_response(
    frequencies: np.ndarray, sample_rate: float, fc: float, q: float, gain_db: float
) -> np.ndarray:
    """RBJ Audio EQ Cookbook peaking biquad response."""
    f = np.asarray(frequencies, dtype=np.float64)
    omega = 2.0 * np.pi * f / sample_rate
    omega0 = 2.0 * np.pi * fc / sample_rate
    cosine0 = np.cos(omega0)
    sine0 = np.sin(omega0)
    a = 10.0 ** (gain_db / 40.0)
    alpha = sine0 / (2.0 * q)
    b0 = 1.0 + alpha * a
    b1 = -2.0 * cosine0
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cosine0
    a2 = 1.0 - alpha / a
    z1 = np.exp(-1j * omega)
    z2 = z1 * z1
    numerator = b0 + b1 * z1 + b2 * z2
    denominator = a0 + a1 * z1 + a2 * z2
    response = np.ones_like(numerator, dtype=np.complex128)
    np.divide(numerator, denominator, out=response, where=np.abs(denominator) > 1e-14)
    return response


def low_shelf_response(
    frequencies: np.ndarray,
    sample_rate: float,
    fc: float,
    gain_db: float,
    slope: float = 1.0,
) -> np.ndarray:
    """RBJ Audio EQ Cookbook low-shelf biquad response.

    ``slope`` is the RBJ "S" shelf-slope parameter, ``0 < S <= 1``; ``S = 1``
    is the steepest transition available without gain overshoot. Unlike
    ``peq_response``'s constant-Q bell, this approaches ``gain_db`` well
    below ``fc`` and 0 dB well above it. ``fit_eq_filters`` may fit one such
    shelf automatically as part of the same correction bank as its PK bands.
    """
    f = np.asarray(frequencies, dtype=np.float64)
    omega = 2.0 * np.pi * f / sample_rate
    omega0 = 2.0 * np.pi * fc / sample_rate
    cosine0 = np.cos(omega0)
    sine0 = np.sin(omega0)
    a = 10.0 ** (gain_db / 40.0)
    sqrt_a = math.sqrt(a)
    alpha = 0.5 * sine0 * math.sqrt((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0)
    b0 = a * ((a + 1.0) - (a - 1.0) * cosine0 + 2.0 * sqrt_a * alpha)
    b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cosine0)
    b2 = a * ((a + 1.0) - (a - 1.0) * cosine0 - 2.0 * sqrt_a * alpha)
    a0 = (a + 1.0) + (a - 1.0) * cosine0 + 2.0 * sqrt_a * alpha
    a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cosine0)
    a2 = (a + 1.0) + (a - 1.0) * cosine0 - 2.0 * sqrt_a * alpha
    z1 = np.exp(-1j * omega)
    z2 = z1 * z1
    numerator = b0 + b1 * z1 + b2 * z2
    denominator = a0 + a1 * z1 + a2 * z2
    response = np.ones_like(numerator, dtype=np.complex128)
    np.divide(numerator, denominator, out=response, where=np.abs(denominator) > 1e-14)
    return response


@dataclass(frozen=True)
class EqOptions:
    target: str = "trend"
    correction_range: tuple[float, float] | None = None
    correction_slope_db_per_octave: float = 48.0
    max_boost_db: float = 0.0
    max_cut_db: float = 18.0
    max_filters: int = 7
    low_shelf: bool = True

    def __post_init__(self) -> None:
        if self.target not in {"trend", "flat", "dsp"}:
            raise ValueError("EQ target must be 'trend', 'flat', or 'dsp'")
        if self.correction_range is not None:
            low, high = self.correction_range
            if low <= 0 or high <= low:
                raise ValueError("EQ correction range must be positive and increasing")
        if not 0.0 <= self.correction_slope_db_per_octave <= 48.0:
            raise ValueError("EQ correction range slope must be between 0 and 48 dB/oct")
        if not 0.0 <= self.max_boost_db <= 12.0:
            raise ValueError("EQ maximum boost must be between 0 and 12 dB")
        if not 0.0 <= self.max_cut_db <= 30.0:
            raise ValueError("EQ maximum cut must be between 0 and 30 dB")
        if not 0 <= self.max_filters <= 16:
            raise ValueError("EQ filter count must be between 0 and 16")


def _correction_range_authority(
    frequencies: np.ndarray,
    correction_range: tuple[float, float],
    slope_db_per_octave: float,
) -> np.ndarray:
    """Return 1 in-range and a configurable correction curtain outside it.

    A zero slope is interpreted as a hard curtain. Positive values attenuate
    correction authority by that many dB per octave outside each boundary.
    """
    low, high = correction_range
    authority = np.ones_like(frequencies, dtype=np.float64)
    below = frequencies < low
    above = frequencies > high
    if slope_db_per_octave == 0.0:
        authority[below | above] = 0.0
        return authority
    outside_octaves = np.zeros_like(frequencies, dtype=np.float64)
    outside_octaves[below] = np.log2(low / frequencies[below])
    outside_octaves[above] = np.log2(frequencies[above] / high)
    authority[below | above] = 10.0 ** (
        -slope_db_per_octave * outside_octaves[below | above] / 20.0
    )
    return authority


def _excess_gd_authority(
    frequencies: np.ndarray, excess_group_delay_ms: np.ndarray
) -> np.ndarray:
    """Gate broad excess-delay regions without following narrow pointwise wiggles.

    Risk is a *maximum* filter over at least a one-third-octave window, not
    an averaging one: a moving average (Gaussian or boxcar) inherently
    dilutes any feature narrower than its own window in proportion to how
    much narrower it is, which used to mean a genuinely severe but narrow
    (a few bins) excess-GD spike was almost entirely ignored while a wider,
    shallower bump of the *same peak height* was heavily gated. A maximum
    filter instead reproduces a feature's true peak height everywhere within
    reach of it regardless of that feature's own width, so a narrow spike
    and a wide bump of equal height are gated the same. Only a light
    denoise (well under one bin FWHM) runs first, just enough to keep a
    single noisy sample from single-handedly setting the gate; the final
    Gaussian pass only softens the gate's edges; the maximum filter already
    fixed the true height by then.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)
    excess_cycles = np.abs(excess_group_delay_ms) * frequencies / 1000.0
    if frequencies.size < 3:
        return 1.0 / (1.0 + (excess_cycles / 0.35) ** 4)
    ppo = _grid_ppo(frequencies)
    denoised_cycles = ndimage.gaussian_filter1d(
        excess_cycles, sigma=0.35, mode="nearest", truncate=3.0
    )
    gate_size = max(1, int(round(ppo / 3.0)))
    risk = ndimage.maximum_filter1d(denoised_cycles, size=gate_size, mode="nearest")
    authority = 1.0 / (1.0 + (risk / 0.35) ** 4)
    return ndimage.gaussian_filter1d(
        authority,
        sigma=max(0.5, (ppo / 4.0) / 2.354820045),
        mode="nearest",
        truncate=3.0,
    )


def _denoised_residual(residual: np.ndarray, ppo: int) -> np.ndarray:
    """Lightly smooth a target-error curve for peak/bandwidth detection only.

    Picking candidate filters directly from raw, single-bin target error is
    sensitive to measurement ripple: a one-bin noise spike can steer a narrow,
    high-Q cut at an artifact instead of a real modal peak. A sub-octave
    (~1/12-octave FWHM) Gaussian suppresses that without blurring genuine room
    modes, which are rarely narrower than this. The resulting filter is still
    accepted or rejected against the true, unsmoothed residual.
    """
    sigma = max(0.6, (ppo / 12.0) / 2.354820045)
    return ndimage.gaussian_filter1d(
        np.asarray(residual, dtype=np.float64), sigma=sigma, mode="nearest", truncate=3.0
    )


def _fitted_low_shelf_response(
    frequencies: np.ndarray,
    sample_rate: float,
    shelf: dict[str, Any] | None,
) -> np.ndarray:
    if not shelf or not shelf.get("active"):
        return np.ones_like(
            np.asarray(frequencies, dtype=np.float64), dtype=np.complex128
        )
    return low_shelf_response(
        frequencies,
        sample_rate,
        float(shelf["freq_hz"]),
        float(shelf["gain_db"]),
        float(shelf.get("slope", 1.0)),
    )


def _low_shelf_bases(
    frequencies: np.ndarray,
    sample_rate: float,
    ppo: int,
    in_range: np.ndarray,
) -> list[tuple[float, np.ndarray]]:
    """Return automatic shelf corners at roughly 12 points per octave."""

    indices = np.flatnonzero(in_range)
    if not indices.size:
        return []
    step = max(1, int(round(ppo / 12.0)))
    candidates = np.unique(np.append(indices[::step], indices[-1]))
    return [
        (
            float(frequencies[index]),
            db20(
                low_shelf_response(
                    frequencies,
                    sample_rate,
                    float(frequencies[index]),
                    1.0,
                )
            ),
        )
        for index in candidates
    ]


def _best_low_shelf(
    frequencies: np.ndarray,
    sample_rate: float,
    total: np.ndarray,
    residual: np.ndarray,
    objective_weights: np.ndarray,
    bases: list[tuple[float, np.ndarray]],
    options: EqOptions,
    threshold_db: float,
    current_error: float,
) -> tuple[float, np.ndarray, dict[str, Any]] | None:
    """Fit one broad LS candidate against the same objective as PK bands."""

    minimum_gain = -options.max_cut_db
    maximum_gain = options.max_boost_db
    if minimum_gain == 0.0 and maximum_gain == 0.0:
        return None

    total_db = db20(total)
    coarse_best: tuple[float, float, float] | None = None
    for fc, unit_db in bases:
        denominator = float(np.sum(objective_weights * unit_db**2))
        if denominator <= EPS:
            continue
        gain_db = float(
            np.clip(
                np.sum(objective_weights * unit_db * residual) / denominator,
                minimum_gain,
                maximum_gain,
            )
        )
        if abs(gain_db) < threshold_db:
            continue
        trial = total * low_shelf_response(
            frequencies, sample_rate, fc, gain_db
        )
        trial_db = db20(trial)
        if gain_db > 0.0 and float(np.max(trial_db)) > maximum_gain + 1e-9:
            low_gain, high_gain = 0.0, gain_db
            for _ in range(24):
                mid_gain = 0.5 * (low_gain + high_gain)
                mid_db = db20(
                    total
                    * low_shelf_response(
                        frequencies, sample_rate, fc, mid_gain
                    )
                )
                if float(np.max(mid_db)) <= maximum_gain + 1e-9:
                    low_gain = mid_gain
                else:
                    high_gain = mid_gain
            gain_db = low_gain
            if abs(gain_db) < threshold_db:
                continue
            trial = total * low_shelf_response(
                frequencies, sample_rate, fc, gain_db
            )
            trial_db = db20(trial)
        trial_error = float(
            np.mean(objective_weights * (residual - (trial_db - total_db)) ** 2)
        )
        if coarse_best is None or trial_error < coarse_best[0]:
            coarse_best = (trial_error, fc, gain_db)

    if coarse_best is None:
        return None
    _, fc, coarse_gain = coarse_best

    # Refine the gain continuously at the best broad corner. The coarse pass
    # above keeps this to one scalar optimization per EQ iteration rather than
    # one optimization per possible corner.
    maximum_allowed_gain = maximum_gain
    if maximum_allowed_gain > 0.0:
        at_limit = db20(
            total
            * low_shelf_response(
                frequencies, sample_rate, fc, maximum_allowed_gain
            )
        )
        if float(np.max(at_limit)) > maximum_gain + 1e-9:
            low_gain, high_gain = 0.0, maximum_allowed_gain
            for _ in range(24):
                mid_gain = 0.5 * (low_gain + high_gain)
                mid_db = db20(
                    total
                    * low_shelf_response(
                        frequencies, sample_rate, fc, mid_gain
                    )
                )
                if float(np.max(mid_db)) <= maximum_gain + 1e-9:
                    low_gain = mid_gain
                else:
                    high_gain = mid_gain
            maximum_allowed_gain = low_gain

    if maximum_allowed_gain <= minimum_gain + 1e-9:
        return None

    def error_for_gain(gain_db: float) -> float:
        trial_db = db20(
            total
            * low_shelf_response(frequencies, sample_rate, fc, gain_db)
        )
        return float(
            np.mean(objective_weights * (residual - (trial_db - total_db)) ** 2)
        )

    refined = optimize.minimize_scalar(
        error_for_gain,
        bounds=(minimum_gain, maximum_allowed_gain),
        method="bounded",
        options={"xatol": 5e-4},
    )
    gain_db = float(refined.x if refined.success else coarse_gain)
    if abs(gain_db) < threshold_db:
        return None
    gain_db = (
        math.floor(gain_db * 1000.0) / 1000.0
        if gain_db > 0.0
        else round(gain_db, 3)
    )
    fc = round(fc, 3)
    trial = total * low_shelf_response(frequencies, sample_rate, fc, gain_db)
    trial_error = float(
        np.mean(objective_weights * (residual - (db20(trial) - total_db)) ** 2)
    )
    if trial_error >= current_error - 1e-6:
        return None
    shelf = {
        "active": True,
        "freq_hz": fc,
        "gain_db": gain_db,
        "slope": 1.0,
    }
    return trial_error, trial, shelf


def fit_eq_filters(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    sample_rate: float,
    ppo: int,
    excess_group_delay_ms: np.ndarray,
    options: EqOptions,
    margin_spectrum: np.ndarray | None = None,
    margin_slice: slice | None = None,
) -> tuple[list[dict[str, float]], np.ndarray, dict[str, Any]]:
    """Fit a bounded EQ bank toward a range- and excess-GD-aware target.

    PK candidates and, when ``options.low_shelf`` is true, one automatic LS
    candidate compete against the same weighted residual. The LS corner and
    boost/cut are fitted per response and a selected shelf consumes one of
    ``max_filters``'s slots. ``filters`` contains the PK bands for backward
    compatibility; the fitted LS parameters are returned in
    ``metadata["shelf"]`` and are already included in the returned response.

    ``margin_spectrum``/``margin_slice`` are an optional wider companion to
    ``spectrum``/``frequencies`` (see ``margin_frequencies``) used only to
    compute the ``trend`` target without the edge bias that
    ``broad_trend_db``'s ``mode="nearest"`` smoothing would otherwise
    introduce at the band boundaries. All other fitting still uses the raw,
    in-band ``spectrum``.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)
    correction_range = options.correction_range or (
        float(frequencies[0]),
        float(frequencies[-1]),
    )
    if correction_range[0] < frequencies[0] or correction_range[1] > frequencies[-1]:
        raise ValueError(
            "EQ correction range must lie within the analysis band "
            f"{frequencies[0]:g}..{frequencies[-1]:g} Hz"
        )
    base_db = db20(spectrum)
    in_range = (frequencies >= correction_range[0]) & (
        frequencies <= correction_range[1]
    )
    if not np.any(in_range):
        raise ValueError("EQ correction range contains no evaluation frequencies")
    if options.target in ("flat", "dsp"):
        percentile = 50.0 if options.max_boost_db > 0.0 else 30.0
        target_level = float(np.percentile(base_db[in_range], percentile))
        nominal_target = np.full_like(base_db, target_level)
    elif margin_spectrum is not None and margin_slice is not None:
        nominal_target = broad_trend_db(db20(margin_spectrum), ppo)[margin_slice]
        target_level = float(np.median(nominal_target[in_range]))
    else:
        nominal_target = broad_trend_db(base_db, ppo)
        target_level = float(np.median(nominal_target[in_range]))

    desired = np.clip(
        nominal_target - base_db, -options.max_cut_db, options.max_boost_db
    )
    range_authority = _correction_range_authority(
        frequencies, correction_range, options.correction_slope_db_per_octave
    )
    # Shrink the EQ target where excess GD indicates that a magnitude-only
    # filter cannot repair the phase-domain cancellation.
    gd_authority = _excess_gd_authority(frequencies, excess_group_delay_ms)
    authority = range_authority * gd_authority
    desired *= authority
    effective_target = base_db + desired

    total = np.ones_like(spectrum, dtype=np.complex128)
    filters: list[dict[str, float]] = []
    fitted_shelf: dict[str, Any] = {
        "active": False,
        "freq_hz": None,
        "gain_db": 0.0,
        "slope": 1.0,
    }
    shelf_bases = (
        _low_shelf_bases(frequencies, sample_rate, ppo, in_range)
        if options.low_shelf
        else []
    )
    threshold_db = 0.35 if options.target in ("flat", "dsp") else 0.75
    objective_weights = np.maximum(0.15, gd_authority)
    for _ in range(options.max_filters):
        total_db = db20(total)
        residual = desired - total_db
        current_error = float(np.mean(objective_weights * residual**2))
        # Peak location, sign, and bandwidth are read from a lightly denoised
        # copy of the residual so single-bin measurement ripple cannot steer a
        # narrow, high-Q cut at a noise artifact. Acceptance below is still
        # judged against the true, unsmoothed residual/desired curve.
        smoothed_residual = _denoised_residual(residual, ppo)
        candidate_score = np.abs(smoothed_residual)
        candidate_score[~in_range] = 0.0
        if options.max_boost_db <= 0.0:
            candidate_score[smoothed_residual > 0.0] = 0.0
        if float(np.max(candidate_score)) < threshold_db:
            break

        # REW-style greedy assignment starts with the largest target errors,
        # but several extrema are evaluated so an uncorrectable narrow null
        # cannot prevent a broader, useful filter from being selected.
        extrema, _ = signal.find_peaks(
            candidate_score,
            distance=max(1, int(round(ppo / 24.0))),
        )
        extrema = np.unique(np.append(extrema, int(np.argmax(candidate_score))))
        ordered = extrema[np.argsort(candidate_score[extrema], kind="stable")][::-1]
        ordered = ordered[:32]
        best: tuple[float, np.ndarray, dict[str, Any], bool] | None = None
        for peak_index in ordered:
            correction_db = float(smoothed_residual[peak_index])
            if abs(correction_db) < threshold_db:
                continue
            sign = 1.0 if correction_db > 0.0 else -1.0
            half = abs(correction_db) / 2.0
            left = int(peak_index)
            right = int(peak_index)
            while left > 0 and sign * smoothed_residual[left] > half:
                left -= 1
            while right < smoothed_residual.size - 1 and sign * smoothed_residual[right] > half:
                right += 1
            fc = float(frequencies[peak_index])
            bandwidth = max(float(frequencies[right] - frequencies[left]), fc / 20.0)
            # Narrow cuts can tame room modes. Boosts are deliberately broad:
            # miniDSP recommends Q <= 1 and warns against filling narrow nulls.
            maximum_q = 1.0 if correction_db > 0.0 else 10.0
            q = float(np.clip(fc / bandwidth, 0.4, maximum_q))
            gain_db = float(
                np.clip(correction_db, -options.max_cut_db, options.max_boost_db)
            )

            def trial_response(gain: float) -> tuple[np.ndarray, np.ndarray]:
                response = total * peq_response(
                    frequencies, sample_rate, fc, q, gain
                )
                return response, db20(response)

            # Enforce max boost on the combined filter bank, not just on each
            # proposed bell. This mirrors the separate overall-boost guard in
            # established automatic EQ tools.
            if gain_db > 0.0:
                _, trial_db_at_limit = trial_response(gain_db)
                if float(np.max(trial_db_at_limit)) > options.max_boost_db + 1e-9:
                    low_gain, high_gain = 0.0, gain_db
                    for _ in range(24):
                        mid_gain = 0.5 * (low_gain + high_gain)
                        _, mid_db = trial_response(mid_gain)
                        if float(np.max(mid_db)) <= options.max_boost_db + 1e-9:
                            low_gain = mid_gain
                        else:
                            high_gain = mid_gain
                    gain_db = low_gain
                if gain_db < threshold_db:
                    continue

            gain_db = (
                math.floor(gain_db * 1000.0) / 1000.0
                if gain_db > 0.0
                else round(gain_db, 3)
            )
            q = round(q, 3)
            fc = round(fc, 3)
            trial, trial_db = trial_response(gain_db)
            trial_error = float(
                np.mean(objective_weights * (desired - trial_db) ** 2)
            )
            if trial_error >= current_error - 1e-6:
                continue
            current = {"fc_hz": fc, "gain_db": gain_db, "q": q}
            if best is None or trial_error < best[0]:
                best = (trial_error, trial, current, False)
        if shelf_bases and not fitted_shelf["active"]:
            shelf_candidate = _best_low_shelf(
                frequencies,
                sample_rate,
                total,
                residual,
                objective_weights,
                shelf_bases,
                options,
                threshold_db,
                current_error,
            )
            if shelf_candidate is not None and (
                best is None or shelf_candidate[0] < best[0]
            ):
                best = (*shelf_candidate, True)
        if best is None:
            break
        _, total, current, is_shelf = best
        if is_shelf:
            fitted_shelf = current
        else:
            filters.append(current)
    metadata: dict[str, Any] = {
        "target": options.target,
        "target_level_db": target_level,
        "correction_range_hz": correction_range,
        "correction_slope_db_per_octave": options.correction_slope_db_per_octave,
        "max_boost_db": options.max_boost_db,
        "effective_target_db": effective_target,
        "nominal_target_db": nominal_target,
        "eq_authority": authority,
        "shelf": fitted_shelf,
        "filter_count": len(filters) + int(bool(fitted_shelf["active"])),
    }
    return filters, total, metadata


def fit_cut_filters(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    sample_rate: float,
    ppo: int,
    maximum: int = 4,
) -> tuple[list[dict[str, float]], np.ndarray]:
    """Backward-compatible conservative, cuts-only fitter."""
    filters, total, _ = fit_eq_filters(
        spectrum,
        frequencies,
        sample_rate,
        ppo,
        np.zeros_like(frequencies),
        EqOptions(max_filters=maximum, low_shelf=False),
    )
    return filters, total


def filters_response(
    frequencies: np.ndarray, sample_rate: float, filters: Sequence[dict[str, float]]
) -> np.ndarray:
    total = np.ones_like(np.asarray(frequencies), dtype=np.complex128)
    for current in filters:
        total *= peq_response(
            frequencies,
            sample_rate,
            float(current["fc_hz"]),
            float(current["q"]),
            float(current["gain_db"]),
        )
    return total


def minimum_phase_log_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """Minimum-phase complex log spectrum via the discrete Hilbert/cepstrum method.

    `spectrum` must be an rFFT computed with the desired zero-padded FFT length.
    The magnitude floor is 160 dB below its peak.
    """
    bins = np.asarray(spectrum, dtype=np.complex128)
    n_fft = 2 * (bins.size - 1)
    magnitude = np.abs(bins)
    peak = max(float(np.max(magnitude)), EPS)
    log_magnitude = np.log(np.maximum(magnitude, peak * 1e-8))
    cepstrum = np.fft.irfft(log_magnitude, n=n_fft)
    minimum_cepstrum = np.zeros_like(cepstrum)
    minimum_cepstrum[0] = cepstrum[0]
    if n_fft % 2 == 0:
        minimum_cepstrum[1 : n_fft // 2] = 2.0 * cepstrum[1 : n_fft // 2]
        minimum_cepstrum[n_fft // 2] = cepstrum[n_fft // 2]
    else:
        minimum_cepstrum[1 : (n_fft + 1) // 2] = 2.0 * cepstrum[
            1 : (n_fft + 1) // 2
        ]
    return np.fft.rfft(minimum_cepstrum, n=n_fft)


MIN_RELIABLE_NATIVE_BINS = 6.0


def gd_smoothing_octaves(
    frequencies: np.ndarray,
    native_resolution_hz: float,
    min_native_bins: float = MIN_RELIABLE_NATIVE_BINS,
) -> np.ndarray:
    """Gaussian smoothing sigma, in octaves, that averages ~min_native_bins native bins.

    ``native_resolution_hz`` (``sample_rate / length`` of the *unpadded* cached
    impulse; see ``AnalysisContext.native_resolution_hz``) is the coarsest
    frequency spacing a sweep/capture of that length actually resolves;
    anything finer comes from zero-padded interpolation, not new
    information. The number of native bins packed into one octave around
    frequency ``f`` is roughly ``f / native_resolution_hz``, so the sigma
    needed to average a fixed count of them grows without bound as ``f``
    falls toward (and below) that native spacing, and is negligible once
    ``f`` is many multiples of it. A short sweep or a low analysis band
    therefore gets progressively more smoothing exactly where excess-GD
    noise is otherwise worst; a long sweep is smoothed almost nowhere.

    ``native_resolution_hz`` is a useful resolution *heuristic*, not a hard
    measurement-theory reliable/unreliable boundary, and ``min_native_bins``
    is a chosen estimator width, not a threshold derived from first
    principles - see ``_smooth_by_variable_octaves``'s ladder cap for the
    corresponding tradeoff this makes against genuine, narrower-than-the-
    kernel low-frequency features.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if native_resolution_hz <= 0.0:
        return np.zeros_like(frequencies)
    return np.log2(1.0 + min_native_bins * native_resolution_hz / frequencies)


# Capped at 4 octaves - already wider than most analysis bands - so a
# pathologically short capture cannot smooth away the entire band; see
# gd_smoothing_octaves. A genuine feature whose own bandwidth is comparable
# to or narrower than the sigma applied at its frequency is still
# attenuated by this smoothing, the same tradeoff any smoothing-based
# denoiser makes - this preferentially preserves broad features relative to
# narrow noise, it does not leave every genuine feature unaffected.
_GD_SMOOTHING_LADDER_OCTAVES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def _smooth_by_variable_octaves(
    values: np.ndarray, ppo: int, sigma_octaves: np.ndarray
) -> np.ndarray:
    """Approximate per-point-variable-sigma Gaussian smoothing on a log grid.

    ``ndimage.gaussian_filter1d`` only takes one sigma per call. This
    precomputes the curve at each rung of ``_GD_SMOOTHING_LADDER_OCTAVES``
    (sigma expressed as a bin count, exact because the grid is uniform in
    log-frequency) and linearly blends, per point, between the two rungs
    bracketing that point's own requested ``sigma_octaves``. Vectorised and
    deterministic; a point requesting 0 octaves reproduces the raw value
    exactly, and a point past the top rung is clamped to the most-smoothed
    curve rather than extrapolated.
    """
    values = np.asarray(values, dtype=np.float64)
    ladder = np.asarray(_GD_SMOOTHING_LADDER_OCTAVES, dtype=np.float64)
    target = np.clip(np.asarray(sigma_octaves, dtype=np.float64), 0.0, ladder[-1])
    rungs = [values]
    for octaves in ladder[1:]:
        rungs.append(
            ndimage.gaussian_filter1d(
                values, sigma=octaves * ppo, mode="nearest", truncate=3.0
            )
        )
    rungs = np.asarray(rungs)
    upper_index = np.clip(np.searchsorted(ladder, target, side="left"), 1, ladder.size - 1)
    lower_index = upper_index - 1
    lower_octaves = ladder[lower_index]
    upper_octaves = ladder[upper_index]
    blend = np.clip(
        (target - lower_octaves) / np.maximum(upper_octaves - lower_octaves, EPS), 0.0, 1.0
    )
    points = np.arange(values.size)
    lower_values = rungs[lower_index, points]
    upper_values = rungs[upper_index, points]
    return lower_values * (1.0 - blend) + upper_values * blend


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    midpoint = 0.5 * float(np.sum(ordered_weights))
    return float(ordered_values[np.searchsorted(np.cumsum(ordered_weights), midpoint)])


def excess_group_delay(
    spectrum: np.ndarray,
    fft_frequencies: np.ndarray,
    evaluation_frequencies: np.ndarray,
    integration_range: tuple[float, float] | None = None,
    native_resolution_hz: float | None = None,
    ppo: int = 48,
) -> tuple[float, np.ndarray]:
    """Return energy-weighted mean absolute excess GD and its de-offset curve.

    The minimum-phase transform and group-delay derivative use the complete
    supplied spectra and evaluation grid. When ``integration_range`` is set,
    only that frequency interval contributes to common-delay removal and the
    reported scalar diagnostic.

    A single constant - this curve's weighted median within
    ``integration_range`` - is treated as the arbitrary common time origin
    and removed, leaving frequency-dependent (excess) storage/decay. An
    earlier, opt-in ``--gd-baseline monotonic`` mode instead fit a per-point
    baseline constrained to be non-increasing in magnitude as frequency
    rose, modelling a genuine low-end group-delay rise as normal rather than
    as excess; it was removed after proving unreliable on real measurements
    - a non-increasing PAVA fit pools *any* later violation backward across
    everything before it, so a single local bump anywhere in the curve
    (ordinary measurement ripple, not just a genuine feature) could inflate
    the baseline into an implausible, near-flat plateau spanning nearly the
    whole band, several times taller than the curve's own genuine excess-GD
    peak. The resolution-aware smoothing below already handles the
    reliability concern that motivated the acoustic-modelling attempt, so a
    single constant baseline is the only mode now.

    When ``native_resolution_hz`` is given (the cache's unpadded frequency
    resolution; see ``AnalysisContext.native_resolution_hz``), the returned
    curve is progressively smoothed, via ``gd_smoothing_octaves``, wherever
    the evaluation grid is finer than that sweep/capture can actually
    resolve - almost always the sub-bass, where a short sweep leaves few
    genuinely independent samples per octave and ordinary measurement noise
    otherwise shows up as large, sign-flipping group-delay swings around
    zero. Leaving this ``None`` (the default) reproduces the original,
    unsmoothed curve, e.g. for callers that supply hand-built curves rather
    than a real cache's resolution. This applies after baseline removal.

    Returns ``(score, curve_ms)``: the scalar score and the de-offset
    (baseline-removed, optionally smoothed) curve in ms.
    """
    log_minimum = minimum_phase_log_spectrum(spectrum)
    minimum_phase = np.imag(log_minimum)
    excess_phase_full = np.unwrap(np.angle(spectrum) - minimum_phase)
    phase = np.interp(evaluation_frequencies, fft_frequencies, excess_phase_full)
    omega = 2.0 * np.pi * evaluation_frequencies
    group_delay = -np.gradient(phase, omega, edge_order=2)
    magnitude = np.abs(
        _interp_complex(fft_frequencies, spectrum, evaluation_frequencies)
    )
    weights = np.maximum(magnitude * magnitude, np.max(magnitude * magnitude) * 1e-6)
    if integration_range is None:
        score_mask = np.ones(evaluation_frequencies.size, dtype=bool)
    else:
        low, high = integration_range
        if low <= 0.0 or high <= low:
            raise ValueError("Excess-GD integration range must be positive and increasing")
        score_mask = (evaluation_frequencies >= low) & (
            evaluation_frequencies <= high
        )
        if np.count_nonzero(score_mask) < 3:
            raise ValueError("Excess-GD integration range contains fewer than three points")
    # A constant group delay is the arbitrary common time origin. Removing
    # its weighted median (computed from the raw, unsmoothed curve, so the
    # smoothing below cannot bias the common-delay estimate itself) leaves
    # frequency-dependent (excess) storage/decay.
    baseline = _weighted_median(group_delay[score_mask], weights[score_mask])
    group_delay = group_delay - baseline
    if native_resolution_hz is not None:
        sigma_octaves = gd_smoothing_octaves(evaluation_frequencies, native_resolution_hz)
        group_delay = _smooth_by_variable_octaves(group_delay, ppo, sigma_octaves)
    score_frequencies = evaluation_frequencies[score_mask]
    score_weights = weights[score_mask]
    score_delays = group_delay[score_mask]
    log_frequency = np.log(score_frequencies)
    numerator = np.trapezoid(score_weights * np.abs(score_delays), x=log_frequency)
    denominator = max(np.trapezoid(score_weights, x=log_frequency), EPS)
    return (
        float(1000.0 * numerator / denominator),
        1000.0 * group_delay,
    )


EXCESS_GD_TAIL_POWER = 1.0


def excess_gd_tail_ms(
    excess_group_delay_ms: np.ndarray,
    frequencies: np.ndarray,
    integration_range: tuple[float, float] | None = None,
    power: float = EXCESS_GD_TAIL_POWER,
) -> float:
    """|excess GD| integrated over log-frequency, unweighted by level.

    ``excess_group_delay``'s scalar is an *energy-weighted* mean, so a badly
    smeared region that happens to sit in a magnitude dip or near a band
    edge (where SPL is naturally low) barely moves it - two sums can look
    equally clean on that metric while one is audibly ringing somewhere the
    ear doesn't need much level to notice it. This integrates |excess
    GD|^power over log-frequency instead (matching how
    ``excess_group_delay``'s own mean already integrates via
    ``np.trapezoid``, rather than a plain index-based average), with every
    frequency in range weighted equally regardless of level, so a sum that
    is flat on magnitude but smeary in phase is still caught.

    The default ``power=1`` is a plain area-weighted average: a narrow,
    severe spike and a wider, shallower bump of the same area (peak height
    times width) score the same, which is what makes this shape-neutral
    rather than favouring one over the other. It replaces an earlier
    percentile-based version of this function, which was an order statistic
    blind to any anomaly narrower than ``100 - percentile`` percent of the
    band, however severe - the opposite failure from a naive peak detector,
    which instead only sees the narrow one. A power greater than 1 would
    weight a large *local* deviation more than proportionally, but at the
    cost of that shape-neutrality: concentrating the same area into a
    narrower, taller feature then scores higher than spreading it out.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)
    values = np.abs(np.asarray(excess_group_delay_ms, dtype=np.float64))
    if integration_range is not None:
        low, high = integration_range
        mask = (frequencies >= low) & (frequencies <= high)
        if np.any(mask):
            frequencies = frequencies[mask]
            values = values[mask]
    if values.size < 2:
        return float(values[0]) if values.size else 0.0
    log_frequency = np.log(frequencies)
    span = max(float(log_frequency[-1] - log_frequency[0]), EPS)
    numerator = np.trapezoid(values**power, x=log_frequency)
    return float((numerator / span) ** (1.0 / power))


EXCESS_GD_PEAK_DENOISE_SIGMA_BINS = 0.35


def excess_gd_peak_ms(
    excess_group_delay_ms: np.ndarray,
    frequencies: np.ndarray,
    integration_range: tuple[float, float] | None = None,
) -> float:
    """Denoised worst-case |excess GD|, width-invariant unlike ``excess_gd_tail_ms``.

    ``excess_gd_tail_ms`` is deliberately *area*-based: a narrow severe spike
    and a wide shallow bump of the same area score the same (see its
    docstring), which is the right property for an overall-smear estimate but
    means one genuinely severe, narrow non-minimum-phase feature can be
    diluted in that average by wide, mild variation elsewhere in the range. A
    plain maximum is already perfectly width-invariant on its own — a spike
    of height H and a plateau of height H both peak at H regardless of how
    wide either one is — so, unlike ``_excess_gd_authority``'s pointwise
    maximum-filter (needed there to build a full-curve gate), this only needs
    the same light pre-denoise ``_excess_gd_authority`` applies before its
    maximum filter, so a single noisy sample cannot set the reported peak by
    itself, and then a single global maximum.

    This complements rather than replaces ``excess_gd_tail_ms``: it exposes a
    single sharp, denoised-real non-minimum-phase excursion which the
    area-based diagnostic would weight like several mild, spread-out ones.
    Neither value changes the usable-output ranking.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)
    values = np.abs(np.asarray(excess_group_delay_ms, dtype=np.float64))
    if integration_range is not None:
        low, high = integration_range
        mask = (frequencies >= low) & (frequencies <= high)
        if np.any(mask):
            frequencies = frequencies[mask]
            values = values[mask]
    if values.size == 0:
        return 0.0
    if values.size < 3:
        return float(np.max(values))
    denoised = ndimage.gaussian_filter1d(
        values, sigma=EXCESS_GD_PEAK_DENOISE_SIGMA_BINS, mode="nearest", truncate=3.0
    )
    return float(np.max(denoised))


def _band_centres(low: float, high: float, ppo: int) -> np.ndarray:
    return log_frequency_grid(low, high, ppo)


ARRIVAL_ONSET_THRESHOLD_DB = -10.0


def arrival_onset_index(
    impulse: np.ndarray,
    sample_rate: float,
    band: tuple[float, float],
    threshold_db: float = ARRIVAL_ONSET_THRESHOLD_DB,
    search_seconds: float = 0.060,
) -> float:
    """Leading edge of the impulse, as a sub-sample index.

    The peak of a band-limited impulse is a poor arrival marker: across two or
    three octaves the response is a slow oscillatory blob many milliseconds
    long, and wherever a room mode rings hard a later half-cycle can exceed the
    direct arrival's, so a peak pick silently jumps a whole cycle.  The leading
    edge does not move when that happens -- the direct sound is still the first
    energy to arrive regardless of which lobe ends up tallest.

    This deliberately returns the threshold crossing, not a physical arrival:
    band-limiting spreads energy earlier than the true onset, so the value sits
    a fixed distance ahead of it.  That bias is common to every measurement in
    a cache and cancels in the differences that are actually used, which is why
    no attempt is made to correct it.
    """

    impulse = np.asarray(impulse, dtype=np.float64)
    if impulse.size < 16:
        raise ValueError("Impulse is too short for onset detection")
    nyquist = 0.5 * sample_rate
    low = max(1.0, min(band[0], 0.45 * nyquist))
    high = min(max(low * 1.5, band[1]), 0.9 * nyquist)
    sos = signal.butter(4, [low, high], btype="bandpass", fs=sample_rate, output="sos")
    filtered = (
        signal.sosfiltfilt(sos, impulse)
        if impulse.size > 3 * (2 * sos.shape[0] + 1)
        else signal.sosfilt(sos, impulse)
    )
    envelope = np.abs(signal.hilbert(filtered))
    peak = int(np.argmax(envelope))
    limit = envelope[peak] * 10.0 ** (threshold_db / 20.0)
    start = max(0, peak - int(round(search_seconds * sample_rate)))
    window = envelope[start : peak + 1]
    crossings = np.flatnonzero(window >= limit)
    if not crossings.size:
        return float(peak)
    index = start + int(crossings[0])
    if index > 0:
        previous, current = envelope[index - 1], envelope[index]
        fraction = 0.0 if current == previous else (limit - previous) / (current - previous)
        index = index - 1 + float(np.clip(fraction, 0.0, 1.0))
    return float(index)


def csd_style_decay(
    impulse: np.ndarray,
    sample_rate: float,
    band: tuple[float, float],
    ppo: int = 3,
    duration_seconds: float = 1.2,
    hop_seconds: float = 0.002,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Constant-percentage-band analytic decay used for tails and heatmaps."""
    impulse = np.asarray(impulse, dtype=np.float64)
    peak = int(np.argmax(np.abs(impulse)))
    pre_samples = int(round(min(0.25, max(8.0 / band[0], 0.08)) * sample_rate))
    post_samples = int(round(duration_seconds * sample_rate))
    start = max(0, peak - pre_samples)
    stop = min(impulse.size, peak + post_samples)
    segment = impulse[start:stop]
    if segment.size < 32:
        raise ValueError("Impulse is too short for decay analysis")
    times = (np.arange(segment.size) + start - peak) / sample_rate
    hop = max(1, int(round(hop_seconds * sample_rate)))
    sampled_times = times[::hop]
    centres = _band_centres(band[0], band[1], ppo)
    decay = np.empty((centres.size, sampled_times.size), dtype=np.float64)
    t20 = np.empty(centres.size, dtype=np.float64)
    nyquist = sample_rate / 2.0
    for row, centre in enumerate(centres):
        lower = max(0.5, centre * 2.0 ** (-1.0 / 6.0))
        upper = min(nyquist * 0.98, centre * 2.0 ** (1.0 / 6.0))
        sos = signal.butter(2, [lower, upper], btype="bandpass", fs=sample_rate, output="sos")
        if segment.size > 3 * (2 * sos.shape[0] + 1):
            filtered = signal.sosfiltfilt(sos, segment)
        else:
            filtered = signal.sosfilt(sos, segment)
        envelope = np.abs(signal.hilbert(filtered)) ** 2
        cycle = max(1, int(round(sample_rate / centre)))
        envelope = ndimage.uniform_filter1d(envelope, size=cycle, mode="nearest")
        reference_window = (times >= -0.05) & (times <= 0.10)
        reference = max(float(np.max(envelope[reference_window])), EPS)
        curve = 10.0 * np.log10(np.maximum(envelope, reference * 1e-10) / reference)
        sampled = curve[::hop]
        decay[row] = sampled
        local = np.flatnonzero((sampled_times >= -0.02) & (sampled_times <= 0.10))
        local_peak = int(local[np.argmax(sampled[local])]) if local.size else int(np.argmax(sampled))
        hold_count = max(1, int(round(max(0.04, 2.0 / centre) / hop_seconds)))
        crossing = sampled_times[-1]
        for index in range(local_peak, max(local_peak + 1, sampled.size - hold_count + 1)):
            if np.max(sampled[index : index + hold_count]) <= -20.0:
                crossing = sampled_times[index]
                break
        t20[row] = max(0.0, 1000.0 * float(crossing - sampled_times[local_peak]))
    return centres, sampled_times, decay, t20


@dataclass
class AnalysisContext:
    measurements: list[CachedMeasurement]
    band: tuple[float, float]
    ppo: int = 48
    minphase_pad_factor: int = 4

    def __post_init__(self) -> None:
        self.sample_rate = self.measurements[0].sample_rate
        self.length = self.measurements[0].impulse.size
        # The *unpadded* capture length sets the native, non-interpolated
        # frequency resolution: zero-padding (used for minimum-phase/CSD
        # work below) makes the spectrum smoother to look at but does not
        # add real information finer than this. See excess_group_delay's
        # use of gd_smoothing_octaves for why this matters most near DC.
        self.native_resolution_hz = self.sample_rate / self.length
        if self.band[1] >= self.sample_rate / 2.0:
            raise ValueError(
                f"Band upper edge {self.band[1]:g} Hz must be below Nyquist "
                f"({self.sample_rate / 2:g} Hz)"
            )
        self._padded_fft_length = next_fast_len(self.length * self.minphase_pad_factor)
        if self._padded_fft_length % 2:
            self._padded_fft_length = next_fast_len(self._padded_fft_length + 1)
        # sum_full/padded_spectra apply each measurement's start-time offset
        # as a pure frequency-domain phase ramp, which is a *circular* shift
        # over the zero-padded FFT frame. If the offset exceeds the available
        # zero padding, real impulse content wraps around the frame instead
        # of landing in the padding, silently corrupting the minimum-phase,
        # excess-GD, and CSD-tail results for that pair. Measurements sharing
        # a loopback-derived time base (the documented assumption) will have
        # offsets of at most a few milliseconds, far inside this margin.
        available_padding_seconds = (
            self._padded_fft_length - self.length
        ) / self.sample_rate
        safe_shift_seconds = 0.5 * available_padding_seconds
        common_start = min(row.start_time_seconds for row in self.measurements)
        for measurement in self.measurements:
            offset_seconds = abs(measurement.start_time_seconds - common_start)
            if offset_seconds > safe_shift_seconds:
                raise ValueError(
                    f"Measurement {measurement.title!r} start time is "
                    f"{offset_seconds * 1000.0:.3f} ms from the earliest "
                    "loaded measurement, which exceeds the safe zero-padded "
                    f"analysis window ({safe_shift_seconds * 1000.0:.3f} ms); "
                    "loaded measurements must share a loopback-derived "
                    "absolute time base"
                )
        self.frequencies = log_frequency_grid(*self.band, self.ppo)
        self.trend_frequencies, self.trend_slice = margin_frequencies(
            self.frequencies, self.ppo, self.ppo
        )
        fft_frequencies = np.fft.rfftfreq(self.length, 1.0 / self.sample_rate)
        peak_absolute = (
            self.measurements[0].start_time_seconds
            + int(np.argmax(np.abs(self.measurements[0].impulse))) / self.sample_rate
        )
        spectra = []
        trend_spectra = []
        for measurement in self.measurements:
            raw = np.fft.rfft(measurement.impulse)
            shift = measurement.start_time_seconds - peak_absolute
            absolute = raw * np.exp(-2j * np.pi * fft_frequencies * shift)
            spectra.append(_interp_complex(fft_frequencies, absolute, self.frequencies))
            trend_spectra.append(
                _interp_complex(fft_frequencies, absolute, self.trend_frequencies)
            )
        self.spectra = np.asarray(spectra, dtype=np.complex128)
        self.trend_spectra = np.asarray(trend_spectra, dtype=np.complex128)
        self._padded_spectra: np.ndarray | None = None
        self._padded_frequencies: np.ndarray | None = None

    def padded_spectra(self) -> tuple[np.ndarray, np.ndarray]:
        if self._padded_spectra is None or self._padded_frequencies is None:
            n_fft = self._padded_fft_length
            frequencies = np.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)
            common_start = min(row.start_time_seconds for row in self.measurements)
            rows = []
            for measurement in self.measurements:
                raw = np.fft.rfft(measurement.impulse, n=n_fft)
                shift = measurement.start_time_seconds - common_start
                rows.append(raw * np.exp(-2j * np.pi * frequencies * shift))
            self._padded_spectra = np.asarray(rows, dtype=np.complex128)
            self._padded_frequencies = frequencies
        return self._padded_spectra, self._padded_frequencies

    def sum_on_grid(
        self, first: int, second: int, polarity: int, delay_ms: float, gain_db: float
    ) -> np.ndarray:
        factor = polarity * 10.0 ** (gain_db / 20.0)
        phase = np.exp(-2j * np.pi * self.frequencies * delay_ms / 1000.0)
        return self.spectra[first] + factor * self.spectra[second] * phase

    def sum_on_trend_grid(
        self, first: int, second: int, polarity: int, delay_ms: float, gain_db: float
    ) -> np.ndarray:
        """Same as ``sum_on_grid`` but on the margin-extended trend grid."""
        factor = polarity * 10.0 ** (gain_db / 20.0)
        phase = np.exp(-2j * np.pi * self.trend_frequencies * delay_ms / 1000.0)
        return self.trend_spectra[first] + factor * self.trend_spectra[second] * phase

    def sum_full(
        self, first: int, second: int, polarity: int, delay_ms: float, gain_db: float
    ) -> tuple[np.ndarray, np.ndarray]:
        spectra, frequencies = self.padded_spectra()
        factor = polarity * 10.0 ** (gain_db / 20.0)
        phase = np.exp(-2j * np.pi * frequencies * delay_ms / 1000.0)
        return spectra[first] + factor * spectra[second] * phase, frequencies


def pair_diagnostics(
    context: AnalysisContext,
    first: int,
    second: int,
    polarity: int,
    delay_ms: float,
    gain_db: float,
    include_decay: bool = False,
    eq_options: EqOptions | None = None,
    include_trends: bool = False,
    score_low_end_weight: float = DEFAULT_SCORE_LOW_END_WEIGHT,
    score_dip_weight: float = DEFAULT_SCORE_DIP_WEIGHT,
) -> dict[str, Any]:
    eq_options = eq_options or EqOptions(correction_range=context.band)
    grid_sum = context.sum_on_grid(first, second, polarity, delay_ms, gain_db)
    # The trend is smoothed over a margin-extended grid with real spectral
    # content beyond the reported band, then cropped back, so it is not
    # biased by edge-replicated ("nearest") smoothing at the band boundaries.
    trend_wide_sum = context.sum_on_trend_grid(first, second, polarity, delay_ms, gain_db)
    full_sum, full_frequencies = context.sum_full(
        first, second, polarity, delay_ms, gain_db
    )
    excess_score, excess_curve = excess_group_delay(
        full_sum,
        full_frequencies,
        context.frequencies,
        integration_range=eq_options.correction_range,
        native_resolution_hz=context.native_resolution_hz,
        ppo=context.ppo,
    )
    filters, eq_grid, eq_metadata = fit_eq_filters(
        grid_sum,
        context.frequencies,
        context.sample_rate,
        context.ppo,
        excess_curve,
        eq_options,
        margin_spectrum=trend_wide_sum,
        margin_slice=context.trend_slice,
    )
    # filters_response() reconstructs only the fitted PK bands. The automatic
    # shelf chosen by the same fitter must be evaluated on these companion
    # grids as well; all of them together still respect max_filters.
    fitted_shelf = eq_metadata["shelf"]
    eq_full = filters_response(
        full_frequencies, context.sample_rate, filters
    ) * _fitted_low_shelf_response(
        full_frequencies, context.sample_rate, fitted_shelf
    )
    # Headroom is a real negative gain in the compared signal path, not a
    # private correction inside one diagnostic. The first sub is the 0 dB
    # reference and the second receives gain_db, so positive pair gain must
    # be removed globally to put the hottest driver back at 0 dB. Post-EQ,
    # remove the fitted response's largest positive boost as well.
    headroom_db = -float(gain_db) if gain_db > 0.0 else 0.0
    eq_drive_boost_db = max(0.0, float(np.max(db20(eq_grid))))
    post_eq_headroom_db = headroom_db - eq_drive_boost_db
    raw_headroom_linear = 10.0 ** (headroom_db / 20.0)
    post_eq_headroom_linear = 10.0 ** (post_eq_headroom_db / 20.0)

    n_fft = 2 * (full_sum.size - 1)
    normalized_full_sum = full_sum * raw_headroom_linear
    post_full = full_sum * eq_full * post_eq_headroom_linear
    pre_ir = np.fft.irfft(normalized_full_sum, n=n_fft)
    post_ir = np.fft.irfft(post_full, n=n_fft)
    _, _, _, raw_tail_by_band = csd_style_decay(
        pre_ir, context.sample_rate, context.band, ppo=3
    )
    _, _, _, tail_by_band = csd_style_decay(
        post_ir, context.sample_rate, context.band, ppo=3
    )
    normalized_grid_sum = grid_sum * raw_headroom_linear
    post_grid = grid_sum * eq_grid * post_eq_headroom_linear
    magnitude_db = db20(normalized_grid_sum)
    post_magnitude_db = db20(post_grid)
    normalized_trend_wide_sum = trend_wide_sum * raw_headroom_linear
    trend_db = broad_trend_db(
        db20(normalized_trend_wide_sum), context.ppo
    )[context.trend_slice]
    dip_db = float(
        smoothed_dip_db(
            db20(normalized_trend_wide_sum),
            context.ppo,
            score_slice=context.trend_slice,
        )
    )
    eq_trend_wide = filters_response(
        context.trend_frequencies, context.sample_rate, filters
    ) * _fitted_low_shelf_response(
        context.trend_frequencies, context.sample_rate, fitted_shelf
    )
    post_trend_wide_sum = (
        trend_wide_sum * eq_trend_wide * post_eq_headroom_linear
    )
    post_trend_db = broad_trend_db(db20(post_trend_wide_sum), context.ppo)[context.trend_slice]
    post_eq_dip_db = float(
        smoothed_dip_db(
            db20(post_trend_wide_sum),
            context.ppo,
            score_slice=context.trend_slice,
        )
    )
    post_excess_score, post_excess_curve = excess_group_delay(
        post_full,
        full_frequencies,
        context.frequencies,
        integration_range=eq_options.correction_range,
        native_resolution_hz=context.native_resolution_hz,
        ppo=context.ppo,
    )
    spl_db = float(
        10.0 * np.log10(max(np.mean(np.abs(normalized_grid_sum) ** 2), EPS))
    )
    post_eq_spl_db = float(
        10.0 * np.log10(max(np.mean(np.abs(post_grid) ** 2), EPS))
    )
    raw_low_end_power_db = float(low_end_power_db(trend_db, context.frequencies))
    post_eq_low_end_power_db = float(
        low_end_power_db(post_trend_db, context.frequencies)
    )
    result: dict[str, Any] = {
        "dip_db": dip_db,
        "sound_power_db": float(
            usable_output_score_db(
                spl_db,
                raw_low_end_power_db,
                0.0,
                score_low_end_weight,
                0.0,
            )
        ),
        "score_db": float(
            usable_output_score_db(
                spl_db,
                raw_low_end_power_db,
                dip_db,
                score_low_end_weight,
                score_dip_weight,
            )
        ),
        "excess_gd_ms": float(excess_score),
        "excess_gd_tail_ms": excess_gd_tail_ms(
            excess_curve, context.frequencies, eq_options.correction_range
        ),
        "excess_gd_peak_ms": excess_gd_peak_ms(
            excess_curve, context.frequencies, eq_options.correction_range
        ),
        "raw_tail_ms": float(np.max(raw_tail_by_band)),
        "raw_tail_by_band_ms": [round(float(value), 6) for value in raw_tail_by_band],
        "low_end_power_db": raw_low_end_power_db,
        "headroom_db": headroom_db,
        "post_eq_dip_db": post_eq_dip_db,
        "post_eq_sound_power_db": float(
            usable_output_score_db(
                post_eq_spl_db,
                post_eq_low_end_power_db,
                0.0,
                score_low_end_weight,
                0.0,
            )
        ),
        "post_eq_score_db": float(
            usable_output_score_db(
                post_eq_spl_db,
                post_eq_low_end_power_db,
                post_eq_dip_db,
                score_low_end_weight,
                score_dip_weight,
            )
        ),
        "post_eq_excess_gd_ms": float(post_excess_score),
        "post_eq_excess_gd_tail_ms": excess_gd_tail_ms(
            post_excess_curve, context.frequencies, eq_options.correction_range
        ),
        "post_eq_excess_gd_peak_ms": excess_gd_peak_ms(
            post_excess_curve, context.frequencies, eq_options.correction_range
        ),
        "post_eq_tail_ms": float(np.max(tail_by_band)),
        "tail_by_band_ms": [round(float(value), 6) for value in tail_by_band],
        "post_eq_low_end_power_db": post_eq_low_end_power_db,
        "post_eq_headroom_db": post_eq_headroom_db,
        "filters": filters,
        "eq_target": eq_metadata["target"],
        "eq_target_level_db": float(
            eq_metadata["target_level_db"] + post_eq_headroom_db
        ),
        "eq_mean_authority": float(np.mean(eq_metadata["eq_authority"])),
        "eq_filter_count": int(eq_metadata["filter_count"]),
        "eq_shelf": dict(fitted_shelf),
        "spl_db": spl_db,
        "post_eq_spl_db": post_eq_spl_db,
    }
    if include_trends:
        # Keep broad trends opt-in so ordinary diagnostic results stay small.
        result["trend_db"] = trend_db
        result["post_eq_trend_db"] = post_trend_db
    if include_decay:
        pre_f, pre_t, pre_decay, _ = csd_style_decay(
            pre_ir, context.sample_rate, context.band, ppo=12
        )
        post_f, post_t, post_decay, _ = csd_style_decay(
            post_ir, context.sample_rate, context.band, ppo=12
        )
        result.update(
            {
                "frequencies": context.frequencies,
                "sum_db": magnitude_db,
                "trend_db": trend_db,
                "solo_first_db": db20(context.spectra[first]),
                "solo_second_db": db20(context.spectra[second]),
                "excess_curve_ms": excess_curve,
                "post_eq_db": post_magnitude_db,
                "post_eq_trend_db": post_trend_db,
                "post_eq_excess_curve_ms": post_excess_curve,
                "eq_target_db": (
                    eq_metadata["effective_target_db"] + post_eq_headroom_db
                ),
                "eq_nominal_target_db": (
                    eq_metadata["nominal_target_db"] + post_eq_headroom_db
                ),
                "eq_authority": eq_metadata["eq_authority"],
                "decay_frequencies": pre_f,
                "decay_times": pre_t,
                "pre_decay_db": pre_decay,
                "post_decay_db": post_decay,
            }
        )
    return result
