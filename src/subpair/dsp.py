"""Numerical primitives used by the search, report, and verifier."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy import ndimage, signal
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


def _interp_complex(source_f: np.ndarray, source_h: np.ndarray, target_f: np.ndarray) -> np.ndarray:
    return np.interp(target_f, source_f, source_h.real) + 1j * np.interp(
        target_f, source_f, source_h.imag
    )


def variable_smooth_db(values: np.ndarray, frequencies: np.ndarray, ppo: int) -> np.ndarray:
    """1/6 octave below 100 Hz, narrowing smoothly toward 1/12 by 200 Hz."""
    values = np.asarray(values, dtype=np.float64)
    sigma_sixth = (ppo / 6.0) / 2.354820045
    sigma_twelfth = (ppo / 12.0) / 2.354820045
    broad = ndimage.gaussian_filter1d(
        values, sigma=max(sigma_sixth, 0.01), axis=-1, mode="nearest", truncate=3.0
    )
    narrow = ndimage.gaussian_filter1d(
        values, sigma=max(sigma_twelfth, 0.01), axis=-1, mode="nearest", truncate=3.0
    )
    blend = np.clip(np.log2(np.maximum(frequencies, 100.0) / 100.0), 0.0, 1.0)
    return broad * (1.0 - blend) + narrow * blend


def broad_trend_db(values: np.ndarray, ppo: int) -> np.ndarray:
    sigma = ppo / 2.354820045  # one-octave FWHM
    return ndimage.gaussian_filter1d(
        np.asarray(values, dtype=np.float64),
        sigma=max(sigma, 0.01),
        axis=-1,
        mode="nearest",
        truncate=3.0,
    )


def null_scores(spectra: np.ndarray, frequencies: np.ndarray, ppo: int) -> np.ndarray:
    magnitude_db = db20(spectra)
    smoothed = variable_smooth_db(magnitude_db, frequencies, ppo)
    trend = broad_trend_db(smoothed, ppo)
    return np.maximum(0.0, np.max(trend - smoothed, axis=-1))


def peq_response(
    frequencies: np.ndarray, sample_rate: float, fc: float, q: float, gain_db: float
) -> np.ndarray:
    """RBJ Audio EQ Cookbook peaking biquad response."""
    f = np.asarray(frequencies, dtype=np.float64)
    omega = 2.0 * np.pi * f / sample_rate
    cosine = np.cos(omega)
    sine = np.sin(omega)
    a = 10.0 ** (gain_db / 40.0)
    alpha = sine / (2.0 * q)
    b0 = 1.0 + alpha * a
    b1 = -2.0 * cosine
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cosine
    a2 = 1.0 - alpha / a
    z1 = np.exp(-1j * omega)
    z2 = z1 * z1
    numerator = b0 + b1 * z1 + b2 * z2
    denominator = a0 + a1 * z1 + a2 * z2
    response = np.ones_like(numerator, dtype=np.complex128)
    np.divide(numerator, denominator, out=response, where=np.abs(denominator) > 1e-14)
    return response


def fit_cut_filters(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    sample_rate: float,
    ppo: int,
    maximum: int = 4,
) -> tuple[list[dict[str, float]], np.ndarray]:
    """Greedily remove broad peaks above a fixed one-octave trend; never boost."""
    target = broad_trend_db(variable_smooth_db(db20(spectrum), frequencies, ppo), ppo)
    total = np.ones_like(spectrum, dtype=np.complex128)
    filters: list[dict[str, float]] = []
    for _ in range(maximum):
        corrected = spectrum * total
        corrected_db = variable_smooth_db(db20(corrected), frequencies, ppo)
        residual = corrected_db - target
        # Avoid treating a truncated edge as a well-defined parametric peak.
        if residual.size > 4:
            residual = residual.copy()
            residual[:2] = -np.inf
            residual[-2:] = -np.inf
        peak_index = int(np.argmax(residual))
        peak_db = float(residual[peak_index])
        if not np.isfinite(peak_db) or peak_db < 0.75:
            break
        half = peak_db / 2.0
        left = peak_index
        right = peak_index
        while left > 0 and residual[left] > half:
            left -= 1
        while right < residual.size - 1 and residual[right] > half:
            right += 1
        fc = float(frequencies[peak_index])
        bandwidth = max(float(frequencies[right] - frequencies[left]), fc / 20.0)
        q = float(np.clip(fc / bandwidth, 0.4, 12.0))
        gain_db = -float(np.clip(peak_db, 0.5, 12.0))
        current = {
            "fc_hz": round(fc, 3),
            "gain_db": round(gain_db, 3),
            "q": round(q, 3),
        }
        filters.append(current)
        total *= peq_response(frequencies, sample_rate, fc, q, gain_db)
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
) -> tuple[float, np.ndarray]:
    """Return energy-weighted mean absolute excess GD and its de-offset curve."""
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
    # A constant group delay is the arbitrary common time origin. Removing its
    # weighted median leaves frequency-dependent (excess) storage/decay.
    group_delay -= _weighted_median(group_delay, weights)
    log_frequency = np.log(evaluation_frequencies)
    numerator = np.trapezoid(weights * np.abs(group_delay), x=log_frequency)
    denominator = max(np.trapezoid(weights, x=log_frequency), EPS)
    return float(1000.0 * numerator / denominator), 1000.0 * group_delay


def _band_centres(low: float, high: float, ppo: int) -> np.ndarray:
    return log_frequency_grid(low, high, ppo)


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
        if self.band[1] >= self.sample_rate / 2.0:
            raise ValueError(
                f"Band upper edge {self.band[1]:g} Hz must be below Nyquist "
                f"({self.sample_rate / 2:g} Hz)"
            )
        self.frequencies = log_frequency_grid(*self.band, self.ppo)
        fft_frequencies = np.fft.rfftfreq(self.length, 1.0 / self.sample_rate)
        peak_absolute = (
            self.measurements[0].start_time_seconds
            + int(np.argmax(np.abs(self.measurements[0].impulse))) / self.sample_rate
        )
        spectra = []
        for measurement in self.measurements:
            raw = np.fft.rfft(measurement.impulse)
            shift = measurement.start_time_seconds - peak_absolute
            absolute = raw * np.exp(-2j * np.pi * fft_frequencies * shift)
            spectra.append(_interp_complex(fft_frequencies, absolute, self.frequencies))
        self.spectra = np.asarray(spectra, dtype=np.complex128)
        self._padded_spectra: np.ndarray | None = None
        self._padded_frequencies: np.ndarray | None = None

    def padded_spectra(self) -> tuple[np.ndarray, np.ndarray]:
        if self._padded_spectra is None or self._padded_frequencies is None:
            n_fft = next_fast_len(self.length * self.minphase_pad_factor)
            if n_fft % 2:
                n_fft = next_fast_len(n_fft + 1)
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
) -> dict[str, Any]:
    grid_sum = context.sum_on_grid(first, second, polarity, delay_ms, gain_db)
    magnitude_db = db20(grid_sum)
    smoothed_db = variable_smooth_db(magnitude_db, context.frequencies, context.ppo)
    trend_db = broad_trend_db(smoothed_db, context.ppo)
    full_sum, full_frequencies = context.sum_full(
        first, second, polarity, delay_ms, gain_db
    )
    excess_score, excess_curve = excess_group_delay(
        full_sum, full_frequencies, context.frequencies
    )
    filters, eq_grid = fit_cut_filters(
        grid_sum, context.frequencies, context.sample_rate, context.ppo
    )
    eq_full = filters_response(full_frequencies, context.sample_rate, filters)
    n_fft = 2 * (full_sum.size - 1)
    pre_ir = np.fft.irfft(full_sum, n=n_fft)
    post_ir = np.fft.irfft(full_sum * eq_full, n=n_fft)
    _, _, _, tail_by_band = csd_style_decay(
        post_ir, context.sample_rate, context.band, ppo=3
    )
    result: dict[str, Any] = {
        "null_score_db": float(np.max(np.maximum(0.0, trend_db - smoothed_db))),
        "excess_gd_ms": float(excess_score),
        "post_eq_tail_ms": float(np.max(tail_by_band)),
        "tail_by_band_ms": [round(float(value), 6) for value in tail_by_band],
        "filters": filters,
        "spl_db": float(10.0 * np.log10(max(np.mean(np.abs(grid_sum) ** 2), EPS))),
    }
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
                "smoothed_db": smoothed_db,
                "trend_db": trend_db,
                "solo_first_db": db20(context.spectra[first]),
                "solo_second_db": db20(context.spectra[second]),
                "excess_curve_ms": excess_curve,
                "post_eq_db": db20(grid_sum * eq_grid),
                "decay_frequencies": pre_f,
                "decay_times": pre_t,
                "pre_decay_db": pre_decay,
                "post_decay_db": post_decay,
            }
        )
    return result
