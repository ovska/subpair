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
    trend = broad_trend_db(magnitude_db, ppo)
    return np.maximum(0.0, np.max(trend - magnitude_db, axis=-1))


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


@dataclass(frozen=True)
class EqOptions:
    target: str = "trend"
    correction_range: tuple[float, float] | None = None
    correction_slope_db_per_octave: float = 48.0
    max_boost_db: float = 0.0
    max_cut_db: float = 18.0
    max_filters: int = 7

    def __post_init__(self) -> None:
        if self.target not in {"trend", "flat"}:
            raise ValueError("EQ target must be 'trend' or 'flat'")
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

    Large peaks are detected on a smoothed cycles-of-delay curve, expanded to
    at least a one-third-octave Gaussian gate, and averaged once more after the
    nonlinear authority mapping.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)
    excess_cycles = np.abs(excess_group_delay_ms) * frequencies / 1000.0
    if frequencies.size < 3:
        return 1.0 / (1.0 + (excess_cycles / 0.35) ** 4)
    steps = np.diff(np.log2(frequencies))
    ppo = max(1.0, 1.0 / float(np.median(steps)))
    analysis_sigma = max(0.5, (ppo / 12.0) / 2.354820045)
    smoothed_cycles = ndimage.gaussian_filter1d(
        excess_cycles, sigma=analysis_sigma, mode="nearest", truncate=3.0
    )
    risk = ndimage.gaussian_filter1d(
        excess_cycles,
        sigma=max(0.5, (ppo / 6.0) / 2.354820045),
        mode="nearest",
        truncate=3.0,
    )
    peaks, properties = signal.find_peaks(
        smoothed_cycles,
        height=0.08,
        prominence=0.03,
        distance=max(1, int(round(ppo / 12.0))),
    )
    if peaks.size:
        widths = signal.peak_widths(smoothed_cycles, peaks, rel_height=0.5)[0]
        indices = np.arange(frequencies.size, dtype=np.float64)
        for peak, height, width in zip(peaks, properties["peak_heights"], widths):
            gate_fwhm = max(float(width), ppo / 3.0)
            gate_sigma = gate_fwhm / 2.354820045
            gate = float(height) * np.exp(
                -0.5 * ((indices - float(peak)) / gate_sigma) ** 2
            )
            risk = np.maximum(risk, gate)
    authority = 1.0 / (1.0 + (risk / 0.35) ** 4)
    return ndimage.gaussian_filter1d(
        authority,
        sigma=max(0.5, (ppo / 6.0) / 2.354820045),
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


def fit_eq_filters(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    sample_rate: float,
    ppo: int,
    excess_group_delay_ms: np.ndarray,
    options: EqOptions,
) -> tuple[list[dict[str, float]], np.ndarray, dict[str, Any]]:
    """Fit a bounded PEQ bank toward a range- and excess-GD-aware target."""
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
    if options.target == "flat":
        percentile = 50.0 if options.max_boost_db > 0.0 else 30.0
        target_level = float(np.percentile(base_db[in_range], percentile))
        nominal_target = np.full_like(base_db, target_level)
    else:
        nominal_target = broad_trend_db(base_db, ppo)
        target_level = float(np.median(nominal_target[in_range]))

    desired = np.clip(
        nominal_target - base_db, -options.max_cut_db, options.max_boost_db
    )
    range_authority = _correction_range_authority(
        frequencies, correction_range, options.correction_slope_db_per_octave
    )
    gd_authority = _excess_gd_authority(frequencies, excess_group_delay_ms)
    authority = range_authority * gd_authority
    desired *= authority
    effective_target = base_db + desired

    total = np.ones_like(spectrum, dtype=np.complex128)
    filters: list[dict[str, float]] = []
    threshold_db = 0.35 if options.target == "flat" else 0.75
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
        best: tuple[float, np.ndarray, dict[str, float]] | None = None
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
                best = (trial_error, trial, current)
        if best is None:
            break
        _, total, current = best
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
        EqOptions(max_filters=maximum),
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
) -> tuple[float, np.ndarray]:
    """Return energy-weighted mean absolute excess GD and its de-offset curve.

    The minimum-phase transform and group-delay derivative use the complete
    supplied spectra and evaluation grid. When ``integration_range`` is set,
    only that frequency interval contributes to common-delay removal and the
    scalar score used for ranking.
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
    # A constant group delay is the arbitrary common time origin. Removing its
    # weighted median leaves frequency-dependent (excess) storage/decay.
    group_delay -= _weighted_median(group_delay[score_mask], weights[score_mask])
    score_frequencies = evaluation_frequencies[score_mask]
    score_weights = weights[score_mask]
    score_delays = group_delay[score_mask]
    log_frequency = np.log(score_frequencies)
    numerator = np.trapezoid(score_weights * np.abs(score_delays), x=log_frequency)
    denominator = max(np.trapezoid(score_weights, x=log_frequency), EPS)
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
    eq_options: EqOptions | None = None,
) -> dict[str, Any]:
    eq_options = eq_options or EqOptions(correction_range=context.band)
    grid_sum = context.sum_on_grid(first, second, polarity, delay_ms, gain_db)
    magnitude_db = db20(grid_sum)
    trend_db = broad_trend_db(magnitude_db, context.ppo)
    full_sum, full_frequencies = context.sum_full(
        first, second, polarity, delay_ms, gain_db
    )
    excess_score, excess_curve = excess_group_delay(
        full_sum,
        full_frequencies,
        context.frequencies,
        integration_range=eq_options.correction_range,
    )
    filters, eq_grid, eq_metadata = fit_eq_filters(
        grid_sum,
        context.frequencies,
        context.sample_rate,
        context.ppo,
        excess_curve,
        eq_options,
    )
    eq_full = filters_response(full_frequencies, context.sample_rate, filters)
    n_fft = 2 * (full_sum.size - 1)
    pre_ir = np.fft.irfft(full_sum, n=n_fft)
    post_full = full_sum * eq_full
    post_ir = np.fft.irfft(post_full, n=n_fft)
    _, _, _, raw_tail_by_band = csd_style_decay(
        pre_ir, context.sample_rate, context.band, ppo=3
    )
    _, _, _, tail_by_band = csd_style_decay(
        post_ir, context.sample_rate, context.band, ppo=3
    )
    post_grid = grid_sum * eq_grid
    post_magnitude_db = db20(post_grid)
    post_trend_db = broad_trend_db(post_magnitude_db, context.ppo)
    post_excess_score, post_excess_curve = excess_group_delay(
        post_full,
        full_frequencies,
        context.frequencies,
        integration_range=eq_options.correction_range,
    )
    result: dict[str, Any] = {
        "null_score_db": float(np.max(np.maximum(0.0, trend_db - magnitude_db))),
        "excess_gd_ms": float(excess_score),
        "raw_tail_ms": float(np.max(raw_tail_by_band)),
        "raw_tail_by_band_ms": [round(float(value), 6) for value in raw_tail_by_band],
        "post_eq_null_score_db": float(
            np.max(np.maximum(0.0, post_trend_db - post_magnitude_db))
        ),
        "post_eq_excess_gd_ms": float(post_excess_score),
        "post_eq_tail_ms": float(np.max(tail_by_band)),
        "tail_by_band_ms": [round(float(value), 6) for value in tail_by_band],
        "filters": filters,
        "eq_target": eq_metadata["target"],
        "eq_target_level_db": float(eq_metadata["target_level_db"]),
        "eq_mean_authority": float(np.mean(eq_metadata["eq_authority"])),
        "spl_db": float(10.0 * np.log10(max(np.mean(np.abs(grid_sum) ** 2), EPS))),
        "post_eq_spl_db": float(
            10.0 * np.log10(max(np.mean(np.abs(post_grid) ** 2), EPS))
        ),
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
                "trend_db": trend_db,
                "solo_first_db": db20(context.spectra[first]),
                "solo_second_db": db20(context.spectra[second]),
                "excess_curve_ms": excess_curve,
                "post_eq_db": post_magnitude_db,
                "post_eq_trend_db": post_trend_db,
                "post_eq_excess_curve_ms": post_excess_curve,
                "eq_target_db": eq_metadata["effective_target_db"],
                "eq_nominal_target_db": eq_metadata["nominal_target_db"],
                "eq_authority": eq_metadata["eq_authority"],
                "decay_frequencies": pre_f,
                "decay_times": pre_t,
                "pre_decay_db": pre_decay,
                "post_decay_db": post_decay,
            }
        )
    return result
