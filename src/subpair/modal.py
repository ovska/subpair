"""Parametric modal decomposition: matrix-pencil pole estimation and stored-energy
metrics.

This module answers a question the magnitude/phase scoring in ``dsp.py`` does not:
how hard does a given placement drive the room's own resonances, and for how long
does that stored energy stay audible? Delay/polarity/gain cannot change a room
mode's frequency or damping (those are properties of the room), but they change how
strongly a given sum excites each mode -- so two placements with near-identical
smoothed frequency response can differ substantially in modal excitation.

Design follows a two-stage, pole-fixed approach rather than per-band Schroeder
decay, because Schroeder integration in fractional-octave bands cannot separate
two modes closer together than the band, and a Q~9 mode at 50 Hz has a modal
bandwidth (~5 Hz) narrower than 1/6-octave smoothing (~8.4 Hz) at that frequency:

- Stage 1 (``estimate_room_poles``) estimates the room's pole set -- frequency and
  decay rate per mode -- jointly from every solo measurement via the matrix-pencil
  method, sweeping model order and requiring a pole to persist across both a
  majority of orders and a majority of measurements before it is trusted. This is
  the room's modal signature and does not depend on which pair is being evaluated.
- Stage 2 (``compute_pair_modal_metrics``) fixes that pole set and solves a linear
  least-squares fit for each candidate pair sum's residues (modal levels) only.
  This is fast, well-conditioned, and directly comparable across candidates,
  because only the fast-varying residues differ pair to pair.

None of this changes ``dsp.py``'s usable-output score by default; see
``engine.SearchOptions.modal_tiebreak`` for the configurable secondary ordering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from scipy import signal

from .dsp import EPS


LN10 = math.log(10.0)
# ln(1000): a T60 is, by definition, the time for level to fall 60 dB, i.e. for
# linear amplitude to fall to 10**(-3); amplitude ~ exp(-alpha t), so
# alpha * T60 = 3 * ln(10).
T60_LN_RATIO = 3.0 * LN10

_ROBUSTNESS_GAIN_STEPS_DB: tuple[float, ...] = (-1.0, 0.0, 1.0)
_ROBUSTNESS_POSITION_JITTER_CM = 10.0
_SPEED_OF_SOUND_M_PER_S = 343.0
_DEFAULT_TIMING_JITTER_MS = 0.5


@dataclass(frozen=True)
class ModalOptions:
    """Parameters governing pole estimation, gating, and metric thresholds.

    Defaults follow the motivating design brief: an 18-200 Hz analysis band
    decimated to 500 Hz for conditioning, a >=600 ms fit window (>=1.5x an
    assumed ~390 ms worst-case T60 at 50 Hz), a 10-60 pole model-order sweep,
    and 60% persistence gates across both order and measurement. These are
    heuristics, not first-principles thresholds; see ``PLAN.md`` for the
    reasoning and revisit them against real-world captures.
    """

    band: tuple[float, float] = (18.0, 200.0)
    decimated_fs_hz: float = 500.0
    window_seconds: float = 0.6
    order_min: int = 10
    order_max: int = 60
    order_step: int = 5
    pencil_ratio: float = 0.5
    freq_tolerance_hz: float = 0.5
    decay_tolerance_fraction: float = 0.2
    order_persistence_fraction: float = 0.6
    measurement_persistence_fraction: float = 0.6
    noise_floor_margin_db: float = 10.0
    high_q_threshold: float = 16.0
    level_gates_db: tuple[float, ...] = (-15.0, -20.0)
    primary_gate_db: float = -20.0
    audible_margin_db: float = 20.0
    min_fixed_pole_fit_r2: float = 0.5
    max_acceptable_discard_fraction: float = 0.5

    def __post_init__(self) -> None:
        low, high = self.band
        if low <= 0.0 or high <= low:
            raise ValueError("Modal analysis band must be positive and increasing")
        if self.decimated_fs_hz <= 2.0 * high:
            raise ValueError(
                "Modal decimated sample rate must exceed twice the analysis "
                "band's upper edge (Nyquist for the already band-limited signal)"
            )
        if self.window_seconds <= 0.0:
            raise ValueError("Modal fit window must be positive")
        if self.order_min < 2 or self.order_max < self.order_min:
            raise ValueError("Modal order sweep range is invalid")
        if self.order_step < 1:
            raise ValueError("Modal order sweep step must be at least 1")
        if not 0.0 < self.pencil_ratio < 1.0:
            raise ValueError("Modal pencil ratio must be between 0 and 1")
        if self.freq_tolerance_hz <= 0.0:
            raise ValueError("Modal frequency tolerance must be positive")
        if not 0.0 < self.decay_tolerance_fraction < 1.0:
            raise ValueError("Modal decay tolerance fraction must be between 0 and 1")
        if not 0.0 < self.order_persistence_fraction <= 1.0:
            raise ValueError("Modal order persistence fraction must be in (0, 1]")
        if not 0.0 < self.measurement_persistence_fraction <= 1.0:
            raise ValueError("Modal measurement persistence fraction must be in (0, 1]")
        if self.high_q_threshold <= 0.0:
            raise ValueError("Modal high-Q threshold must be positive")
        if not self.level_gates_db:
            raise ValueError("Modal level gates must be non-empty")
        if self.primary_gate_db not in self.level_gates_db:
            raise ValueError("Modal primary gate must be one of the reported level gates")


# Reference window for the direct-sound level (Ln's denominator): the spec's
# simpler alternative to a minimum-phase-equivalent early-response peak. Also
# used as the skip-ahead offset before the decay-fit window starts (see
# _prepare_segments below) so the fit window models only modal ringing, not
# the direct arrival itself.
DIRECT_REFERENCE_WINDOW_SECONDS = 0.02


def _bandlimited_decimated_impulse(
    impulse: np.ndarray,
    source_fs: float,
    band: tuple[float, float],
    target_fs: float,
) -> tuple[np.ndarray, float]:
    """Band-limit in the time domain and decimate to (approximately) ``target_fs``.

    This is the "frequency zoom" preprocessing step that conditions the
    matrix-pencil estimator (running it at the full capture rate is
    numerically poor, per Karjalainen et al.). A zero-phase Butterworth
    bandpass (mirroring ``dsp.csd_style_decay``'s filtering convention) is
    used rather than a hard frequency-domain mask: a brick-wall mask applied
    to an impulsive signal rings almost undamped (Gibbs), which would
    contaminate the whole fit window with non-physical ripple indistinguishable
    from a real, very-high-Q pole; a smooth filter's own impulse response
    decays quickly relative to real room-mode decay times.
    """
    nyquist = source_fs / 2.0
    low = max(band[0], 0.5)
    high = min(band[1], nyquist * 0.98)
    sos = signal.butter(2, [low, high], btype="bandpass", fs=source_fs, output="sos")
    pad = 3 * (2 * sos.shape[0] + 1)
    limited = (
        signal.sosfiltfilt(sos, impulse)
        if impulse.size > pad
        else signal.sosfilt(sos, impulse)
    )
    up = int(round(target_fs))
    down = int(round(source_fs))
    divisor = math.gcd(up, down)
    up //= divisor
    down //= divisor
    decimated = signal.resample_poly(limited, up, down)
    actual_fs = source_fs * up / down
    return decimated, actual_fs


def _windowed_segment(
    decimated_ir: np.ndarray, fs: float, peak_index: int, offset_seconds: float, duration_seconds: float
) -> tuple[np.ndarray, float]:
    """Segment of up to ``duration_seconds``, ``offset_seconds`` after ``peak_index``."""
    start = max(0, min(decimated_ir.size, peak_index + int(round(offset_seconds * fs))))
    end = min(decimated_ir.size, start + int(round(duration_seconds * fs)))
    segment = np.asarray(decimated_ir[start:end], dtype=np.float64)
    return segment, segment.size / fs


_NOISE_FLOOR_DB_FLOOR = -300.0


def _noise_floor_db(segment: np.ndarray) -> float:
    """RMS of the window's tail, in dB relative to the window's peak sample."""
    if segment.size < 16:
        # Too short to estimate a tail at all; a large-but-finite floor keeps
        # this JSON-serializable (the caller's allow_nan=False write) without
        # ever making a real mode look artificially far above the noise floor.
        return _NOISE_FLOOR_DB_FLOOR
    tail_length = max(8, segment.size // 5)
    tail = segment[-tail_length:]
    floor_rms = float(np.sqrt(np.mean(tail**2)))
    peak = max(float(np.max(np.abs(segment))), EPS)
    return 20.0 * math.log10(max(floor_rms, EPS) / peak)


def _prepare_segments(
    impulse: np.ndarray, source_fs: float, options: ModalOptions
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return ``(direct_segment, fit_segment, decimated_fs, fit_window_seconds)``.

    ``direct_segment`` covers the direct arrival (``DIRECT_REFERENCE_WINDOW_SECONDS``
    starting at the band-limited impulse's peak) and is used only to compute
    ``Ln``'s 0 dB reference. ``fit_segment`` starts immediately after it and
    is what the matrix-pencil estimator and the fixed-pole residue fit run
    against, so a broadband direct-sound transient -- which a handful of
    narrowband high-Q poles cannot represent -- is not mistaken for
    unexplained modal residual.
    """
    decimated, actual_fs = _bandlimited_decimated_impulse(
        impulse, source_fs, options.band, options.decimated_fs_hz
    )
    peak_index = int(np.argmax(np.abs(decimated)))
    direct_segment, _ = _windowed_segment(
        decimated, actual_fs, peak_index, 0.0, DIRECT_REFERENCE_WINDOW_SECONDS
    )
    fit_segment, achieved_seconds = _windowed_segment(
        decimated, actual_fs, peak_index, DIRECT_REFERENCE_WINDOW_SECONDS, options.window_seconds
    )
    return direct_segment, fit_segment, actual_fs, achieved_seconds


def _matrix_pencil_poles(segment: np.ndarray, order: int, pencil_ratio: float) -> np.ndarray:
    """Discrete poles ``z_i`` of ``segment`` via the matrix-pencil method.

    Hua & Sarkar's formulation: build an (N-L)x(L+1) Hankel matrix from the
    signal, split its top ``order`` right singular vectors into two
    row-shifted halves ``V1``/``V2``, and take the eigenvalues of
    ``pinv(V1) @ V2`` -- the generalized-eigenvalue solution of
    ``V2 v = z V1 v``. Preferred over Prony here because it goes through the
    SVD, which is markedly less noise-sensitive than solving Prony's
    companion-matrix normal equations directly.
    """
    n = segment.size
    order = max(1, min(order, max(1, n // 3)))
    pencil_length = int(round(pencil_ratio * n))
    pencil_length = max(order + 1, min(pencil_length, n - order - 1))
    if pencil_length < 2 or n - pencil_length < 2:
        return np.array([], dtype=np.complex128)
    hankel = np.lib.stride_tricks.sliding_window_view(segment, pencil_length + 1)
    try:
        _, _, vh = np.linalg.svd(hankel, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.array([], dtype=np.complex128)
    rank = min(order, vh.shape[0])
    if rank < 1:
        return np.array([], dtype=np.complex128)
    v = vh.conj().T[:, :rank]
    v1, v2 = v[:-1, :], v[1:, :]
    try:
        pencil = np.linalg.pinv(v1) @ v2
        return np.linalg.eigvals(pencil)
    except np.linalg.LinAlgError:
        return np.array([], dtype=np.complex128)


def _eigenvalues_to_modes(
    eigenvalues: np.ndarray, fs: float, band: tuple[float, float]
) -> list[tuple[float, float]]:
    """Keep only decaying, positive-frequency, in-band poles as ``(f_hz, alpha_per_s)``."""
    modes: list[tuple[float, float]] = []
    for z in eigenvalues:
        magnitude = abs(z)
        if not (0.0 < magnitude < 1.0 - 1e-9):
            continue  # non-decaying or unstable/growing: not a physical resonance
        angle = float(np.angle(z))
        if angle <= 1e-9:
            continue  # keep one pole per conjugate pair
        f = angle * fs / (2.0 * math.pi)
        if not (band[0] <= f <= band[1]):
            continue
        alpha = -fs * math.log(magnitude)
        if not math.isfinite(alpha) or alpha <= 0.0:
            continue
        modes.append((f, alpha))
    return modes


def _cluster_candidates(
    candidates: Sequence[tuple[int, float, float]],
    freq_tolerance_hz: float,
    decay_tolerance_fraction: float,
) -> list[list[tuple[int, float, float]]]:
    """Group ``(tag, f, alpha)`` candidates whose frequency and decay rate agree.

    A greedy frequency-sorted merge: this is not a globally optimal clustering
    (a group's running mean shifts as members are added), but with well-
    separated room modes and physically tight tolerances it recovers the
    intended groups reliably and stays deterministic and cheap.
    """
    ordered = sorted(candidates, key=lambda item: item[1])
    groups: list[list[tuple[int, float, float]]] = []
    for tag, f, alpha in ordered:
        placed = False
        for group in groups:
            group_f = sum(item[1] for item in group) / len(group)
            group_alpha = sum(item[2] for item in group) / len(group)
            if (
                abs(f - group_f) <= freq_tolerance_hz
                and abs(alpha - group_alpha) <= decay_tolerance_fraction * group_alpha
            ):
                group.append((tag, f, alpha))
                placed = True
                break
        if not placed:
            groups.append([(tag, f, alpha)])
    return groups


def _measurement_consensus_poles(
    segment: np.ndarray, fs: float, options: ModalOptions
) -> tuple[list[tuple[float, float, float]], int, int]:
    """Order-sweep + persistence filter for one measurement.

    Returns ``(retained, raw_candidate_count, kept_candidate_count)``, where
    ``retained`` is a list of ``(f_hz, alpha_per_s, order_persistence)``.
    """
    orders = list(range(options.order_min, options.order_max + 1, options.order_step))
    if not orders:
        raise ValueError("Modal order sweep range is empty")
    candidates: list[tuple[int, float, float]] = []
    for order_index, order in enumerate(orders):
        eigenvalues = _matrix_pencil_poles(segment, order, options.pencil_ratio)
        for f, alpha in _eigenvalues_to_modes(eigenvalues, fs, options.band):
            candidates.append((order_index, f, alpha))
    groups = _cluster_candidates(
        candidates, options.freq_tolerance_hz, options.decay_tolerance_fraction
    )
    retained: list[tuple[float, float, float]] = []
    kept_count = 0
    for group in groups:
        distinct_orders = {order_index for order_index, _, _ in group}
        persistence = len(distinct_orders) / len(orders)
        if persistence >= options.order_persistence_fraction:
            f_mean = float(np.mean([f for _, f, _ in group]))
            alpha_mean = float(np.mean([alpha for _, _, alpha in group]))
            retained.append((f_mean, alpha_mean, persistence))
            kept_count += len(group)
    return retained, len(candidates), kept_count


@dataclass(frozen=True)
class RoomMode:
    frequency_hz: float
    decay_rate_per_s: float
    measurement_persistence: float


@dataclass(frozen=True)
class RoomModalSignature:
    """The room's pooled modal signature: Stage 1's output.

    ``per_measurement`` holds, for each solo measurement, its own (pre-pooling)
    consensus poles matched to a pooled-mode index where possible -- this is
    the data the report's invariance-check plot renders directly.
    """

    modes: tuple[RoomMode, ...]
    decimated_fs_hz: float
    window_seconds: float
    discard_fraction: float
    per_measurement: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]
    valid: bool


def _match_to_pooled(
    poles: Sequence[tuple[float, float, float]],
    pooled: Sequence[RoomMode],
    options: ModalOptions,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for f, alpha, _persistence in poles:
        best_index: int | None = None
        best_distance: float | None = None
        for index, mode in enumerate(pooled):
            if abs(f - mode.frequency_hz) <= options.freq_tolerance_hz and abs(
                alpha - mode.decay_rate_per_s
            ) <= options.decay_tolerance_fraction * mode.decay_rate_per_s:
                distance = abs(f - mode.frequency_hz)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_index = index
        matches.append(
            {"frequency_hz": f, "t60_s": T60_LN_RATIO / alpha, "pooled_index": best_index}
        )
    return matches


def estimate_room_poles(
    measurements: Sequence[tuple[str, np.ndarray]],
    source_fs: float,
    options: ModalOptions,
) -> RoomModalSignature:
    """Stage 1: jointly estimate the room's pole set from every solo measurement.

    ``measurements`` is ``(title, impulse)`` pairs at ``source_fs``. Poles are
    pooled across measurements requiring ``measurement_persistence_fraction``
    agreement (default 60%), mirroring the per-measurement order-sweep
    persistence gate. If no pole clears both gates, ``valid`` is false and a
    warning explains why -- callers must not report modal scores in that case.
    """
    if len(measurements) < 2:
        raise ValueError("Joint modal pole estimation needs at least two measurements")
    warnings: list[str] = []
    per_measurement_raw: list[dict[str, Any]] = []
    all_candidates: list[tuple[int, float, float]] = []
    total_raw = 0
    total_kept = 0
    actual_fs = options.decimated_fs_hz
    for measurement_index, (title, impulse) in enumerate(measurements):
        _direct_segment, segment, actual_fs, achieved_seconds = _prepare_segments(
            impulse, source_fs, options
        )
        if achieved_seconds < options.window_seconds - 1e-9:
            warnings.append(
                f"{title}: modal fit window is {achieved_seconds * 1000.0:.0f} ms, "
                f"short of the requested {options.window_seconds * 1000.0:.0f} ms"
            )
        retained, raw_count, kept_count = _measurement_consensus_poles(
            segment, actual_fs, options
        )
        total_raw += raw_count
        total_kept += kept_count
        per_measurement_raw.append({"title": title, "poles": retained})
        for f, alpha, _persistence in retained:
            all_candidates.append((measurement_index, f, alpha))

    groups = _cluster_candidates(
        all_candidates, options.freq_tolerance_hz, options.decay_tolerance_fraction
    )
    n_measurements = len(measurements)
    pooled: list[RoomMode] = []
    for group in groups:
        distinct_measurements = {tag for tag, _, _ in group}
        persistence = len(distinct_measurements) / n_measurements
        if persistence >= options.measurement_persistence_fraction:
            f_mean = float(np.mean([f for _, f, _ in group]))
            alpha_mean = float(np.mean([alpha for _, _, alpha in group]))
            pooled.append(RoomMode(f_mean, alpha_mean, persistence))
    pooled.sort(key=lambda mode: mode.frequency_hz)

    discard_fraction = 1.0 - (total_kept / total_raw if total_raw else 0.0)
    valid = bool(pooled)
    if not valid:
        warnings.append(
            "No candidate pole was consistent enough, across model orders and "
            "measurements, to retain; modal metrics are not being reported"
        )
    elif discard_fraction > options.max_acceptable_discard_fraction:
        warnings.append(
            f"{discard_fraction * 100.0:.0f}% of candidate poles were discarded as "
            "noise or order-inconsistent; retained modal metrics may be unreliable"
        )

    per_measurement = tuple(
        {"title": entry["title"], "modes": _match_to_pooled(entry["poles"], pooled, options)}
        for entry in per_measurement_raw
    )
    return RoomModalSignature(
        modes=tuple(pooled),
        decimated_fs_hz=actual_fs,
        window_seconds=options.window_seconds,
        discard_fraction=discard_fraction,
        per_measurement=per_measurement,
        warnings=tuple(warnings),
        valid=valid,
    )


def fit_mode_residues(
    segment: np.ndarray, fs: float, poles: Sequence[tuple[float, float]]
) -> tuple[np.ndarray, float]:
    """Stage 2: fixed-pole linear least-squares fit for modal amplitudes.

    Each mode ``i`` contributes ``exp(-alpha_i t) * (a_i cos(omega_i t) + b_i
    sin(omega_i t))`` to the real segment; solving for ``(a_i, b_i)`` is
    ordinary linear least squares once the poles are fixed, which is what
    makes Stage 2 cheap enough to run per candidate. The reported amplitude is
    each mode's envelope value at ``t=0``, ``sqrt(a_i**2 + b_i**2)``.

    Returns ``(amplitudes, fit_r2)``; ``fit_r2`` is 1 minus the residual/total
    energy ratio and is used as the "does the fixed room pole set actually
    explain this sum" validity check from the design brief.
    """
    if not poles:
        return np.array([]), 0.0
    n = segment.size
    t = np.arange(n) / fs
    columns = []
    for f, alpha in poles:
        decay = np.exp(-alpha * t)
        omega = 2.0 * math.pi * f
        columns.append(decay * np.cos(omega * t))
        columns.append(decay * np.sin(omega * t))
    design = np.stack(columns, axis=1)
    coeffs, *_ = np.linalg.lstsq(design, segment, rcond=None)
    residual = segment - design @ coeffs
    residual_energy = float(np.sum(residual**2))
    total_energy = float(np.sum(segment**2)) + float(EPS)
    fit_r2 = float(1.0 - residual_energy / total_energy)
    amplitudes = np.hypot(coeffs[0::2], coeffs[1::2])
    return amplitudes, fit_r2


def mode_metrics(
    poles: Sequence[tuple[float, float]],
    amplitudes: np.ndarray,
    direct_reference: float,
    noise_floor_db: float,
    options: ModalOptions,
) -> list[dict[str, Any]]:
    """Per-mode ``f_n``/``T60_n``/``Q_n``/``L_n``/``t_audible,n``.

    Modes whose fitted level is within ``noise_floor_margin_db`` of the
    window's noise floor are dropped ("abort the fit" for that mode) rather
    than reported with a fabricated confidence.
    """
    direct_reference = max(direct_reference, EPS)
    modes: list[dict[str, Any]] = []
    for (f, alpha), amplitude in zip(poles, amplitudes):
        if alpha <= 0.0 or not math.isfinite(alpha):
            continue
        level_db = 20.0 * math.log10(max(float(amplitude), EPS) / direct_reference)
        if level_db - noise_floor_db < options.noise_floor_margin_db:
            continue
        t60 = T60_LN_RATIO / alpha
        q = math.pi * f * t60 / T60_LN_RATIO
        decline_db_per_s = 20.0 * alpha / LN10
        t_audible = (
            0.0
            if level_db <= -options.audible_margin_db
            else (level_db + options.audible_margin_db) / decline_db_per_s
        )
        modes.append(
            {
                "frequency_hz": f,
                "decay_rate_per_s": alpha,
                "t60_s": t60,
                "q": q,
                "level_db": level_db,
                "t_audible_s": t_audible,
            }
        )
    modes.sort(key=lambda mode: mode["frequency_hz"])
    return modes


def aggregate_modal_metrics(
    modes: Sequence[dict[str, Any]], options: ModalOptions
) -> dict[str, Any]:
    """``n_highQ`` (at every configured gate), the worst offender, stored energy,
    and ``ringing_ms``.

    Gating is on *both* Q and level: a high-Q pole far enough below the direct
    sound is not audible and must not inflate the count. ``sum_modal_energy_db``
    uses only the primary gate so it stays comparable across pairs; the other
    configured gates are reported purely to show the metric's sensitivity to
    the threshold, per the design brief.

    ``ringing_ms`` is ``max(t_audible_n)`` over *every* retained mode (not
    just the Q-gated ones -- a moderate-Q mode ringing loudly for a while is
    still audible ringing even if it never counts toward ``n_highQ``), i.e.
    the worst-case time for this pair's excitation to fall below the
    audibility margin relative to its own direct sound. Unlike a fixed
    -20 dB-from-local-peak CSD envelope crossing in a 1/3-octave band
    (``dsp.csd_style_decay``'s ``raw_tail_ms``/``post_eq_tail_ms``), this is
    referenced to the actual direct-sound level of the same sum and isn't
    blurred across modes narrower than a fractional-octave band -- see
    ``PLAN.md`` for the comparison this was chosen over. It is ``None`` when
    no mode survived the noise-floor gate in ``mode_metrics``.
    """
    by_gate: dict[str, int] = {}
    for gate_db in options.level_gates_db:
        gated = [
            m for m in modes if m["q"] > options.high_q_threshold and m["level_db"] > gate_db
        ]
        by_gate[f"{gate_db:g}"] = len(gated)
    primary_gated = [
        m
        for m in modes
        if m["q"] > options.high_q_threshold and m["level_db"] > options.primary_gate_db
    ]
    if primary_gated:
        worst = max(primary_gated, key=lambda m: m["q"])
        q_max = worst["q"]
        q_max_triple = {
            "frequency_hz": worst["frequency_hz"],
            "q": worst["q"],
            "level_db": worst["level_db"],
        }
        energy_linear = sum(
            10.0 ** (m["level_db"] / 10.0) / m["decay_rate_per_s"] for m in primary_gated
        )
        sum_modal_energy_db: float | None = 10.0 * math.log10(max(energy_linear, EPS))
    else:
        q_max = 0.0
        q_max_triple = None
        sum_modal_energy_db = None
    ringing_ms = 1000.0 * max(m["t_audible_s"] for m in modes) if modes else None
    return {
        "modes": modes,
        "n_high_q": len(primary_gated),
        "n_high_q_by_gate_db": by_gate,
        "q_max": q_max,
        "q_max_triple": q_max_triple,
        "sum_modal_energy_db": sum_modal_energy_db,
        "ringing_ms": ringing_ms,
        "high_q_threshold": options.high_q_threshold,
        "primary_gate_db": options.primary_gate_db,
    }


def compute_pair_modal_metrics(
    signature: RoomModalSignature,
    impulse: np.ndarray,
    source_fs: float,
    options: ModalOptions,
) -> dict[str, Any]:
    """Stage 2 + metrics for one candidate pair sum's time-domain impulse.

    ``impulse`` is the pair sum's full-band impulse response (e.g.
    ``np.fft.irfft`` of ``AnalysisContext.sum_full``'s complex spectrum, the
    same construction ``dsp.pair_diagnostics`` already uses for its own
    decay/CSD diagnostics) at ``source_fs``.
    """
    if not signature.valid:
        return {"valid": False, "warnings": list(signature.warnings)}
    direct_segment, fit_segment, actual_fs, achieved_seconds = _prepare_segments(
        impulse, source_fs, options
    )
    noise_floor_db = _noise_floor_db(fit_segment)
    direct_reference = float(np.max(np.abs(direct_segment)))
    poles = [(mode.frequency_hz, mode.decay_rate_per_s) for mode in signature.modes]
    amplitudes, fit_r2 = fit_mode_residues(fit_segment, actual_fs, poles)
    modes = mode_metrics(poles, amplitudes, direct_reference, noise_floor_db, options)
    aggregate = aggregate_modal_metrics(modes, options)
    warnings = list(signature.warnings)
    if achieved_seconds < options.window_seconds - 1e-9:
        warnings.append(
            f"modal fit window is {achieved_seconds * 1000.0:.0f} ms, short of the "
            f"requested {options.window_seconds * 1000.0:.0f} ms"
        )
    aggregate.update(
        {
            "valid": True,
            "fixed_pole_fit_r2": fit_r2,
            "fixed_pole_fit_flagged": fit_r2 < options.min_fixed_pole_fit_r2,
            "window_seconds": achieved_seconds,
            "noise_floor_db": noise_floor_db,
            "warnings": warnings,
        }
    )
    return aggregate


def modal_robustness(
    signature: RoomModalSignature,
    sum_impulse: Callable[[float, float], np.ndarray],
    source_fs: float,
    nominal_delay_ms: float,
    nominal_gain_db: float,
    options: ModalOptions,
    timing_jitter_ms: float = _DEFAULT_TIMING_JITTER_MS,
) -> dict[str, Any]:
    """Fraction of a small (delay, gain) neighbourhood where ``n_highQ`` holds.

    ``sum_impulse(delay_ms, gain_db)`` must return the pair's full-band
    time-domain impulse at the given delay/gain (polarity fixed by the
    caller's closure) -- see ``compute_pair_modal_metrics`` for the expected
    construction.

    The neighbourhood combines +-``timing_jitter_ms`` (measurement/timing
    uncertainty, default 0.5 ms) with a +-10 cm placement-uncertainty
    equivalent (about 0.29 ms at 343 m/s) via RSS, crossed with +-1 dB gain
    drift -- this codebase has no continuous-position search axis, so
    "position" uncertainty is folded into the one continuous axis (delay) that
    stands in for it, rather than left unmodelled.
    """
    if not signature.valid:
        return {"valid": False}
    position_jitter_ms = (
        1000.0 * (_ROBUSTNESS_POSITION_JITTER_CM / 100.0) / _SPEED_OF_SOUND_M_PER_S
    )
    delay_jitter_ms = math.hypot(timing_jitter_ms, position_jitter_ms)
    delay_steps = (-delay_jitter_ms, 0.0, delay_jitter_ms)
    nominal_n_high_q: int | None = None
    outcomes: list[int | None] = []
    for delay_offset in delay_steps:
        for gain_offset in _ROBUSTNESS_GAIN_STEPS_DB:
            impulse = sum_impulse(
                nominal_delay_ms + delay_offset, nominal_gain_db + gain_offset
            )
            metrics = compute_pair_modal_metrics(signature, impulse, source_fs, options)
            n_high_q = metrics["n_high_q"] if metrics.get("valid") else None
            outcomes.append(n_high_q)
            if delay_offset == 0.0 and gain_offset == 0.0:
                nominal_n_high_q = n_high_q
    if nominal_n_high_q is None:
        return {"valid": False}
    stable = sum(1 for value in outcomes if value == nominal_n_high_q)
    return {
        "valid": True,
        "nominal_n_high_q": nominal_n_high_q,
        "neighbourhood_size": len(outcomes),
        "fraction_stable": stable / len(outcomes),
        "delay_jitter_ms": delay_jitter_ms,
        "gain_jitter_db": max(_ROBUSTNESS_GAIN_STEPS_DB),
    }
