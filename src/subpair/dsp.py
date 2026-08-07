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


def _grid_ppo(frequencies: np.ndarray) -> float:
    """Points-per-octave implied by a log-frequency grid's actual spacing."""
    if frequencies.size < 2:
        return 48.0
    steps = np.diff(np.log2(np.asarray(frequencies, dtype=np.float64)))
    return max(1.0, 1.0 / float(np.median(steps)))


WIDE_DIP_MARGIN_OCTAVES = 1.0


def _two_sided_wide_dip_db(magnitude_db: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    """Dip below the higher of the best level well to the left AND well to the right.

    A genuine cancellation dip, however wide, sits between normal-level
    content on both sides and this flags it at close to its true depth. A
    smooth monotonic rolloff never recovers on at least one side, so it
    always scores zero here regardless of how much total range it spans:
    that's the deliberate difference from a single global reference (a
    percentile, a single wide trend), which cannot tell "sustained decline"
    from "bounded dip" apart. The margin before a side starts counting keeps
    this from just re-detecting narrow, already-``broad_trend_db``-covered
    notches off their own immediate shoulders, and near the array's own
    edges - where one side has no room left for a margin plus a reference -
    it stays silent rather than guessing.
    """
    magnitude_db = np.asarray(magnitude_db, dtype=np.float64)
    n = magnitude_db.shape[-1]
    margin_bins = int(round(_grid_ppo(frequencies) * WIDE_DIP_MARGIN_OCTAVES))
    if n <= margin_bins:
        return np.zeros_like(magnitude_db)
    left_reference = np.full_like(magnitude_db, -np.inf)
    right_reference = np.full_like(magnitude_db, -np.inf)
    left_cummax = np.maximum.accumulate(magnitude_db, axis=-1)
    right_cummax = np.maximum.accumulate(magnitude_db[..., ::-1], axis=-1)[..., ::-1]
    left_reference[..., margin_bins:] = left_cummax[..., : n - margin_bins]
    right_reference[..., : n - margin_bins] = right_cummax[..., margin_bins:]
    reference = np.minimum(left_reference, right_reference)
    return np.maximum(0.0, reference - magnitude_db)


def dip_below_trend_db(
    magnitude_db: np.ndarray, trend_db: np.ndarray, frequencies: np.ndarray
) -> np.ndarray:
    """Per-frequency dip depth: below the narrow trend OR a two-sided wide check.

    ``trend_db`` (see ``broad_trend_db``) is a ~1-octave smoothing of the
    same curve it's compared against, so it is a good reference for narrow
    comb-filtering nulls but a dip much wider than that window is largely
    absorbed into the trend itself and under-reported (the trend just
    follows it down). ``_two_sided_wide_dip_db`` catches that case without
    the false positives a single global baseline gives on an ordinary
    monotonic rolloff.
    """
    magnitude_db = np.asarray(magnitude_db, dtype=np.float64)
    narrow_dip = np.maximum(0.0, np.asarray(trend_db, dtype=np.float64) - magnitude_db)
    wide_dip = _two_sided_wide_dip_db(magnitude_db, frequencies)
    return np.maximum(narrow_dip, wide_dip)


def null_scores(
    spectra: np.ndarray,
    frequencies: np.ndarray,
    ppo: int,
    score_slice: slice | None = None,
) -> np.ndarray:
    """Max dip of magnitude below its one-octave trend or a wide two-sided check.

    When ``score_slice`` is given, ``spectra``/``frequencies`` are assumed to
    carry real spectral content beyond the reported band (see
    ``AnalysisContext.trend_frequencies``) so neither the trend nor the wide
    check in ``dip_below_trend_db`` are biased by edge-replicated or
    edge-truncated data; the max-dip search itself is still restricted to
    the reported band via the slice.
    """
    magnitude_db = db20(spectra)
    trend = broad_trend_db(magnitude_db, ppo)
    dip = dip_below_trend_db(magnitude_db, trend, frequencies)
    if score_slice is not None:
        dip = dip[..., score_slice]
    return np.max(dip, axis=-1)


LOW_END_EXTENSION_F3_THRESHOLD_DB = 3.0
LOW_END_EXTENSION_F6_THRESHOLD_DB = 6.0


def _two_sided_envelope_db(trend_db: np.ndarray) -> np.ndarray:
    """Two-sided envelope: at each point, the higher of what's attainable from either side.

    ``envelope[i] = min(max(trend_db[:i+1]), max(trend_db[i:]))``. A narrow
    dip/null that fully recovers on both sides cannot pull this down - both
    directions "see past" it to the higher level beyond - but a genuine,
    sustained decline *does* pull it down, since the declining side's own
    running max keeps tracking the decline while the other (undeclined) side
    stays pinned at the higher passband level, and the envelope takes
    whichever side is lower. This is the same idea as
    ``_two_sided_wide_dip_db``'s two-sided reference, without that function's
    inter-metric margin: there is nothing else here it needs to avoid
    double-counting against, and no margin means every index - including
    both array edges - gets a well-defined value from a plain two-sided
    cummax rather than needing edge-case fallbacks.
    """
    trend_db = np.asarray(trend_db, dtype=np.float64)
    ascending = np.maximum.accumulate(trend_db)
    descending = np.maximum.accumulate(trend_db[::-1])[::-1]
    return np.minimum(ascending, descending)


def low_end_extension_hz(
    trend_db: np.ndarray,
    frequencies: np.ndarray,
    threshold_db: float = LOW_END_EXTENSION_F3_THRESHOLD_DB,
    reference_db: float | None = None,
) -> float | None:
    """Lowest frequency the broad trend holds up before permanently falling ``threshold_db`` below a reference.

    An F3/F6-style extension estimate (``threshold_db`` selects which one -
    ``LOW_END_EXTENSION_F3_THRESHOLD_DB``/``LOW_END_EXTENSION_F6_THRESHOLD_DB``),
    deliberately measuring the *envelope* (``_two_sided_envelope_db``) rather
    than the raw trend: a narrow, recoverable dip or null is a placement
    defect the null-score metric already scores on its own terms, not a
    low-end-extension defect, so it must not be able to drag the reported
    extension up to its own frequency by itself the way scanning the raw
    trend would. A curve that is flat down to 25 Hz but has one isolated
    -5 dB notch at 100 Hz still reports ~25 Hz extension, not ~100 Hz,
    because the envelope's two-sided cummax sees past the notch on both
    sides. A *sustained* rolloff is not treated this way: below the corner,
    only the low side of the envelope keeps tracking the decline (the high
    side has nothing left to see past), so the envelope decline still lands
    close to where the raw curve actually crosses the threshold.

    The scan always starts at the envelope's own *peak*, wherever in the
    band it occurs - not the value at the top of the band. A two-subwoofer
    sum is routinely bandpass-shaped (it rises out of the bottom of the
    band, peaks somewhere in the middle, and rolls off again toward
    crossover) rather than staying flat all the way to the top edge;
    starting the scan at the top-of-band sample specifically is fragile in
    that case - if the curve is already declining well before the top edge
    (true for a completely ordinary, well-behaved response, not a defect),
    that edge sample can sit more than ``threshold_db`` below the response's
    own peak, which would make the *entire* scan fail immediately and
    misreport a permanently-collapsed low end regardless of how
    well-extended the actual low end is. Starting at the envelope's own peak
    (its highest sustained plateau, found via ``np.argmax`` - which lands on
    the *lowest* frequency of that plateau when the peak is a broad flat
    region, not a single sample) is unaffected by whatever happens to the
    response *above* the peak, which is a high-end/crossover concern this
    metric is not about. When the peak is already at the top of the band (a
    monotonically-rising passband, e.g. a simple single-corner rolloff),
    this is identical to scanning down from the top edge, so ordinary
    single-corner responses are scored exactly as before.

    ``trend_db`` need not be an absolute level. ``reference_db`` sets what
    level the ``threshold_db`` drop is measured from:

    - Leaving ``reference_db`` ``None`` uses the envelope's own peak value,
      making the metric fully self-referential: "how far does this
      placement's own low end extend relative to its own best-supported
      level," regardless of how that level compares to any other placement.
    - Passing ``trend_db - best_curve`` (an elementwise difference against
      the *best* value found at each frequency across every candidate in a
      search - not necessarily from the same candidate throughout, one
      value per frequency, i.e. compute the subtraction *before* calling
      this function) together with ``reference_db=0.0`` answers a genuinely
      cross-pair-comparable question instead: "how far does this
      placement's low end extend before falling behind what the *best*
      candidate here delivers at that same frequency." The scan position
      still starts at this placement's own best-relative-to-the-best-curve
      point (found from the departure curve's own envelope, for the
      bandpass-shape reason above), but the threshold itself no longer
      depends on where that point is or how high it is - it is fixed at
      "0 dB departure from the best curve," so a placement that is
      uniformly quieter everywhere, not just at one frequency, cannot hide
      that by construction. An elementwise *average* curve was tried and
      reverted: most real placements naturally roll off toward the bottom
      of the band to some degree, so the average curve already has a
      "typical" rolloff baked into its own shape, which hides exactly a
      normal amount of rolloff in any placement that isn't unusually bad -
      the *best* curve has no such baked-in rolloff, since it is literally
      the best SPL any candidate actually delivers at each frequency.

    A placement whose own peak (of whatever curve was passed) already falls
    more than ``threshold_db`` below the reference legitimately has no
    frequency at which it is within spec, so this returns ``None`` - not a
    number, since any Hz value would misleadingly suggest a real crossing
    point exists. Report/CLI consumers should render that as an empty/blank
    cell, not a number.

    Scanning downward from the peak, this returns the highest frequency at
    which the *running minimum* of the envelope first falls ``threshold_db``
    below the reference and does not recover, or the band's own lower edge
    if it never does (fully extended through the analyzed range). This is
    purely diagnostic - it is not part of the raw or EQ'd ranking key - so a
    placement's own null/excess-GD/tail severity always decides the winner;
    it only summarizes how far down the winning (or any) placement's
    underlying passband shape reaches.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)
    trend_db = np.asarray(trend_db, dtype=np.float64)
    if frequencies.size == 0:
        return 0.0
    envelope = _two_sided_envelope_db(trend_db)
    peak_index = int(np.argmax(envelope))
    reference = float(envelope[peak_index]) if reference_db is None else float(reference_db)
    threshold = reference - threshold_db
    segment = envelope[: peak_index + 1]
    running_min_from_peak = np.minimum.accumulate(segment[::-1])[::-1]
    meets_threshold = running_min_from_peak >= threshold
    if not np.any(meets_threshold):
        return None
    return float(frequencies[np.argmax(meets_threshold)])


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
    below ``fc`` and 0 dB well above it, by design: it is a broad tonal
    control, not a corrective filter, and is deliberately not fitted by
    ``fit_eq_filters`` (see ``ShelfOptions``).
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
class ShelfOptions:
    """A fixed, user-specified low-shelf tonal control.

    Set via ``EqOptions.shelf`` and ``cli.py``'s ``search`` subcommand's
    ``--low-shelf-*`` flags, exactly like ``max_boost_db``/``max_filters``:
    a search-time EQ configuration choice, fitted into the post-EQ response
    every ranking-relevant score is computed from (see ``fit_eq_filters``).
    Changing it means re-running ``search``, the same as changing any other
    ``EqOptions`` field - ``report``/``verify`` read whichever shelf a given
    ``search-results.json`` was generated with and cannot override it.
    Unlike the bounded, corrective bell bank ``fit_eq_filters`` greedily
    fits, this is a broad tonality preference set directly by the user
    (more/less sub-bass): the bells are fitted completely unaware of it (see
    ``fit_eq_filters``'s docstring), so a deliberate tonal tilt is not
    fought/cancelled by the corrective fitter the way a genuine response
    defect at the same frequencies would be.
    """

    freq_hz: float | None = None
    gain_db: float = 0.0
    slope: float = 1.0

    def __post_init__(self) -> None:
        if self.gain_db != 0.0 and self.freq_hz is None:
            raise ValueError("A nonzero low-shelf gain requires a low-shelf frequency")
        if self.freq_hz is not None and self.freq_hz <= 0.0:
            raise ValueError("Low-shelf frequency must be positive")
        if not -15.0 <= self.gain_db <= 15.0:
            raise ValueError("Low-shelf gain must be between -15 and 15 dB")
        if not 0.1 <= self.slope <= 1.0:
            raise ValueError("Low-shelf slope must be between 0.1 and 1.0")

    @property
    def active(self) -> bool:
        return self.gain_db != 0.0 and self.freq_hz is not None

    def response(self, frequencies: np.ndarray, sample_rate: float) -> np.ndarray:
        if not self.active:
            return np.ones_like(np.asarray(frequencies, dtype=np.float64), dtype=np.complex128)
        return low_shelf_response(frequencies, sample_rate, self.freq_hz, self.gain_db, self.slope)


@dataclass(frozen=True)
class EqOptions:
    target: str = "trend"
    correction_range: tuple[float, float] | None = None
    correction_slope_db_per_octave: float = 48.0
    max_boost_db: float = 0.0
    max_cut_db: float = 18.0
    max_filters: int = 7
    shelf: ShelfOptions = ShelfOptions()

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


DIP_GD_SEVERITY_WEIGHT = 1.5  # up to +150% dip severity where excess GD is worst
DSP_TARGET_MIN_PHASE_DIP_WEIGHT = 0.2  # 'dsp' target: minimum-phase dips barely count


def gd_weighted_null_score(
    magnitude_db: np.ndarray,
    trend_db: np.ndarray,
    frequencies: np.ndarray,
    excess_group_delay_ms: np.ndarray,
    dsp_target: bool = False,
) -> float:
    """Max magnitude dip below trend, scaled up where it coincides with excess GD.

    Dip depth itself comes from ``dip_below_trend_db``: below the narrow
    trend OR a two-sided wide check, so a dip much wider than the trend's
    ~1-octave window is not missed. A plain magnitude dip is scored
    the same whether it is a shallow, EQ-fixable amplitude ripple or a
    genuine destructive-interference null (acoustically irreparable and
    audible as smearing). This reuses the same excess-GD risk gate as the EQ
    authority curve (``_excess_gd_authority``) to inflate dip severity where
    it coincides with real excess group delay, so the reported/ranked score
    reflects how *fixable* a dip is, not just how deep it looks in magnitude
    alone. A dip with zero depth stays zero regardless of nearby group
    delay; group-delay-only regions are already handled separately by the
    EQ authority curve.

    A magnitude peak above the trend is deliberately *not* scored as a dip
    is, even at the same excess-GD risk: reinforcement adds output rather
    than destructively cancelling it. But a peak that only exists because of
    real excess group delay - not minimum-phase, i.e. not explained by its
    own magnitude shape - is a resonance/ringing signature (comb reinforcement
    with genuine energy storage), not a benign constructive bump, and is
    scored the same way a dip's severity is inflated: proportional to
    ``gd_risk`` alone, so a minimum-phase peak (``gd_risk`` near 0) still
    scores exactly zero, and only a non-minimum-phase peak counts at all.

    ``dsp_target=True`` (the ``'dsp'`` EQ target) is for placements that
    will be corrected by a full-featured external DSP rather than subpair's
    own conservative fitter: a *minimum-phase* dip is, by definition, fully
    correctable by any minimum-phase EQ (a boost that follows the same
    minimum-phase relationship exactly restores both magnitude and phase),
    so it barely counts here at ``gd_risk`` near 0
    (``DSP_TARGET_MIN_PHASE_DIP_WEIGHT`` instead of the usual full weight).
    But a *non*-minimum-phase dip remains a genuine, DSP-unfixable
    cancellation regardless of target, so both modes converge to the same
    maximum severity as ``gd_risk`` rises to 1 - the ranking in ``dsp`` mode
    ends up preferring flat excess group delay over flat raw magnitude, since
    magnitude-only problems are assumed to be someone else's problem to fix
    later. Peak scoring is unaffected by ``dsp_target``: minimum-phase peaks
    already score zero in every mode.

    This is deliberately not used inside the fast exhaustive delay/gain/
    polarity search: true excess group delay needs a minimum-phase
    extraction per candidate, which is too expensive to run over that whole
    grid (and coarse-grid phase unwrapping is fragile exactly at deep
    nulls). It is computed once per finalist instead.
    """
    magnitude_db = np.asarray(magnitude_db, dtype=np.float64)
    trend_db = np.asarray(trend_db, dtype=np.float64)
    dip_db = dip_below_trend_db(magnitude_db, trend_db, frequencies)
    peak_db = np.maximum(0.0, magnitude_db - trend_db)
    gd_risk = 1.0 - _excess_gd_authority(frequencies, excess_group_delay_ms)
    max_dip_multiplier = 1.0 + DIP_GD_SEVERITY_WEIGHT
    base_dip_multiplier = DSP_TARGET_MIN_PHASE_DIP_WEIGHT if dsp_target else 1.0
    dip_multiplier = base_dip_multiplier + (max_dip_multiplier - base_dip_multiplier) * gd_risk
    dip_severity_db = dip_db * dip_multiplier
    peak_severity_db = peak_db * DIP_GD_SEVERITY_WEIGHT * gd_risk
    severity_db = np.maximum(dip_severity_db, peak_severity_db)
    if severity_db.size == 0:
        return 0.0
    return float(np.max(severity_db))


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
    margin_spectrum: np.ndarray | None = None,
    margin_slice: slice | None = None,
) -> tuple[list[dict[str, float]], np.ndarray, dict[str, Any]]:
    """Fit a bounded PEQ bank toward a range- and excess-GD-aware target.

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
    # Same excess-GD risk gate as gd_weighted_null_score, opposite use: there
    # it inflates a dip's reported severity where GD is bad (it's a real,
    # unfixable cancellation); here it shrinks the EQ target there (a filter
    # can't repair phase-domain cancellation by boosting/cutting magnitude).
    gd_authority = _excess_gd_authority(frequencies, excess_group_delay_ms)
    authority = range_authority * gd_authority
    desired *= authority
    effective_target = base_db + desired

    total = np.ones_like(spectrum, dtype=np.complex128)
    filters: list[dict[str, float]] = []
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
    # The shelf is deliberately applied *after* the bell-fitting loop above,
    # not folded into `desired`/`residual`: the bells there still target the
    # raw, unshelved response exactly as if the shelf were inactive, so a
    # deliberate tonal tilt is never fought/cancelled by the corrective
    # fitter as if it were a defect at the same frequencies. `filters`
    # itself stays a pure PK-bell list (the shelf isn't a peaking biquad and
    # has no fc/gain/q representation in that schema); callers reconstructing
    # a full EQ'd response from `filters` via `filters_response()` must also
    # multiply in `options.shelf.response(...)` at their own frequency grid
    # to stay consistent with the `total` returned here.
    total = total * options.shelf.response(frequencies, sample_rate)
    metadata: dict[str, Any] = {
        "target": options.target,
        "target_level_db": target_level,
        "correction_range_hz": correction_range,
        "correction_slope_db_per_octave": options.correction_slope_db_per_octave,
        "max_boost_db": options.max_boost_db,
        "effective_target_db": effective_target,
        "nominal_target_db": nominal_target,
        "eq_authority": authority,
        "shelf": options.shelf,
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


GD_BASELINE_MODES = ("flat", "monotonic")

# np.gradient(..., edge_order=2) uses a one-sided finite difference at the
# very first/last evaluated frequency instead of the centered difference
# used everywhere else, which can leave a single, sharply elevated
# |group_delay| sample right at the edge - see _monotonic_gd_baseline's
# call site in excess_group_delay for why this specifically breaks a
# non-increasing PAVA fit. Small and odd so the window is centered; not
# tied to native_resolution_hz because the artifact is a property of the
# differencing formula at the array boundary, not of measurement noise or
# capture length.
MONOTONIC_BASELINE_EDGE_DENOISE_BINS = 5


def _isotonic_non_increasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted least-squares fit that is non-increasing in index order (PAVA).

    The pool-adjacent-violators algorithm: walk the array left to right,
    keeping a stack of pools (weighted mean, weight, size). Whenever the new
    point's pool would score *higher* than the pool immediately to its left
    - a violation of "non-increasing" - merge the two pools into one at
    their combined weighted mean and re-check the pool now to their left,
    since merging can itself create a new violation further back. This is
    the standard exact solution to weighted isotonic regression, not a
    smoothing heuristic: for any non-increasing sequence it reproduces that
    sequence exactly, and it flattens (weighted-averages) any run that
    isn't, using no more merging than the data forces.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    block_sums: list[float] = []
    block_weights: list[float] = []
    block_sizes: list[int] = []
    for value, weight in zip(values, weights):
        block_sums.append(value * weight)
        block_weights.append(weight)
        block_sizes.append(1)
        while (
            len(block_sums) >= 2
            and block_sums[-2] / block_weights[-2] < block_sums[-1] / block_weights[-1]
        ):
            sum_b, weight_b, size_b = (
                block_sums.pop(),
                block_weights.pop(),
                block_sizes.pop(),
            )
            block_sums[-1] += sum_b
            block_weights[-1] += weight_b
            block_sizes[-1] += size_b
    result = np.empty(values.size, dtype=np.float64)
    index = 0
    for total, weight, size in zip(block_sums, block_weights, block_sizes):
        result[index : index + size] = total / weight
        index += size
    return result


def _monotonic_gd_baseline_from_gradient(
    group_delay: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """``_monotonic_gd_baseline``, pre-denoised against a np.gradient edge artifact.

    A non-increasing PAVA fit (see ``_isotonic_non_increasing``) only pools
    a point into its neighbours when a *later* point violates
    monotonicity; a point that happens to be the largest value remaining in
    the array never gets that chance and is adopted into the baseline
    completely unfiltered. That is almost always the very first
    (lowest-frequency) sample of a real ``group_delay`` curve, because
    ``np.gradient(..., edge_order=2)`` uses a one-sided finite difference at
    the array boundary instead of the centered difference used everywhere
    else, which can leave a single, sharply elevated sample right there -
    unrelated to measurement noise or capture length, so it is not covered
    by the native-resolution smoothing elsewhere in this module. A small
    median pre-filter - robust to a single outlier, unlike a moving average
    - with a boundary mode that mirrors real interior values rather than
    replicating the edge sample into its own window removes that specific
    artifact before the fit sees it, without smoothing away a genuine,
    broader low-frequency rise (``_isotonic_non_increasing`` still governs
    the actual shape).
    """
    denoised = ndimage.median_filter(
        np.asarray(group_delay, dtype=np.float64),
        size=MONOTONIC_BASELINE_EDGE_DENOISE_BINS,
        mode="mirror",
    )
    return _monotonic_gd_baseline(denoised, weights)


def _monotonic_gd_baseline(group_delay: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Signed group-delay baseline whose magnitude is non-increasing with frequency.

    Models the physically-expected case of a real, benign group-delay rise
    toward the bottom of the band (room/port/driver behaviour smoothly
    declining as frequency rises) as the *baseline* rather than as excess,
    while still catching a bump anywhere else: the non-increasing constraint
    on ``|group_delay|`` means the fit can only be elevated at a given
    frequency if it was already at least that elevated everywhere below it,
    so a rise that appears after a lower value earlier in the band cannot be
    absorbed into the baseline - it is structurally excess, regardless of
    how wide or gentle it is. Fit on magnitude (not the signed curve) so a
    genuine low-end rise is captured symmetrically whichever sign it has,
    matching every other excess-GD consumer's ``abs()`` treatment.
    """
    magnitude_envelope = _isotonic_non_increasing(np.abs(group_delay), weights)
    sign = np.where(group_delay >= 0.0, 1.0, -1.0)
    return sign * magnitude_envelope


def excess_group_delay(
    spectrum: np.ndarray,
    fft_frequencies: np.ndarray,
    evaluation_frequencies: np.ndarray,
    integration_range: tuple[float, float] | None = None,
    native_resolution_hz: float | None = None,
    ppo: int = 48,
    gd_baseline: str = "flat",
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return energy-weighted mean absolute excess GD, its de-offset curve, and the baseline removed.

    The minimum-phase transform and group-delay derivative use the complete
    supplied spectra and evaluation grid. When ``integration_range`` is set,
    only that frequency interval contributes to common-delay removal and the
    scalar score used for ranking.

    ``gd_baseline`` selects what is treated as the arbitrary reference that
    "excess" is measured from, over ``evaluation_frequencies`` in full
    (regardless of ``integration_range``, so a real sub-bass rise is judged
    against the actual sub-bass data even when the correction/score range is
    narrower):

    - ``"flat"`` (default): a single constant, this curve's weighted median
      within ``integration_range``. Removing it leaves frequency-dependent
      (excess) storage/decay relative to the arbitrary common time origin.
    - ``"monotonic"``: a per-point baseline via ``_monotonic_gd_baseline``,
      constrained to be non-increasing in magnitude as frequency rises. This
      treats a genuine, physically-expected group-delay rise toward the
      bottom of the band as normal rather than as excess, while still
      flagging a bump anywhere the non-increasing constraint cannot explain
      - by construction, not via a separate width heuristic. This is an
      explicit, opt-in acoustic assumption (real low-frequency non-minimum-
      phase structure looks like this), not a measurement-reliability
      correction; ``"flat"`` remains the default because it makes no such
      assumption.

    When ``native_resolution_hz`` is given (the cache's unpadded frequency
    resolution; see ``AnalysisContext.native_resolution_hz``), the returned
    curve is progressively smoothed, via ``gd_smoothing_octaves``, wherever
    the evaluation grid is finer than that sweep/capture can actually
    resolve - almost always the sub-bass, where a short sweep leaves few
    genuinely independent samples per octave and ordinary measurement noise
    otherwise shows up as large, sign-flipping group-delay swings around
    zero. Leaving this ``None`` (the default) reproduces the original,
    unsmoothed curve, e.g. for callers that supply hand-built curves rather
    than a real cache's resolution. This applies after baseline removal in
    either mode.

    Returns ``(score, curve_ms, baseline_ms)``: the scalar score, the
    de-offset (baseline-removed, optionally smoothed) curve in ms, and the
    baseline itself in ms - a full-length array in both modes (a constant
    broadcast in ``"flat"`` mode) so a caller can display or diff it the
    same way regardless of which mode produced it. ``curve_ms + baseline_ms``
    reconstructs the raw (pre-removal) group-delay curve.
    """
    if gd_baseline not in GD_BASELINE_MODES:
        raise ValueError(f"gd_baseline must be one of {GD_BASELINE_MODES}")
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
    if gd_baseline == "monotonic":
        # Fit over the full curve (see docstring), computed from the raw,
        # unsmoothed curve so the smoothing below cannot bias the baseline
        # (see _monotonic_gd_baseline_from_gradient for the one exception:
        # a targeted denoise of a specific np.gradient edge artifact).
        baseline = _monotonic_gd_baseline_from_gradient(group_delay, weights)
    else:
        # A constant group delay is the arbitrary common time origin. Removing
        # its weighted median (computed from the raw, unsmoothed curve, so the
        # smoothing below cannot bias the common-delay estimate itself) leaves
        # frequency-dependent (excess) storage/decay.
        baseline = np.full_like(
            group_delay, _weighted_median(group_delay[score_mask], weights[score_mask])
        )
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
        1000.0 * baseline,
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

    This is a lexicographic tie-break placed after ``excess_gd_tail_ms``, not
    a replacement for it: it exists to separate two placements whose smeared
    *area* looks equally clean but where one has a single sharp, denoised-real
    non-minimum-phase excursion the area-based tail metric alone would not
    weight any differently from several mild, spread-out ones.
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
    gd_baseline: str = "flat",
) -> dict[str, Any]:
    eq_options = eq_options or EqOptions(correction_range=context.band)
    grid_sum = context.sum_on_grid(first, second, polarity, delay_ms, gain_db)
    magnitude_db = db20(grid_sum)
    # The trend is smoothed over a margin-extended grid with real spectral
    # content beyond the reported band, then cropped back, so it is not
    # biased by edge-replicated ("nearest") smoothing at the band boundaries.
    trend_wide_sum = context.sum_on_trend_grid(first, second, polarity, delay_ms, gain_db)
    trend_db = broad_trend_db(db20(trend_wide_sum), context.ppo)[context.trend_slice]
    full_sum, full_frequencies = context.sum_full(
        first, second, polarity, delay_ms, gain_db
    )
    excess_score, excess_curve, excess_baseline_curve = excess_group_delay(
        full_sum,
        full_frequencies,
        context.frequencies,
        integration_range=eq_options.correction_range,
        native_resolution_hz=context.native_resolution_hz,
        ppo=context.ppo,
        gd_baseline=gd_baseline,
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
    # filters_response() only reconstructs the fitted PK bells; the shelf
    # (already folded into eq_grid by fit_eq_filters) must be multiplied in
    # again here to stay consistent, since eq_full/eq_trend_wide are
    # reconstructed from `filters` on different frequency grids, not derived
    # from eq_grid itself.
    eq_full = filters_response(
        full_frequencies, context.sample_rate, filters
    ) * eq_options.shelf.response(full_frequencies, context.sample_rate)
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
    eq_trend_wide = filters_response(
        context.trend_frequencies, context.sample_rate, filters
    ) * eq_options.shelf.response(context.trend_frequencies, context.sample_rate)
    post_trend_wide_sum = trend_wide_sum * eq_trend_wide
    post_trend_db = broad_trend_db(db20(post_trend_wide_sum), context.ppo)[context.trend_slice]
    post_excess_score, post_excess_curve, post_excess_baseline_curve = excess_group_delay(
        post_full,
        full_frequencies,
        context.frequencies,
        integration_range=eq_options.correction_range,
        native_resolution_hz=context.native_resolution_hz,
        ppo=context.ppo,
        gd_baseline=gd_baseline,
    )
    dsp_target = eq_options.target == "dsp"
    result: dict[str, Any] = {
        "null_score_db": gd_weighted_null_score(
            magnitude_db, trend_db, context.frequencies, excess_curve, dsp_target=dsp_target
        ),
        "magnitude_only_null_score_db": float(
            np.max(dip_below_trend_db(magnitude_db, trend_db, context.frequencies))
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
        # Self-referential (own-peak) defaults, overwritten by run_search's
        # search-wide second pass with the group-average-referenced version
        # actually used in reports; kept here so pair_diagnostics() alone
        # still returns complete, sensible diagnostics for callers that
        # don't go through run_search (a self-referential value is always a
        # real number - see low_end_extension_hz's docstring - so these are
        # never None).
        "low_end_extension_f3_hz": low_end_extension_hz(
            trend_db, context.frequencies, threshold_db=LOW_END_EXTENSION_F3_THRESHOLD_DB
        ),
        "low_end_extension_f6_hz": low_end_extension_hz(
            trend_db, context.frequencies, threshold_db=LOW_END_EXTENSION_F6_THRESHOLD_DB
        ),
        "post_eq_null_score_db": gd_weighted_null_score(
            post_magnitude_db,
            post_trend_db,
            context.frequencies,
            post_excess_curve,
            dsp_target=dsp_target,
        ),
        "post_eq_magnitude_only_null_score_db": float(
            np.max(dip_below_trend_db(post_magnitude_db, post_trend_db, context.frequencies))
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
        "post_eq_low_end_extension_f3_hz": low_end_extension_hz(
            post_trend_db, context.frequencies, threshold_db=LOW_END_EXTENSION_F3_THRESHOLD_DB
        ),
        "post_eq_low_end_extension_f6_hz": low_end_extension_hz(
            post_trend_db, context.frequencies, threshold_db=LOW_END_EXTENSION_F6_THRESHOLD_DB
        ),
        "filters": filters,
        "eq_target": eq_metadata["target"],
        "eq_target_level_db": float(eq_metadata["target_level_db"]),
        "eq_mean_authority": float(np.mean(eq_metadata["eq_authority"])),
        "eq_shelf": {
            "freq_hz": eq_options.shelf.freq_hz,
            "gain_db": eq_options.shelf.gain_db,
            "slope": eq_options.shelf.slope,
            "active": eq_options.shelf.active,
        },
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
                "excess_baseline_ms": excess_baseline_curve,
                "post_eq_db": post_magnitude_db,
                "post_eq_trend_db": post_trend_db,
                "post_eq_excess_curve_ms": post_excess_curve,
                "post_eq_excess_baseline_ms": post_excess_baseline_curve,
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
