# Scoring / EQ improvement plan

Source: code review of `dsp.py` (scoring primitives, target-curve/EQ fitting)
and `engine.py` (search/ranking). Items are ordered by priority. Each item is
removed from this file once implemented (see git history for the commit that
closed it).

## Now implementing

1. **EQ candidate peak-picking is noise-sensitive.**
   `fit_eq_filters` runs `scipy.signal.find_peaks` directly on the raw,
   unsmoothed per-bin `abs(residual)` curve, with cuts allowed up to Q=10.
   Single-bin measurement ripple can drive narrow high-Q cut filters that
   chase noise rather than real modal peaks. The half-max bandwidth walk
   (same function) has the same problem: a noisy wiggle near a true peak can
   halt the walk early, understating bandwidth (overstating Q). Fix: detect
   peaks and estimate bandwidth on a lightly smoothed copy of the residual,
   while still fitting gain against the true unsmoothed target.

2. **Edge bias in `broad_trend_db` near band boundaries.**
   The one-octave Gaussian trend used for both `null_scores` and the
   `trend` EQ target is computed with `mode="nearest"` directly on
   `context.frequencies`, which starts/ends exactly at the search band with
   no margin. Near the band edges this replicates the edge sample instead of
   using real out-of-band content, biasing the trend (and hence null score)
   right where users often care most (e.g. near 25 Hz). Fix: compute the
   trend over a slightly widened internal grid and crop back to the
   requested band.

3. **No sanity check on inter-measurement absolute time offsets.**
   `AnalysisContext.padded_spectra` applies a spectral delay derived from
   `start_time_seconds` before an FFT zero-padded 4x. If measurements don't
   actually share a loopback-derived time base (user error / REW
   misconfiguration), a large relative offset can silently alias/wrap
   instead of raising a clear error. Add a guard that raises `CacheError`
   when the spread of `start_time_seconds` exceeds a safe fraction of the
   padded analysis window.

4. **Dead code / documentation.**
   - `engine._best_configuration` (singular) is unused everywhere; delete it.
   - Document `--max-cut` in the README next to the existing `--max-boost`
     documentation.

## Deferred (design questions, not clear-cut bugs)

These change ranking/scoring philosophy that the README documents as
intentional ("no weighted blend", dip-only null scoring). Implementing them
as opt-in flags (default = current behavior) rather than changing defaults
outright, once the items above are done:

5. **Ranking is strictly lexicographic across pairs**, so `excess_gd_ms` and
   `raw_tail_ms` essentially never affect which *pair* ranks #1 (only which
   delay/gain/polarity setting of the *same* pair). Proposal: optional
   `--tie-tolerance-db` that groups near-equal primary scores before
   applying the secondary/tertiary sort keys. Default 0.0 (no behavior
   change) to preserve existing tests/semantics.

6. **No robustness/plateau preference in delay/gain optimum search.**
   The exact grid minimum of `null_scores` can be a razor's-edge optimum
   that's very sensitive to small real-world delay drift. Proposal: report a
   sensitivity/plateau-width diagnostic alongside the chosen configuration.

7. **`null_scores` is self-referential and dip-only.** Because the "trend"
   is a ~1-octave smoothing of the same measured curve, dips wider than
   roughly an octave are partly absorbed into the trend and under-scored,
   and peaks above trend are never penalized. This is documented/intentional
   for narrow comb-filtering nulls; flagged here in case a supplementary
   wide-window check is wanted later.
