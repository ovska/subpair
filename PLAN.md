# Scoring / EQ improvement plan

Source: code review of `dsp.py` (scoring primitives, target-curve/EQ fitting)
and `engine.py` (search/ranking). All items from the original review are now
implemented (see git history), including:

- The wide-dip detection blind spot: `dip_below_trend_db` also flags a dip
  via a two-sided check (best level well to the left *and* well to the right
  of each point) so a dip much wider than the one-octave trend's smoothing
  window is no longer absorbed into that trend and under-scored, without
  flagging an ordinary monotonic rolloff as a false positive (a rolloff
  never recovers on at least one side).
- The peaks-are-never-penalized question: a magnitude peak above trend still
  scores zero on its own (reinforcement isn't a summing-position problem the
  way a null is), but `gd_weighted_null_score` now also penalizes a peak
  that only exists with real excess group delay — non-minimum-phase, i.e.
  not explained by the peak's own magnitude shape — since that's a
  resonance/ringing signature, not benign constructive reinforcement.
  Scored proportional to GD risk alone, so a minimum-phase peak still scores
  exactly zero.
- A new ranking criterion for overall smear that the energy-weighted excess-
  GD mean can miss: `excess_gd_tail_ms`/`post_eq_excess_gd_tail_ms`,
  `|excess GD|` integrated over log-frequency across the same range as the
  mean, unweighted by level. Inserted as a new tie-break level right after
  the mean in both the raw and EQ'd rankings (and in the finalist
  delay/gain/polarity tie-break), so a sum that's flat on magnitude but
  smeary somewhere quiet no longer slips through.
- Both the EQ-authority gate and that tail metric were originally built on
  a moving average / percentile, verified (numerically, not just in theory)
  to dilute or completely miss a narrow, severe excess-GD spike relative to
  a wider, shallower feature of the same peak height or area: e.g. a 1-bin,
  1-cycle spike left `_excess_gd_authority` at 0.88 (almost fully trusting
  EQ there) while a 3-bin spike of the same height correctly dropped it to
  0.09, and `excess_gd_tail_ms` reported exactly 0.0 for any feature
  narrower than its percentile's own width cutoff, however severe.
  `_excess_gd_authority` now uses a *maximum* filter (not an average) over
  the same minimum window, so peak height - not width - drives the gate;
  `excess_gd_tail_ms` now integrates `|excess GD|` over log-frequency
  (`np.trapezoid`, matching how the energy-weighted mean already integrates)
  instead of taking a percentile, so a narrow severe spike and a wider
  shallower bump of the same area score the same.

Nothing is currently deferred.
