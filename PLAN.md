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

Nothing is currently deferred.
