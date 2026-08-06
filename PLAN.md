# Scoring / EQ improvement plan

Source: code review of `dsp.py` (scoring primitives, target-curve/EQ fitting)
and `engine.py` (search/ranking). All items from the original review are now
implemented (see git history), including the wide-dip detection blind spot:
`dip_below_trend_db` now also flags a dip via a two-sided check (best level
well to the left *and* well to the right of each point) so a dip much wider
than the one-octave trend's smoothing window is no longer absorbed into that
trend and under-scored, without flagging an ordinary monotonic rolloff as a
false positive (a rolloff never recovers on at least one side).

## Deferred

1. **Peaks above trend are never penalized.** Only dips (magnitude below the
   trend/baseline) count toward `null_score_db`; a reinforcement peak scores
   zero regardless of size. This is documented/intentional — a peak adds
   output rather than destructively cancelling it, and is generally less of
   a summing-position problem than a null — but is flagged here in case a
   symmetric (peak-aware) variant is wanted later.
