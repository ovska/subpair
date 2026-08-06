# Scoring / EQ improvement plan

Source: code review of `dsp.py` (scoring primitives, target-curve/EQ fitting)
and `engine.py` (search/ranking). All items from the original review —
including the three deferred design-philosophy proposals (tie-tolerance
ranking, delay/gain plateau diagnostics, excess-GD-aware dip scoring) — are
now implemented (see git history). What remains is a narrower, still-open
question about the null-score metric.

## Deferred

1. **`null_scores`'s one-octave trend is self-referential and still misses
   wide dips / peaks.** Because the "trend" is a ~1-octave smoothing of the
   same measured curve, dips wider than roughly an octave are partly
   absorbed into the trend and under-scored, and peaks above trend are never
   penalized. Excess-GD weighting (now implemented) addresses the *severity*
   of a detected dip, not this detection blind spot — a wide, phase-benign
   shelf-like suck-out from path-length differences could still slip
   through. This is documented/intentional for narrow comb-filtering nulls;
   flagged here in case a supplementary wide-window or peak-aware check is
   wanted later.
