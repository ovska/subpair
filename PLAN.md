# Scoring / EQ improvement plan

Source: code review of `dsp.py` (scoring primitives, target-curve/EQ fitting)
and `engine.py` (search/ranking). All clear-cut correctness/robustness
findings from that review have been implemented (see git history). What
remains are design-philosophy questions the README documents as intentional
("no weighted blend", dip-only null scoring) — left here as future,
opt-in-flag proposals rather than changes to current defaults.

## Deferred (design questions, not clear-cut bugs)

1. **Ranking is strictly lexicographic across pairs**, so `excess_gd_ms` and
   `raw_tail_ms` essentially never affect which *pair* ranks #1 (only which
   delay/gain/polarity setting of the *same* pair). Proposal: optional
   `--tie-tolerance-db` that groups near-equal primary scores before
   applying the secondary/tertiary sort keys. Default 0.0 (no behavior
   change) to preserve existing tests/semantics.

2. **No robustness/plateau preference in delay/gain optimum search.**
   The exact grid minimum of `null_scores` can be a razor's-edge optimum
   that's very sensitive to small real-world delay drift. Proposal: report a
   sensitivity/plateau-width diagnostic alongside the chosen configuration.

3. **`null_scores` is self-referential and dip-only.** Because the "trend"
   is a ~1-octave smoothing of the same measured curve, dips wider than
   roughly an octave are partly absorbed into the trend and under-scored,
   and peaks above trend are never penalized. This is documented/intentional
   for narrow comb-filtering nulls; flagged here in case a supplementary
   wide-window check is wanted later.
