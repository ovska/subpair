# Target-specific scoring architecture

## Summary

Replace the current shared scoring path with explicit, independently testable scoring
algorithms for the `trend`, `flat`, and `dsp` target types.

The correction curve and the ranking objective must become separate concepts. `flat`
and `dsp` may continue to fit the same flat PEQ curve, but they must not share a score
implementation merely because their fitted filters are identical. Each target policy
must control:

- the inexpensive score used to shortlist polarity/delay/gain configurations;
- the expensive score used to choose the finalist for each placement pair;
- the raw and post-EQ pair-ranking keys;
- the score components and explanations written to result JSON;
- the metric labels, columns, and explanations rendered in the report.

The implementation must stay deterministic and offline-first, and it must retain a
bounded, vectorised fast path so a normal search does not require minimum-phase
extraction for every point in the full delay/gain/polarity grid.

## Problem statement

The current implementation conflates three concerns:

1. `EqOptions.target` selects the correction curve used by the PEQ fitter.
2. A boolean `dsp_target` slightly changes one term inside the shared
   `gd_weighted_null_score()` function.
3. `engine.py` uses the same magnitude-only exhaustive search and the same ranking
   tuple for every target.

This has several consequences:

- `flat` and `dsp` intentionally generate the same filters, but can also accidentally
  generate identical scores because their shared peak term dominates the shared
  `max()` reduction.
- The target-specific intent is hidden inside a boolean instead of being represented
  as a complete scoring policy.
- The exhaustive search retains only exact minima from the shared magnitude score.
  A target-specific finalist cannot recover a slightly worse magnitude candidate that
  is much better under that target's actual objective.
- Result JSON records a single scalar `null_score_db`, so it is hard to explain why a
  target did or did not change a rank.
- The report hard-codes one set of metrics and prose even when different targets need
  different definitions or ordering.

The recent six-position cache demonstrates the failure mode: all 15 raw and all 15
post-EQ scores were dominated by the target-independent non-minimum-phase peak term,
so the `flat` and `dsp` reports were otherwise identical. This is valid under the
current formula, but it defeats the purpose of offering distinct target policies.

## Goals

- Give `trend`, `flat`, and `dsp` separate scoring implementations with explicit
  semantics and versioned identifiers.
- Allow each target to shortlist and select different polarity/delay/gain settings.
- Allow each target to rank the same pair differently before and after EQ.
- Keep score components separate until lexicographic ordering; do not hide unrelated
  failure modes inside one maximum unless that is explicitly part of a target policy.
- Make every rank explainable from serialized component values.
- Preserve deterministic output for fixed cache contents, settings, and code version.
- Bound runtime and memory growth relative to the current vectorised search.
- Preserve verification behavior and enough legacy fields for a controlled migration.
- Add synthetic tests that prove target policies differ for the situations they are
  designed to treat differently.

## Non-goals

- Do not change REW fetching, cache formats, impulse alignment, minimum-phase
  extraction, CSD calculation, or PEQ biquad simulation unless a scoring test exposes
  an independent defect in one of those primitives.
- Do not require the three targets to produce different results on every real dataset.
  Legitimate convergence remains possible and must be explainable from components.
- Do not make the internal shortlist size or scoring implementation configurable from
  the CLI in the first version.
- Do not replace lexicographic ranking with an undocumented weighted blend.
- Do not make network access part of search or report generation.

## Target semantics

Before implementation, encode the following definitions in tests and documentation.
Exact thresholds may be tuned against synthetic fixtures, but the ordering intent must
not change without updating the scoring algorithm version.

### `trend`: conservative placement and correction

Purpose: choose a naturally well-behaved sum that follows its own broad response and
requires conservative correction.

Proposed raw ordering:

1. destructive dip severity below the broad trend, increased where excess GD shows
   that the dip is non-minimum-phase;
2. non-minimum-phase resonance severity above the trend;
3. energy-weighted excess GD;
4. level-independent excess-GD tail;
5. decay tail and configuration robustness.

Proposed post-EQ ordering uses the same component definitions on the corrected
response. Magnitude-only peaks that are minimum-phase remain correctable and do not
outrank destructive dips.

The trend correction profile remains the broad-trend target. Existing cut/boost limits
continue to come from `EqOptions`.

### `flat`: best result from subpair's bounded PEQ fitter

Purpose: choose the pair that subpair's configured filters can bring closest to a flat
in-range response without spending correction on unfixable cancellations.

Proposed raw ordering:

1. bounded-correction feasibility: worst target deficit or excess that remains after
   applying the current boost/cut and excess-GD authority limits;
2. unfixable cancellation severity;
3. correction demand/headroom, including boost-limit and filter-count saturation;
4. excess-GD mean and tail;
5. decay tail and configuration robustness.

Proposed post-EQ ordering:

1. worst residual against the effective, range- and GD-aware flat target;
2. unfixable cancellation severity after EQ;
3. integrated/typical target error so a single narrow bin is not the only measure of
   fit quality;
4. excess-GD mean and tail;
5. decay tail and configuration robustness.

The flat scorer may use the fitted filter bank during finalist evaluation, but the fast
grid stage needs a vectorised feasibility proxy rather than fitting filters for every
candidate.

### `dsp`: placement for correction by an external DSP

Purpose: choose acoustically correctable placement geometry, assuming a capable
external minimum-phase DSP will handle ordinary amplitude ripple later.

Proposed raw ordering:

1. unfixable cancellation severity, defined from dip depth and excess-GD risk with
   minimum-phase dip depth removed or heavily discounted;
2. level-independent excess-GD tail;
3. energy-weighted excess GD;
4. non-minimum-phase resonance severity as its own component, rather than taking a
   shared maximum with the cancellation component;
5. delay/gain plateau width and decay tail;
6. magnitude-only correction demand as a late tie-breaker.

Proposed post-EQ ordering applies the same correctability objective to the simulated
response. The report should make clear that the suggested PEQs are illustrative: the
ranking assumes later correction by an external DSP.

The DSP correction profile remains the same flat curve as `flat` unless a future CLI
option explicitly introduces a different house curve. Sharing a correction profile
must not imply sharing a scorer.

## Architecture

### Separate target policy from correction profile

Introduce an internal `TargetPolicy` selected by the existing `--eq-target` value:

```text
CLI target    correction profile    scoring algorithm
trend         trend-v1              trend-v1
flat          flat-v1               flat-v1
dsp           flat-v1               dsp-v1
```

`TargetPolicy` should own or reference:

- `name`: stable CLI/result value;
- `scoring_algorithm`: versioned identifier such as `dsp-v1`;
- `correction_profile`: versioned PEQ-target identifier;
- a vectorised fast scorer;
- a finalist scorer;
- raw and post-EQ `ScoreSpec` definitions;
- report labels, units, ordering directions, and human-readable descriptions.

Keep the registry explicit, for example `TARGET_POLICIES = {"trend": ..., ...}`.
Unknown values must fail at CLI parsing and again at the policy boundary for callers
that bypass the CLI.

### Add a scoring module

Create `src/subpair/scoring.py` for policy and score composition. Keep signal-processing
primitives such as trend extraction, excess GD, authority curves, and PEQ response in
`dsp.py`. This avoids adding more target branches to an already large signal-processing
module.

Suggested types:

- `CandidateFeatures`: vectorised magnitude/trend features available for every fast
  grid candidate;
- `DiagnosticFeatures`: arrays and scalar diagnostics computed for a shortlisted
  finalist, before any target-specific reduction;
- `ScoreComponentSpec`: stable key, label, unit, direction, tolerance behavior, and
  description;
- `TargetScore`: ordered component values plus optional display-only diagnostics;
- `ScoreSpec`: component order and deterministic lexicographic key generation;
- `TargetPolicy`: fast scoring, shortlist selection, raw scoring, post-EQ scoring, and
  correction-profile selection.

Use concrete classes or immutable callables for `TrendScorer`, `FlatScorer`, and
`DspScorer`. Do not add another `dsp_target: bool` argument to shared functions.

### Compute neutral diagnostics before target reduction

Refactor `pair_diagnostics()` so it returns neutral facts rather than a target-reduced
`null_score_db`:

- raw/post magnitude and broad trend;
- raw/post dip-depth curves;
- raw/post peak-height curves;
- raw/post excess-GD curves and authority/risk curves;
- raw/post EQ target and residual curves;
- fitted filters and filter-budget usage;
- boost/cut headroom and uncorrected target deficit/excess;
- excess-GD mean/tail and decay metrics;
- SPL and plateau diagnostics.

The selected `TargetPolicy` then reduces those facts into named score components.
Expensive arrays should remain in memory only for shortlisted configurations and
report-selected pairs; serialized pair rows should contain compact scalar components.

## Search pipeline changes

### 1. Vectorised fast scoring

Change `_best_configurations()` to accept a target policy and return a deterministic
shortlist, not only exact minima from the shared magnitude score.

Each policy supplies a fast score or tuple derived from the already-vectorised complex
candidate sums. The first implementation should avoid minimum-phase extraction inside
the full grid.

- `trend`: broad-trend dip proxy, close to the existing fast score.
- `flat`: flat-target correction-feasibility proxy, including obvious boost/cut
  overflow where it can be estimated cheaply.
- `dsp`: a deliberately broader amplitude shortlist, because ordinary magnitude error
  is secondary. Preserve candidates across polarity and multiple score bands so the
  later excess-GD evaluation can choose a slightly worse-looking magnitude response.

Always include the current magnitude-only optimum as a compatibility/reference
candidate.

### 2. Deterministic shortlist strategy

Choose the internal shortlist bound from benchmarks rather than intuition. Evaluate at
least `K = 1, 4, 8, 16, 32` against an exhaustive oracle on reduced synthetic grids.
Select the smallest `K` that consistently recovers the oracle winner for every target.

The shortlist must:

- have a stable sort order and explicit tie-breakers;
- avoid dependence on hash/dictionary iteration order;
- include diversity across polarity and nearby delay/gain plateaus;
- avoid duplicate configurations;
- expose shortlist size and fast-score values in debug/test diagnostics;
- keep peak memory bounded by processing one placement pair at a time.

If no practical `K` recovers DSP winners, add a cheap phase-smoothness proxy or a
two-pass coarse-to-fine search rather than silently reverting to the magnitude-only
winner.

### 3. Expensive finalist evaluation

For each shortlisted configuration:

1. compute the neutral raw diagnostics;
2. fit the correction profile selected by the target policy;
3. compute neutral post-EQ diagnostics;
4. ask the policy for raw and post-EQ `TargetScore` objects;
5. select the per-pair configuration using the policy's raw finalist key;
6. retain both score objects for pair ranking and reporting.

Document whether configuration selection is based on the raw score, post-EQ score, or
a target-specific selection score. The proposed default is:

- `trend`: raw score, favoring naturally robust placement;
- `flat`: post-EQ score, because the target explicitly asks what the bounded fitter can
  achieve;
- `dsp`: raw correctability score, because the external DSP is not the simulated PEQ
  bank.

This choice must be locked down by tests; it must not be an incidental shared behavior.

### 4. Pair ranking

Replace hard-coded `null_score_db` sort tuples in `engine.py` with `ScoreSpec.sort_key()`.
Raw and EQ'd tables may use different specs. Apply `--tie-tolerance-db` only to score
components declared compatible with a dB tolerance; do not quantize millisecond or
dimensionless components with a dB value.

Continue to add deterministic placement-index tie-breakers after every policy-defined
component is exhausted.

## Result schema and compatibility

Bump `format_version` and serialize the policy explicitly. A proposed structure is:

```json
{
  "settings": {
    "target_policy": {
      "name": "dsp",
      "scoring_algorithm": "dsp-v1",
      "correction_profile": "flat-v1",
      "raw_components": ["unfixable_cancellation_db", "excess_gd_tail_ms"],
      "eq_components": ["post_unfixable_cancellation_db", "post_excess_gd_tail_ms"]
    }
  },
  "pairs": [
    {
      "scores": {
        "raw": {
          "unfixable_cancellation_db": 2.1,
          "excess_gd_tail_ms": 4.3
        },
        "eq": {
          "post_unfixable_cancellation_db": 1.8,
          "post_excess_gd_tail_ms": 4.2
        }
      }
    }
  ]
}
```

Migration rules:

- retain `rank`, `eq_rank`, pair configuration, filters, SPL, and verification fields;
- retain legacy scalar score fields for one format version if doing so keeps external
  consumers working, but mark them as compatibility aliases in settings metadata;
- make `html_report.load_results()` reject unsupported old scoring formats with a
  precise instruction to rerun `subpair search`;
- ensure `verify` reads pair configuration and filters without depending on a specific
  target's score component names;
- never silently interpret an old shared `null_score_db` as a new target-specific
  component.

## Report changes

Make ranking tables data-driven from the serialized `ScoreSpec` metadata.

- Replace the hard-coded `Null dB` column with the target's primary component label.
- Add columns for the target's important secondary components while keeping the table
  compact.
- Show the scoring-algorithm and correction-profile identifiers near the report title.
- In pair details, show a component breakdown for raw and post-EQ scores.
- Explain which components are correctable, uncorrectable, or informational for the
  active target.
- Keep magnitude, excess-GD, CSD, and PEQ plots shared where their underlying data is
  genuinely shared.
- State explicitly when two targets fit identical filters but rank them differently.
- If two target runs converge, the component breakdown must make the reason visible
  rather than leaving only an identical scalar.
- Continue embedding all data and Plotly locally; do not introduce a CDN dependency.

Metric coloring and client-side sorting must read direction and unit metadata rather
than assuming smaller `null_score_db` is always the primary objective.

## CLI and documentation

Keep `--eq-target {trend,flat,dsp}` for compatibility, but update its help text to say
that the option selects both a target policy and a correction profile.

Update `README.md` with:

- a concise purpose statement for each target;
- the raw and EQ'd lexicographic component order for each target;
- the distinction between correction profile and scoring algorithm;
- the finalist shortlist behavior and performance rationale;
- a warning that real datasets can legitimately converge;
- the scoring-algorithm identifiers stored in result JSON.

Do not expose formula constants as CLI flags until their semantics and stability are
proven. Version the algorithm instead.

## Implementation phases

### Phase 0: freeze behavior and construct fixtures

- [ ] Capture the current shared scorer's outputs for focused synthetic cases.
- [ ] Add fixtures for minimum-phase dips/peaks, non-minimum-phase cancellations and
      resonances, broad shelves, monotonic rolloff, correction-limit saturation,
      filter-count saturation, and clean excess-GD tails without large magnitude error.
- [ ] Add a synthetic multi-placement cache where `trend`, `flat`, and `dsp` have known
      and intentionally different winners.
- [ ] Add a reduced-grid exhaustive reference search used only by tests.
- [ ] Record current runtime and peak memory for a six-position, 201-delay, 13-gain
      search as a performance baseline.

### Phase 1: introduce policies without changing output

- [ ] Add `scoring.py`, policy types, component specs, and the explicit registry.
- [ ] Implement a `LegacyScorer` adapter that reproduces the current results.
- [ ] Resolve the CLI target to a policy once, near `run_search()` entry.
- [ ] Remove target-name conditionals from orchestration code where policy dispatch can
      replace them.
- [ ] Prove the adapter produces byte-identical result metrics for existing synthetic
      tests.

### Phase 2: neutral diagnostic extraction

- [ ] Split array/scalar feature calculation from target-specific score reduction.
- [ ] Remove `dsp_target: bool` from `gd_weighted_null_score()` or retire that function
      after equivalent primitives are covered by tests.
- [ ] Give dip, resonance, correction residual, excess-GD, and headroom components
      independent unit tests.
- [ ] Confirm neutral diagnostics do not depend on the selected target when the
      correction profile is the same.

### Phase 3: target-specific finalist scorers

- [ ] Implement and version `TrendScorer`.
- [ ] Implement and version `FlatScorer`.
- [ ] Implement and version `DspScorer`.
- [ ] Encode raw, post-EQ, and configuration-selection keys independently.
- [ ] Add component-level debug output usable from tests without polluting normal CLI
      output.
- [ ] Verify the constructed multi-placement fixture yields the expected different
      winners and ranks.

### Phase 4: target-specific shortlisting

- [ ] Refactor `_best_configurations()` to accept vectorised policy scoring.
- [ ] Implement deterministic top-K/diversity shortlists.
- [ ] Compare every target's shortlisted winner with the exhaustive reduced-grid
      oracle.
- [ ] Tune `K` and document the accuracy/runtime tradeoff.
- [ ] Add a regression case where DSP deliberately selects a slightly worse
      magnitude-only candidate with substantially better unfixable behavior.

### Phase 5: schema, report, and verification migration

- [ ] Bump result format and serialize policy/component metadata.
- [ ] Preserve or deliberately retire legacy score aliases.
- [ ] Convert report tables and metric coloring to serialized component specs.
- [ ] Add pair-level component explanations.
- [ ] Decouple verification from hard-coded score field names.
- [ ] Test clear failure messages for old result files.

### Phase 6: validation and documentation

- [ ] Run the full unit suite and deterministic repeated-search tests.
- [ ] Generate reports for all three targets from the same synthetic cache.
- [ ] Re-run against the local six-position cache without checking its private data into
      the repository.
- [ ] Inspect whether any convergence is supported by matching component keys rather
      than caused by lost target dispatch.
- [ ] Benchmark runtime and memory against Phase 0.
- [ ] Update README, CLI help, and result-schema explanations.

## Test matrix

### Scorer unit tests

For every target, cover raw and post-EQ forms of:

- smooth broad trend and monotonic rolloff;
- shallow and deep minimum-phase dips;
- destructive non-minimum-phase dips at low and high SPL;
- minimum-phase peaks correctable by cuts;
- non-minimum-phase resonant peaks;
- narrow and broad excess-GD features with equal area;
- correction demand just below and just above boost/cut limits;
- fewer and more modal features than the filter budget;
- response features at correction-range boundaries;
- exact ties and differences just inside/outside tie tolerance.

Required cross-target assertions:

- `trend` values natural broad-trend behavior over nominal flatness.
- `flat` prefers the response with the best bounded post-EQ residual.
- `dsp` discounts a correctable minimum-phase amplitude problem.
- `dsp` does not discount a non-minimum-phase cancellation.
- A target-independent resonance component cannot erase DSP-specific cancellation
  ordering merely by being combined through a shared maximum.
- `flat` and `dsp` can produce identical filters while producing different component
  scores and ranks.
- Identical ranks are allowed when every ordered component legitimately ties.

### Search integration tests

- Each target can select a different configuration for the same placement pair.
- Shortlist results match the exhaustive oracle on reduced grids.
- Repeated runs produce byte-identical JSON.
- Pair ordering remains stable across equal NumPy values and exact ties.
- `--tie-tolerance-db 0` remains strictly lexicographic.
- Nonzero tolerance affects only declared dB components.
- `--eq-bands 0`, boost/cut extremes, and correction-range edges remain valid.

### Report and schema tests

- Target, algorithm version, and correction profile are correct in embedded JSON.
- Table headers and metric directions match the selected policy.
- Client-side sorting agrees with server-side ranks.
- Pair details contain every ranked score component.
- Old formats are rejected or migrated according to the documented rule.
- The report remains self-contained and deterministic.

## Performance and determinism requirements

- Establish a measured budget after Phase 0; initial goal is no more than 2x current
  wall time and no more than 1.5x peak memory on the six-position baseline.
- Preserve pair-at-a-time vectorisation and avoid retaining full candidate arrays after
  a pair is complete.
- Never perform minimum-phase extraction or PEQ fitting across the complete exhaustive
  grid unless benchmarks prove it practical.
- Use stable sorts and explicit final tie-breakers everywhere.
- Round only for display/serialization compatibility; rank using full-precision values.
- Store scoring-algorithm versions so formula changes cannot masquerade as identical
  settings.

## Risks and decisions to resolve

1. **Shortlist miss risk:** a DSP-optimal configuration may look mediocre on every
   magnitude proxy. Resolve with exhaustive reduced-grid tests, shortlist diversity,
   and possibly a coarse phase proxy.
2. **Metric proliferation:** dynamic score components can make reports unreadable.
   Limit tables to ranked components and put diagnostic-only values in pair details.
3. **Target-level normalization:** flat target level currently varies per pair. Decide
   whether comparing residuals requires a common level convention; document and test
   the choice.
4. **Raw versus post-EQ configuration selection:** especially for `flat`, choosing by
   post-EQ outcome may change the meaning of the raw table. Make the policy explicit in
   metadata and report prose.
5. **Peak semantics:** decide separately for each policy whether a non-minimum-phase
   peak is primary, secondary, or purely diagnostic. Do not reuse one peak penalty by
   convenience.
6. **External compatibility:** unknown consumers may read `null_score_db`. Retain a
   versioned alias for one release or make the format bump and error message prominent.
7. **Legitimate convergence:** tests must prove dispatch and component differences on
   controlled fixtures, not assert that all real reports must differ.

## Acceptance criteria

The work is complete when:

- each CLI target resolves to a distinct versioned scorer;
- correction profile and scorer are separate in code and result metadata;
- no target-specific boolean remains inside the shared scalar null-score function;
- each target controls fast shortlisting, finalist selection, raw ranking, and post-EQ
  ranking;
- reduced-grid shortlist searches recover exhaustive target-specific winners;
- synthetic fixtures demonstrate intentionally different target winners;
- score components and lexicographic order are serialized and visible in reports;
- flat and DSP reports can explain both divergence and legitimate convergence;
- verification works without hard-coded target score names;
- repeated searches remain deterministic;
- the full test suite passes within the agreed runtime and memory budget;
- README and CLI help accurately describe the implemented semantics.

## Expected file changes

- `src/subpair/scoring.py`: new policy registry, target scorers, score/component types.
- `src/subpair/dsp.py`: neutral signal diagnostics and correction profiles; remove
  target-specific score reduction.
- `src/subpair/engine.py`: policy-driven shortlist, finalist selection, and ranking.
- `src/subpair/cli.py`: policy resolution/help text and unchanged public target names.
- `src/subpair/html_report.py`: dynamic score columns and component explanations.
- `src/subpair/verification.py`: consume target-neutral pair/configuration fields.
- `tests/test_pipeline.py`: scorer and search regression coverage.
- Additional focused test modules/fixtures if `test_pipeline.py` becomes too large.
- `README.md`: target semantics, versioning, result fields, and search behavior.

---

# Low-frequency excess-group-delay reliability and an independent low-shelf tool

**Status: implemented, with a post-review documentation correction.** Both
parts landed as designed below, including the Section B.1 scope decision
(report/verify-time only, never reaching `search` or any ranking key).
`format_version` is now `5`. See
`dsp.native_resolution_hz`/`gd_smoothing_octaves`/`_smooth_by_variable_octaves`/
`ShelfOptions`/`low_shelf_response`, `engine.py`'s `native_resolution` settings
block, and the `--low-shelf-*` flags on `report`/`verify`. Phase checklists
below are left in place as a record of what was done.

**Section B.1 update: the report/verify-only scoping was later reversed by
explicit user request** (`format_version` bumped again, to `12`). The shelf
is now a `search`-time `EqOptions.shelf` field, folded into every post-EQ
score exactly like `max_boost_db`/`max_filters`, and `--low-shelf-*` moved
from `report`/`verify` to `search`. The alternative flagged at the end of
the original B.1 write-up below ("a `SearchOptions`/`EqOptions` field
serialized into settings and picked up by `report`/`verify` automatically")
is what was actually built, not the report/verify-overlay design the rest
of this section describes. The original rationale (a tonal preference
should not silently change which placement wins) is preserved in spirit by
keeping `fit_eq_filters`'s greedy PK-bell loop completely unaware of the
shelf - the bells still target the raw, unshelved response, so the shelf
does not get "corrected away"; it is applied once, at the end, exactly like
a fixed hardware EQ would be. It is a deliberate choice that the shelf *can*
now change which placement wins, since the whole point of moving it into
scoring was to let it. The rest of Section B.1 is left as a historical
record of the reasoning behind the design that was later superseded, not as
a description of current behaviour.

Code review after the initial implementation found the docs overclaimed that
genuine low-frequency GD features are "unaffected" by the smoothing (Risk #3
below); they were corrected to say the smoothing *preferentially preserves*
broad features over narrow noise, which is what the algorithm and tests
actually support, and the smoothing ladder's ceiling was tightened from 8 to
4 octaves.

## Summary

Two related, independently shippable pieces of work to make sub-bass behavior more
trustworthy and more controllable:

- **Part A** makes `excess_group_delay()` aware of the *native* frequency resolution
  implied by the cached impulse response's length (i.e. the sweep/capture duration),
  and progressively widens smoothing of the excess-GD curve as evaluated frequency
  approaches that resolution limit, so near-zero measurement noise in the sub-bass
  stops masquerading as real excess group delay in scores, authority gating, and
  plots.
- **Part B** adds a broad low-shelf filter as a fixed, user-specified tonal control,
  separate from the greedy PEQ bell fitter (`fit_eq_filters`) and excluded from
  ranking, for people who want a general "more/less sub-bass" tilt rather than
  (or in addition to) corrective EQ.

Both parts touch `dsp.py` primitives; Part A also touches `engine.py`'s result schema,
Part B only touches `report`/`verify` and does not touch `search` or ranking at all.
They can be implemented and shipped independently of each other and independently of
the target-specific scoring plan above (that plan reduces neutral diagnostics into
target-specific scores; this plan improves one of those diagnostics upstream of any
target, and adds a control that deliberately sits outside the scored diagnostics).

## Problem statement

### Part A: sweep length sets an absolute-Hz resolution that becomes huge in octave terms near DC

`AnalysisContext` builds its evaluation grid with `log_frequency_grid()`, a constant
points-per-octave (`ppo`, default 48) grid. `excess_group_delay()` computes the
minimum-phase excess phase from the *zero-padded* full-band spectrum
(`AnalysisContext.sum_full`, padded to `next_fast_len(length * minphase_pad_factor)`),
interpolates it onto that log grid, and differentiates it with `np.gradient(phase,
omega, edge_order=2)`.

Zero-padding does not add real spectral information; it only presents a smooth,
exact sinc interpolation of the frequency-domain content that a sweep of the
*original, unpadded* length (`context.length` samples at `context.sample_rate`, i.e.
capture duration `T = length / sample_rate`) actually contains. That capture sets a
native resolution `Δf_native = sample_rate / length` that is constant in Hz,
regardless of `ppo` or the padding factor. The number of native-resolution samples
packed into one octave around frequency `f` is proportional to `f / Δf_native`, which
shrinks linearly as `f` falls. A sweep long enough to give excellent, densely-sampled
resolution at 200 Hz can have only a handful of genuinely independent samples per
octave below 30 Hz.

Any real-world artifact with a characteristic width of roughly one native bin —
mic/preamp noise, HVAC or traffic rumble picked up in the tail, nonlinear distortion
residue, or ordinary DFT leakage from a decaying IR that has not fully settled within
`length` samples — therefore looks, on the finely-sampled log-frequency evaluation
grid, like a small number of nearly-uncorrelated phase perturbations spread across a
large fraction of an octave near DC. `np.gradient` divides those phase differences by
the (very small, in absolute Hz/rad·s⁻¹) step between adjacent log-grid points, so the
same absolute phase jitter that is invisible at 100 Hz turns into large, sign-flipping
group-delay swings — "squiggles hovering around ±0" — below roughly a few multiples of
`Δf_native`. Two consequences already observed in practice, both driven by the same
root cause:

- `excess_gd_ms`/`post_eq_excess_gd_ms` (an *energy-weighted* mean, tie-break #2) and
  especially `excess_gd_tail_ms`/`post_eq_excess_gd_tail_ms` (a deliberately
  *level-independent* integral, tie-break #3 — see `README.md`) can swing
  meaningfully between two searches that differ only in how long the REW sweep/capture
  was, with no change to the actual acoustic placement.
- `_excess_gd_authority()` (the same curve, used to throttle PEQ correction authority
  and to inflate `gd_weighted_null_score()`'s dip/peak severity) can gate away
  legitimate correction authority — or inflate a null score — in the sub-bass based on
  resolution noise rather than a genuine non-minimum-phase problem.

`_excess_gd_authority()` already applies a *fixed*-width (in log-frequency bins, hence
constant in octaves) light Gaussian denoise before its maximum-filter gate. A
fixed-octave-width smoothing is the right idea in principle — this codebase already
uses that pattern for `broad_trend_db()` and the `_excess_gd_authority` gate itself —
but it does not know that a fixed octave width corresponds to far fewer *real*
samples near DC than it does an octave higher up, so it under-smooths exactly where
smoothing is needed most.

### Part B: there is no dedicated tonality control

`fit_eq_filters()` only ever proposes RBJ constant-Q *peaking* (bell) biquads
(`peq_response()`), constrained to correct measured deviation from a target curve
within `--max-boost`/`--max-cut`/`--eq-range`/excess-GD authority. There is no shelf
biquad in `dsp.py`, and no way to ask for "more sub-bass" or "less sub-bass" as a
deliberate, broad tonal preference independent of what the corrective fitter would
choose. A user who wants a house-curve-style low shelf currently has to apply it
outside subpair entirely, losing the ability to see it in the report or fold it into
`verify`.

## Goals

- Make `excess_gd_ms`, `post_eq_excess_gd_ms`, `excess_gd_tail_ms`,
  `post_eq_excess_gd_tail_ms`, the excess-GD authority gate, and the report's excess-GD
  plot materially more stable across sweep-length/capture-duration changes that do not
  reflect a real change in the measured system, especially below a few multiples of
  the cache's native frequency resolution.
- Preserve full sensitivity to genuine, resolution-supported low-frequency excess-GD
  features (a real port/passive-radiator resonance, a real destructive null with
  measurable group-delay signature) — this is smoothing away *noise*, not lowering
  detection sensitivity for real problems.
- Surface the native resolution and the reliability threshold it implies, so users can
  see when their sweep is marginal for the band they asked to analyze.
- Add a broad low-shelf boost/cut as a fixed, explicitly-configured tonal filter,
  independent of `fit_eq_filters()`'s bell bank and of `--eq-bands`/`--max-boost`/
  `--max-cut`/`--eq-range` budgets, and excluded from every ranking key.
- Keep both changes deterministic, offline-first, and consistent with the existing
  "no weighted blend, strict lexicographic ranking" principle.

## Non-goals

- Do not change minimum-phase extraction itself (`minimum_phase_log_spectrum`,
  `minphase_pad_factor`, the cepstral method, or the 160 dB magnitude floor). Zero
  padding for *interpolation smoothness* is unrelated to the *native resolution* limit
  this plan addresses, and remains as-is.
- Do not change magnitude-based dip/peak detection (`dip_below_trend_db`,
  `null_scores`, `broad_trend_db`). Those already compare raw magnitude against an
  octave-smoothed trend rather than differentiating it, which is a fundamentally
  different (and already reasonably robust) design; this plan is scoped to the
  group-delay *derivative* path specifically, per the reported symptom.
- Do not make `fit_eq_filters()` aware of, or compensate for, the low shelf. The two
  stay independent by construction (Part B design decision below).
- Do not change `search`'s CLI surface, `SearchOptions`, or `engine.py` ranking keys
  for Part B; the shelf never participates in `null_score_db`/`excess_gd_ms`/tail
  scoring or pair ordering.
- Do not add resampling or any reinterpretation of measurement data; the fix is purely
  about how much to trust/smooth already-correct spectral estimates, not about
  changing what was measured.

## Part A: resolution-aware group-delay smoothing

### A.1 Native frequency resolution as a first-class quantity

Add `native_resolution_hz = sample_rate / length` to `AnalysisContext.__post_init__`
(computed from the *unpadded* `self.length`, not `self._padded_fft_length`), stored
alongside the existing `sample_rate`/`length` attributes. This is the one new fact the
rest of Part A derives from.

### A.2 A reliability-driven smoothing width

Add a small helper, e.g.:

```python
def gd_smoothing_octaves(
    frequencies: np.ndarray,
    native_resolution_hz: float,
    min_native_bins: float = MIN_RELIABLE_NATIVE_BINS,
) -> np.ndarray:
    """Half-width, in octaves, needed so ~min_native_bins native bins fall inside it."""
    return np.log2(1.0 + min_native_bins * native_resolution_hz / np.asarray(frequencies))
```

`MIN_RELIABLE_NATIVE_BINS` (proposed default 6, tune against Phase A0 fixtures) is the
number of independent native-resolution samples a local average needs to meaningfully
suppress zero-mean noise. This intentionally mirrors `_excess_gd_authority`'s existing
"fixed width in octaves" pattern, but the width itself now depends on position: it is
negligible once `f >> native_resolution_hz` (long sweep, or well above the sub-bass)
and grows without bound as `f → 0` or as the sweep gets shorter.

### A.3 Variable-width smoothing implementation

`scipy.ndimage.gaussian_filter1d`/`maximum_filter1d` only support a single fixed
`sigma`/`size` per call, so a genuinely per-point-variable-width smoother needs new
code. Two implementation strategies are acceptable; pick one after a Phase A0
brute-force comparison:

1. **Sigma-ladder blend (recommended starting point):** precompute the curve smoothed
   at a small fixed ladder of octave widths (e.g. `0, 0.25, 0.5, 1, 2, 4, 8` octaves via
   repeated `gaussian_filter1d` calls on the *log-frequency-indexed* array, which is
   already uniformly spaced so a bin-count sigma is exact), then for each grid point
   linearly blend between the two ladder rungs bracketing that point's
   `gd_smoothing_octaves()` value. Fully vectorized, deterministic, `O(K·N)` for a
   small constant `K`.
2. **Exact variable-width boxcar via prefix sums:** since the evaluation grid is
   sorted, a per-point window `[f / 2**w(f), f * 2**w(f)]` can be resolved with
   `np.searchsorted` on the log-frequency axis and a cumulative-sum array in `O(N log
   N)`. Simpler to prove exactly correct, but a boxcar's harder edge is more likely to
   show up as visible plot artifacts than the Gaussian blend.

Whichever is chosen, write it as a private, directly testable function (this
codebase's convention — see `_denoised_residual`, `_excess_gd_authority` — is to keep
such helpers private but import them by name in `tests/test_pipeline.py`), and add a
slow, obviously-correct reference implementation (an explicit per-index Python loop
building a local Gaussian/boxcar window from `gd_smoothing_octaves()`) used only in
tests to check the vectorized version point-by-point on small synthetic arrays.

### A.4 Where it plugs into `excess_group_delay()`

Inside `excess_group_delay()`:

1. Compute the raw `group_delay` via `np.gradient` exactly as today.
2. Remove the energy-weighted median common delay from the *raw* curve, over
   `score_mask`, exactly as today — the common-delay estimate should stay maximally
   robust and should not be biased by smoothing choices made afterward.
3. Apply the new adaptive smoothing to the resulting zero-centered curve, over the
   *entire* evaluation grid (not just `score_mask`/`integration_range` — the smoothed
   curve is also what gets returned as `excess_curve_ms` for full-band consumers like
   `_excess_gd_authority` and the report plot).
4. Use the smoothed curve for both the returned curve and the energy-weighted scalar
   integral (`excess_score`), so every downstream consumer —
   `excess_gd_ms`/`post_eq_excess_gd_ms`, `excess_gd_tail_ms`, `_excess_gd_authority`
   (and therefore `fit_eq_filters`'s target shrinkage and
   `gd_weighted_null_score`'s dip/peak inflation) and the report's excess-GD trace —
   sees one consistent, resolution-aware curve without duplicating the smoothing logic
   at each call site.

`_excess_gd_authority()`'s own fixed-width denoise-then-maximum-filter-then-smooth
pipeline is left unchanged; it now simply operates on an already-denoised curve. Its
existing behavior at frequencies well above the native-resolution limit (i.e. every
case the current tests cover, since existing synthetic fixtures use short, clean
impulses with plenty of native resolution across the tested band) must stay
byte-identical — that is the key regression guarantee for this change, and should be
asserted directly in Phase A1.

### A.5 Result schema, settings metadata, and report visibility

`excess_gd_ms`, `post_eq_excess_gd_ms`, `excess_gd_tail_ms`, and
`post_eq_excess_gd_tail_ms` are ranking-relevant serialized fields
(`engine.py`'s `pairs` rows), so this is a scoring-affecting change and must bump
`result["format_version"]` (currently `4`). Update `html_report.py`'s `format_version
< 4` guard to the new minimum and its error message.

Add a `native_resolution` block next to the existing `minimum_phase`/`ranking`
metadata in `run_search()`'s serialized `settings`, e.g.:

```json
"native_resolution": {
  "hz": 0.73,
  "min_reliable_native_bins": 6,
  "note": "excess-GD curves are progressively smoothed below roughly N x this value; wider octave smoothing near DC compensates for how few independent frequency samples a sweep of this length provides there"
}
```

Consider (Phase A4, optional polish) a non-fatal CLI warning from `fetch`/`search` when
`native_resolution_hz` is a large fraction of `band[0]` (e.g. fewer than
`MIN_RELIABLE_NATIVE_BINS` native bins exist below the requested lower band edge at
all), pointing the user at a longer sweep/capture. This is advisory only; it must not
block `search` from running.

## Part B: an independent low-shelf tonality tool

### B.1 Scope decision: report/verify-time overlay, not a search-time or scored option

**Superseded — see the note at the top of this section.** This scoping was later
reversed by explicit user request: the shelf is now a `search`-time, scored
`EqOptions` field. The reasoning below explains why the *original* implementation
chose the opposite, and is kept for historical context, not as current behaviour.

**Design decision — confirm before implementing.** The shelf is implemented as a
fixed filter applied only when generating a `report` or running `verify`, using flags
passed to that command, not stored in or read from `search-results.json`. It never
reaches `engine.py`, `SearchOptions`, or any ranking key. Rationale:

- "Independent of other filters" is read as independent of `fit_eq_filters()`'s
  correction budget and objective, not merely "a different filter type inside the same
  budget." A house-curve tonal preference is orthogonal to *placement* quality
  (dip/null/excess-GD severity), which is what the search ranks. Coupling the shelf
  into `broad_trend_db`/dip-vs-trend/excess-GD-authority math would let a purely
  aesthetic choice quietly change which placement wins, which conflicts with this
  project's existing "no weighted blend, objective ranking" stance.
  If a future user wants the shelf to influence which pair wins, that is a materially
  different feature (a target-policy-level house curve) and belongs with the
  target-specific scoring plan above, not here.
  If this scoping is wrong for the intended use case, the alternative — a
  `SearchOptions`/`EqOptions` field serialized into settings and picked up by
  `report`/`verify` automatically — is a small extension of the same filter code; flag
  this back before Phase B0 if that's actually wanted.
- Keeping it at report/verify time means changing shelf gain/frequency is instant (no
  re-running the potentially slow exhaustive `search`), and adds zero risk to
  `engine.py`.
- A well-formed RBJ shelf biquad is itself minimum-phase, so even though it changes
  magnitude, it does not introduce "excess" (non-minimum-phase) group delay by the
  definitions this codebase already uses — `gd_weighted_null_score`/
  `_excess_gd_authority` would not flag a *correctly implemented* shelf as a resonance
  even if it were later included in a GD-aware calculation. "Ignoring group delay" is
  therefore mostly moot for a shelf specifically (unlike a narrow bell mis-tuned to
  fight a real cancellation); the real reason to exclude it from scoring is the
  taste-vs-objective-ranking argument above, not a group-delay concern.

Because the ranking table's numbers describe the corrected-but-unshelved response, the
report must not silently show a plot/PEQ block that includes the shelf next to metrics
that don't. Label the shelf trace and PEQ text block explicitly as not reflected in the
ranking metrics (see B.4).

### B.2 Filter design

Add `low_shelf_response(frequencies, sample_rate, fc, gain_db, slope=1.0) ->
np.ndarray` to `dsp.py`, parallel in style to `peq_response()`: the RBJ Audio EQ
Cookbook low-shelf biquad (`A = 10**(gain_db/40)`, `alpha` from `slope` via the
standard shelf-slope formula, `b0..b2`/`a0..a2`, then the same `z1 = exp(-1j*omega)`
complex-response evaluation `peq_response` already uses). `slope` is the RBJ "S"
shelf-slope parameter, `0 < S <= 1` (`S = 1` is the steepest shelf without gain
overshoot); expose it bounded `0.1..1.0`, default `1.0`.

Tests: DC gain converges to `gain_db` as `fc, f -> 0` relative spacing; response at
frequencies well above `fc` converges to 0 dB; response is monotonic between those
limits for `slope = 1`; a specific `(fc, gain_db, slope)` triple matches a
hand-computed reference value at `fc` itself (a standard cookbook sanity check).

### B.3 CLI surface

Add to both `report` and `verify` subparsers in `cli.py`:

- `--low-shelf-freq HZ` (float, default `None`)
- `--low-shelf-gain DB` (bounded e.g. `-15..15`, default `0.0`)
- `--low-shelf-slope` (bounded `0.1..1.0`, default `1.0`, `argparse.SUPPRESS`-level
  advanced knob like `--ppo`)

Validate (in `cli.py`, matching the existing `_bounded_float`/`argparse.ArgumentError`
style, not deep in `dsp.py`): `--low-shelf-gain` nonzero requires `--low-shelf-freq`;
`--low-shelf-freq` without a nonzero gain is accepted but inert (documented as a no-op,
not an error, so scripts can pass a fixed frequency and toggle gain to `0` to disable).
Do not tie these bounds to `--max-boost`/`--max-cut` — they are a different, wider
control surface with a different purpose.

### B.4 Report integration

In `build_report()`, when the shelf is active, compute `low_shelf_response()` once
over `context.frequencies` (for the pair-detail magnitude plot) and over the
`padded_spectra()` frequency grid (for CSD/tail if extended there — see below), and
combine it with each displayed pair's already-fitted filter response
(`eq_grid`/`eq_full` from `pair_diagnostics`) purely for a new, separately computed
"with shelf" curve. Do not fold it into `data["post_eq_db"]`/`data["filters"]` or any
field the ranking table reads.

- Add an extra, clearly labeled Plotly trace to `_magnitude_figure` (e.g. "Post-EQ +
  low shelf (tonal, not scored)"), toggleable like the existing "Combined PEQ response"
  legend-click behavior.
- Extend `_peq_text` to append a visually separated block when the shelf is active,
  e.g. a blank line followed by `LS Fc {fc:.1f} Hz  Gain {gain:+.1f} dB  Slope
  {slope:.2f}  (tonal, not scored)`, so the exported filter text a user pastes into
  their own DSP includes it without conflating it with the fitted `PK` lines above.
- Add one sentence to the report's settings/legend text and to the pair-detail
  configuration line making clear the shelf does not affect ranking. Do not touch
  `_ranking_table`, `_overview_figure`, or any ranking/rank/score field.

### B.5 Verify integration

`run_verification()` predicts a specific pair's summed response
(`context.sum_on_grid`) and compares it to one new physical measurement. When the
shelf flags are passed to `verify`, apply `low_shelf_response()` to the *predicted*
curve before computing `deviation`/`max_deviation_db`, since `verify` is inherently
about validating one concrete, fully-specified configuration — if the user intends to
run the shelf on their real DSP (or already has), the prediction should include it so
the comparison is meaningful. Label the verification plot/summary line with the active
shelf settings when present, matching the report's "not part of the placement search"
framing is unnecessary here since `verify` has no ranking to protect — the shelf is
just part of "the configuration being checked" for this command.

### B.6 What stays untouched

`fit_eq_filters()`, `EqOptions`, `SearchOptions`, `engine.py`, `run_search()`'s
serialized settings/pairs, and `search`'s CLI surface are unmodified by Part B in the
design above. `search-results.json`'s `format_version` is unaffected by Part B (it may
still bump for Part A, independently).

## Implementation phases

### Phase A0: freeze behavior and build fixtures

- [x] Add a short-sweep synthetic fixture: reuse `_synthetic_ir()`'s pattern but with a
      short `length` (coarse `native_resolution_hz`) and injected small, zero-mean,
      seeded-random phase/amplitude perturbation concentrated in the sub-bass, so the
      "squiggle" failure mode is directly reproducible in a test.
- [x] Add a long-sweep fixture covering the same band, to confirm fine native
      resolution produces negligible extra smoothing (near-identity behavior).
      (Implemented as a treble-region-vs-sub-bass-region reduction-ratio comparison
      within one fixture rather than a separate long-sweep cache; same guarantee.)
- [x] Add a fixture with a genuine, resolution-supported low-frequency excess-GD
      feature (e.g. a synthetic secondary reflection/mode with real group-delay
      signature well above the noise floor) to prove real problems are still caught.
- [x] Record current (pre-change) `excess_gd_ms`/`excess_gd_tail_ms`/
      `_excess_gd_authority` output for all of the above as a regression baseline.
      (Done ad hoc via scratch scripts to calibrate test thresholds rather than
      committed as a separate baseline artifact.)

### Phase A1: native resolution and adaptive smoothing primitives

- [x] Add `AnalysisContext.native_resolution_hz`.
- [x] Implement `gd_smoothing_octaves()` and the chosen variable-width smoother
      (Section A.3: sigma-ladder blend), with a slow reference check
      (`test_smooth_by_variable_octaves_matches_fixed_sigma_at_a_ladder_rung`).
- [x] Unit-test the smoother directly against the reference implementation on small
      synthetic arrays, independent of the rest of the pipeline.

### Phase A2: wire into `excess_group_delay()`

- [x] Apply the smoother per Section A.4's ordering (common-delay removal before
      smoothing).
- [x] Confirm all pre-existing `test_pipeline.py` excess-GD tests (`test_excess_gd_*`,
      `test_gd_weighted_null_score_*`) are unaffected: they call `excess_group_delay`
      without `native_resolution_hz` (default `None` disables smoothing) or call
      `gd_weighted_null_score`/`_excess_gd_authority` directly with hand-built curves,
      so none of them exercise the new path.
- [x] Confirm the short-sweep noisy fixture's roughness drops materially in the
      sub-bass and much less so in the treble
      (`test_excess_group_delay_native_resolution_smooths_subbass_noise_more_than_treble`).
- [x] Confirm the genuine-low-frequency-feature fixture's score is not meaningfully
      reduced
      (`test_excess_group_delay_native_resolution_preserves_a_genuine_broad_feature`).

### Phase A3: schema, settings, and report

- [x] Bump `format_version` (now `5`); update `html_report.py`'s minimum-version guard
      and error text.
- [x] Add the `native_resolution` settings block in `run_search()`.
- [x] Confirm the report's excess-GD plot and pair-detail text read the new curve with
      no other changes required (it already consumes `data["excess_curve_ms"]`/
      `post_eq_excess_curve_ms"` from `pair_diagnostics`).

### Phase A4: polish and documentation

- [ ] Optional non-fatal low-resolution CLI warning (Section A.5) — skipped for this
      pass; not required for the core fix, revisit if short sweeps turn out to be
      common in practice.
- [x] Document the native-resolution concept and `MIN_RELIABLE_NATIVE_BINS` in
      `README.md` alongside the existing excess-GD prose.
- [x] Update `CLAUDE.md`.

### Phase B0: low-shelf primitive

- [x] Implement `low_shelf_response()` and its direct unit tests (Section B.2).
- [x] Confirm it composes correctly in cascade with `filters_response()`'s existing
      multiplicative pattern (a shelf response is just another factor multiplied into
      the total complex response). (Applied via `db20` addition in `html_report.py`
      and direct complex multiplication in `verification.py`, both equivalent to
      cascading the biquad.)

### Phase B1: CLI and validation

- [x] Add `--low-shelf-freq`/`--low-shelf-gain`/`--low-shelf-slope` to `report` and
      `verify` parsers, with the gain/frequency co-requirement validated in
      `ShelfOptions.__post_init__` (`dsp.py`) rather than `cli.py`, so the same
      validation applies to any caller, not just the CLI.
- [x] CLI parsing tests mirroring `test_search_max_cut_and_tie_tolerance_arguments`
      (`test_low_shelf_cli_arguments_on_report_and_verify`).

### Phase B2: report integration

- [x] Wire the shelf into `build_report()`/`_magnitude_figure()`/`_peq_text()` per
      Section B.4.
- [x] Integration test: build a report twice (shelf on/off) from the same
      `search-results.json` and confirm every ranking-table cell and `rank`/`eq_rank`
      value is byte-identical, while the shelf trace/PEQ text block differs
      (part of `test_synthetic_search_and_report`).

### Phase B3: verify integration

- [x] Wire the shelf into `run_verification()`'s predicted curve per Section B.5.
- [ ] Integration test comparing `max_deviation_db` with/without a known synthetic
      shelf applied to both the "measured" and predicted curves — skipped: there was
      no existing test coverage of `run_verification()` to extend (it requires a REW
      HTTP client, network-mocked or otherwise, which `test_pipeline.py` does not set
      up for any command). Revisit alongside adding baseline `verify` test coverage.

## Test matrix

### Part A

- Adaptive smoother matches its slow reference implementation on synthetic arrays.
- Long/fine-resolution sweeps: `excess_group_delay`, `excess_gd_tail_ms`,
  `_excess_gd_authority`, `gd_weighted_null_score` all byte-identical to pre-change
  behavior.
- Short/coarse-resolution sweeps with injected zero-mean sub-bass noise: scores are
  materially smaller and stable across different random seeds/noise realizations of
  the same underlying (noise-free) response.
- A genuine, resolution-supported low-frequency excess-GD feature is preferentially
  preserved relative to noise, retaining a substantial majority of its severity
  (not "essentially unchanged" - see Risk #3 below for the actual measured retention
  and why a hard "unaffected" claim would overstate what a smoothing-based approach
  can guarantee).
- `format_version` bump rejects old `search-results.json` files with a precise
  "rerun `subpair search`" message, exactly like the existing `< 4` guard.
- Full existing `test_pipeline.py` suite passes unmodified in intent (values may
  change only where the fixtures are specifically the short-sweep/noisy case).

### Part B

- `low_shelf_response()` DC/high-frequency asymptotes, monotonicity, and a
  hand-computed reference point.
- CLI bounds/co-requirement validation for `--low-shelf-*` on both `report` and
  `verify`.
- Report byte-identical ranking output with shelf on vs off (Phase B2's key test).
- Report PEQ text and magnitude plot clearly separate shelf output from fitted `PK`
  filters.
- Verify's predicted-vs-measured comparison applies the shelf only where intended
  (Phase B3's key test).
- `search` and `SearchOptions`/`EqOptions` remain completely unaware of the shelf
  (no new fields, no CLI flags on `search`).

## Risks and decisions to resolve

1. **Scope of Part B (Section B.1) — resolved:** implemented as report/verify-only,
   per the plan's default recommendation. `search`/`SearchOptions`/`EqOptions` are
   untouched; `ShelfOptions` lives only in `dsp.py` and is threaded through
   `cli.py`/`html_report.py`/`verification.py`. Revisit only if a user actually wants
   the shelf to influence `search` (a materially different feature).
2. **`MIN_RELIABLE_NATIVE_BINS` (= 6) and the sigma-ladder rungs are empirical
   constants**, chosen from interactive experimentation against synthetic fixtures
   (see the test-threshold calibration in `test_pipeline.py`'s Part A tests) rather
   than a formal sweep. Revisit if real-world short-sweep reports still look
   under- or over-smoothed.
3. **Over-smoothing risk — confirmed real, not eliminated.** Post-review, the initial
   docs overclaimed that a "genuine, resolution-supported low-frequency excess-GD
   feature is unaffected." That is not generally true: this is Gaussian smoothing
   with sigma up to several octaves at the extreme, and any real feature whose own
   bandwidth is comparable to or narrower than the sigma applied at its frequency is
   attenuated by construction, the same tradeoff any smoothing-based denoiser makes.
   `sample_rate / length` is a resolution *heuristic*, and `MIN_RELIABLE_NATIVE_BINS`
   is a chosen estimator width, neither is a hard measurement-theory reliable/
   unreliable boundary. The corrected, accurate claim (now in `README.md`/`dsp.py`/
   `engine.py`) is that this smoothing *preferentially preserves* broad features
   relative to narrow noise. Measured retention for one roughly one-octave-wide
   synthetic feature under moderately coarse resolution: ~72% peak amplitude, ~87%
   of scalar score — real but bounded, not "smoothed into invisibility," and not
   "unaffected" either. The ladder's top rung was also tightened from 8 to 4 octaves
   so a pathologically short capture cannot smooth away an entire realistic analysis
   band. This was validated against exactly one feature shape/width, not the "wider
   family of physically plausible GD features" a fuller validation would use;
   revisit with broader synthetic coverage if real-world short-sweep reports turn out
   to under-report genuine narrow low-frequency features. If that happens, shrink
   `MIN_RELIABLE_NATIVE_BINS` or the sigma-ladder's top rung rather than raising the
   floor another way.
4. **Boxcar vs Gaussian ladder-blend (Section A.3) — resolved:** implemented the
   sigma-ladder blend (option 1); it composes cleanly with the existing
   `ndimage.gaussian_filter1d`-based helpers elsewhere in `dsp.py` and needed no new
   dependency.
5. **Shelf gain bound — resolved as `-15..15` dB.** Not revisited against real usage
   yet; still just a "sane CLI default," not an acoustically-derived limit. Loosen if
   it turns out to bind in practice.

## Acceptance criteria

- Excess-GD scores and authority gating are materially stable across sweep-length
  changes that do not reflect a real acoustic difference, verified by the Phase A0
  noisy-fixture test.
- Genuine low-frequency excess-GD features remain fully detected, verified by the
  Phase A0 genuine-feature fixture.
- Existing (fine-resolution) synthetic tests are unaffected.
- `format_version` is bumped and old result files fail with a clear message.
- Report exposes the native-resolution context near the existing minimum-phase/ranking
  explanation text.
- A broad low-shelf boost/cut is available on `report` and `verify`, is clearly
  distinguished from fitted PEQ bells in both the plot and the exported filter text,
  and provably does not change `search`'s output, ranking, or any `null_score_db`/
  `excess_gd_*` value.
- `verify` correctly reflects an active shelf in its predicted-vs-measured comparison.
- Full test suite passes; README documents both features.

## Expected file changes

- `src/subpair/dsp.py`: `AnalysisContext.native_resolution_hz`,
  `gd_smoothing_octaves()`, the variable-width smoother, updated
  `excess_group_delay()`; `low_shelf_response()`.
- `src/subpair/engine.py`: `format_version` bump and `native_resolution` settings
  block (Part A only).
- `src/subpair/cli.py`: `--low-shelf-freq`/`--low-shelf-gain`/`--low-shelf-slope` on
  `report` and `verify`, with validation (Part B only).
- `src/subpair/html_report.py`: `format_version` guard bump (Part A); shelf trace,
  `_peq_text` extension, settings-block rendering (Part B).
- `src/subpair/verification.py`: shelf application to the predicted curve (Part B
  only).
- `tests/test_pipeline.py`: adaptive-smoother reference tests, short/long-sweep
  fixtures, low-shelf unit and integration tests; additional fixture helpers alongside
  `_synthetic_ir()`.
- `README.md`: native-resolution/adaptive-smoothing explanation next to the existing
  excess-GD prose; low-shelf usage under a new or existing `subpair report`/`subpair
  verify` section.
- `CLAUDE.md`: update the `dsp.py`/architecture summary once implemented, per its own
  instruction to keep pace with `PLAN.md`'s status.
