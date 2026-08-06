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
