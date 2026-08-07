# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`subpair` is a deterministic, offline-first Python CLI that picks two subwoofer
positions from solo Room EQ Wizard (REW) measurements. REW is used read-only:
`subpair fetch` pulls impulse responses from REW's HTTP API into a local NumPy
cache, and every other command (`search`, `report`, `verify`) operates only on
that cache — no network access, no REW dependency, and byte-for-byte
reproducible output for fixed inputs and options.

The full command semantics (ranking metrics, EQ targets, plateau diagnostics,
verification behaviour) are documented in detail in `README.md` — read it
before changing scoring/search/report behaviour, since the prose there is the
spec, not just user docs.

## Commands

```sh
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -e .          # editable install; registers the `subpair` console script
```

Run the test suite with `unittest` (no pytest config/dependency in this repo):

```sh
python -m unittest discover -s tests -v      # full suite
python -m unittest tests.test_pipeline.PipelineTests.test_synthetic_search_and_report  # single test
```

There is no configured linter, formatter, or type checker in `pyproject.toml`
— don't assume `ruff`/`black`/`mypy` conventions are enforced by CI here.

CLI usage (see `README.md` for full flag semantics):

```sh
subpair fetch --count 12
subpair search --band 25 150 --delay-range -10 10 0.1 --gain-range -3 3 0.5 --eq-bands 7 --top 10
subpair report --top 5 --limit 15 --output subpair-report.html
subpair verify --rank 1 --output verification.html
```

## Architecture

Four pipeline stages, each a thin CLI wrapper (`cli.py`) around a module with
the real logic. Data flows strictly one-way through `.subpair-cache/`:

1. **`api.py` (`RewClient`)** — talks to REW's beta HTTP API. Never assumes a
   fixed route shape: `discover()` fetches REW's root, locates its advertised
   OpenAPI/Swagger document (falling back to a validated `doc.json` probe only
   for old Swagger UI builds that don't expose a spec URL), and *semantically*
   scores GET operations by summary/description/operationId text to find the
   measurement-list and impulse-response routes. Never hardcode a REW path;
   route resolution must stay data-driven from the spec.

2. **`cache.py`** — validates and atomically persists fetched measurements as
   `.subpair-cache/measurement-*.npz` + `manifest.json`. Enforces the core
   invariant used everywhere downstream: every cached measurement must share
   the same sample rate and impulse length (mismatches raise `CacheError`
   rather than resampling/padding — subpair never silently reinterprets
   measurement data). Writes are atomic (temp file + `os.replace`).

3. **`dsp.py`** — the numerical core; everything else is built on
   `AnalysisContext` and the functions below it. No CLI or I/O code here.
   - `AnalysisContext` builds a shared log-frequency grid from the cache,
     phase-aligns every measurement onto one absolute time base (via each
     measurement's `start_time_seconds`), and rejects caches whose relative
     start-time offset would wrap around the zero-padded FFT frame used for
     minimum-phase/CSD work (`__post_init__` raises `ValueError`).
   - Two rankings are computed per pair — **raw** magnitude-only and
     **EQ'd** (post-PEQ) — each a strict lexicographic tuple (never a
     weighted blend): GD-weighted null score → excess GD → excess-GD tail →
     excess-GD peak → decay tail. `excess_gd_peak_ms` is the width-invariant
     counterpart to the area-based excess-GD tail: a denoised, plain maximum
     of `|excess GD|`, so a narrow severe spike and a wide bump of equal peak
     height score the same, breaking ties the area-based tail leaves exact.
     `--tie-tolerance-db` only widens what counts as a tie in the *primary*
     metric before falling through to the rest.
   - `gd_weighted_null_score` inflates magnitude-dip severity (and scores
     magnitude peaks) only where they coincide with real excess group delay,
     using `_excess_gd_authority` as the shared risk gate — the same gate
     also throttles PEQ correction authority in `fit_eq_filters`, so "how bad
     is this dip" and "how much can EQ fix it" derive from one signal.
   - `fit_eq_filters` is a bounded greedy RBJ-biquad fitter (constant-Q
     bells), constrained by `EqOptions` (`target` ∈ `trend`/`flat`/`dsp`,
     boost/cut limits, correction range + slope, max filter count).
   - Minimum-phase/excess-GD extraction is real-cepstrum-based and
     deliberately only run once per finalist (not inside the fast grid
     search) because it's too expensive to vectorize over the full
     delay/gain/polarity grid.
   - `excess_group_delay` is resolution-aware: `AnalysisContext.native_resolution_hz`
     (`sample_rate / length` of the *unpadded* cache) sets how much of the
     sub-bass a given capture can actually resolve, and `gd_smoothing_octaves`
     + `_smooth_by_variable_octaves` progressively smooth the excess-GD curve
     below that limit so ordinary measurement noise near DC doesn't read as
     excess group delay. This feeds every downstream consumer of the curve
     (`excess_gd_ms`/`excess_gd_tail_ms`, `_excess_gd_authority`,
     `gd_weighted_null_score`, the report's excess-GD plot) from one place.
   - `excess_group_delay`'s `gd_baseline` argument (`SearchOptions.gd_baseline`,
     CLI `--gd-baseline`, default `"flat"`) selects what "excess" is measured
     from: `"flat"` removes a single weighted-median constant (the existing,
     unchanged behaviour); `"monotonic"` instead removes a per-point baseline
     from `_monotonic_gd_baseline` (weighted isotonic regression via PAVA,
     `_isotonic_non_increasing`), constrained non-increasing in magnitude as
     frequency rises, over the full band regardless of `integration_range`.
     This is a deliberate, opt-in *acoustic* assumption (a genuine low-end GD
     rise is normal; a bump anywhere the non-increasing fit can't explain
     still counts in full, by construction) — unlike the resolution-based
     smoothing above, it is not a measurement-reliability correction, changes
     rankings, and is not the default. `excess_group_delay` returns
     `(score, curve_ms, baseline_ms)`; every caller unpacks three values now.
   - `low_end_extension_f3_hz`/`low_end_extension_f6_hz` (and their
     `post_eq_` counterparts, from `low_end_extension_hz()`) are
     diagnostic-only F3/F6-style extension estimates reported in
     `search`/`report` tables. They are deliberately **not** part of either
     ranking tuple — see "Key invariants" below. `low_end_extension_hz()`
     itself is self-referential (scores a curve against its own envelope
     peak) unless a caller passes `reference_db`; `run_search` uses this by
     computing the *elementwise maximum* trend curve (the best SPL any pair
     actually achieves, per frequency — not necessarily from the same pair
     throughout) across every pair in the search and calling it with each
     pair's `(trend - best_curve)` departure curve and `reference_db=0.0`,
     so the reported Hz is a same-frequency, cross-pair-comparable answer
     rather than a self-referential shape estimate. Two alternatives were
     tried and reverted: comparing only against a single scalar (a pair's
     own peak, or the loudest pair's peak) breaks down for a bandpass-shaped
     sum, since a pair whose peak sits at a different frequency than the
     reference can read as *less* extended while actually being louder at
     the low end; comparing against the *elementwise average* curve breaks
     down because most real placements roll off toward the bottom of the
     band to some degree, so the average curve already has a "typical"
     rolloff baked into its own shape, hiding exactly that amount of rolloff
     in any pair that isn't unusually bad — confirmed on a real search where
     two pairs a genuine 6 dB apart at 30 Hz both read as fully extended
     against their average. The function returns `None` (not a misleading
     in-band or edge number) when a curve's departure never gets within
     threshold of the reference even at its own best point; render that as
     an empty/gray cell, never as a
     number.
   - `ShelfOptions`/`low_shelf_response` add a fixed, user-specified broad
     low-shelf tonal control. It is a `search`-time `EqOptions.shelf` field
     (`cli.py`'s `--low-shelf-*` flags live on the `search` subcommand),
     exactly like `max_boost_db`/`max_filters` — it *can* change which
     placement wins, since it's folded into every post-EQ score. `report`/
     `verify` no longer take `--low-shelf-*`; they read whichever shelf a
     given `search-results.json` already has baked into `settings.eq.shelf`.
     `fit_eq_filters`'s greedy PK-bell loop is fitted completely unaware of
     the shelf (targets the raw, unshelved response exactly as if it were
     inactive) and the shelf is multiplied into the returned response only
     at the very end, so a deliberate tonal tilt is never fought/cancelled
     by the corrective fitter as if it were a defect at the same
     frequencies. `filters_response()` only ever reconstructs the PK-bell
     list (the shelf has no fc/gain/q representation there); every site
     that reconstructs a full EQ'd response from `filters` — `eq_full`/
     `eq_trend_wide` in `pair_diagnostics` — must also multiply in
     `eq_options.shelf.response(...)` at its own frequency grid to stay
     consistent with `fit_eq_filters`'s own returned `total`. Raw (pre-EQ)
     diagnostics are untouched by the shelf, exactly like every other
     `EqOptions` field only affecting `post_eq_*`.
   - Read the module-level docstrings before touching any scoring function —
     most encode a specific, previously-debugged failure mode (e.g. why dip
     detection uses a two-sided wide check in addition to the one-octave
     trend, why the excess-GD gate uses a maximum filter instead of an
     average, why the tail metric is shape-neutral). `PLAN.md` documents an
     in-progress redesign to give `trend`/`flat`/`dsp` fully independent
     scoring policies instead of sharing `gd_weighted_null_score` — check its
     status before assuming the current shared-scorer behaviour is final.

4. **`engine.py` (`run_search`)** — enumerates every measurement pair and,
   for each, exhaustively grid-searches polarity × delay × gain (vectorised
   NumPy, magnitude-only for speed) to find the best raw candidate(s), then
   runs the expensive per-finalist diagnostics (`dsp.pair_diagnostics`) only
   on ties from that fast stage. Writes `search-results.json`
   (`format_version` is bumped whenever the result schema changes — keep
   `verification.py`/`html_report.py` in sync with it).

5. **`html_report.py`** / **`verification.py`** — consume
   `search-results.json` (+ cache) to produce self-contained HTML (Plotly
   inlined, no CDN/network at view time). `verification.py` additionally
   talks to `RewClient` once, to fetch the one new physical measurement being
   checked against a predicted sum. Neither takes its own shelf flags; both
   reconstruct `ShelfOptions` from the loaded `search-results.json`'s
   `settings.eq.shelf` (see above) and pass it through `EqOptions.shelf` -
   `build_report` via `pair_diagnostics`, `run_verification` by applying it
   to the *predicted* curve before computing deviation, matching whatever
   was actually scored.

### Key invariants to preserve

- **No resampling, no zero-padding to match lengths.** Mismatched sample
  rates or impulse lengths between measurements are hard errors everywhere
  (`cache.py`, `AnalysisContext`), not something to coerce.
- **Determinism.** Same cache + same CLI options must produce byte-identical
  `search-results.json`. Avoid introducing unseeded randomness, dict-order
  dependence, or wall-clock-dependent output into `engine.py`/`dsp.py`.
- **Lexicographic ranking only.** Don't collapse the ranking tuples in
  `engine.py` into a single weighted score; each later metric exists
  specifically to break ties in the one before it.
- **Diagnostic-only fields stay out of ranking.** `low_end_extension_f3_hz`/
  `low_end_extension_f6_hz` are deliberately excluded from every ranking
  tuple in `engine.py`/`run_search` (the low-shelf is *not* one of these —
  it is a scored `EqOptions` field, see above). If you add another
  informational metric,
  don't fold it into `raw_bands`/`eq_bands`'s sort keys without an explicit
  decision to do so — the whole point of these two is that they summarize a
  placement without being able to change which one wins.
- **Offline-first.** `search`, `report`, and the scoring/EQ logic in `dsp.py`
  must never require network access; only `fetch` and `verify` talk to REW.
