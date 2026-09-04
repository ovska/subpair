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
subpair search --band 25 150 --delay-range -10 10 0.05 --gain-range -3 3 0.5 --eq-bands 7 --top 10
subpair report --top 5 --limit 15 --output subpair-report.html
subpair verify --rank 1 --output verification.html
```

## Workflow

Commit after each self-contained change, with a clear message describing
what changed and why — don't batch unrelated changes into one commit, and
don't leave a commit that couldn't be checked out on its own without
something broken or unfinished (failing tests, a half-done refactor, dead
imports). If a change can't yet stand on its own, keep going rather than
committing a broken intermediate state.

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
   - Raw and post-EQ results use one higher-is-better usable-output score:
     `(1-low_end_weight)*SPL + low_end_weight*low_end_power -
     dip_weight*smoothed_dip`. Defaults are 0.5/1.0 and the CLI exposes both
     weights. The displayed relative score sets the best pair to 0 dB.
   - `smoothed_dip_db` is the worst negative deviation from a
     one-third-octave-FWHM Gaussian smoothing, evaluated with real spectral
     margin outside the scored band. It deliberately has no old wide-null
     heuristic or group-delay multiplier.
   - Low-end power is the one-octave broad response through 100 Hz weighted by
     the `f^-4` excursion/amplifier cost (+12.04 dB/octave downward). Raw and
     post-EQ headroom must already be present in every response before SPL,
     low-end power, dip, or score is calculated.
   - Excess GD, excess-GD tail/peak, and CSD decay remain diagnostics and still
     gate unsafe EQ boost, but are not score terms or hidden tie-breakers.
   - `fit_eq_filters` is a bounded greedy RBJ-biquad fitter (constant-Q
     bells), constrained by `EqOptions` (`target` ∈ `trend`/`flat`/`dsp`,
     boost/cut limits, correction range + slope, max filter count). `dsp` is
     now a descriptive alias of `flat`, with no score-specific exception.
   - Minimum-phase/excess-GD extraction is real-cepstrum-based and run only for
     shortlisted configurations, not every exhaustive grid point. Engine
     retains up to eight highest raw-score and eight lowest-dip configurations
     per pair, then selects the reported tuple by fitted post-EQ score.
   - `excess_group_delay` remains resolution-aware via
     `AnalysisContext.native_resolution_hz`,
     `gd_smoothing_octaves`, and `_smooth_by_variable_octaves`; its
     weighted-median common delay is removed before diagnostic output and EQ
     authority gating.
   - `low_shelf_response` implements the RBJ LS biquad.
     `EqOptions.low_shelf` enables one automatic LS candidate by default;
     `cli.py` exposes only `--low-shelf on|off`. `fit_eq_filters` chooses the
     shelf corner and boost/cut per pair inside the same greedy loop and
     objective as PK filters. A selected shelf obeys the EQ gain/range/GD
     constraints and consumes one `max_filters` slot. The returned `filters`
     list remains PK-only, while `metadata["shelf"]`/the persisted
     per-pair `eq_shelf` dict hold the fitted LS parameters. Any companion
     frequency grid reconstructed in `pair_diagnostics` must multiply
     `_fitted_low_shelf_response(...)` alongside `filters_response(...)`.
     Raw diagnostics and physical-sum verification apply neither PK nor LS;
     every fitted band affects only `post_eq_*`.
   - Read the module-level docstrings before touching any scoring function —
     they encode why score smoothing uses real frequency margins, why low-end
     power is normalized, and why the excess-GD EQ gate uses a maximum filter
     instead of an average.

4. **`engine.py` (`run_search`)** — enumerates every measurement pair and,
   for each, exhaustively grid-searches polarity × delay × gain (vectorised
   NumPy) to form the bounded high-score/low-dip shortlist, then runs the
   expensive per-finalist diagnostics (`dsp.pair_diagnostics`) and selects by
   post-EQ usable-output score. Writes `search-results.json`
   (`format_version` is bumped whenever the result schema changes — keep
   `verification.py`/`html_report.py` in sync with it). When
   `SearchOptions.modal` is set, also computes `modal.estimate_room_poles`
   once (Stage 1, over every solo measurement) and, per finalist pair,
   `modal.compute_pair_modal_metrics`/`modal.modal_robustness` against the raw
   sum plus a second `compute_pair_modal_metrics` pass against the sum after
   applying its already-fitted (not re-derived) EQ bank (Stage 2, cheap
   fixed-pole linear fits) — diagnostic fields only, never part of
   `score_db`/`post_eq_score_db` unless `SearchOptions.modal_tiebreak` opts a
   pair's `(n_highQ, sum_modal_energy_db)` into the sort key strictly after
   the primary score. `effective_tail_ms`/`post_eq_effective_tail_ms` are
   always populated (regardless of `SearchOptions.modal`): a pair's own modal
   `ringing_ms` when valid, else `dsp.csd_style_decay`'s original
   `raw_tail_ms`/`post_eq_tail_ms`. The report's/CLI's "Tail" column instead
   reads `effective_tail_db`/`post_eq_effective_tail_db` (that pair's modal
   `worst_mode_level_db`, the loudest detected mode's level relative to
   direct sound) whenever the source is modal, falling back to the ms fields
   otherwise: `ringing_ms` saturates at 0 for every mode below
   `ModalOptions.audible_margin_db`, which a well-controlled room can do for
   every pair, so the dB figure — which keeps varying below that floor — is
   preferred whenever it's available.

5. **`modal.py`** — parametric modal decomposition, independent of
   `dsp.py`'s magnitude/phase scoring. Matrix-pencil pole estimation
   (`_matrix_pencil_poles`) on an 18–200 Hz band-limited, 500 Hz-decimated,
   Butterworth-filtered (not a hard frequency mask — see the module
   docstring for why a brick-wall mask rings unacceptably on an impulsive
   signal) segment; poles are swept across model order and pooled jointly
   across every solo measurement (`estimate_room_poles`), retaining only
   poles that persist across a majority of both. A fixed-pole linear
   least-squares residue fit (`fit_mode_residues`/`compute_pair_modal_metrics`)
   is what makes per-pair metrics cheap. No CLI or I/O code here; `engine.py`
   is the only caller. `L_n`'s 0 dB reference (`compute_pair_modal_metrics`'s
   `direct_reference`) is the RMS of the band-limited direct arrival over
   `_direct_reference_window_seconds` (at least one period of the lowest
   in-band mode, floored at 20 ms), not a single peak sample in a fixed
   20 ms window — a window shorter than the mode's own period, scored by
   peak sample, measures the onset transient rather than the mode and
   systematically understates every mode's level.

6. **`html_report.py`** / **`verification.py`** — consume
   `search-results.json` (+ cache) to produce self-contained HTML (Plotly
   inlined, no CDN/network at view time). `verification.py` additionally
   talks to `RewClient` once, to fetch the one new physical measurement being
   checked against a predicted sum. Neither takes its own shelf flag.
   `build_report` reads the search-time automatic-shelf enablement and reruns
   `pair_diagnostics`; `run_verification` compares the unequalized physical
   and predicted sums, applying neither PK nor LS filters. `build_report`
   also renders a "Modal analysis" section (pole map, per-position invariance
   check, per-pair mode tables) whenever `search-results.json` carries a
   valid `modal_signature`; it is omitted entirely for older or
   `--modal off` results. `build_report`'s own `--room` flag (report-only,
   not stored in `search-results.json`) overlays `room_mode_frequencies`'
   purely geometric rigid-box eigenfrequencies, capped at 3rd order per axis
   index by default, as legend-toggleable line traces (`_room_mode_traces`)
   on every frequency-domain chart and, as horizontal lines, the CSD
   heatmaps — a deliberately distinct, cheaper, less trustworthy sibling to
   `modal.py`'s measured poles, not a replacement for them. Only the axial
   trace is drawn by default (`visible=True`); tangential/oblique start
   `visible="legendonly"` (still toggleable, not hidden from the legend)
   since they're usually weaker and numerous enough to clutter the chart.

### Key invariants to preserve

- **No resampling, no zero-padding to match lengths.** Mismatched sample
  rates or impulse lengths between measurements are hard errors everywhere
  (`cache.py`, `AnalysisContext`), not something to coerce.
- **Determinism.** Same cache + same CLI options must produce byte-identical
  `search-results.json`. Avoid introducing unseeded randomness, dict-order
  dependence, or wall-clock-dependent output into `engine.py`/`dsp.py`.
- **One documented scalar score.** Keep score components explicit in result
  JSON and preserve the CLI weights. Do not reintroduce hidden lexicographic
  tie-breakers or multiply visible dip depth by an unrelated phase metric.
- **Equal-drive scoring.** Headroom must be applied to the complete response
  before SPL, low-end power, residual dip, plots, and the final EQ sum are
  derived. Low-end power and SPL are score components; excess-GD and CSD/tail
  metrics are diagnostics and EQ-authority inputs only.
- **Offline-first.** `search`, `report`, and the scoring/EQ logic in `dsp.py`
  must never require network access; only `fetch` and `verify` talk to REW.
- **Modal metrics stay a separate, opt-in axis.** `modal.py`'s per-pair
  high-Q/stored-energy metrics must never silently enter `score_db`/
  `post_eq_score_db`; the only sanctioned path is the explicit
  `SearchOptions.modal_tiebreak` secondary sort key, inserted strictly after
  the primary score, never before it.
