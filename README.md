# subpair

`subpair` is a deterministic, offline-first Python CLI for choosing two
subwoofer positions from solo Room EQ Wizard (REW) measurements. REW is used
read-only: `fetch` retrieves impulse responses, and every expensive operation
runs from a local NumPy cache.

## Install

Python 3.11 or newer is required.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Start REW's API server (default `http://127.0.0.1:4735`) and load at least
two solo measurements made with the same loopback timing reference (with
only two, there's exactly one possible pair — `search` still runs the full
scoring/EQ/diagnostic pipeline on it, just with nothing to rank it against).
Then:

```sh
subpair fetch --count 12
subpair search --band 25 150 \
  --delay-range -10 10 0.05 --gain-range -3 3 0.5 --eq-bands 7 --top 10
subpair report --top 5 --limit 15 --output subpair-report.html
```

For an aggressive, range-limited flat target with bounded boost:

```sh
subpair search --aggressive-correction \
  --eq-range 30 90 --eq-range-slope 48 --max-boost 6
subpair report --top 5
```

Omit `--count` to use every loaded measurement, or use `--indices 1,3,5,7`
to select explicit entries. `fetch` prints the complete REW measurement list
before selecting anything. It rejects mixed sample rates or response lengths;
responses are never resampled or extended to make them match.

The defaults write `measurement-*.npz`, `manifest.json`, and
`search-results.json` below `.subpair-cache/`.

## Commands

### `subpair fetch`

The REW API is beta. `subpair` first loads the self-documenting root, locates
the OpenAPI document referenced by it, and only then resolves the measurement
list and impulse-response GET operations from the advertised paths. A
validated `/doc.json` compatibility probe is used only when old Swagger UI
HTML does not expose its specification URL. Arrays are decoded as big-endian
32-bit floats. The unwindowed, non-normalised impulse response is requested
when those query parameters are advertised.

### `subpair search`

All pair, polarity, relative-delay, and relative-gain settings are enumerated.
The first sub is the 0 dB reference; the second receives the reported gain,
polarity, and delay. The grid includes both range endpoints when they lie on
the requested step. Delay is always evaluated at 0.05 ms resolution or finer;
if a coarser step is requested, subpair automatically uses 0.05 ms and records
the effective step as `settings.delay_grid_step_ms`.

Raw and EQ'd results are both rated with one scalar **usable-output
score**. Higher is better, and the displayed score is relative to the best
pair in that mode (best = 0 dB):

```text
sound power = (1 - low-end weight) × equal-drive full-band SPL
            + low-end weight × excursion-weighted low-end power
score       = sound power - dip weight × residual dip
```

The defaults are a 0.5 low-end weight and a 1.0 dip weight. In other words,
ordinary in-band pressure and low-frequency extension contribute equally in
dB, and each dB of the worst remaining local dip costs one score dB.
`--score-low-end-weight 0..1` moves the output term from full-band SPL to
low-end power; `--score-dip-weight 0..4` controls how strongly response
smoothness matters. A dip weight of zero rates output alone. The absolute
`score_db` values retain the cache's arbitrary acoustic level reference, but
subtracting the best value gives the stable, directly comparable
`relative_score_db` shown as **Score** in the CLI and report.

The residual-dip component compares each raw or corrected response with a
one-third-octave-FWHM Gaussian smoothing of that same response and takes only
the largest negative deviation. Real measured response outside the analysis
band is included as smoothing margin before the result is cropped back to the
scored band. This makes the number visually auditable: a broad roll-off is
followed by the smoothed reference, while a narrow notch remains visible.
There is no two-sided null-recovery heuristic and no excess-group-delay
multiplier, so a 3 dB feature cannot be reported as a 19 dB magnitude null.

Excess group delay, its shape-neutral tail integral, its width-invariant peak,
and CSD decay time remain in the result and report as diagnostics. They still
gate unsafe automatic EQ boost, but they do not silently inflate the residual
dip or act as score tie-breakers.

The exhaustive polarity/delay/gain grid uses the same equal-drive raw formula
as a fast first pass. Full EQ fitting for every grid point would be
prohibitively expensive, so each pair first chooses the robust delay for every
polarity/gain and retains up to eight lowest-robust-objective configurations
and eight lowest-residual-dip configurations. Fitted post-EQ score then selects
the pair's reported polarity and gain from that robust-delay shortlist. This
two-objective shortlist matters because a slightly quieter, smoother raw sum
can require less attenuation and win after EQ. Raw and EQ'd tables report the
same selected physical configuration so their before/after values remain
comparable.

Each pair also reports `delay_plateau_ms`/`gain_plateau_db`: how far delay
or gain can drift while the raw score remains within 0.5 dB of its value at
the selected configuration. A wide plateau is forgiving; a narrow one is
more sensitive to real-world delay drift, temperature, or DSP quantization.

#### Delay-basin robustness and physical timing

A sharp single-position timing cancellation can score as well as genuinely
complementary room coupling, so the raw optimum is not automatically the delay
subpair recommends. For each shortlisted polarity/gain, subpair defines the
lower-is-better objective `f = -raw usable-output score` and evaluates

```text
f_robust(tau) = E[f(tau + epsilon)]
epsilon ~ Normal(0, sigma_tau^2)
```

by Gaussian convolution of the already-computed exhaustive delay curve. It
does not re-run the acoustic sum for jitter samples. Independent gain jitter
with sigma 0.5 dB (roughly a +/-1 dB excursion at two sigma) is folded into
the same expectation. The result reports both the unconstrained raw `tau_star`
and the physically constrained `tau_robust`; `delay_ms` is the latter for
backward compatibility. It also keeps `f_tau_star` and
`f_robust_tau_robust` explicit so disagreement is visible.

The pair JSON and comparison table include:

- `fragility`: `f_robust(tau_star) - f(tau_star)` in dB;
- `basin_w03` and `basin_w05`: the contiguous basin containing `tau_star`,
  not the total duration of every disconnected near-optimal region;
- `worst_case` at 0.5, 1.0, and 1.5 ms around `tau_star`;
- `n_competing`: local minima within 0.3 dB of the raw global minimum; and
- `geometric_pass`: whether `basin_w03` covers the pair's geometric excursion.

The report plots raw and jitter-averaged `f(tau)`, shades the +0.3 dB
contiguous basin, and separately marks the measured physical delay,
`tau_star`, and `tau_robust`.

REW's loopback-referenced arrival delay is normalized into the cache at fetch
time. Because delay is applied to sub 2, a pair's physical compensation is
`arrival_1 - arrival_2`. Robust delay selection is restricted to that value
+/- `--physical-delay-window` (1.5 ms by default). The unconstrained raw
optimum is still reported and marked `non_physical_solution` when it lies
outside the window. Such pairs, pairs whose window does not intersect the scan,
and pairs involving an arrival-delay outlier are disqualified ahead of score;
score remains the unchanged ordering within the physically credible group.
Subpair warns about delays above 1.5 times the median and, when room dimensions
are configured, path lengths longer than the room diagonal. Old caches without
arrival metadata remain usable, but their affected pairs explicitly say the
physical constraint is unavailable; re-run `subpair fetch` to populate it.

Timing jitter comes from the change in differential arrival time for a listener
displacement `d`:

```text
delta_tau_max = (d / c) * |u_A - u_B|
sigma_tau = delta_tau_max / 2
```

`d` defaults to 0.25 m (`--listener-movement`) and `c` to 343 m/s
(`--speed-of-sound`). Pass coordinates with `--geometry-config` (or
`--geometry`):

```json
{
  "listening_position_m": [2.5, 2.0, 1.0],
  "sub_positions_m": {
    "1": [0.0, 0.0, 0.0],
    "2": [5.0, 0.0, 0.0],
    "3": [0.0, 4.0, 0.0]
  },
  "room_dimensions_m": [5.0, 4.0, 2.5]
}
```

Sub-position keys are the 1-based cached positions. A simple array in cache
order is also accepted. If either coordinate is unavailable, subpair uses the
conservative `2d/c` bound and marks that fact. Opposite-side pairs reach this
bound; same-direction pairs can have much smaller differential jitter.

These metrics are intentionally a disqualifier, not a certificate. A narrow
basin proves that the tuning is fragile. A wide basin does **not** prove that a
pair is robust across seats: moving the microphone also changes each sub's
magnitude response as it samples a different point in the room's modal pressure
field, and this delay-only test cannot capture that. Only multi-position
measurements can validate a listening area.

Low-end power replaces the old F3/F6 thresholds. It energy-averages the
one-octave broad response over the analyzed range through 100 Hz and weights
each frequency by the approximate amplifier and excursion cost of producing
pressure there. In the pistonic region, pressure is proportional to frequency
squared times cone displacement. Holding pressure constant one octave lower
therefore takes four times the displacement and, with the simple
voltage-proportional-to-displacement model available without driver data,
about sixteen times the amplifier power. The resulting `f^-4` weight is
+12.04 dB per octave downward.

The searched gain is included through `headroom_db`/`post_eq_headroom_db`,
a negative gain applied to the actual compared response. The first sub is at
0 dB and the second receives the reported relative gain, so raw headroom is
the negative of any positive pair gain. Post-EQ headroom additionally removes
the fitted EQ response's largest in-band boost. The hottest driver is
therefore at 0 dB for every pair, and EQ boost cannot manufacture equal-drive
output capability.

Headroom is applied once to the complete raw or post-EQ sum before calculating
every scoring component, magnitude comparison, and final combined EQ
response. The report's Headroom column and copyable `Preamp ... dB` setting
use that same value. Exact electrical watts or cone excursion would require
driver impedance, motor, enclosure, limiter, and built-in DSP transfer data
which an acoustic REW impulse response does not contain, so low-end power is
an excursion-cost proxy rather than a claim of absolute output capability.

For minimum phase, subpair uses the real-cepstrum form of the Hilbert
transform on the *full available 0-to-Nyquist magnitude*, not a brick-wall
copy of the scoring band. The impulse is zero-padded by 4x for this transform,
and log magnitude is floored 160 dB below its maximum. This limits circular
Hilbert/cepstral wrap while making the bandwidth convention explicit. The
energy-weighted constant component of excess delay is removed: it represents
the arbitrary common timing offset, while relative arrival time remains in
the complex sum.

The cached impulse's own length sets a native frequency resolution
(`sample_rate / length`) that zero-padding cannot improve — it can only
interpolate smoothly between what that capture actually resolved. Near DC
that native resolution covers a large fraction of an octave, so a short
sweep leaves few genuinely independent samples per octave in the sub-bass;
differentiating an interpolated phase there amplifies ordinary measurement
noise into large, sign-flipping excess-group-delay swings that have nothing
to do with the placement itself. `excess_gd_ms`/`post_eq_excess_gd_ms`,
`excess_gd_tail_ms`/`post_eq_excess_gd_tail_ms`, the excess-GD authority
gate, and the report's excess-GD plot are therefore progressively smoothed
below roughly six times the cache's native resolution per octave (capped at
4 octaves of smoothing so an unusually short capture can't smooth away the
entire analysis band) — negligible for a long sweep or well above the
sub-bass, strongest right where a short sweep's own resolution runs out.

`sample_rate / length` is a useful resolution *heuristic*, not a hard
measurement-theory cutoff, and "six native bins" is a chosen, tunable
estimator width (`MIN_RELIABLE_NATIVE_BINS`), not a threshold derived from
first principles. Consequently this smoothing *preferentially preserves*
genuine, resolution-supported low-frequency excess-GD features (a real
reflection or port resonance many bins wide relative to the cache's native
resolution) over noise concentrated near or below that resolution — it does
not leave every genuine feature bit-for-bit unaffected. A real feature whose
own bandwidth is comparable to or narrower than the smoothing width applied
at its frequency will still be attenuated somewhat, the same tradeoff any
smoothing-based denoiser makes; only pathologically short captures (well
outside normal REW usage) push the smoothing width large enough for this to
matter for an ordinarily-wide feature. The cache's native resolution and
this threshold are recorded in `search-results.json`'s
`settings.native_resolution` block.

The automatic EQ simulator uses constrained greedy target matching with RBJ
constant-Q PK filters and the optional automatic low shelf. It evaluates the
largest raw-magnitude target errors and retains the PK or LS candidate that
most reduces the weighted global error. `--eq-bands COUNT`
allows 0–16 bands and defaults to 7; zero cleanly disables EQ. The default
`--eq-target trend` follows the broad response and `--max-boost 0` preserves
cuts-only behaviour. `--eq-target flat` or `--aggressive-correction` uses a
flat in-range target; `--max-boost` permits 0–12 dB of *combined* boost.
`--max-cut` bounds any single filter's cut depth, 0–30 dB (default 18).

`--eq-target dsp` fits the same flat curve as `flat`, for placements you
intend to correct with a full-featured external DSP rather than subpair's
own conservative fitter. It is retained as a descriptive alias; unlike the
retired null/GD ranking, usable-output scoring has no target-specific `dsp`
exception, so `dsp` and `flat` produce the same numerical result for otherwise
identical options.

Boost filters are capped at Q 1 so the fitter cannot use a sharp resonant bell
to fill a narrow cancellation. Cuts may use Q up to 10 for modal peaks. The
combined response is checked against `--max-boost` after every candidate, and
areas with large excess group delay receive less fitting authority.
These safeguards follow the [REW automatic-EQ controls](https://www.roomeqwizard.com/help/help_en-GB/html/eqwindow.html)
and [miniDSP room-EQ guidance](https://www.minidsp.com/applications/home-theater-tuning/surround-equalization-with-10x10hd).

`--eq-range LOW HIGH` constrains filter centres. The target correction is
attenuated outside that range by `--eq-range-slope` (0–48 dB/oct); zero means
a hard target curtain. The fitted EQ bands may still have their natural skirts
outside the range. Excess GD (`excess_gd_ms`/`excess_gd_tail_ms`/
`excess_gd_peak_ms` and their post-EQ counterparts) is integrated over this
same range, since it doubles as the EQ authority gate. `--eq-bands 0`
disables fitting entirely, and in that case the range has nothing left to
scope: it falls back to the full analysis `--band` regardless of any
`--eq-range` also passed, so a leftover narrower `--eq-range` from another
run cannot silently shrink the diagnostics of an un-EQ'd search.

EQ authority is reduced where absolute excess group delay is large relative
to the local period. Subpair lightly denoises delay on the log-frequency
grid (well under one bin, just enough that a single noisy sample can't set
a gate by itself), then takes a *maximum* — not a moving average — over at
least a one-third-octave window before smoothing only the resulting gate's
edges. A moving average would dilute a peak in proportion to how much
narrower it is than the averaging window, so a genuinely severe but narrow
excess-GD spike could end up almost entirely ignored while a wider, shallower
bump of the very same peak height was heavily gated; the maximum filter
instead gates a narrow spike and a wide bump of equal height alike. It
therefore cannot follow narrow point-to-point GD wiggles or jump abruptly
from low to full authority over a couple of hertz, but does still respond
fully to a genuinely narrow, severe spike. Authority falls rapidly as gated
excess GD approaches 0.35 cycles, so the aggressive target does not blindly
boost phase-storage nulls which are unlikely to respond to EQ. The effective,
range- and excess-GD-aware target is available as a hidden trace on each
magnitude plot.

The post-EQ tail score is a deterministic, one-third-octave CSD-style
analytic-envelope estimate and should be treated as a comparative metric, not
a room RT60 measurement.

#### Low shelf

`--low-shelf on|off` controls whether the automatic EQ fitter may use one RBJ
low-shelf band; it defaults to `on`. When enabled, the fitter chooses both the
corner frequency and the boost or attenuation independently for every pair.
The shelf competes directly with PK candidates against the same correction
target, obeys `--max-boost`, `--max-cut`, `--eq-range`, and the excess-GD-aware
objective, and consumes one of the slots allowed by `--eq-bands` when chosen.

`on` enables the candidate but does not force a shelf when it cannot improve
the fit. `off` leaves the entire band budget available to PK filters. The RBJ
slope is fixed internally at 1, the steepest transition without gain
overshoot; there are no manual shelf-frequency, gain, or slope flags. A fitted
shelf appears alongside the PK filters in the report as an `LS Fc ... Gain ...
Slope ...` line and is included in every post-EQ plot and score.

#### Modal decay and high-Q resonance metrics

`--modal on|off` (default `off`) computes a parametric modal decomposition —
matrix-pencil pole estimation — separate from the magnitude/phase-based
scoring above. It answers a question null suppression and excess GD do not:
how hard does a given placement drive the room's own resonances, and for how
long does that stored energy stay audible? Delay/polarity/gain cannot change
a room mode's frequency or damping (those are properties of the room), but
they change how strongly a given sum excites it, so two placements with
near-identical smoothed response can still differ substantially in modal
excitation and audible tail length.

Estimation is two-stage:

- **Stage 1** jointly estimates the room's pole set — frequency `f_n` and
  decay rate `α_n` per mode — from every solo measurement's impulse, using the
  matrix-pencil method (preferred over Prony here for its SVD-based noise
  robustness) on a signal band-limited to 18–200 Hz and decimated to 500 Hz
  ("frequency zoom" conditioning, essential for the estimator). Model order is
  swept 10–60; a candidate pole is retained only if it recurs, within
  tolerance, across a majority of orders tried *and* a majority of solo
  measurements. This is the room's modal signature and does not depend on
  which pair is being evaluated.
- **Stage 2** fixes that pole set and solves an ordinary linear least-squares
  fit for each candidate pair sum's modal amplitudes only. This is fast and
  well-conditioned, which is what makes per-pair modal metrics affordable —
  only Stage 1 (once per search) runs the expensive order sweep.

Per retained mode, `subpair` reports `f_n` (Hz), `T60_n` (s), `Q_n`, `L_n`
(initial modal level in dB relative to the same sum's direct-sound peak — the
quantity delay/gain actually controls), and `t_audible_n` (time for the mode
to fall a configured margin below the direct sound). The 0 dB reference for
`L_n` is the RMS of the band-limited direct arrival over a window spanning at
least one period of the lowest in-band mode (floored at 20 ms), not a single
peak sample in a fixed 20 ms window: the lowest in-band mode can have a
period longer than 20 ms (e.g. 1/18 Hz ≈ 55.6 ms), so a fixed 20 ms window
doesn't even complete one cycle of it and a peak sample within it mostly
measures the band-limited impulse's broadband onset transient — which a
handful of narrowband poles cannot represent — rather than any one mode's
sustained level. Aggregate per pair: `n_highQ` (count of modes with `Q > 16`
*and* `L_n` above a level gate — a high-Q pole far enough below the direct
sound is not audible and must not inflate the count; reported at two gate
levels, -15 and -20 dB, so the metric's sensitivity to the threshold is
visible), `Q_max` with its `(f, Q, L)` triple, `sum_modal_energy_db`, a
total-stored-energy proxy over the gated modes, `ringing_ms` —
`max(t_audible_n)` over *every* retained mode, not just the Q-gated ones (a
moderate-Q mode ringing loudly is still audible ringing even if it never
counts toward `n_highQ`) — and `worst_mode_level_db`, `max(L_n)` over that
same set: the loudest mode's level regardless of whether it actually crosses
the audibility margin. This is computed both for the raw sum (`modal`) and,
independently, for the sum after applying its already-fitted EQ bank
(`post_eq_modal` — the filters are held fixed, not re-derived, matching how
the rest of the pipeline treats a chosen EQ as fixed once selected). Every
pair's raw fit also carries a robustness check: the fraction of a small ±0.5
ms timing / ±10 cm placement-equivalent / ±1 dB gain neighbourhood in which
`n_highQ` holds at its nominal value, since a modal advantage that evaporates
under a little drift is not a real advantage.

`ringing_ms` is a better "how much does this pair audibly ring" answer than
`raw_tail_ms`/`post_eq_tail_ms` (a fixed -20 dB-from-local-peak CSD envelope
crossing in a 1/3-octave band): it's referenced to the actual direct-sound
level rather than an arbitrary local peak, and it isn't blurred across modes
narrower than a fractional-octave band — the same resolution problem the
"Method" section above describes for Schroeder decay. `subpair` does not
replace the CSD-based tail outright, since it is cheap and always available
while modal estimation is heavier and can fail (LTI violations, a too-short
capture, insufficient pole persistence); instead, `effective_tail_ms`/
`post_eq_effective_tail_ms` use that pair's own `ringing_ms` whenever its
modal fit is valid, and fall back to the CSD tail otherwise. These two fields
are always present regardless of `--modal`; `effective_tail_is_modal`/
`post_eq_effective_tail_is_modal` record which source produced the value.

The ranking table's and CLI's "Tail" column, though, shows
`effective_tail_db`/`post_eq_effective_tail_db` (that pair's own
`worst_mode_level_db`/`post_eq_modal`'s equivalent) instead of the ms value
whenever the source is modal, falling back to the ms value (with no dB
equivalent) otherwise: `ringing_ms` saturates at 0 the instant a mode's level
drops below `audible_margin_db`, which a well-controlled room with no strong
modes can do for every single pair — a wall of identical, uninformative 0 ms
values. The dB figure keeps varying below that floor, so pairs stay
distinguishable even when none of them cross the audibility margin. The
report marks a modal-sourced value with "(modal)" in the pair summary.

These metrics are diagnostic-only by default: they do not affect `score_db`
or `post_eq_score_db`, and are never a hidden tie-breaker. `--modal-tiebreak
on` (requires `--modal on`) inserts `(n_highQ, sum_modal_energy_db)` — both
lower-is-better — strictly after the primary usable-output score, before the
deterministic pair-index tie-break, so the weighting stays opt-in and
inspectable rather than assumed. The pooled room pole set, per-solo-position
invariance data, discard fraction, and any estimation warnings are written to
`search-results.json`'s top-level `modal_signature`; `subpair report` renders
a pole map (`f` vs `Q`, marker size ∝ level), a per-position invariance check
(confirms `f_n`/`T60_n` agree across solo positions, as a real room mode
should), and a sortable per-mode table for every displayed pair, all inside a
"Modal analysis" section that only appears when `--modal` was enabled.

This is a genuinely more expensive analysis than the rest of `search` and is
new/less battle-tested than the magnitude-based scoring; it requires strict
LTI behaviour (a rattling panel or fan noise will confidently produce
confident-looking, meaningless poles) and consistent absolute timing across
every solo measurement, the same requirement `search` already has for delay
alignment. If the retained pole count looks implausibly low, or the report's
warnings mention a high discard fraction, treat the modal metrics for that
search with real skepticism rather than as a settled number.

**`--modal` is still experimental and not fully tested against real
captures** — validation so far is mostly synthetic fixtures plus a handful of
real-world spot checks, not a broad survey of rooms/capture conditions.
`estimate_room_poles`'s joint persistence gate (`measurement_persistence_fraction`,
default 60%) needs near-unanimous agreement across solo measurements, which
gets stricter the fewer of them there are — with only 2, both must agree,
so a real mode that's weakly excited at one position, or two positions whose
individually-detected peaks simply don't line up, can both legitimately
report `modal_signature.valid: false` with no further explanation of which
case occurred. Prefer more solo measurements, and cross-check against
`--room`'s geometric eigenfrequencies, before trusting an empty or
suspiciously sparse modal signature as a real "no significant modes" result.

Relative SPL is the energy-mean in-band SPL after applying the corresponding
raw or post-EQ headroom, so it compares equal maximum driver drive rather than
nominal searched levels. It contributes to sound power according to
`--score-low-end-weight`; the standalone table value is relative to the
highest-scoring pair in that mode. Gains are referenced to an equal 1 kHz
electrical drive.

### `subpair report`

Produces one HTML file containing Plotly itself, the sortable EQ'd ranking,
top-pair response diagnostics, post-EQ CSD heatmaps, and copyable EQ text.
Pass `--raw` to show the raw ranking and raw diagnostics instead, omitting the
EQ-specific plots and filter controls entirely. The table has selection checkboxes
(its top five are checked by default); the selected pairs feed the combined
overview and appear as pair tabs below the table for fast one-at-a-time
diagnostics. The overview switches between magnitude and excess group delay.
Number keys 1–9 open the corresponding selected pair tab. Metric cells use a
best-to-worst colour scale. `report --top N` changes the initial selection
count. `report --limit N` limits the table and its selectable diagnostics to
the top N results (default 15). No network connection or CDN is used when the
report is viewed.

The selected overview and pair diagnostics share one magnitude Y-axis range
and one excess-group-delay Y-axis range, recalculated whenever the selection
changes. The combined EQ response uses that same magnitude axis; when it is
shown, small markers identify every fitted PK/LS band's configured frequency
and gain. EQ response and band gains are included in the shared range so those
markers remain visible. Excess-GD plots include zero but clamp their lower
scale limit to -20 ms so large negative measurement-noise excursions do not
flatten the useful part of the curves.

Each CSD heatmap includes the corresponding zero-referenced excess-group-delay
curve and a solid 0 ms reference. A vertical overlay indicates constant delay;
frequency-dependent bends expose excess storage without relying on visual
estimation of the heatmap ridge. CSD figures are static to avoid accidental
zooming or panning and appear before the separate excess-group-delay graph,
except when `--room` is given (see below), which needs legend clicks to stay
available and so relaxes this to a fixed-axis (no zoom/pan) but otherwise
interactive plot instead.

The table is initially sorted by the computed Score rather than an ordinal
Rank column. It also shows the score's residual-dip, low-end-power, and
Relative-SPL components plus the applied Headroom gain. Low-end power and
Score use higher-is-better colouring; residual dip uses lower-is-better.

#### Room mode overlay

`--room LxWxHcm` (e.g. `--room 345x274x248`) overlays theoretical rigid-
rectangular-room eigenfrequencies on every chart: vertical lines on the
frequency-domain charts (magnitude, excess GD, and their overview panels)
and horizontal lines on the CSD heatmaps, since that plot's y-axis is
frequency. Modes are computed from the standard rigid-box formula
`f(nx,ny,nz) = (c/2) * sqrt((nx/L)^2 + (ny/W)^2 + (nz/H)^2)` up to the
search band's upper edge and up to 3rd order on each axis index (`nx`/`ny`/
`nz` each capped at 3 — past low single digits, higher-order modes are both
weak in a typical room and numerous enough, particularly in tangential/
oblique combinations, to clutter the chart), and classified by how many of
the three integer indices are nonzero: axial (one wall pair, solid line),
tangential (two wall pairs, dashed), or oblique (all three, dotted). This is
a purely geometric reference for a perfectly rigid, empty box — it does not
know about absorption, furniture, openings, or non-rectangular geometry, so
treat it as a rough guide to where axial modes are *likely* to fall, not a
prediction of the room's actual measured behaviour; `subpair search --modal
on` measures the room's real poles directly from the cached impulse
responses instead (see below) and is the more trustworthy source when the
two disagree.

Each mode type is its own legend entry (`Room mode: axial` etc.); click it to
show or hide that type independently, on every chart on the page. Only axial
modes are drawn by default — tangential and oblique modes are usually much
weaker and numerous enough to clutter the chart on their own — but both stay
in the legend and one click makes either visible everywhere. Omitting
`--room` leaves every chart exactly as before — this is a purely additive,
opt-in visual aid with no effect on ranking, scoring, or any other report
content.

### `subpair verify`

After physically measuring one selected sum, leave that new measurement in
REW and run:

```sh
subpair verify --rank 1 --output verification.html
```

The command chooses the sole REW measurement not present in the cache. Use
`--measurement INDEX_OR_UUID` if more than one new measurement is loaded. It
checks sample rate and length, overlays measured versus predicted magnitude,
and prints the maximum in-band deviation. The comparison removes only one
constant level offset by default (`--keep-level` disables that); it does not
hide frequency-dependent mutual-coupling error.

Verification compares the physical sum with the unequalized prediction. It
applies the raw `headroom_db` global gain, but not PK filters, post-EQ
headroom, or the automatically fitted low shelf; the shelf follows exactly
the same verification semantics as every other EQ band.

## Important assumptions

- Solo measurements must share a loopback-derived absolute time base.
- The cache records each IR start time. Spectra are shifted onto that absolute
  time axis, then one common reference delay is removed. Only relative delay
  can affect the search.
- `verify` is the mutual-coupling sanity check. Linear superposition can fail
  when nearby drivers/enclosures interact acoustically or electrically.
- Search output is deterministic for a fixed cache and options. Fetch metadata
  naturally reflects whatever REW returns.
