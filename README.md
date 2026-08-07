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
three solo measurements made with the same loopback timing reference. Then:

```sh
subpair fetch --count 12
subpair search --band 25 150 \
  --delay-range -10 10 0.1 --gain-range -3 3 0.5 --eq-bands 7 --top 10
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
the requested step.

Two rankings are calculated, both strictly lexicographic:

1. **Raw:** an excess-GD-weighted maximum dip of the raw magnitude below a
   one-octave broad trend or a wide two-sided check, energy-weighted mean
   absolute excess group delay, the shape-neutral excess-GD tail integral,
   then worst raw time-to-minus-20-dB across one-third-octave bands.
2. **EQ'd:** the same three measurements after applying the fitted PEQs.

The one-octave trend is itself a smoothing of the curve it's compared
against, so a dip much wider than that window (a broadband suck-out from a
large path-length difference between the two subs, say) is largely absorbed
into the trend and under-reported. A second check catches this: at each
frequency it looks for the best level well to the left *and* well to the
right (past a one-octave margin so it isn't just re-detecting a narrow
notch off its own shoulders); if both sides recover to a normal level, the
gap between that and the current level counts as a dip, however wide. A
plain monotonic rolloff — expected at a subwoofer's low end — never
recovers on at least one side, so it is never flagged by this check
regardless of its total range; only a single global reference (a
percentile, a single very wide trend) would get that wrong.

A magnitude dip that coincides with real excess group delay is a genuine
destructive-interference null — acoustically irreparable and audible as
smearing, not just a shallow, EQ-fixable amplitude ripple. The null-score
metric scales dip severity up (by up to +150%) where it overlaps excess GD,
using the same risk gate that reduces EQ authority there (see below); a dip
with no excess GD nearby scores the same as before. The plain, unweighted
dip depth is still reported separately as `magnitude_only_null_score_db` /
`post_eq_magnitude_only_null_score_db`.

A magnitude peak above trend still scores nothing on its own — reinforcement
adds output rather than destructively cancelling it, and isn't the kind of
summing-position problem a null is. But a peak that only exists *with* real
excess group delay is a resonance/ringing signature (comb reinforcement with
genuine energy storage that its own magnitude doesn't explain), not a benign
constructive bump, and is scored the same way a dip's severity is inflated:
proportional to the excess-GD risk alone. A minimum-phase peak (negligible
excess GD) still scores exactly zero; only a non-minimum-phase one counts.

This GD weighting (both the dip-severity boost and the peak penalty) is
applied once per finalist, not inside the fast exhaustive delay/gain/polarity
search itself:
true excess GD needs a minimum-phase extraction per candidate, which is too
expensive to run over that whole grid.

The excess-GD scalar used by both rankings is normalized and integrated only
over `--eq-range` (or the complete analysis band when `--eq-range` is not
specified). Minimum-phase extraction and phase differentiation still use the
full cached bandwidth and analysis grid so correction-range boundaries do not
create Hilbert or numerical derivative artifacts.

That scalar is an *energy-weighted* mean, so a badly smeared region that
happens to sit in a magnitude dip or near a band edge (where SPL is
naturally low) barely moves it — two sums can look equally clean on that
metric while one is audibly ringing somewhere the ear doesn't need much
level to notice it. `excess_gd_tail_ms`/`post_eq_excess_gd_tail_ms` — the
third tie-break level in each ranking — integrates `|excess GD|` over
log-frequency across the same range, with every frequency weighted equally
regardless of level, using the same `np.trapezoid`-style integration as the
energy-weighted scalar rather than a plain index-based average. It is
deliberately *shape-neutral*: a narrow, severe spike and a wider, shallower
bump of the same area (peak height times width) score the same, rather than
a peak detector (which only sees the narrow one) or a percentile (which is
blind to anything narrower than its own width cutoff, however severe, the
opposite failure). It exists to catch a sum that looks flat and clean on
magnitude but is smeary in phase somewhere — a case the energy-weighted
mean and the null-score metric (which only look where magnitude itself is
unusual) can both miss.

`excess_gd_peak_ms`/`post_eq_excess_gd_peak_ms` — a fourth, final tie-break —
is the deliberately *width-invariant* counterpart to the area-based tail
metric above: the single worst denoised `|excess GD|` sample in the range,
lightly pre-denoised the same way `_excess_gd_authority`'s own gate is (so
one noisy sample cannot set it by itself), then a plain maximum. A maximum
is unaffected by a feature's own width — a narrow spike and a wide plateau
of the same height already score identically — so this only ever separates
two placements whose smeared *area* ties but where one still has a single
sharper, more severe excursion the area-based tail metric alone cannot see.
It does not replace `excess_gd_tail_ms`; like every metric in this ranking,
it only breaks ties the earlier ones leave exact.

All four excess-GD-derived metrics above — the null-score GD weighting,
`excess_gd_ms`, `excess_gd_tail_ms`, and `excess_gd_peak_ms` — measure
"excess" relative to a baseline selected by `--gd-baseline`, which is
`flat` by default: a single constant (this curve's weighted median), so any
frequency-dependent group delay at all counts as excess. `--gd-baseline
monotonic` instead fits a per-point baseline, via weighted isotonic
regression (pool-adjacent-violators), constrained to be non-increasing in
magnitude as frequency rises, over the complete analysis band. This treats a
genuine, physically-expected group-delay rise confined to the bottom of the
band as normal rather than as excess — the fit can trace a high-to-low
descent through it — while a bump anywhere the non-increasing constraint
cannot explain (a rise appearing after a lower value earlier in the band)
still counts in full, regardless of how wide or gentle it is; this falls out
of the monotonic constraint itself, not a separate width heuristic. This is
an explicit, opt-in *acoustic* assumption about what a benign low end looks
like, not a measurement-reliability correction like the native-resolution
smoothing below — it changes rankings, so validate it against real
measurements before trusting it over the default. When active, the report's
per-pair excess-GD plot overlays the fitted baseline curve.

There is no weighted blend. Each later metric only breaks an exact tie in the
earlier metrics. `--tie-tolerance-db` (0–3 dB, default 0) widens "tie" to any
null-score difference within that many dB, so the fast-search's finalist
tie-break and the raw/EQ'd pair rankings fall through to excess-GD and tail
time between practically-indistinguishable null scores instead of a
below-audibility difference deciding the winner outright. The default of 0
preserves strict lexicographic behaviour, which means a null-score gap of a
few tenths of a dB — well within the resolution/noise of the underlying
measurements — can otherwise decide a pair outright over another candidate
that is dramatically better on every other metric (excess GD, tail, SPL).
A nonzero tolerance in roughly the 1–2 dB range is a reasonable starting
point for real measurements; `generate-reports.sh` uses 1.5. Magnitude
scoring and PEQ fitting
use the raw log-grid samples without fractional-octave or variable
smoothing. The one-octave trend is only the broad reference from which raw
dips and the conservative target are measured. Both rankings use the same
per-pair polarity, delay, and gain tuple selected by the exhaustive raw
search; the EQ'd ranking answers which of those selected sums responds best
to the requested correction.

Each pair also reports `delay_plateau_ms`/`gain_plateau_db`: how far delay
or gain can drift from the chosen value while the raw magnitude null score
stays within 0.5 dB of its optimum. A wide plateau is a forgiving setting;
a narrow one is a razor's-edge optimum easily upset by real-world delay
drift, temperature, or DSP quantization.

Each pair also reports `low_end_extension_hz`/`post_eq_low_end_extension_hz`:
an in-band, F3-style diagnostic giving the lowest frequency the broad
trend's *envelope* holds up, scanning down from the envelope's own **peak**
— wherever in the band it occurs — before permanently falling 3 dB below
that peak and not recovering. The peak is not assumed to sit at the top of
the band: a two-subwoofer sum is routinely bandpass-shaped (it rises out of
the bottom of the band, peaks somewhere in the middle, and rolls off again
toward crossover), and an earlier version of this metric anchored to the
top-of-band sample specifically, which is fragile exactly in that ordinary
case — a curve already declining well before the top edge could sit more
than 3 dB below its own peak there, misreporting a fine low end as
completely collapsed. Anchoring to the envelope's own peak instead is
unaffected by whatever happens *above* the peak (a separate, high-end/
crossover concern), and is identical to the old top-anchored behaviour for
an ordinary monotonically-rising passband. The envelope — the higher of the
best level attained scanning up from the bottom of the band and the best
level still attainable scanning down from the top — is used to find both
the peak and the corner, deliberately not the raw trend: an isolated,
recoverable notch is a placement defect the null score already measures on
its own terms, so a response that is flat down to 25 Hz with one unrelated
-5 dB notch at 100 Hz still reports ~25 Hz extension, not ~100 Hz. A
genuine, sustained rolloff is not masked this way — below the corner, the
envelope still tracks the decline — so it is still reported close to where
the raw trend actually crosses the threshold. Lower is more extended. This
is purely informational — it is shown in `subpair report`'s tables and
printed by `subpair search`, but it is not a raw or EQ'd ranking key, so it
never changes which placement wins; a placement's own null/excess-GD/tail
severity always decides.

This metric is deliberately self-referential — each pair is scored against
its *own* peak, not a level shared across pairs. A shared cross-pair
reference was tried and reverted: it made any pair whose own peak fell more
than 3 dB below that shared level collapse to the same "no extension"
answer everywhere, regardless of how good its actual low-end shape was,
which is not useful and actively misleading for a bandpass-shaped sum.
Cross-pair absolute output is a genuinely different question, already
answered directly by the `relative_spl_db`/`post_eq_relative_spl_db`
columns in `subpair report`; check those alongside Extension if you want to
know how loud two placements are relative to each other, not just how each
one's own low end holds up.

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

The PEQ simulator uses constrained greedy target matching with RBJ constant-Q
bells: it evaluates the largest raw-magnitude target errors and retains the
candidate that most reduces the weighted global error. `--eq-bands COUNT`
allows 0–16 bands and defaults to 7; zero cleanly disables EQ. The default
`--eq-target trend` follows the broad response and `--max-boost 0` preserves
cuts-only behaviour. `--eq-target flat` or `--aggressive-correction` uses a
flat in-range target; `--max-boost` permits 0–12 dB of *combined* boost.
`--max-cut` bounds any single filter's cut depth, 0–30 dB (default 18).

`--eq-target dsp` fits the same flat curve as `flat`, for placements you
intend to correct with a full-featured external DSP rather than subpair's
own conservative fitter. It changes *ranking*, not just the suggested PEQ:
`null_score_db`/`post_eq_null_score_db` barely count a minimum-phase dip at
low excess GD, since any minimum-phase EQ (which describes essentially all
DSP/PEQ hardware) can restore both its magnitude and phase exactly, as long
as the correction fits inside your own `--max-boost`/`--max-cut`. A
non-minimum-phase dip — real excess group delay, a genuine destructive-
interference null — still scores up to the same maximum severity as the
other targets, since no amount of magnitude-only correction fixes that.
Practically, `dsp` mode ends up preferring flat excess group delay over
flat raw magnitude: with ordinary magnitude problems assumed fixable later,
what's left to differentiate placements is largely what a DSP can't fix.
Minimum-phase peaks already score zero in every target, `dsp` included.

Boost filters are capped at Q 1 so the fitter cannot use a sharp resonant bell
to fill a narrow cancellation. Cuts may use Q up to 10 for modal peaks. The
combined response is checked against `--max-boost` after every candidate, and
areas with large excess group delay receive less fitting authority.
These safeguards follow the [REW automatic-EQ controls](https://www.roomeqwizard.com/help/help_en-GB/html/eqwindow.html)
and [miniDSP room-EQ guidance](https://www.minidsp.com/applications/home-theater-tuning/surround-equalization-with-10x10hd).

`--eq-range LOW HIGH` constrains filter centres. The target correction is
attenuated outside that range by `--eq-range-slope` (0–48 dB/oct); zero means
a hard target curtain. The fitted PEQs may still have their natural skirts
outside the range.

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

Relative SPL is reported, not ranked. It is the energy-mean in-band SPL at the
searched gain settings (gains are referenced to an equal 1 kHz electrical
drive). Raw and EQ'd modes each reference their own rank 1.

### `subpair report`

Produces one HTML file containing Plotly itself, the sortable EQ'd ranking,
top-pair response diagnostics, post-EQ CSD heatmaps, and copyable PEQ text.
Pass `--raw` to show the raw ranking and raw diagnostics instead, omitting the
EQ-specific plots and PEQ controls entirely. The table has selection checkboxes
(its top five are checked by default); the selected pairs feed the combined
overview and appear as pair tabs below the table for fast one-at-a-time
diagnostics. The overview switches between magnitude and excess group delay.
Number keys 1–9 open the corresponding selected pair tab. Metric cells use a
best-to-worst colour scale. `report --top N` changes the initial selection
count. `report --limit N` limits the table and its selectable diagnostics to
the top N results (default 15). No network connection or CDN is used when the
report is viewed.

Each CSD heatmap includes the corresponding zero-referenced excess-group-delay
curve and a solid 0 ms reference. A vertical overlay indicates constant delay;
frequency-dependent bends expose excess storage without relying on visual
estimation of the heatmap ridge. CSD figures are static to avoid accidental
zooming or panning and appear before the separate excess-group-delay graph.

The ranking table also shows a colour-rated, informational
`low_end_extension_hz`/`post_eq_low_end_extension_hz` column (see above); it
uses lower-is-better colouring but does not affect sorting or which pairs are
recommended.

#### Low shelf

`--low-shelf-freq HZ --low-shelf-gain DB [--low-shelf-slope S]` adds a fixed,
broad RBJ low-shelf boost or cut on top of the fitted PEQ bank, for people who
want a general tonality control (more or less sub-bass) rather than — or in
addition to — corrective EQ. `--low-shelf-gain` is required to be nonzero for
the shelf to take effect (-15..15 dB); `--low-shelf-freq` alone is inert.
`--low-shelf-slope` is the RBJ "S" shelf-slope parameter (0.1..1, default 1,
the steepest transition without gain overshoot).

The shelf is deliberately independent of `fit_eq_filters`'s bounded bell
fitter: it is never counted against `--eq-bands`/`--max-boost`/`--max-cut`/
`--eq-range`, and it never reaches `subpair search` or any ranking key —
placement selection stays purely about acoustic correctness, and a tonal
preference can never change which pair wins. In the report it appears as a
separate, clearly labelled trace ("Post-EQ + low shelf (tonal, not scored)")
and PEQ-text block (`LS Fc ... Gain ... Slope ...`), left out of every scored
metric and out of the ranking tables entirely.

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

`verify` accepts the same `--low-shelf-freq`/`--low-shelf-gain`/
`--low-shelf-slope` flags as `report`. Unlike `report`, `verify` applies the
shelf to the *predicted* curve before computing the deviation — pass it when
the physical measurement being checked already has that shelf applied
(e.g. on an external DSP), so the comparison stays meaningful.

## Important assumptions

- Solo measurements must share a loopback-derived absolute time base.
- The cache records each IR start time. Spectra are shifted onto that absolute
  time axis, then one common reference delay is removed. Only relative delay
  can affect the search.
- `verify` is the mutual-coupling sanity check. Linear superposition can fail
  when nearby drivers/enclosures interact acoustically or electrically.
- Search output is deterministic for a fixed cache and options. Fetch metadata
  naturally reflects whatever REW returns.
