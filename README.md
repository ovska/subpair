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
  --delay-range -10 10 0.1 --gain-range -3 3 0.5 --top 10
subpair report --top 5 --output subpair-report.html
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

Ranking is strictly lexicographic:

1. maximum dip of a 1/6-octave-smoothed magnitude below a one-octave trend;
2. energy-weighted mean absolute excess group delay;
3. worst post-EQ time-to-minus-20-dB across one-third-octave bands.

There is no weighted blend. Smoothing is deliberately applied before null
depth is measured, so thin comb notches do not win or lose the search.

For minimum phase, subpair uses the real-cepstrum form of the Hilbert
transform on the *full available 0-to-Nyquist magnitude*, not a brick-wall
copy of the scoring band. The impulse is zero-padded by 4x for this transform,
and log magnitude is floored 160 dB below its maximum. This limits circular
Hilbert/cepstral wrap while making the bandwidth convention explicit. The
energy-weighted constant component of excess delay is removed: it represents
the arbitrary common timing offset, while relative arrival time remains in
the complex sum.

The PEQ simulator greedily fits at most four RBJ constant-Q bell cuts against
the broad trend. It never boosts. The post-EQ tail score is a deterministic,
one-third-octave CSD-style analytic-envelope estimate and should be treated as
a comparative metric, not a room RT60 measurement.

Relative SPL is reported, not ranked. It is the energy-mean in-band SPL at the
searched gain settings (gains are referenced to an equal 1 kHz electrical
drive), relative to rank 1.

### `subpair report`

Produces one HTML file containing Plotly itself, a sortable table, top-pair
response diagnostics, CSD-style pre/post-EQ heatmaps, and copyable PEQ text.
No network connection or CDN is used when the report is viewed.

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

## Important assumptions

- Solo measurements must share a loopback-derived absolute time base.
- The cache records each IR start time. Spectra are shifted onto that absolute
  time axis, then one common reference delay is removed. Only relative delay
  can affect the search.
- `verify` is the mutual-coupling sanity check. Linear superposition can fail
  when nearby drivers/enclosures interact acoustically or electrically.
- Search output is deterministic for a fixed cache and options. Fetch metadata
  naturally reflects whatever REW returns.

