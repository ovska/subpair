"""Physical-sum verification against one newly loaded REW measurement."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from .api import RewClient
from .cache import CachedMeasurement, load_cache
from .dsp import AnalysisContext, db20
from .html_report import load_results


class VerificationError(RuntimeError):
    pass


def _select_measurement(
    client: RewClient,
    summaries: list[dict[str, Any]],
    cached: list[CachedMeasurement],
    requested: str | None,
) -> dict[str, Any]:
    if requested is not None:
        if requested.isdigit():
            index = int(requested)
            if not 1 <= index <= len(summaries):
                raise VerificationError(f"Measurement index {index} is out of range")
            return summaries[index - 1]
        matches = [row for row in summaries if client.measurement_uuid(row) == requested]
        if len(matches) != 1:
            raise VerificationError(
                f"Expected one measurement with UUID {requested!r}, found {len(matches)}"
            )
        return matches[0]

    cached_uuids = {row.uuid for row in cached if row.uuid}
    cached_indices = {row.source_index for row in cached}
    candidates = []
    for row in summaries:
        uuid = client.measurement_uuid(row)
        index = int(row["_index"])
        is_new = uuid not in cached_uuids if uuid else index not in cached_indices
        if is_new:
            candidates.append(row)
    if len(candidates) != 1:
        raise VerificationError(
            "Expected exactly one REW measurement not present in the cache, found "
            f"{len(candidates)}; pass --measurement INDEX_OR_UUID"
        )
    return candidates[0]


def run_verification(
    cache_dir: Path,
    results_path: Path,
    output_path: Path,
    root_url: str,
    rank: int = 1,
    measurement_id: str | None = None,
    keep_level: bool = False,
    band_override: tuple[float, float] | None = None,
) -> dict[str, Any]:
    cached, _ = load_cache(cache_dir)
    results = load_results(results_path)
    matches = [row for row in results["pairs"] if int(row["rank"]) == rank]
    if len(matches) != 1:
        raise VerificationError(f"Rank {rank} is not present in {results_path}")
    pair = matches[0]
    band = band_override or tuple(float(v) for v in results["settings"]["band_hz"])
    context = AnalysisContext(cached, band, int(results["settings"]["ppo"]))

    client = RewClient(root_url)
    routes = client.discover()
    summaries = client.list_measurements()
    selected = _select_measurement(client, summaries, cached, measurement_id)
    impulse, ir_metadata = client.fetch_impulse(int(selected["_index"]))
    sample_rate = float(ir_metadata["sample_rate"])
    if sample_rate != context.sample_rate:
        raise VerificationError(
            f"Verification sample rate {sample_rate:g} Hz does not match cache "
            f"{context.sample_rate:g} Hz; refusing to resample"
        )
    if impulse.size != context.length:
        raise VerificationError(
            f"Verification response has {impulse.size} samples, cache has {context.length}; "
            "refusing to zero-pad"
        )

    predicted = context.sum_on_grid(
        int(pair["first"]) - 1,
        int(pair["second"]) - 1,
        int(pair["polarity"]),
        float(pair["delay_ms"]),
        float(pair["gain_db"]),
    )
    bins = np.fft.rfftfreq(impulse.size, 1.0 / sample_rate)
    measured_fft = np.fft.rfft(impulse)
    measured = np.interp(context.frequencies, bins, measured_fft.real) + 1j * np.interp(
        context.frequencies, bins, measured_fft.imag
    )
    predicted_db = db20(predicted)
    measured_db = db20(measured)
    level_offset = 0.0 if keep_level else float(np.median(predicted_db - measured_db))
    aligned_measured_db = measured_db + level_offset
    deviation = aligned_measured_db - predicted_db
    max_deviation = float(np.max(np.abs(deviation)))

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=context.frequencies, y=predicted_db, name="Predicted sum")
    )
    figure.add_trace(
        go.Scatter(x=context.frequencies, y=aligned_measured_db, name="Physical measurement")
    )
    figure.add_trace(
        go.Scatter(
            x=context.frequencies,
            y=deviation,
            name="Deviation",
            yaxis="y2",
            line={"color": "#fb7185", "width": 1.7},
        )
    )
    figure.update_layout(
        template="plotly_dark",
        title=f"Rank {rank} predicted vs physical sum",
        xaxis={"type": "log", "title": "Frequency (Hz)"},
        yaxis={"title": "Level (dB)"},
        yaxis2={"title": "Deviation (dB)", "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "y": -0.2},
        height=620,
    )
    plot = figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"displaylogo": False, "responsive": True},
        div_id="verification-plot",
    )
    title = client.measurement_title(selected)
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>subpair verification</title>
<style>body{{margin:2rem auto;max-width:1200px;background:#07111f;color:#e5edf8;font:15px/1.5 system-ui,sans-serif}}
.summary{{padding:1rem 1.25rem;background:#0f1b2d;border:1px solid #26364d;border-radius:10px}}</style>
<script>{get_plotlyjs()}</script></head><body><h1>subpair verification</h1>
<div class="summary"><strong>{html.escape(title)}</strong><br>Maximum absolute deviation in band:
{max_deviation:.3f} dB · constant measured-level offset: {level_offset:+.3f} dB<br>
API routes discovered from {html.escape(routes.spec_url)}</div>{plot}</body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return {
        "rank": rank,
        "measurement_index": int(selected["_index"]),
        "measurement_name": title,
        "max_deviation_db": max_deviation,
        "level_offset_db": level_offset,
        "output": str(output_path),
    }
