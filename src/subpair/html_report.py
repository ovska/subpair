"""Self-contained Plotly report generation."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from .cache import load_cache
from .dsp import AnalysisContext, EqOptions, ShelfOptions, db20, pair_diagnostics


class ReportError(RuntimeError):
    pass


def load_results(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReportError(f"Search results not found: {path}; run 'subpair search' first") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"Cannot read search results {path}: {exc}") from exc
    if not isinstance(value.get("pairs"), list) or not value["pairs"]:
        raise ReportError(f"Search results {path} contain no ranked pairs")
    return value


def _plot_html(figure: go.Figure, div_id: str, *, static: bool = False) -> str:
    config = {"displaylogo": False, "responsive": True}
    if static:
        config.update({"staticPlot": True, "displayModeBar": False})
    return figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        config=config,
        div_id=div_id,
    )


def _magnitude_figure(pair: dict[str, Any], data: dict[str, Any]) -> go.Figure:
    f = data["frequencies"]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=f,
            y=data["solo_first_db"],
            name=pair["first_name"],
            line={"color": "#fb7185", "width": 1.4},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=f,
            y=data["solo_second_db"],
            name=pair["second_name"],
            line={"color": "#fbbf24", "width": 1.4},
        )
    )
    figure.add_trace(
        go.Scatter(x=f, y=data["sum_db"], name="Raw sum", line={"color": "#7dd3fc", "width": 1})
    )
    figure.add_trace(
        go.Scatter(
            x=f,
            y=data["eq_target_db"],
            name="EQ target (range/GD aware)",
            line={"color": "#e879f9", "width": 1.5},
            visible="legendonly",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=f,
            y=data["post_eq_db"],
            name="Post-EQ sum",
            line={"color": "#86efac", "width": 1.7},
            visible="legendonly",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=f,
            y=np.asarray(data["post_eq_db"]) - np.asarray(data["sum_db"]),
            name="Combined PEQ response (all bands)",
            line={"color": "#c4b5fd", "width": 1.7},
            visible="legendonly",
            yaxis="y2",
        )
    )
    if "post_eq_shelf_db" in data:
        figure.add_trace(
            go.Scatter(
                x=f,
                y=data["post_eq_shelf_db"],
                name="Post-EQ + low shelf (tonal, not scored)",
                line={"color": "#fdba74", "width": 1.7},
                visible="legendonly",
            )
        )
    figure.update_layout(
        title="Magnitude: solos and optimised sum",
        xaxis={"type": "log", "title": "Frequency (Hz)"},
        yaxis={"title": "Level (dB; cache reference)"},
        yaxis2={
            "title": "Combined PEQ gain (dB)",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "visible": False,
        },
        margin={"l": 62, "r": 70, "t": 52, "b": 55},
        legend={"orientation": "h", "y": -0.22},
        template="plotly_dark",
        height=510,
    )
    return figure


def _overview_figure(
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
    mode: str,
    selected_keys: set[str],
) -> go.Figure:
    figure = go.Figure()
    eq = mode == "eq"
    for index, (pair, data) in enumerate(rows):
        key = f"{int(pair['first'])}-{int(pair['second'])}"
        figure.add_trace(
            go.Scatter(
                x=data["frequencies"],
                y=data["post_eq_db" if eq else "sum_db"],
                name=(
                    f"#{pair['eq_rank' if eq else 'rank']} · "
                    f"{pair['first']}+{pair['second']} — "
                    f"{pair['first_name']} + {pair['second_name']}"
                ),
                line={
                    "color": f"hsl({(index * 137.508) % 360:.1f},72%,67%)",
                    "width": 2.2,
                },
                meta={"pair_key": key},
                visible=key in selected_keys,
            )
        )
    figure.update_layout(
        title="Selected pair EQ’d sums" if eq else "Selected pair raw sums",
        xaxis={"type": "log", "title": "Frequency (Hz)"},
        yaxis={
            "title": (
                "Post-EQ summed level (dB; cache reference)"
                if eq
                else "Raw summed level (dB; cache reference)"
            )
        },
        margin={"l": 62, "r": 24, "t": 52, "b": 55},
        legend={"orientation": "h", "y": -0.22},
        template="plotly_dark",
        height=540,
    )
    return figure


def _overview_excess_figure(
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
    mode: str,
    selected_keys: set[str],
) -> go.Figure:
    figure = go.Figure()
    eq = mode == "eq"
    for index, (pair, data) in enumerate(rows):
        key = f"{int(pair['first'])}-{int(pair['second'])}"
        figure.add_trace(
            go.Scatter(
                x=data["frequencies"],
                y=data["post_eq_excess_curve_ms" if eq else "excess_curve_ms"],
                name=(
                    f"#{pair['eq_rank' if eq else 'rank']} · "
                    f"{pair['first']}+{pair['second']} — "
                    f"{pair['first_name']} + {pair['second_name']}"
                ),
                line={
                    "color": f"hsl({(index * 137.508) % 360:.1f},72%,67%)",
                    "width": 2.0,
                    "shape": "spline",
                    "smoothing": 1.0,
                },
                meta={"pair_key": key},
                visible=key in selected_keys,
            )
        )
    figure.add_hline(y=0.0, line={"color": "#64748b", "width": 1})
    figure.update_layout(
        title=(
            "Selected pair post-EQ excess group delay"
            if eq
            else "Selected pair raw excess group delay"
        ),
        xaxis={"type": "log", "title": "Frequency (Hz)"},
        yaxis={"title": "Excess GD (ms)"},
        margin={"l": 62, "r": 24, "t": 52, "b": 55},
        legend={"orientation": "h", "y": -0.22},
        template="plotly_dark",
        height=540,
    )
    return figure


def _excess_figure(data: dict[str, Any]) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=data["frequencies"],
            y=data["excess_curve_ms"],
            line={"color": "#c4b5fd", "width": 2, "shape": "spline", "smoothing": 1.0},
            name="Excess GD",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=data["frequencies"],
            y=100.0 * np.asarray(data["eq_authority"]),
            line={"color": "#86efac", "width": 1.5},
            name="EQ authority",
            yaxis="y2",
        )
    )
    figure.add_hline(y=0.0, line={"color": "#64748b", "width": 1})
    peak_ms = float(data["excess_gd_peak_ms"])
    for sign in (1.0, -1.0):
        figure.add_hline(
            y=sign * peak_ms,
            line={"color": "#fbbf24", "width": 1, "dash": "dot"},
        )
    figure.add_annotation(
        xref="paper",
        x=1.0,
        y=peak_ms,
        text=f"peak {peak_ms:.2f} ms",
        showarrow=False,
        font={"color": "#fbbf24", "size": 11},
        xanchor="right",
        yanchor="bottom",
    )
    figure.update_layout(
        title="Excess group delay (display spline; raw data used for score)",
        xaxis={"type": "log", "title": "Frequency (Hz)"},
        yaxis={"title": "Excess GD (ms)"},
        yaxis2={
            "title": "EQ authority (%)",
            "overlaying": "y",
            "side": "right",
            "range": [0, 105],
        },
        margin={"l": 62, "r": 24, "t": 52, "b": 55},
        template="plotly_dark",
        height=390,
    )
    return figure


def _decay_figure(data: dict[str, Any]) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=("Pre-EQ", "Post-EQ (bounded PEQ)"),
        horizontal_spacing=0.08,
    )
    common = {
        "x": 1000.0 * np.asarray(data["decay_times"]),
        "y": data["decay_frequencies"],
        "zmin": -40,
        "zmax": 0,
        "colorscale": "Turbo",
        "colorbar": {"title": "dB"},
    }
    figure.add_trace(
        go.Heatmap(z=data["pre_decay_db"], showscale=False, **common), row=1, col=1
    )
    figure.add_trace(
        go.Heatmap(z=data["post_decay_db"], showscale=True, **common), row=1, col=2
    )
    overlay_common = {
        "mode": "lines",
        "line": {
            "color": "#f8fafc",
            "width": 2.5,
            "shape": "spline",
            "smoothing": 1.0,
        },
        "hovertemplate": "%{y:.1f} Hz · %{x:.2f} ms<extra></extra>",
    }
    figure.add_trace(
        go.Scatter(
            x=data["excess_curve_ms"],
            y=data["frequencies"],
            name="Pre-EQ excess GD",
            line=overlay_common["line"],
            mode=overlay_common["mode"],
            hovertemplate=overlay_common["hovertemplate"],
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=data["post_eq_excess_curve_ms"],
            y=data["frequencies"],
            name="Post-EQ excess GD",
            line=overlay_common["line"],
            mode=overlay_common["mode"],
            hovertemplate=overlay_common["hovertemplate"],
        ),
        row=1,
        col=2,
    )
    figure.add_vline(x=0.0, line={"color": "#94a3b8", "width": 1}, row=1, col=1)
    figure.add_vline(x=0.0, line={"color": "#94a3b8", "width": 1}, row=1, col=2)
    figure.update_xaxes(title_text="Time from sum peak (ms)")
    figure.update_yaxes(type="log", title_text="Frequency (Hz)", row=1, col=1)
    figure.update_layout(
        title="CSD-style decay with zero-referenced excess-GD overlay",
        template="plotly_dark",
        height=520,
        legend={"orientation": "h", "y": -0.18},
        margin={"l": 66, "r": 70, "t": 72, "b": 82},
    )
    return figure


def _peq_text(filters: list[dict[str, float]], shelf: ShelfOptions | None = None) -> str:
    lines = [
        f"PK Fc {item['fc_hz']:.1f} Hz  Gain {item['gain_db']:.1f} dB  Q {item['q']:.3f}"
        for item in filters
    ]
    if not lines:
        lines.append("No filters fitted")
    if shelf is not None and shelf.active:
        lines.append("")
        lines.append(
            f"LS Fc {shelf.freq_hz:.1f} Hz  Gain {shelf.gain_db:+.1f} dB  "
            f"Slope {shelf.slope:.2f}  (tonal, not scored)"
        )
    return "\n".join(lines)


def _ranking_table(
    pairs: list[dict[str, Any]],
    mode: str,
    table_id: str,
    selected_keys: set[str],
) -> str:
    eq = mode == "eq"
    rank_key = "eq_rank" if eq else "rank"
    null_key = "post_eq_null_score_db" if eq else "null_score_db"
    excess_key = "post_eq_excess_gd_ms" if eq else "excess_gd_ms"
    tail_key = "post_eq_tail_ms" if eq else "raw_tail_ms"
    extension_key = "post_eq_low_end_extension_hz" if eq else "low_end_extension_hz"
    spl_key = "post_eq_relative_spl_db" if eq else "relative_spl_db"
    columns = [
        (rank_key, "Rank", "number"),
        ("pair", "Pair", "text"),
        ("polarity", "Pol 2", "number"),
        ("delay_ms", "Delay 2 (ms)", "number"),
        ("gain_db", "Gain 2 (dB)", "number"),
        (null_key, "Worst null (dB)", "number"),
        (excess_key, "Excess GD (ms)", "number"),
        (tail_key, "Tail (ms)", "number"),
        (extension_key, "Extension −3dB (Hz)", "number"),
        (spl_key, "Relative SPL (dB)", "number"),
    ]
    heading = '<th class="selection-heading">Show</th>' + "".join(
        f'<th data-key="{key}" data-type="{kind}" data-column-index="{index + 1}">'
        f'{html.escape(label)}</th>'
        for index, (key, label, kind) in enumerate(columns)
    )
    metric_directions = {
        null_key: "low",
        excess_key: "low",
        tail_key: "low",
        spl_key: "high",
    }
    metric_ranges: dict[str, tuple[float, float]] = {}
    for key in metric_directions:
        numeric = [float(pair[key]) for pair in pairs]
        metric_ranges[key] = (min(numeric), max(numeric))

    def score_style(key: str, value: float) -> str:
        if key not in metric_directions:
            return ""
        low, high = metric_ranges[key]
        if high == low:
            worstness = 0.0
        elif metric_directions[key] == "low":
            worstness = (value - low) / (high - low)
        else:
            worstness = (high - value) / (high - low)
        hue = 138.0 * (1.0 - float(np.clip(worstness, 0.0, 1.0)))
        best = worstness <= 1e-12
        outline = "box-shadow:inset 0 0 0 2px rgba(167,243,208,.85);" if best else ""
        return f"background:hsla({hue:.1f},72%,38%,.48);{outline}"

    rows = []
    for pair in pairs:
        key_value = f"{int(pair['first'])}-{int(pair['second'])}"
        values: dict[str, tuple[str, str]] = {
            rank_key: (str(pair[rank_key]), str(pair[rank_key])),
            "pair": (
                f"{pair['first']} + {pair['second']}",
                f"{pair['first']:04d}-{pair['second']:04d}",
            ),
            "polarity": ("+" if pair["polarity"] > 0 else "−", str(pair["polarity"])),
            "delay_ms": (f"{pair['delay_ms']:+.3f}", str(pair["delay_ms"])),
            "gain_db": (f"{pair['gain_db']:+.2f}", str(pair["gain_db"])),
            null_key: (f"{pair[null_key]:.3f}", str(pair[null_key])),
            excess_key: (f"{pair[excess_key]:.3f}", str(pair[excess_key])),
            tail_key: (f"{pair[tail_key]:.1f}", str(pair[tail_key])),
            extension_key: (f"{pair[extension_key]:.1f}", str(pair[extension_key])),
            spl_key: (f"{pair[spl_key]:+.2f}", str(pair[spl_key])),
        }
        cells = []
        for key, _, _ in columns:
            style = score_style(key, float(pair[key])) if key in metric_directions else ""
            style_attribute = f' style="{style}"' if style else ""
            cells.append(
                f'<td data-value="{html.escape(values[key][1])}"{style_attribute}>'
                f'{html.escape(values[key][0])}</td>'
            )
        checked = " checked" if key_value in selected_keys else ""
        checkbox = (
            '<td class="selection-cell"><input class="pair-select" type="checkbox" '
            f'data-mode="{mode}" data-pair-key="{key_value}"{checked} '
            f'aria-label="Show pair {pair["first"]}+{pair["second"]}"></td>'
        )
        rows.append(
            f'<tr data-pair-key="{key_value}">{checkbox}{"".join(cells)}</tr>'
        )
    return (
        f'<table id="{table_id}" class="ranking-table"><thead><tr>{heading}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def build_report(
    cache_dir: Path,
    results_path: Path,
    output_path: Path,
    top: int = 5,
    limit: int = 15,
    shelf: ShelfOptions | None = None,
) -> Path:
    if limit < 1:
        raise ReportError("Report result limit must be at least 1")
    shelf = shelf or ShelfOptions()
    results = load_results(results_path)
    measurements, _ = load_cache(cache_dir)
    if len(measurements) != int(results.get("measurement_count", -1)):
        raise ReportError("Cache measurement count does not match the search results")
    settings = results["settings"]
    band = tuple(float(value) for value in settings["band_hz"])
    eq_settings = settings.get("eq", {})
    eq_range = tuple(float(value) for value in eq_settings.get("correction_range_hz", band))
    eq_options = EqOptions(
        target=str(eq_settings.get("target", "trend")),
        correction_range=eq_range,
        correction_slope_db_per_octave=float(
            eq_settings.get("correction_slope_db_per_octave", 48.0)
        ),
        max_boost_db=float(eq_settings.get("max_boost_db", 0.0)),
        max_cut_db=float(eq_settings.get("max_cut_db", 18.0)),
        max_filters=int(eq_settings.get("max_filters", 7)),
    )
    required_ranking_fields = {
        "rank",
        "eq_rank",
        "raw_tail_ms",
        "post_eq_null_score_db",
        "post_eq_excess_gd_ms",
        "post_eq_relative_spl_db",
        "low_end_extension_hz",
        "post_eq_low_end_extension_hz",
    }
    if int(results.get("format_version", 0)) < 6:
        raise ReportError(
            "Search results predate the width-invariant excess-GD peak "
            "tie-break; run 'subpair search' again"
        )
    if any(
        not required_ranking_fields.issubset(pair)
        for pair in results["pairs"]
    ):
        raise ReportError(
            "Search results predate dual raw/EQ ranking; run 'subpair search' again"
        )
    context = AnalysisContext(measurements, band, int(settings["ppo"]))
    raw_pairs = sorted(results["pairs"], key=lambda pair: int(pair["rank"]))[:limit]
    eq_pairs = sorted(results["pairs"], key=lambda pair: int(pair["eq_rank"]))[:limit]

    def pair_key(pair: dict[str, Any]) -> str:
        return f"{int(pair['first'])}-{int(pair['second'])}"

    default_count = max(0, min(top, len(raw_pairs)))
    raw_default_keys = {pair_key(pair) for pair in raw_pairs[:default_count]}
    eq_default_keys = {pair_key(pair) for pair in eq_pairs[:default_count]}
    detail_sections = []
    diagnostic_by_key: dict[str, dict[str, Any]] = {}
    # Every displayed row can be selected interactively. Pre-render the union
    # of the limited raw and EQ'd rankings so either table has instant, offline
    # diagnostics without retaining every pair in the report.
    displayed_pairs = list(raw_pairs)
    displayed_keys = {pair_key(pair) for pair in displayed_pairs}
    for pair in eq_pairs:
        if pair_key(pair) not in displayed_keys:
            displayed_pairs.append(pair)
            displayed_keys.add(pair_key(pair))
    for pair in displayed_pairs:
        data = pair_diagnostics(
            context,
            int(pair["first"]) - 1,
            int(pair["second"]) - 1,
            int(pair["polarity"]),
            float(pair["delay_ms"]),
            float(pair["gain_db"]),
            include_decay=True,
            eq_options=eq_options,
        )
        key = pair_key(pair)
        if shelf.active:
            shelf_response = shelf.response(context.frequencies, context.sample_rate)
            data["post_eq_shelf_db"] = np.asarray(data["post_eq_db"]) + db20(shelf_response)
        diagnostic_by_key[key] = data
        peq = _peq_text(data["filters"], shelf)
        detail_sections.append(
            f"""
            <section class="pair-detail" data-pair-key="{key}"
              data-pair-label="{pair['first']}+{pair['second']}"
              data-raw-rank="{pair['rank']}" data-eq-rank="{pair['eq_rank']}"
              hidden>
              <h2><span data-mode-copy="raw">#{pair['rank']} Raw</span><span data-mode-copy="eq">#{pair['eq_rank']} EQ’d</span>:
                {html.escape(pair['first_name'])} + {html.escape(pair['second_name'])}</h2>
              <p class="configuration">Sub 2: {'normal' if pair['polarity'] > 0 else 'inverted'},
                delay {pair['delay_ms']:+.3f} ms, gain {pair['gain_db']:+.2f} dB<br>
                <span data-mode-copy="raw">Raw: null {pair['null_score_db']:.3f} dB ·
                excess GD {pair['excess_gd_ms']:.3f} ms ·
                peak {pair['excess_gd_peak_ms']:.2f} ms · tail {pair['raw_tail_ms']:.1f} ms</span>
                <span data-mode-copy="eq">EQ’d: null {pair['post_eq_null_score_db']:.3f} dB ·
                excess GD {pair['post_eq_excess_gd_ms']:.3f} ms ·
                peak {pair['post_eq_excess_gd_peak_ms']:.2f} ms · tail {pair['post_eq_tail_ms']:.1f} ms</span><br>
                EQ: {html.escape(eq_options.target)} target, {eq_range[0]:g}–{eq_range[1]:g} Hz,
                {eq_options.correction_slope_db_per_octave:g} dB/oct curtain,
                max boost {eq_options.max_boost_db:g} dB, up to
                {eq_options.max_filters} PEQ bands; excess-GD guarded.<br>
                {f'Low shelf: {shelf.gain_db:+.1f} dB at {shelf.freq_hz:g} Hz, slope {shelf.slope:.2f} — a fixed tonal control, not reflected in the ranking metrics above.<br>' if shelf.active else ''}
                CSD overlay: excess GD with common delay removed; a vertical line is frequency-independent delay.</p>
              {_plot_html(_magnitude_figure(pair, data), f'magnitude-{key}')}
              {_plot_html(_excess_figure(data), f'excess-{key}')}
              {_plot_html(_decay_figure(data), f'decay-{key}', static=True)}
              <div class="peq"><h3>Fitted PEQ filters</h3><button onclick="copyPeq(this)">Copy</button>
              <pre>{html.escape(peq)}</pre></div>
            </section>
            """.strip()
        )

    raw_overview = [(pair, diagnostic_by_key[pair_key(pair)]) for pair in raw_pairs]
    eq_overview = [(pair, diagnostic_by_key[pair_key(pair)]) for pair in eq_pairs]
    raw_default_json = json.dumps(
        [pair_key(pair) for pair in raw_pairs[:default_count]], separators=(",", ":")
    )
    eq_default_json = json.dumps(
        [pair_key(pair) for pair in eq_pairs[:default_count]], separators=(",", ":")
    )

    settings_json = html.escape(json.dumps(settings, sort_keys=True, indent=2))
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>subpair ranking</title>
<style>
:root {{ color-scheme: dark; --bg:#07111f; --card:#0f1b2d; --line:#26364d; --text:#e5edf8; --muted:#9fb0c7; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.5 ui-sans-serif,system-ui,sans-serif; }}
main {{ width:min(1500px,96vw); margin:0 auto; padding:36px 0 80px; }}
h1 {{ font-size:2.2rem; margin:0 0 4px; }} h2 {{ margin-top:0; }}
.lede,.configuration,.note {{ color:var(--muted); }}
.mode-tabs {{ display:inline-flex; gap:4px; margin:16px 0 20px; padding:4px; border:1px solid var(--line); border-radius:10px; background:#091322; }}
.mode-tab {{ border:0; border-radius:7px; padding:9px 18px; color:var(--muted); background:transparent; cursor:pointer; font-weight:700; }}
.mode-tab.active {{ color:#07111f; background:#7dd3fc; }}
.chart-tabs,.pair-tabs {{ display:flex; flex-wrap:wrap; gap:6px; margin:14px 0; }}
.chart-tab,.pair-tab {{ border:1px solid #3b506d; border-radius:7px; padding:7px 13px; color:var(--muted); background:#101e31; cursor:pointer; font-weight:650; }}
.chart-tab.active,.pair-tab.active {{ color:#07111f; border-color:#7dd3fc; background:#7dd3fc; }}
.pair-tabs {{ min-height:39px; margin:14px 0 2px; }}
.empty-selection {{ align-self:center; color:var(--muted); }}
[hidden] {{ display:none !important; }}
[data-mode-copy="eq"] {{ display:none; }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:12px; background:var(--card); }}
table {{ width:100%; border-collapse:collapse; white-space:nowrap; }}
th,td {{ padding:10px 13px; text-align:right; border-bottom:1px solid var(--line); }}
th:nth-child(3),td:nth-child(3) {{ text-align:left; }}
th {{ position:sticky; top:0; cursor:pointer; background:#142238; color:#c9d8eb; }}
th:hover {{ background:#1b2d48; }} tbody tr:hover {{ background:#132238; }}
.selection-heading,.selection-cell {{ text-align:center; }}
.selection-heading {{ cursor:default; }} .selection-heading:hover {{ background:#142238; }}
.pair-select {{ width:17px; height:17px; accent-color:#7dd3fc; cursor:pointer; vertical-align:middle; }}
.pair-detail {{ margin-top:18px; padding:24px; border:1px solid var(--line); border-radius:14px; background:var(--card); }}
.peq {{ position:relative; padding:16px; background:#091322; border-radius:9px; }}
.peq h3 {{ margin:0 0 10px; }} .peq pre {{ margin:0; white-space:pre-wrap; color:#a7f3d0; }}
.peq button {{ position:absolute; right:14px; top:14px; border:1px solid #49607e; border-radius:6px; padding:6px 11px; color:var(--text); background:#1c2d47; cursor:pointer; }}
details {{ margin:22px 0; }} details pre {{ overflow:auto; color:var(--muted); }}
</style>
<script>{get_plotlyjs()}</script>
</head>
<body><main>
<h1>subpair ranking</h1>
<p class="lede">{results['measurement_count']} positions · {band[0]:g}–{band[1]:g} Hz · {settings['ppo']} points/octave · dual lexicographic ranking · showing up to {limit} pairs per mode</p>
<div class="mode-tabs" role="tablist" aria-label="Ranking response mode">
  <button class="mode-tab active" id="raw-mode-tab" role="tab" aria-selected="true" onclick="setReportMode('raw')">Raw</button>
  <button class="mode-tab" id="eq-mode-tab" role="tab" aria-selected="false" onclick="setReportMode('eq')">EQ’d</button>
</div>
<p class="note">Check table rows to choose comparison pairs. Each mode starts with its top {default_count}; the pair tabs below the table open one full diagnostic at a time. Hotkeys 1–9 open the first nine selected tabs.</p>
<div data-mode-panel="raw">
  <div class="chart-tabs" role="tablist" aria-label="Raw overview chart">
    <button class="chart-tab active" data-overview-mode="raw" data-overview-view="magnitude" role="tab" aria-selected="true" onclick="setOverviewView('raw','magnitude')">Magnitude</button>
    <button class="chart-tab" data-overview-mode="raw" data-overview-view="excess" role="tab" aria-selected="false" onclick="setOverviewView('raw','excess')">Excess GD</button>
  </div>
  <div data-overview-panel="raw" data-overview-view="magnitude">
    {_plot_html(_overview_figure(raw_overview, 'raw', raw_default_keys), 'selected-pairs-magnitude-raw')}
  </div>
  <div data-overview-panel="raw" data-overview-view="excess" hidden>
    {_plot_html(_overview_excess_figure(raw_overview, 'raw', raw_default_keys), 'selected-pairs-excess-raw')}
  </div>
  <p class="note">Raw ranking: raw-magnitude null depth, raw excess group delay, then raw tail.</p>
  <div class="table-wrap">{_ranking_table(raw_pairs, 'raw', 'ranking-raw', raw_default_keys)}</div>
  <div class="pair-tabs" data-pair-tabs="raw" role="tablist" aria-label="Selected raw pairs"></div>
</div>
<div data-mode-panel="eq" hidden>
  <div class="chart-tabs" role="tablist" aria-label="EQ’d overview chart">
    <button class="chart-tab active" data-overview-mode="eq" data-overview-view="magnitude" role="tab" aria-selected="true" onclick="setOverviewView('eq','magnitude')">Magnitude</button>
    <button class="chart-tab" data-overview-mode="eq" data-overview-view="excess" role="tab" aria-selected="false" onclick="setOverviewView('eq','excess')">Excess GD</button>
  </div>
  <div data-overview-panel="eq" data-overview-view="magnitude">
    {_plot_html(_overview_figure(eq_overview, 'eq', eq_default_keys), 'selected-pairs-magnitude-eq')}
  </div>
  <div data-overview-panel="eq" data-overview-view="excess" hidden>
    {_plot_html(_overview_excess_figure(eq_overview, 'eq', eq_default_keys), 'selected-pairs-excess-eq')}
  </div>
  <p class="note">EQ’d ranking: post-EQ raw-magnitude null depth, post-EQ excess group delay, then post-EQ tail.</p>
  <div class="table-wrap">{_ranking_table(eq_pairs, 'eq', 'ranking-eq', eq_default_keys)}</div>
  <div class="pair-tabs" data-pair-tabs="eq" role="tablist" aria-label="Selected EQ’d pairs"></div>
</div>
<p class="note">Click a table heading to sort. Metric cells run from green (best) to red (worst); lower is better except relative SPL, where higher is better. Each mode references its own rank 1 for relative SPL. Extension is an uncolored, informational F3-style estimate (lower is more extended) and is not part of the ranking.</p>
<div id="pair-details">{''.join(detail_sections)}</div>
<details><summary>Analysis settings and minimum-phase convention</summary><pre>{settings_json}</pre></details>
</main>
<script>
document.querySelectorAll('.ranking-table th[data-type]').forEach(th=>{{
  th.addEventListener('click',()=>{{
    const index=Number(th.dataset.columnIndex);
    const table=th.closest('table');
    const body=table.querySelector('tbody');
    const rows=Array.from(body.querySelectorAll('tr'));
    const ascending=th.dataset.order!=='asc';
    table.querySelectorAll('th').forEach(x=>delete x.dataset.order);
    th.dataset.order=ascending?'asc':'desc';
    rows.sort((a,b)=>{{
      let av=a.children[index].dataset.value, bv=b.children[index].dataset.value;
      if(th.dataset.type==='number') {{ av=Number(av); bv=Number(bv); }}
      return (av<bv?-1:av>bv?1:0)*(ascending?1:-1);
    }});
    rows.forEach(row=>body.appendChild(row));
  }});
}});
document.querySelectorAll('.pair-detail .plotly-graph-div[id^="magnitude-"]').forEach(plot=>{{
  plot.on('plotly_legendclick',event=>{{
    const trace=plot.data[event.curveNumber];
    if(trace && trace.name==='Combined PEQ response (all bands)') {{
      const willShow=trace.visible==='legendonly';
      Plotly.relayout(plot,{{'yaxis2.visible':willShow}});
    }}
  }});
}});
const selectedPairs={{
  raw:new Set({raw_default_json}),
  eq:new Set({eq_default_json})
}};
const activePairs={{
  raw:Array.from(selectedPairs.raw)[0]||null,
  eq:Array.from(selectedPairs.eq)[0]||null
}};
const overviewViews={{raw:'magnitude',eq:'magnitude'}};
function sectionForKey(key) {{
  return Array.from(document.querySelectorAll('.pair-detail')).find(
    section=>section.dataset.pairKey===key
  )||null;
}}
function orderedSelectedKeys(mode) {{
  const rankField=mode==='eq'?'eqRank':'rawRank';
  return Array.from(selectedPairs[mode]).filter(key=>sectionForKey(key)).sort((a,b)=>{{
    return Number(sectionForKey(a).dataset[rankField])-Number(sectionForKey(b).dataset[rankField]);
  }});
}}
function updateOverview(mode) {{
  ['magnitude','excess'].forEach(view=>{{
    const plot=document.getElementById('selected-pairs-'+view+'-'+mode);
    if(!plot||!plot.data) return;
    plot.data.forEach((trace,index)=>{{
      const visible=selectedPairs[mode].has(trace.meta.pair_key);
      if(trace.visible!==visible) Plotly.restyle(plot,{{visible:visible}},[index]);
    }});
  }});
}}
function setOverviewView(mode,view) {{
  overviewViews[mode]=view;
  document.querySelectorAll('[data-overview-panel="'+mode+'"]').forEach(panel=>{{
    panel.hidden=panel.dataset.overviewView!==view;
  }});
  document.querySelectorAll('.chart-tab[data-overview-mode="'+mode+'"]').forEach(button=>{{
    const active=button.dataset.overviewView===view;
    button.classList.toggle('active',active);
    button.setAttribute('aria-selected',String(active));
  }});
  window.dispatchEvent(new Event('resize'));
}}
function syncDetailPlot(section,mode) {{
  const plot=document.getElementById('magnitude-'+section.dataset.pairKey);
  if(!plot||!plot.data) return;
  const rawIndex=plot.data.findIndex(trace=>trace.name==='Raw sum');
  const eqIndex=plot.data.findIndex(trace=>trace.name==='Post-EQ sum');
  if(rawIndex>=0) Plotly.restyle(plot,{{visible:mode==='raw'?true:'legendonly'}},[rawIndex]);
  if(eqIndex>=0) Plotly.restyle(plot,{{visible:mode==='eq'?true:'legendonly'}},[eqIndex]);
}}
function renderActiveDetail(mode) {{
  document.querySelectorAll('.pair-detail').forEach(section=>section.hidden=true);
  const key=activePairs[mode];
  if(!key) return;
  const section=sectionForKey(key);
  if(!section) return;
  section.hidden=false;
  syncDetailPlot(section,mode);
  window.dispatchEvent(new Event('resize'));
}}
function activatePair(mode,key) {{
  if(!selectedPairs[mode].has(key)) return;
  activePairs[mode]=key;
  renderPairTabs(mode);
  if(document.body.dataset.reportMode===mode) renderActiveDetail(mode);
}}
function renderPairTabs(mode) {{
  const strip=document.querySelector('[data-pair-tabs="'+mode+'"]');
  const keys=orderedSelectedKeys(mode);
  if(!keys.includes(activePairs[mode])) activePairs[mode]=keys[0]||null;
  strip.replaceChildren();
  if(!keys.length) {{
    const empty=document.createElement('span');
    empty.className='empty-selection';
    empty.textContent='No pairs selected';
    strip.appendChild(empty);
    return;
  }}
  keys.forEach((key,index)=>{{
    const section=sectionForKey(key);
    const button=document.createElement('button');
    button.type='button';
    button.className='pair-tab'+(key===activePairs[mode]?' active':'');
    button.setAttribute('role','tab');
    button.setAttribute('aria-selected',String(key===activePairs[mode]));
    if(index<9) {{
      button.title='Hotkey '+String(index+1);
      button.setAttribute('aria-keyshortcuts',String(index+1));
    }}
    button.textContent=section.dataset.pairLabel;
    button.addEventListener('click',()=>activatePair(mode,key));
    strip.appendChild(button);
  }});
}}
document.querySelectorAll('.pair-select').forEach(checkbox=>{{
  checkbox.addEventListener('change',()=>{{
    const mode=checkbox.dataset.mode, key=checkbox.dataset.pairKey;
    if(checkbox.checked) selectedPairs[mode].add(key);
    else selectedPairs[mode].delete(key);
    renderPairTabs(mode);
    updateOverview(mode);
    if(document.body.dataset.reportMode===mode) renderActiveDetail(mode);
  }});
}});
document.addEventListener('keydown',event=>{{
  if(event.repeat||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey) return;
  if(!/^[1-9]$/.test(event.key)) return;
  const target=event.target;
  if(target instanceof HTMLElement && (
    target.isContentEditable||['INPUT','TEXTAREA','SELECT'].includes(target.tagName)
  )) return;
  const mode=document.body.dataset.reportMode||'raw';
  const key=orderedSelectedKeys(mode)[Number(event.key)-1];
  if(!key) return;
  event.preventDefault();
  activatePair(mode,key);
}});
function setReportMode(mode) {{
  document.body.dataset.reportMode=mode;
  document.querySelectorAll('[data-mode-panel]').forEach(panel=>{{
    panel.hidden=panel.dataset.modePanel!==mode;
  }});
  document.querySelectorAll('.mode-tab').forEach(button=>{{
    const active=button.id===mode+'-mode-tab';
    button.classList.toggle('active',active);
    button.setAttribute('aria-selected',String(active));
  }});
  document.querySelectorAll('[data-mode-copy]').forEach(item=>{{
    item.style.display=item.dataset.modeCopy===mode?'inline':'none';
  }});
  renderPairTabs(mode);
  renderActiveDetail(mode);
  setOverviewView(mode,overviewViews[mode]);
}}
function copyPeq(button) {{
  navigator.clipboard.writeText(button.parentElement.querySelector('pre').innerText);
  const old=button.innerText; button.innerText='Copied'; setTimeout(()=>button.innerText=old,900);
}}
renderPairTabs('raw');
renderPairTabs('eq');
setReportMode('raw');
</script></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path
