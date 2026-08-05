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
from .dsp import AnalysisContext, EqOptions, pair_diagnostics


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


def _plot_html(figure: go.Figure, div_id: str) -> str:
    return figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"displaylogo": False, "responsive": True},
        div_id=div_id,
    )


def _magnitude_figure(pair: dict[str, Any], data: dict[str, Any]) -> go.Figure:
    f = data["frequencies"]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=f, y=data["solo_first_db"], name=pair["first_name"], line={"dash": "dot"})
    )
    figure.add_trace(
        go.Scatter(x=f, y=data["solo_second_db"], name=pair["second_name"], line={"dash": "dot"})
    )
    figure.add_trace(
        go.Scatter(x=f, y=data["sum_db"], name="Raw sum", line={"color": "#7dd3fc", "width": 1})
    )
    if data["eq_target"] == "flat":
        figure.add_trace(
            go.Scatter(
                x=f,
                y=data["eq_nominal_target_db"],
                name="Nominal flat target",
                line={"color": "#e879f9", "width": 1.2, "dash": "dot"},
                visible="legendonly",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=f,
            y=data["eq_target_db"],
            name="EQ target (range/GD aware)",
            line={"color": "#f0abfc", "width": 1.5, "dash": "longdash"},
            visible="legendonly",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=f,
            y=data["trend_db"],
            name="1-oct trend",
            line={"color": "#fb7185", "width": 2},
            visible="legendonly",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=f,
            y=data["post_eq_db"],
            name="Post-EQ sum",
            line={"color": "#86efac", "width": 1.5, "dash": "dash"},
            visible="legendonly",
        )
    )
    figure.update_layout(
        title="Magnitude: solos and optimised sum",
        xaxis={"type": "log", "title": "Frequency (Hz)"},
        yaxis={"title": "Level (dB; cache reference)"},
        margin={"l": 62, "r": 24, "t": 52, "b": 55},
        legend={"orientation": "h", "y": -0.22},
        template="plotly_dark",
        height=510,
    )
    return figure


def _overview_figure(
    rows: list[tuple[dict[str, Any], dict[str, Any]]], mode: str
) -> go.Figure:
    colors = ["#7dd3fc", "#fbbf24", "#c4b5fd", "#86efac", "#fb7185", "#f9a8d4"]
    figure = go.Figure()
    eq = mode == "eq"
    for index, (pair, data) in enumerate(rows):
        figure.add_trace(
            go.Scatter(
                x=data["frequencies"],
                y=data["post_eq_db" if eq else "sum_db"],
                name=(
                    f"#{pair['eq_rank' if eq else 'rank']} · "
                    f"{pair['first']}+{pair['second']} — "
                    f"{pair['first_name']} + {pair['second_name']}"
                ),
                line={"color": colors[index % len(colors)], "width": 2.2},
            )
        )
    figure.update_layout(
        title="Top pair EQ’d sums" if eq else "Top pair raw sums",
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
            line={"color": "#86efac", "width": 1.5, "dash": "dash"},
            name="EQ authority",
            yaxis="y2",
        )
    )
    figure.add_hline(y=0.0, line={"color": "#64748b", "width": 1})
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
    figure.update_xaxes(title_text="Time from sum peak (ms)")
    figure.update_yaxes(type="log", title_text="Frequency (Hz)", row=1, col=1)
    figure.update_layout(
        title="CSD-style constant-percentage-band decay",
        template="plotly_dark",
        height=480,
        margin={"l": 66, "r": 70, "t": 72, "b": 55},
    )
    return figure


def _peq_text(filters: list[dict[str, float]]) -> str:
    if not filters:
        return "No cuts fitted"
    return "\n".join(
        f"PK Fc {item['fc_hz']:.1f} Hz  Gain {item['gain_db']:.1f} dB  Q {item['q']:.3f}"
        for item in filters
    )


def _ranking_table(pairs: list[dict[str, Any]], mode: str, table_id: str) -> str:
    eq = mode == "eq"
    rank_key = "eq_rank" if eq else "rank"
    null_key = "post_eq_null_score_db" if eq else "null_score_db"
    excess_key = "post_eq_excess_gd_ms" if eq else "excess_gd_ms"
    tail_key = "post_eq_tail_ms" if eq else "raw_tail_ms"
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
        (spl_key, "Relative SPL (dB)", "number"),
    ]
    heading = "".join(
        f'<th data-key="{key}" data-type="{kind}">{html.escape(label)}</th>'
        for key, label, kind in columns
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
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        f'<table id="{table_id}" class="ranking-table"><thead><tr>{heading}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def build_report(
    cache_dir: Path,
    results_path: Path,
    output_path: Path,
    top: int = 5,
) -> Path:
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
        max_filters=int(eq_settings.get("max_filters", 4)),
    )
    required_ranking_fields = {
        "rank",
        "eq_rank",
        "raw_tail_ms",
        "post_eq_null_score_db",
        "post_eq_excess_gd_ms",
        "post_eq_relative_spl_db",
    }
    if int(results.get("format_version", 0)) < 4:
        raise ReportError(
            "Search results predate raw-magnitude scoring and configurable EQ bands; "
            "run 'subpair search' again"
        )
    if any(
        not required_ranking_fields.issubset(pair)
        for pair in results["pairs"]
    ):
        raise ReportError(
            "Search results predate dual raw/EQ ranking; run 'subpair search' again"
        )
    context = AnalysisContext(measurements, band, int(settings["ppo"]))
    raw_pairs = sorted(results["pairs"], key=lambda pair: int(pair["rank"]))
    eq_pairs = sorted(results["pairs"], key=lambda pair: int(pair["eq_rank"]))
    raw_selected = raw_pairs[: max(0, min(top, len(raw_pairs)))]
    eq_selected = eq_pairs[: max(0, min(top, len(eq_pairs)))]
    pair_key = lambda pair: f"{int(pair['first'])}-{int(pair['second'])}"
    selected_keys = {pair_key(pair) for pair in [*raw_selected, *eq_selected]}
    selected = [pair for pair in raw_pairs if pair_key(pair) in selected_keys]
    detail_sections = []
    diagnostic_by_key: dict[str, dict[str, Any]] = {}
    for pair in selected:
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
        diagnostic_by_key[key] = data
        peq = _peq_text(data["filters"])
        detail_sections.append(
            f"""
            <section class="pair-detail" data-pair-key="{key}"
              data-raw-rank="{pair['rank']}" data-eq-rank="{pair['eq_rank']}"
              style="order:{pair['rank']}">
              <h2><span data-mode-copy="raw">#{pair['rank']} Raw</span><span data-mode-copy="eq">#{pair['eq_rank']} EQ’d</span>:
                {html.escape(pair['first_name'])} + {html.escape(pair['second_name'])}</h2>
              <p class="configuration">Sub 2: {'normal' if pair['polarity'] > 0 else 'inverted'},
                delay {pair['delay_ms']:+.3f} ms, gain {pair['gain_db']:+.2f} dB<br>
                <span data-mode-copy="raw">Raw: null {pair['null_score_db']:.3f} dB ·
                excess GD {pair['excess_gd_ms']:.3f} ms · tail {pair['raw_tail_ms']:.1f} ms</span>
                <span data-mode-copy="eq">EQ’d: null {pair['post_eq_null_score_db']:.3f} dB ·
                excess GD {pair['post_eq_excess_gd_ms']:.3f} ms · tail {pair['post_eq_tail_ms']:.1f} ms</span><br>
                EQ: {html.escape(eq_options.target)} target, {eq_range[0]:g}–{eq_range[1]:g} Hz,
                {eq_options.correction_slope_db_per_octave:g} dB/oct curtain,
                max boost {eq_options.max_boost_db:g} dB, up to
                {eq_options.max_filters} PEQ bands; excess-GD guarded</p>
              {_plot_html(_magnitude_figure(pair, data), f'magnitude-{key}')}
              {_plot_html(_excess_figure(data), f'excess-{key}')}
              {_plot_html(_decay_figure(data), f'decay-{key}')}
              <div class="peq"><h3>Fitted PEQ filters</h3><button onclick="copyPeq(this)">Copy</button>
              <pre>{html.escape(peq)}</pre></div>
            </section>
            """
        )

    raw_overview = [
        (pair, diagnostic_by_key[pair_key(pair)]) for pair in raw_selected[:5]
    ]
    eq_overview = [
        (pair, diagnostic_by_key[pair_key(pair)]) for pair in eq_selected[:5]
    ]

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
[hidden] {{ display:none !important; }}
[data-mode-copy="eq"] {{ display:none; }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:12px; background:var(--card); }}
table {{ width:100%; border-collapse:collapse; white-space:nowrap; }}
th,td {{ padding:10px 13px; text-align:right; border-bottom:1px solid var(--line); }}
th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
th {{ position:sticky; top:0; cursor:pointer; background:#142238; color:#c9d8eb; }}
th:hover {{ background:#1b2d48; }} tbody tr:hover {{ background:#132238; }}
.pair-detail {{ margin-top:32px; padding:24px; border:1px solid var(--line); border-radius:14px; background:var(--card); }}
#pair-details {{ display:flex; flex-direction:column; }}
.peq {{ position:relative; padding:16px; background:#091322; border-radius:9px; }}
.peq h3 {{ margin:0 0 10px; }} .peq pre {{ margin:0; white-space:pre-wrap; color:#a7f3d0; }}
.peq button {{ position:absolute; right:14px; top:14px; border:1px solid #49607e; border-radius:6px; padding:6px 11px; color:var(--text); background:#1c2d47; cursor:pointer; }}
details {{ margin:22px 0; }} details pre {{ overflow:auto; color:var(--muted); }}
</style>
<script>{get_plotlyjs()}</script>
</head>
<body><main>
<h1>subpair ranking</h1>
<p class="lede">{results['measurement_count']} positions · {band[0]:g}–{band[1]:g} Hz · {settings['ppo']} points/octave · dual lexicographic ranking</p>
<div class="mode-tabs" role="tablist" aria-label="Ranking response mode">
  <button class="mode-tab active" id="raw-mode-tab" role="tab" aria-selected="true" onclick="setReportMode('raw')">Raw</button>
  <button class="mode-tab" id="eq-mode-tab" role="tab" aria-selected="false" onclick="setReportMode('eq')">EQ’d</button>
</div>
<div data-mode-panel="raw">
  {_plot_html(_overview_figure(raw_overview, 'raw'), 'top-pairs-overview-raw') if raw_overview else ''}
  <p class="note">Raw ranking: raw-magnitude null depth, raw excess group delay, then raw tail.</p>
  <div class="table-wrap">{_ranking_table(raw_pairs, 'raw', 'ranking-raw')}</div>
</div>
<div data-mode-panel="eq" hidden>
  {_plot_html(_overview_figure(eq_overview, 'eq'), 'top-pairs-overview-eq') if eq_overview else ''}
  <p class="note">EQ’d ranking: post-EQ raw-magnitude null depth, post-EQ excess group delay, then post-EQ tail.</p>
  <div class="table-wrap">{_ranking_table(eq_pairs, 'eq', 'ranking-eq')}</div>
</div>
<p class="note">Click a table heading to sort. Metric cells run from green (best) to red (worst); lower is better except relative SPL, where higher is better. Each mode references its own rank 1 for relative SPL.</p>
<details><summary>Analysis settings and minimum-phase convention</summary><pre>{settings_json}</pre></details>
<div id="pair-details">{''.join(detail_sections)}</div>
</main>
<script>
document.querySelectorAll('.ranking-table th').forEach((th,index)=>{{
  th.addEventListener('click',()=>{{
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
const reportTop={int(top)};
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
  document.querySelectorAll('.pair-detail').forEach(section=>{{
    const rank=Number(mode==='eq'?section.dataset.eqRank:section.dataset.rawRank);
    section.hidden=rank>reportTop;
    section.style.order=String(rank);
    const plot=document.getElementById('magnitude-'+section.dataset.pairKey);
    if(plot && plot.data) {{
      const rawIndex=plot.data.findIndex(trace=>trace.name==='Raw sum');
      const eqIndex=plot.data.findIndex(trace=>trace.name==='Post-EQ sum');
      if(rawIndex>=0) Plotly.restyle(plot,{{visible:mode==='raw'?true:'legendonly'}},[rawIndex]);
      if(eqIndex>=0) Plotly.restyle(plot,{{visible:mode==='eq'?true:'legendonly'}},[eqIndex]);
    }}
  }});
  window.dispatchEvent(new Event('resize'));
}}
function copyPeq(button) {{
  navigator.clipboard.writeText(button.parentElement.querySelector('pre').innerText);
  const old=button.innerText; button.innerText='Copied'; setTimeout(()=>button.innerText=old,900);
}}
setReportMode('raw');
</script></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path
