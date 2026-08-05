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
from .dsp import AnalysisContext, pair_diagnostics


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
    figure.add_trace(
        go.Scatter(
            x=f,
            y=data["smoothed_db"],
            name="Variable smoothed",
            line={"color": "#fbbf24", "width": 2},
        )
    )
    figure.add_trace(
        go.Scatter(x=f, y=data["trend_db"], name="1-oct trend", line={"color": "#fb7185", "width": 2})
    )
    figure.add_trace(
        go.Scatter(
            x=f,
            y=data["post_eq_db"],
            name="Post-EQ sum",
            line={"color": "#86efac", "width": 1.5, "dash": "dash"},
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


def _excess_figure(data: dict[str, Any]) -> go.Figure:
    figure = go.Figure(
        go.Scatter(
            x=data["frequencies"],
            y=data["excess_curve_ms"],
            line={"color": "#c4b5fd", "width": 2},
            name="Excess GD",
        )
    )
    figure.add_hline(y=0.0, line={"color": "#64748b", "width": 1})
    figure.update_layout(
        title="Excess group delay (constant delay removed)",
        xaxis={"type": "log", "title": "Frequency (Hz)"},
        yaxis={"title": "Excess GD (ms)"},
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
        subplot_titles=("Pre-EQ", "Post-EQ (cuts only)"),
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


def _ranking_table(pairs: list[dict[str, Any]]) -> str:
    columns = [
        ("rank", "Rank", "number"),
        ("pair", "Pair", "text"),
        ("polarity", "Pol 2", "number"),
        ("delay_ms", "Delay 2 (ms)", "number"),
        ("gain_db", "Gain 2 (dB)", "number"),
        ("null_score_db", "Worst null (dB)", "number"),
        ("excess_gd_ms", "Excess GD (ms)", "number"),
        ("post_eq_tail_ms", "Post-EQ tail (ms)", "number"),
        ("relative_spl_db", "Relative SPL (dB)", "number"),
    ]
    heading = "".join(
        f'<th data-key="{key}" data-type="{kind}">{html.escape(label)}</th>'
        for key, label, kind in columns
    )
    rows = []
    for pair in pairs:
        values: dict[str, tuple[str, str]] = {
            "rank": (str(pair["rank"]), str(pair["rank"])),
            "pair": (
                f"{pair['first']} + {pair['second']}",
                f"{pair['first']:04d}-{pair['second']:04d}",
            ),
            "polarity": ("+" if pair["polarity"] > 0 else "−", str(pair["polarity"])),
            "delay_ms": (f"{pair['delay_ms']:+.3f}", str(pair["delay_ms"])),
            "gain_db": (f"{pair['gain_db']:+.2f}", str(pair["gain_db"])),
            "null_score_db": (f"{pair['null_score_db']:.3f}", str(pair["null_score_db"])),
            "excess_gd_ms": (f"{pair['excess_gd_ms']:.3f}", str(pair["excess_gd_ms"])),
            "post_eq_tail_ms": (f"{pair['post_eq_tail_ms']:.1f}", str(pair["post_eq_tail_ms"])),
            "relative_spl_db": (f"{pair['relative_spl_db']:+.2f}", str(pair["relative_spl_db"])),
        }
        cells = "".join(
            f'<td data-value="{html.escape(values[key][1])}">{html.escape(values[key][0])}</td>'
            for key, _, _ in columns
        )
        rows.append(f"<tr>{cells}</tr>")
    return f'<table id="ranking"><thead><tr>{heading}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


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
    context = AnalysisContext(measurements, band, int(settings["ppo"]))
    selected = results["pairs"][: max(0, min(top, len(results["pairs"])))]
    detail_sections = []
    for pair in selected:
        data = pair_diagnostics(
            context,
            int(pair["first"]) - 1,
            int(pair["second"]) - 1,
            int(pair["polarity"]),
            float(pair["delay_ms"]),
            float(pair["gain_db"]),
            include_decay=True,
        )
        rank = int(pair["rank"])
        peq = _peq_text(data["filters"])
        detail_sections.append(
            f"""
            <section class="pair-detail">
              <h2>#{rank}: {html.escape(pair['first_name'])} + {html.escape(pair['second_name'])}</h2>
              <p class="configuration">Sub 2: {'normal' if pair['polarity'] > 0 else 'inverted'},
                delay {pair['delay_ms']:+.3f} ms, gain {pair['gain_db']:+.2f} dB ·
                null {pair['null_score_db']:.3f} dB · excess GD {pair['excess_gd_ms']:.3f} ms ·
                post-EQ tail {pair['post_eq_tail_ms']:.1f} ms</p>
              {_plot_html(_magnitude_figure(pair, data), f'magnitude-{rank}')}
              {_plot_html(_excess_figure(data), f'excess-{rank}')}
              {_plot_html(_decay_figure(data), f'decay-{rank}')}
              <div class="peq"><h3>Fitted PEQ cuts</h3><button onclick="copyPeq(this)">Copy</button>
              <pre>{html.escape(peq)}</pre></div>
            </section>
            """
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
.table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:12px; background:var(--card); }}
table {{ width:100%; border-collapse:collapse; white-space:nowrap; }}
th,td {{ padding:10px 13px; text-align:right; border-bottom:1px solid var(--line); }}
th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
th {{ position:sticky; top:0; cursor:pointer; background:#142238; color:#c9d8eb; }}
th:hover {{ background:#1b2d48; }} tbody tr:hover {{ background:#132238; }}
.pair-detail {{ margin-top:32px; padding:24px; border:1px solid var(--line); border-radius:14px; background:var(--card); }}
.peq {{ position:relative; padding:16px; background:#091322; border-radius:9px; }}
.peq h3 {{ margin:0 0 10px; }} .peq pre {{ margin:0; white-space:pre-wrap; color:#a7f3d0; }}
.peq button {{ position:absolute; right:14px; top:14px; border:1px solid #49607e; border-radius:6px; padding:6px 11px; color:var(--text); background:#1c2d47; cursor:pointer; }}
details {{ margin:22px 0; }} details pre {{ overflow:auto; color:var(--muted); }}
</style>
<script>{get_plotlyjs()}</script>
</head>
<body><main>
<h1>subpair ranking</h1>
<p class="lede">{results['measurement_count']} positions · {band[0]:g}–{band[1]:g} Hz · {settings['ppo']} points/octave · lexicographic ranking</p>
<p class="note">Click a table heading to sort. Relative SPL is referenced to rank 1 and is not a ranking input.</p>
<div class="table-wrap">{_ranking_table(results['pairs'])}</div>
<details><summary>Analysis settings and minimum-phase convention</summary><pre>{settings_json}</pre></details>
{''.join(detail_sections)}
</main>
<script>
document.querySelectorAll('#ranking th').forEach((th,index)=>{{
  th.addEventListener('click',()=>{{
    const body=document.querySelector('#ranking tbody');
    const rows=Array.from(body.querySelectorAll('tr'));
    const ascending=th.dataset.order!=='asc';
    document.querySelectorAll('#ranking th').forEach(x=>delete x.dataset.order);
    th.dataset.order=ascending?'asc':'desc';
    rows.sort((a,b)=>{{
      let av=a.children[index].dataset.value, bv=b.children[index].dataset.value;
      if(th.dataset.type==='number') {{ av=Number(av); bv=Number(bv); }}
      return (av<bv?-1:av>bv?1:0)*(ascending?1:-1);
    }});
    rows.forEach(row=>body.appendChild(row));
  }});
}});
function copyPeq(button) {{
  navigator.clipboard.writeText(button.parentElement.querySelector('pre').innerText);
  const old=button.innerText; button.innerText='Copied'; setTimeout(()=>button.innerText=old,900);
}}
</script></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path
