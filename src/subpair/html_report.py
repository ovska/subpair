"""Self-contained Plotly report generation."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from .cache import load_cache
from .dsp import AnalysisContext, EqOptions, pair_diagnostics


# Hover tooltip number formats, shared across every chart: dB/ms values get
# one decimal place, Hz and percentage values are shown as full/whole
# numbers - explicit templates rather than Plotly's own default formatting,
# which varies by trace and is often needlessly precise.
_HOVER_HZ_DB = "%{x:.0f} Hz<br>%{y:.1f} dB<extra></extra>"
_HOVER_EQ_BAND = (
    "%{customdata}<br>Fc %{x:.1f} Hz<br>Gain %{y:+.1f} dB<extra></extra>"
)
_HOVER_HZ_MS = "%{x:.0f} Hz<br>%{y:.1f} ms<extra></extra>"
_HOVER_HZ_PERCENT = "%{x:.0f} Hz<br>%{y:.0f}%<extra></extra>"
_HOVER_MS_HZ_DB = "%{x:.1f} ms<br>%{y:.0f} Hz<br>%{z:.1f} dB<extra></extra>"
_HOVER_HZ_MS_OVERLAY = "%{y:.0f} Hz · %{x:.1f} ms<extra></extra>"
_EXCESS_GD_LOWER_LIMIT_MS = -20.0


class ReportError(RuntimeError):
    pass


def _eq_band_points(
    data: dict[str, Any],
) -> tuple[list[float], list[float], list[str]]:
    """Return configured frequency/gain coordinates for fitted EQ bands."""

    frequencies: list[float] = []
    gains: list[float] = []
    labels: list[str] = []
    for item in data.get("filters", []):
        frequencies.append(float(item["fc_hz"]))
        gains.append(float(item["gain_db"]))
        labels.append(f"PK band · Q {float(item['q']):.3f}")
    shelf = data.get("eq_shelf")
    if shelf and shelf.get("active"):
        frequencies.append(float(shelf["freq_hz"]))
        gains.append(float(shelf["gain_db"]))
        labels.append(f"LS band · slope {float(shelf['slope']):.2f}")
    return frequencies, gains, labels


def _finite_axis_range(
    series: list[Any],
    *,
    lower_limit: float | None = None,
    include_zero: bool = False,
) -> tuple[float, float] | None:
    """Return finite extrema suitable for a shared Plotly axis."""

    low = 0.0 if include_zero else np.inf
    high = 0.0 if include_zero else -np.inf
    for values in series:
        array = np.asarray(values, dtype=np.float64)
        finite = array[np.isfinite(array)]
        if finite.size:
            low = min(low, float(np.min(finite)))
            high = max(high, float(np.max(finite)))
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    if lower_limit is not None:
        low = max(float(lower_limit), low)
    if high <= low:
        high = low + max(1.0, abs(low) * 0.05)
    return low, high


def _diagnostic_axis_ranges(
    data: dict[str, Any], *, raw: bool
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Bounds for all traces that share an axis in one pair diagnostic."""

    magnitude_series = [
        data["solo_first_db"],
        data["solo_second_db"],
        data["sum_db" if raw else "post_eq_db"],
    ]
    if not raw:
        magnitude_series.append(
            np.asarray(data["post_eq_db"]) - np.asarray(data["sum_db"])
        )
        _, band_gains, _ = _eq_band_points(data)
        magnitude_series.append(band_gains)
    magnitude = _finite_axis_range(magnitude_series, include_zero=not raw)
    excess = _finite_axis_range(
        [data["excess_curve_ms" if raw else "post_eq_excess_curve_ms"]],
        lower_limit=_EXCESS_GD_LOWER_LIMIT_MS,
        include_zero=True,
    )
    return magnitude, excess


def _selected_axis_ranges(
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    raw: bool,
    selected_keys: set[str],
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Combine per-pair bounds so selected diagnostics use identical axes."""

    magnitude_bounds: list[tuple[float, float]] = []
    excess_bounds: list[tuple[float, float]] = []
    for pair, data in rows:
        key = f"{int(pair['first'])}-{int(pair['second'])}"
        if key not in selected_keys:
            continue
        magnitude, excess = _diagnostic_axis_ranges(data, raw=raw)
        if magnitude is not None:
            magnitude_bounds.append(magnitude)
        if excess is not None:
            excess_bounds.append(excess)
    magnitude = _finite_axis_range(magnitude_bounds) if magnitude_bounds else None
    excess = (
        _finite_axis_range(
            excess_bounds,
            lower_limit=_EXCESS_GD_LOWER_LIMIT_MS,
            include_zero=True,
        )
        if excess_bounds
        else None
    )
    return magnitude, excess


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


def _magnitude_figure(
    pair: dict[str, Any],
    data: dict[str, Any],
    *,
    raw: bool = False,
    y_range: tuple[float, float] | None = None,
) -> go.Figure:
    f = data["frequencies"]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=f,
            y=data["solo_first_db"],
            name=pair["first_name"],
            line={"color": "#fb7185", "width": 1.4},
            hovertemplate=_HOVER_HZ_DB,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=f,
            y=data["solo_second_db"],
            name=pair["second_name"],
            line={"color": "#fbbf24", "width": 1.4},
            hovertemplate=_HOVER_HZ_DB,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=f,
            y=data["sum_db"],
            name="Raw sum",
            line={"color": "#7dd3fc", "width": 1},
            visible=True if raw else "legendonly",
            hovertemplate=_HOVER_HZ_DB,
        )
    )
    if not raw:
        figure.add_trace(
            go.Scatter(
                x=f,
                y=data["eq_target_db"],
                name="EQ target (range/GD aware)",
                line={"color": "#e879f9", "width": 1.5},
                visible="legendonly",
                hovertemplate=_HOVER_HZ_DB,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=f,
                y=data["post_eq_db"],
                name="Post-EQ sum",
                line={"color": "#86efac", "width": 1.7},
                hovertemplate=_HOVER_HZ_DB,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=f,
                y=np.asarray(data["post_eq_db"]) - np.asarray(data["sum_db"]),
                name="Combined EQ response (all bands)",
                line={"color": "#c4b5fd", "width": 1.7},
                visible="legendonly",
                legendgroup="eq-bands",
                hovertemplate=_HOVER_HZ_DB,
            )
        )
        band_frequencies, band_gains, band_labels = _eq_band_points(data)
        if band_frequencies:
            figure.add_trace(
                go.Scatter(
                    x=band_frequencies,
                    y=band_gains,
                    customdata=band_labels,
                    name="EQ band settings",
                    mode="markers",
                    marker={
                        "color": "#ddd6fe",
                        "line": {"color": "#6d28d9", "width": 1},
                        "size": 7,
                    },
                    visible="legendonly",
                    legendgroup="eq-bands",
                    showlegend=False,
                    hovertemplate=_HOVER_EQ_BAND,
                )
            )
    yaxis: dict[str, Any] = {"title": "Level (dB; cache reference)"}
    if y_range is not None:
        yaxis["range"] = y_range
    layout: dict[str, Any] = {
        "title": (
            "Magnitude: solos and raw sum" if raw else "Magnitude: solos and EQ’d sum"
        ),
        "xaxis": {"type": "log", "title": "Frequency (Hz)"},
        "yaxis": yaxis,
        "margin": {"l": 62, "r": 70, "t": 52, "b": 55},
        "legend": {
            "orientation": "h",
            "y": -0.22,
            "groupclick": "togglegroup",
        },
        "template": "plotly_dark",
        "height": 510,
    }
    figure.update_layout(**layout)
    return figure


def _overview_figure(
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
    mode: str,
    selected_keys: set[str],
    y_range: tuple[float, float] | None = None,
) -> go.Figure:
    figure = go.Figure()
    eq = mode == "eq"
    score_key = "post_eq_relative_score_db" if eq else "relative_score_db"
    for index, (pair, data) in enumerate(rows):
        key = f"{int(pair['first'])}-{int(pair['second'])}"
        figure.add_trace(
            go.Scatter(
                x=data["frequencies"],
                y=data["post_eq_db" if eq else "sum_db"],
                name=(
                    f"Score {pair[score_key]:+.2f} dB · "
                    f"{pair['first']}+{pair['second']} — "
                    f"{pair['first_name']} + {pair['second_name']}"
                ),
                line={
                    "color": f"hsl({(index * 137.508) % 360:.1f},72%,67%)",
                    "width": 2.2,
                },
                meta={"pair_key": key},
                visible=key in selected_keys,
                hovertemplate=_HOVER_HZ_DB,
            )
        )
    if y_range is None:
        y_range = _finite_axis_range(
            [
                data["post_eq_db" if eq else "sum_db"]
                for pair, data in rows
                if f"{int(pair['first'])}-{int(pair['second'])}" in selected_keys
            ]
        )
    yaxis: dict[str, Any] = {
        "title": (
            "Post-EQ summed level (dB; cache reference)"
            if eq
            else "Raw summed level (dB; cache reference)"
        )
    }
    if y_range is not None:
        yaxis["range"] = y_range
    figure.update_layout(
        title="Selected pair EQ’d sums" if eq else "Selected pair raw sums",
        xaxis={"type": "log", "title": "Frequency (Hz)"},
        yaxis=yaxis,
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
    y_range: tuple[float, float] | None = None,
) -> go.Figure:
    figure = go.Figure()
    eq = mode == "eq"
    score_key = "post_eq_relative_score_db" if eq else "relative_score_db"
    for index, (pair, data) in enumerate(rows):
        key = f"{int(pair['first'])}-{int(pair['second'])}"
        figure.add_trace(
            go.Scatter(
                x=data["frequencies"],
                y=data["post_eq_excess_curve_ms" if eq else "excess_curve_ms"],
                name=(
                    f"Score {pair[score_key]:+.2f} dB · "
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
                hovertemplate=_HOVER_HZ_MS,
            )
        )
    figure.add_hline(y=0.0, line={"color": "#64748b", "width": 1})
    if y_range is None:
        y_range = _finite_axis_range(
            [
                data["post_eq_excess_curve_ms" if eq else "excess_curve_ms"]
                for pair, data in rows
                if f"{int(pair['first'])}-{int(pair['second'])}" in selected_keys
            ],
            lower_limit=_EXCESS_GD_LOWER_LIMIT_MS,
            include_zero=True,
        )
    yaxis: dict[str, Any] = {
        "title": "Excess GD (ms)",
        "minallowed": _EXCESS_GD_LOWER_LIMIT_MS,
    }
    if y_range is not None:
        yaxis["range"] = y_range
    figure.update_layout(
        title=(
            "Selected pair post-EQ excess group delay"
            if eq
            else "Selected pair raw excess group delay"
        ),
        xaxis={"type": "log", "title": "Frequency (Hz)"},
        yaxis=yaxis,
        margin={"l": 62, "r": 24, "t": 52, "b": 55},
        legend={"orientation": "h", "y": -0.22},
        template="plotly_dark",
        height=540,
    )
    return figure


def _excess_figure(
    data: dict[str, Any],
    *,
    raw: bool = False,
    y_range: tuple[float, float] | None = None,
) -> go.Figure:
    figure = go.Figure()
    prefix = "" if raw else "post_eq_"
    label = "Raw" if raw else "Post-EQ"
    figure.add_trace(
        go.Scatter(
            x=data["frequencies"],
            y=data[f"{prefix}excess_curve_ms"],
            line={"color": "#c4b5fd", "width": 2, "shape": "spline", "smoothing": 1.0},
            name=f"{label} excess GD",
            hovertemplate=_HOVER_HZ_MS,
        )
    )
    if not raw:
        figure.add_trace(
            go.Scatter(
                x=data["frequencies"],
                y=100.0 * np.asarray(data["eq_authority"]),
                line={"color": "#86efac", "width": 1.5},
                name="EQ authority",
                yaxis="y2",
                hovertemplate=_HOVER_HZ_PERCENT,
            )
        )
    figure.add_hline(y=0.0, line={"color": "#64748b", "width": 1})
    yaxis: dict[str, Any] = {
        "title": "Excess GD (ms)",
        "minallowed": _EXCESS_GD_LOWER_LIMIT_MS,
    }
    if y_range is not None:
        yaxis["range"] = y_range
    layout = {
        "title": f"{label} excess group delay (display spline; diagnostic only)",
        "xaxis": {"type": "log", "title": "Frequency (Hz)"},
        "yaxis": yaxis,
        "margin": {"l": 62, "r": 24, "t": 52, "b": 55},
        "template": "plotly_dark",
        "height": 390,
    }
    if not raw:
        layout["yaxis2"] = {
            "title": "EQ authority (%)",
            "overlaying": "y",
            "side": "right",
            "range": [0, 105],
        }
    figure.update_layout(**layout)
    return figure


def _decay_figure(data: dict[str, Any], *, raw: bool = False) -> go.Figure:
    figure = go.Figure()
    label = "Raw" if raw else "Post-EQ"
    figure.add_trace(
        go.Heatmap(
            x=1000.0 * np.asarray(data["decay_times"]),
            y=data["decay_frequencies"],
            z=data["pre_decay_db" if raw else "post_decay_db"],
            zmin=-40,
            zmax=0,
            colorscale="Turbo",
            colorbar={"title": "dB"},
            hovertemplate=_HOVER_MS_HZ_DB,
        )
    )
    overlay_common = {
        "mode": "lines",
        "line": {
            "color": "#f8fafc",
            "width": 2.5,
            "shape": "spline",
            "smoothing": 1.0,
        },
        "hovertemplate": _HOVER_HZ_MS_OVERLAY,
    }
    figure.add_trace(
        go.Scatter(
            x=data["excess_curve_ms" if raw else "post_eq_excess_curve_ms"],
            y=data["frequencies"],
            name=f"{label} excess GD",
            line=overlay_common["line"],
            mode=overlay_common["mode"],
            hovertemplate=overlay_common["hovertemplate"],
        ),
    )
    figure.add_vline(x=0.0, line={"color": "#94a3b8", "width": 1})
    figure.update_xaxes(title_text="Time from sum peak (ms)")
    figure.update_yaxes(type="log", title_text="Frequency (Hz)")
    figure.update_layout(
        title=f"{label} CSD-style decay with zero-referenced excess-GD overlay",
        template="plotly_dark",
        height=520,
        legend={"orientation": "h", "y": -0.18},
        margin={"l": 66, "r": 70, "t": 72, "b": 82},
    )
    return figure


def _peq_text(
    filters: list[dict[str, float]],
    shelf: dict[str, Any] | None = None,
    headroom_db: float = 0.0,
) -> str:
    filter_lines = [
        f"PK Fc {item['fc_hz']:.1f} Hz  Gain {item['gain_db']:.1f} dB  Q {item['q']:.3f}"
        for item in filters
    ]
    shelf_active = shelf is not None and shelf.get("active")
    if shelf_active:
        filter_lines.append(
            f"LS Fc {shelf['freq_hz']:.1f} Hz  Gain {shelf['gain_db']:+.1f} dB  "
            f"Slope {shelf['slope']:.2f}  (automatically fitted EQ band)"
        )
    lines = [f"Preamp {headroom_db:+.1f} dB  (equal-drive headroom)"]
    lines.extend(filter_lines or ["No filters fitted"])
    return "\n".join(lines)


def _ranking_table(
    pairs: list[dict[str, Any]],
    mode: str,
    table_id: str,
    selected_keys: set[str],
) -> str:
    eq = mode == "eq"
    score_key = "post_eq_relative_score_db" if eq else "relative_score_db"
    dip_key = "post_eq_dip_db" if eq else "dip_db"
    excess_key = "post_eq_excess_gd_ms" if eq else "excess_gd_ms"
    tail_key = "post_eq_tail_ms" if eq else "raw_tail_ms"
    headroom_key = "post_eq_headroom_db" if eq else "headroom_db"
    low_end_power_key = (
        "post_eq_relative_low_end_power_db" if eq else "relative_low_end_power_db"
    )
    spl_key = "post_eq_relative_spl_db" if eq else "relative_spl_db"
    columns = [
        (score_key, "Score (dB)", "number"),
        ("pair", "Pair", "text"),
        ("polarity", "Pol 2", "number"),
        ("delay_ms", "Delay 2 (ms)", "number"),
        ("gain_db", "Gain 2 (dB)", "number"),
        (headroom_key, "Headroom (dB)", "number"),
        (dip_key, "Residual dip (dB)", "number"),
        (excess_key, "Excess GD (ms)", "number"),
        (tail_key, "Tail (ms)", "number"),
        (low_end_power_key, "Low-end power (dB)", "number"),
        (spl_key, "Relative SPL (dB)", "number"),
    ]
    heading = '<th class="selection-heading">Show</th>' + "".join(
        f'<th data-key="{key}" data-type="{kind}" data-column-index="{index + 1}">'
        f'{html.escape(label)}</th>'
        for index, (key, label, kind) in enumerate(columns)
    )
    metric_directions = {
        score_key: "high",
        dip_key: "low",
        excess_key: "low",
        tail_key: "low",
        low_end_power_key: "high",
        spl_key: "high",
    }
    # Keep missing future metrics out of the colour-scaled range rather than
    # coercing them to numbers that could skew the best/worst endpoints.
    metric_ranges: dict[str, tuple[float, float]] = {}
    for key in metric_directions:
        numeric = [float(pair[key]) for pair in pairs if pair[key] is not None]
        metric_ranges[key] = (min(numeric), max(numeric)) if numeric else (0.0, 0.0)

    def score_style(key: str, value: float | None) -> str:
        if key not in metric_directions or value is None:
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
            score_key: (f"{pair[score_key]:+.2f}", str(pair[score_key])),
            "pair": (
                f"{pair['first']} + {pair['second']}",
                f"{pair['first']:04d}-{pair['second']:04d}",
            ),
            "polarity": ("+" if pair["polarity"] > 0 else "−", str(pair["polarity"])),
            "delay_ms": (f"{pair['delay_ms']:+.3f}", str(pair["delay_ms"])),
            "gain_db": (f"{pair['gain_db']:+.2f}", str(pair["gain_db"])),
            headroom_key: (
                f"{pair[headroom_key]:+.2f}",
                str(pair[headroom_key]),
            ),
            dip_key: (f"{pair[dip_key]:.3f}", str(pair[dip_key])),
            excess_key: (f"{pair[excess_key]:.3f}", str(pair[excess_key])),
            tail_key: (f"{pair[tail_key]:.1f}", str(pair[tail_key])),
            low_end_power_key: (
                f"{pair[low_end_power_key]:+.2f}",
                str(pair[low_end_power_key]),
            ),
            spl_key: (f"{pair[spl_key]:+.2f}", str(pair[spl_key])),
        }
        cells = []
        for key, _, _ in columns:
            is_metric = key in metric_directions
            raw_value = pair.get(key) if is_metric else None
            numeric_value = float(raw_value) if raw_value is not None else None
            style = score_style(key, numeric_value) if is_metric else ""
            empty_class = " is-empty" if is_metric and numeric_value is None else ""
            style_attribute = f' style="{style}"' if style else ""
            cells.append(
                f'<td class="metric-cell{empty_class}" '
                f'data-value="{html.escape(values[key][1])}"{style_attribute}>'
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
    raw: bool = False,
) -> Path:
    if limit < 1:
        raise ReportError("Report result limit must be at least 1")
    results = load_results(results_path)
    measurements, _ = load_cache(cache_dir)
    if len(measurements) != int(results.get("measurement_count", -1)):
        raise ReportError("Cache measurement count does not match the search results")
    settings = results["settings"]
    band = tuple(float(value) for value in settings["band_hz"])
    eq_settings = settings.get("eq", {})
    eq_range = tuple(float(value) for value in eq_settings.get("correction_range_hz", band))
    shelf_settings = eq_settings.get("shelf", {})
    eq_options = EqOptions(
        target=str(eq_settings.get("target", "trend")),
        correction_range=eq_range,
        correction_slope_db_per_octave=float(
            eq_settings.get("correction_slope_db_per_octave", 48.0)
        ),
        max_boost_db=float(eq_settings.get("max_boost_db", 0.0)),
        max_cut_db=float(eq_settings.get("max_cut_db", 18.0)),
        max_filters=int(eq_settings.get("max_filters", 7)),
        low_shelf=bool(shelf_settings.get("enabled", True)),
    )
    required_ranking_fields = {
        "rank",
        "eq_rank",
        "score_db",
        "relative_score_db",
        "post_eq_score_db",
        "post_eq_relative_score_db",
        "dip_db",
        "post_eq_dip_db",
        "raw_tail_ms",
        "post_eq_excess_gd_ms",
        "post_eq_relative_spl_db",
        "eq_filter_count",
        "eq_shelf",
        "headroom_db",
        "post_eq_headroom_db",
        "low_end_power_db",
        "relative_low_end_power_db",
        "post_eq_low_end_power_db",
        "post_eq_relative_low_end_power_db",
    }
    if int(results.get("format_version", 0)) < 6:
        raise ReportError(
            "Search results predate the width-invariant excess-GD peak "
            "tie-break; run 'subpair search' again"
        )
    if int(results.get("format_version", 0)) < 14:
        raise ReportError(
            "Search results predate the current F3/F6 low-end extension "
            "fields; run 'subpair search' again"
        )
    if int(results.get("format_version", 0)) < 15:
        raise ReportError(
            "Search results predate removal of the monotonic GD baseline "
            "mode; run 'subpair search' again"
        )
    if int(results.get("format_version", 0)) < 16:
        raise ReportError(
            "Search results predate automatic low-shelf EQ fitting; "
            "run 'subpair search' again"
        )
    if int(results.get("format_version", 0)) < 17:
        raise ReportError(
            "Search results predate shared-reference F3/F6 extension; "
            "run 'subpair search' again"
        )
    if int(results.get("format_version", 0)) < 18:
        raise ReportError(
            "Search results predate excursion-weighted low-end power; "
            "run 'subpair search' again"
        )
    if int(results.get("format_version", 0)) < 19:
        raise ReportError(
            "Search results predate response-wide headroom normalization; "
            "run 'subpair search' again"
        )
    if int(results.get("format_version", 0)) < 20:
        raise ReportError(
            "Search results predate usable-output scoring; "
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
    mode = "raw" if raw else "eq"
    mode_label = "Raw" if raw else "EQ’d"
    rank_key = "rank" if raw else "eq_rank"
    score_key = "relative_score_db" if raw else "post_eq_relative_score_db"
    pairs = sorted(results["pairs"], key=lambda pair: -float(pair[score_key]))[:limit]
    score_settings = settings.get("ranking", {}).get("score", {})
    score_low_end_weight = float(score_settings.get("low_end_weight", 0.5))
    score_dip_weight = float(score_settings.get("dip_weight", 1.0))

    def pair_key(pair: dict[str, Any]) -> str:
        return f"{int(pair['first'])}-{int(pair['second'])}"

    default_count = max(0, min(top, len(pairs)))
    default_keys = {pair_key(pair) for pair in pairs[:default_count]}
    initial_active_key = pair_key(pairs[0]) if default_count else None
    diagnostic_by_key: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        diagnostic_by_key[pair_key(pair)] = pair_diagnostics(
            context,
            int(pair["first"]) - 1,
            int(pair["second"]) - 1,
            int(pair["polarity"]),
            float(pair["delay_ms"]),
            float(pair["gain_db"]),
            include_decay=True,
            eq_options=eq_options,
            score_low_end_weight=score_low_end_weight,
            score_dip_weight=score_dip_weight,
        )

    overview = [(pair, diagnostic_by_key[pair_key(pair)]) for pair in pairs]
    magnitude_range, excess_range = _selected_axis_ranges(
        overview,
        raw=raw,
        selected_keys=default_keys,
    )
    detail_sections = []
    for pair in pairs:
        key = pair_key(pair)
        data = diagnostic_by_key[key]
        pair_magnitude_range, pair_excess_range = _diagnostic_axis_ranges(data, raw=raw)
        axis_attributes = ""
        if pair_magnitude_range is not None:
            axis_attributes += (
                f' data-magnitude-min="{pair_magnitude_range[0]:.17g}"'
                f' data-magnitude-max="{pair_magnitude_range[1]:.17g}"'
            )
        if pair_excess_range is not None:
            axis_attributes += (
                f' data-excess-min="{pair_excess_range[0]:.17g}"'
                f' data-excess-max="{pair_excess_range[1]:.17g}"'
            )
        detail_class = (
            "pair-detail" if key == initial_active_key else "pair-detail is-inactive"
        )
        metric_summary = (
            f"Raw score {pair['relative_score_db']:+.2f} dB · "
            f"residual dip {pair['dip_db']:.3f} dB · "
            f"excess GD {pair['excess_gd_ms']:.3f} ms · "
            f"peak {pair['excess_gd_peak_ms']:.2f} ms · "
            f"tail {pair['raw_tail_ms']:.1f} ms"
            if raw
            else (
                f"EQ’d score {pair['post_eq_relative_score_db']:+.2f} dB · "
                f"residual dip {pair['post_eq_dip_db']:.3f} dB · "
                f"excess GD {pair['post_eq_excess_gd_ms']:.3f} ms · "
                f"peak {pair['post_eq_excess_gd_peak_ms']:.2f} ms · "
                f"tail {pair['post_eq_tail_ms']:.1f} ms"
            )
        )
        eq_description = ""
        peq_html = ""
        if not raw:
            eq_description = (
                f"EQ: {html.escape(eq_options.target)} target, "
                f"{eq_range[0]:g}–{eq_range[1]:g} Hz, "
                f"{eq_options.correction_slope_db_per_octave:g} dB/oct curtain, "
                f"max boost {eq_options.max_boost_db:g} dB, up to "
                f"{eq_options.max_filters} EQ bands; excess-GD guarded.<br>"
            )
            if eq_options.low_shelf:
                eq_description += (
                    "Automatic low shelf enabled: its corner and gain/cut "
                    "compete with PK filters and consume one EQ-band slot "
                    "when selected.<br>"
                )
            peq = _peq_text(
                data["filters"],
                data.get("eq_shelf"),
                float(data["post_eq_headroom_db"]),
            )
            peq_html = (
                '<div class="peq"><h3>Headroom + Fitted EQ filters</h3>'
                '<button onclick="copyPeq(this)">Copy</button>'
                f"<pre>{html.escape(peq)}</pre></div>"
            )
        detail_sections.append(
            f"""
            <section class="{detail_class}" data-pair-key="{key}"
              data-pair-label="{pair['first']}+{pair['second']}"
              data-rank="{pair[rank_key]}"{axis_attributes}>
              <h2>{mode_label} score {pair[score_key]:+.2f} dB:
                {html.escape(pair['first_name'])} + {html.escape(pair['second_name'])}</h2>
              <p class="configuration">Sub 2: {'normal' if pair['polarity'] > 0 else 'inverted'},
                delay {pair['delay_ms']:+.3f} ms, gain {pair['gain_db']:+.2f} dB,
                headroom {pair['post_eq_headroom_db' if not raw else 'headroom_db']:+.2f} dB<br>
                {metric_summary}<br>
                {eq_description}
                CSD overlay: excess GD with common delay removed; a vertical line is frequency-independent delay.</p>
              {_plot_html(
                  _magnitude_figure(pair, data, raw=raw, y_range=magnitude_range),
                  f'magnitude-{key}',
              )}
              {_plot_html(_decay_figure(data, raw=raw), f'decay-{key}', static=True)}
              {_plot_html(_excess_figure(data, raw=raw, y_range=excess_range), f'excess-{key}')}
              {peq_html}
            </section>
            """.strip()
        )

    default_json = json.dumps(
        [pair_key(pair) for pair in pairs[:default_count]], separators=(",", ":")
    )
    copy_peq_script = "" if raw else """
function copyPeq(button) {
  navigator.clipboard.writeText(button.parentElement.querySelector('pre').innerText);
  const old=button.innerText; button.innerText='Copied'; setTimeout(()=>button.innerText=old,900);
}
""".strip()

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
.chart-tabs,.pair-tabs {{ display:flex; flex-wrap:wrap; gap:6px; margin:14px 0; }}
.chart-tab,.pair-tab {{ border:1px solid #3b506d; border-radius:7px; padding:7px 13px; color:var(--muted); background:#101e31; cursor:pointer; font-weight:650; }}
.chart-tab.active,.pair-tab.active {{ color:#07111f; border-color:#7dd3fc; background:#7dd3fc; }}
.pair-tabs {{ min-height:39px; margin:14px 0 2px; }}
.empty-selection {{ align-self:center; color:var(--muted); }}
.overview-panels,#pair-details {{ position:relative; }}
.overview-panel,.pair-detail {{ width:100%; }}
.overview-panel.is-inactive,.pair-detail.is-inactive {{ position:absolute; inset:0; visibility:hidden; pointer-events:none; }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:12px; background:var(--card); }}
table {{ width:100%; border-collapse:collapse; white-space:nowrap; }}
th,td {{ padding:10px 13px; text-align:right; border-bottom:1px solid var(--line); }}
th:nth-child(3),td:nth-child(3) {{ text-align:left; }}
th {{ position:sticky; top:0; cursor:pointer; background:#142238; color:#c9d8eb; }}
th:hover {{ background:#1b2d48; }} tbody tr:hover {{ background:#132238; }}
.selection-heading,.selection-cell {{ text-align:center; }}
.metric-cell.is-empty {{ background:rgba(148,163,184,.14) !important; color:var(--muted); box-shadow:none !important; }}
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
<p class="lede">{results['measurement_count']} positions · {band[0]:g}–{band[1]:g} Hz · {settings['ppo']} points/octave · {mode_label} usable-output score · showing up to {limit} pairs</p>
<p class="note">Check table rows to choose comparison pairs. The top {default_count} start selected; the pair tabs below the table open one full diagnostic at a time. Hotkeys 1–9 open the first nine selected tabs.</p>
<div class="chart-tabs" role="tablist" aria-label="{mode_label} overview chart">
  <button class="chart-tab active" data-overview-view="magnitude" role="tab" aria-selected="true" onclick="setOverviewView('magnitude')">Magnitude</button>
  <button class="chart-tab" data-overview-view="excess" role="tab" aria-selected="false" onclick="setOverviewView('excess')">Excess GD</button>
</div>
<div class="overview-panels">
  <div class="overview-panel" data-overview-panel data-overview-view="magnitude">
    {_plot_html(
        _overview_figure(overview, mode, default_keys, magnitude_range),
        f'selected-pairs-magnitude-{mode}',
    )}
  </div>
  <div class="overview-panel is-inactive" data-overview-panel data-overview-view="excess">
    {_plot_html(
        _overview_excess_figure(overview, mode, default_keys, excess_range),
        f'selected-pairs-excess-{mode}',
    )}
  </div>
</div>
<p class="note">{('Raw score' if raw else 'EQ’d score')}: {(1.0 - score_low_end_weight):g} × full-band equal-drive SPL + {score_low_end_weight:g} × excursion-weighted low-end power − {score_dip_weight:g} × worst dip below the one-third-octave smoothed response. Higher is better; the best pair is 0 dB.</p>
<div class="table-wrap">{_ranking_table(pairs, mode, f'ranking-{mode}', default_keys)}</div>
<div class="pair-tabs" data-pair-tabs role="tablist" aria-label="Selected {mode_label} pairs"></div>
<p class="note">The table is ordered by Score, and clicking any heading re-sorts it. Metric cells run from green (best) to red (worst); higher is better for Score, low-end power, and Relative SPL, while lower is better for residual dip, excess GD, and tail. Headroom is the negative global gain which removes positive pair gain—and, post-EQ, the fitted response’s maximum boost—so every pair uses the same maximum driver drive. It is applied to the magnitude comparisons, final summed response, scoring inputs, low-end power, and Relative SPL. Low-end power weights the broad response through 100 Hz by the amplifier/excursion cost of producing pressure at each frequency (+12.04 dB per octave downward). Excess GD and tail remain diagnostics; they do not alter Score.</p>
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
const reportMode={json.dumps(mode)};
const excessGdLowerLimitMs={_EXCESS_GD_LOWER_LIMIT_MS:g};
const selectedPairs=new Set({default_json});
let activePair=Array.from(selectedPairs)[0]||null;
function sectionForKey(key) {{
  return Array.from(document.querySelectorAll('.pair-detail')).find(
    section=>section.dataset.pairKey===key
  )||null;
}}
function orderedSelectedKeys() {{
  return Array.from(selectedPairs).filter(key=>sectionForKey(key)).sort((a,b)=>{{
    return Number(sectionForKey(a).dataset.rank)-Number(sectionForKey(b).dataset.rank);
  }});
}}
function selectedAxisRange(kind) {{
  const bounds=orderedSelectedKeys().map(key=>{{
    const section=sectionForKey(key);
    return [Number(section.dataset[kind+'Min']),Number(section.dataset[kind+'Max'])];
  }}).filter(([low,high])=>Number.isFinite(low)&&Number.isFinite(high));
  if(!bounds.length) return null;
  let low=Math.min(...bounds.map(bound=>bound[0]));
  let high=Math.max(...bounds.map(bound=>bound[1]));
  if(kind==='excess') {{
    low=Math.max(excessGdLowerLimitMs,low);
    high=Math.max(0,high);
  }}
  if(high<=low) high=low+Math.max(1,Math.abs(low)*0.05);
  return [low,high];
}}
function updateSharedYAxisRanges() {{
  const ranges={{
    magnitude:selectedAxisRange('magnitude'),
    excess:selectedAxisRange('excess'),
  }};
  ['magnitude','excess'].forEach(view=>{{
    const range=ranges[view];
    const plots=[document.getElementById('selected-pairs-'+view+'-'+reportMode)];
    orderedSelectedKeys().forEach(key=>plots.push(document.getElementById(view+'-'+key)));
    plots.filter(Boolean).forEach(plot=>{{
      if(range) Plotly.relayout(plot,{{'yaxis.range':range,'yaxis.autorange':false}});
      else Plotly.relayout(plot,{{'yaxis.autorange':true}});
    }});
  }});
}}
function updateOverview() {{
  ['magnitude','excess'].forEach(view=>{{
    const plot=document.getElementById('selected-pairs-'+view+'-'+reportMode);
    if(!plot||!plot.data) return;
    plot.data.forEach((trace,index)=>{{
      const visible=selectedPairs.has(trace.meta.pair_key);
      if(trace.visible!==visible) Plotly.restyle(plot,{{visible:visible}},[index]);
    }});
  }});
  updateSharedYAxisRanges();
}}
function setOverviewView(view) {{
  document.querySelectorAll('[data-overview-panel]').forEach(panel=>{{
    panel.classList.toggle('is-inactive',panel.dataset.overviewView!==view);
  }});
  document.querySelectorAll('.chart-tab').forEach(button=>{{
    const active=button.dataset.overviewView===view;
    button.classList.toggle('active',active);
    button.setAttribute('aria-selected',String(active));
  }});
}}
function renderActiveDetail() {{
  document.querySelectorAll('.pair-detail').forEach(section=>{{
    section.classList.toggle('is-inactive',section.dataset.pairKey!==activePair);
  }});
}}
function activatePair(key) {{
  if(!selectedPairs.has(key)) return;
  activePair=key;
  renderPairTabs();
  renderActiveDetail();
}}
function renderPairTabs() {{
  const strip=document.querySelector('[data-pair-tabs]');
  const keys=orderedSelectedKeys();
  if(!keys.includes(activePair)) activePair=keys[0]||null;
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
    button.className='pair-tab'+(key===activePair?' active':'');
    button.setAttribute('role','tab');
    button.setAttribute('aria-selected',String(key===activePair));
    if(index<9) {{
      button.title='Hotkey '+String(index+1);
      button.setAttribute('aria-keyshortcuts',String(index+1));
    }}
    button.textContent=section.dataset.pairLabel;
    button.addEventListener('click',()=>activatePair(key));
    strip.appendChild(button);
  }});
}}
document.querySelectorAll('.pair-select').forEach(checkbox=>{{
  checkbox.addEventListener('change',()=>{{
    const key=checkbox.dataset.pairKey;
    if(checkbox.checked) selectedPairs.add(key);
    else selectedPairs.delete(key);
    renderPairTabs();
    updateOverview();
    renderActiveDetail();
  }});
}});
document.addEventListener('keydown',event=>{{
  if(event.repeat||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey) return;
  if(!/^[1-9]$/.test(event.key)) return;
  const target=event.target;
  if(target instanceof HTMLElement && (
    target.isContentEditable||['INPUT','TEXTAREA','SELECT'].includes(target.tagName)
  )) return;
  const key=orderedSelectedKeys()[Number(event.key)-1];
  if(!key) return;
  event.preventDefault();
  activatePair(key);
}});
{copy_peq_script}
renderPairTabs();
setOverviewView('magnitude');
renderActiveDetail();
updateSharedYAxisRanges();
</script></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path
