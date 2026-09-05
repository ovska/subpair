"""Self-contained Plotly report generation."""

from __future__ import annotations

import html
import json
import math
from collections import Counter
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import quote

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

_GRAIN_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="180" height="180" viewBox="0 0 180 180">
<filter id="grain"><feTurbulence type="fractalNoise" baseFrequency="0.8" numOctaves="4" stitchTiles="stitch"/></filter>
<rect width="100%" height="100%" filter="url(#grain)" opacity="0.62"/>
</svg>"""

SPEED_OF_SOUND_M_PER_S = 343.0

_ROOM_MODE_LINE_STYLE = {
    "axial": {"color": "#e2e8f0", "dash": "solid", "width": 1.3, "visible": True},
    # Tangential/oblique modes are usually much weaker than axial ones and a
    # full order sweep of them clutters the chart fast, so they start hidden
    # ("legendonly": Plotly still lists and can toggle the trace, it's just
    # not drawn) rather than being left out of the report entirely.
    "tangential": {"color": "#94a3b8", "dash": "dash", "width": 1.0, "visible": "legendonly"},
    "oblique": {"color": "#64748b", "dash": "dot", "width": 0.9, "visible": "legendonly"},
}

# Order here is each axis index (nx/ny/nz), not the frequency-domain "filter
# order" sense of the word: a 4th-order-or-higher axial/tangential/oblique
# mode is both weak in a typical room and, past this point, mostly clutters
# the chart with lines a subwoofer placement search has little control over.
_ROOM_MODE_MAX_ORDER = 3


class ReportError(RuntimeError):
    pass


def _svg_data_uri(svg: str) -> str:
    """Encode an SVG for a self-contained HTML/CSS data URL."""

    return "data:image/svg+xml," + quote(svg, safe="")


def _brand_assets() -> tuple[str, str]:
    """Return the inline logo and a favicon data URL derived from the same SVG."""

    logo = files("subpair").joinpath("assets/subpair-logo.svg").read_text(encoding="utf-8")
    inline_logo = logo.replace(
        "<svg ",
        '<svg class="brand-mark" aria-hidden="true" focusable="false" ',
        1,
    )
    return inline_logo, _svg_data_uri(logo)


def room_mode_frequencies(
    dimensions_cm: tuple[float, float, float],
    max_frequency_hz: float,
    max_order: int = _ROOM_MODE_MAX_ORDER,
) -> list[dict[str, Any]]:
    """Axial/tangential/oblique rigid-rectangular-room eigenfrequencies, <= a limit.

    ``f(nx,ny,nz) = (c/2) * sqrt((nx/Lx)^2 + (ny/Ly)^2 + (nz/Lz)^2)`` is the
    standard rigid rectangular-room mode formula (Lx/Ly/Lz in metres, one
    integer index per axis). A mode is axial when exactly one index is
    nonzero (energy bouncing between one pair of parallel walls), tangential
    when two are, oblique when all three are. This is a purely geometric
    visual reference for a perfectly rigid box -- not a substitute for the
    measured poles in ``modal.py``: a real room's absorption, non-rigid
    boundaries, furniture, and non-rectangular geometry all shift and damp
    its actual modes away from this idealization.

    ``max_order`` caps every axis index (``nx``/``ny``/``nz``) in addition to
    ``max_frequency_hz``: past low single digits, higher-order modes are both
    weak in a typical room and numerous enough (particularly tangential and
    oblique combinations) to clutter the chart, so they're excluded from the
    reference entirely rather than just hidden.
    """
    length_m, width_m, height_m = (value / 100.0 for value in dimensions_cm)
    if min(length_m, width_m, height_m) <= 0.0:
        raise ValueError("Room dimensions must be positive")
    if not math.isfinite(max_frequency_hz) or max_frequency_hz <= 0.0:
        raise ValueError("Room mode frequency limit must be positive")
    axis_limits = tuple(
        min(
            max_order,
            max(0, math.floor(2.0 * max_frequency_hz * dimension / SPEED_OF_SOUND_M_PER_S)),
        )
        for dimension in (length_m, width_m, height_m)
    )
    modes: list[dict[str, Any]] = []
    for nx in range(axis_limits[0] + 1):
        for ny in range(axis_limits[1] + 1):
            for nz in range(axis_limits[2] + 1):
                if nx == ny == nz == 0:
                    continue
                frequency = 0.5 * SPEED_OF_SOUND_M_PER_S * math.sqrt(
                    (nx / length_m) ** 2 + (ny / width_m) ** 2 + (nz / height_m) ** 2
                )
                if frequency > max_frequency_hz:
                    continue
                nonzero = (nx > 0) + (ny > 0) + (nz > 0)
                mode_type = {1: "axial", 2: "tangential", 3: "oblique"}[nonzero]
                modes.append(
                    {"frequency_hz": frequency, "type": mode_type, "indices": (nx, ny, nz)}
                )
    modes.sort(key=lambda mode: mode["frequency_hz"])
    return modes


def _room_mode_traces(
    modes: list[dict[str, Any]] | None,
    span: tuple[float, float] | None,
    orientation: str,
) -> list[go.Scatter]:
    """One legend-toggleable line trace per mode type, across ``span``.

    ``orientation="vertical"`` draws a line at each mode's frequency on the
    x-axis (frequency/excess-GD charts); ``"horizontal"`` draws it on the
    y-axis (the CSD heatmap, whose y-axis is frequency). Each trace packs
    every mode of one type into a single multi-segment line (``None``-
    separated), so toggling one legend entry shows/hides every mode of that
    type together rather than needing one entry per mode.
    """
    if not modes or span is None:
        return []
    low, high = span
    traces = []
    for mode_type, style in _ROOM_MODE_LINE_STYLE.items():
        frequencies = [mode["frequency_hz"] for mode in modes if mode["type"] == mode_type]
        if not frequencies:
            continue
        xs: list[float | None] = []
        ys: list[float | None] = []
        for frequency in frequencies:
            if orientation == "vertical":
                xs.extend([frequency, frequency, None])
                ys.extend([low, high, None])
            else:
                xs.extend([low, high, None])
                ys.extend([frequency, frequency, None])
        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"color": style["color"], "width": style["width"], "dash": style["dash"]},
                opacity=0.55,
                name=f"Room mode: {mode_type}",
                legendgroup=f"room-mode-{mode_type}",
                hoverinfo="skip",
                visible=style["visible"],
            )
        )
    return traces


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
    room_modes: list[dict[str, Any]] | None = None,
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
    resolved_y_range = y_range
    if resolved_y_range is None:
        resolved_y_range = _finite_axis_range(
            [data["solo_first_db"], data["solo_second_db"], data["sum_db" if raw else "post_eq_db"]],
            include_zero=not raw,
        )
    for trace in _room_mode_traces(room_modes, resolved_y_range, "vertical"):
        figure.add_trace(trace)
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


def _robustness_figure(pair: dict[str, Any]) -> go.Figure:
    """Delay-objective landscape with basin and physical reference markers."""

    robustness = pair["robustness"]
    tau = np.asarray(robustness["tau_grid_ms"], dtype=np.float64)
    objective = np.asarray(robustness["objective_db"], dtype=np.float64)
    robust = np.asarray(robustness["robust_objective_db"], dtype=np.float64)
    star_index = int(np.argmin(np.abs(tau - float(robustness["tau_star_ms"]))))
    robust_index = int(np.argmin(np.abs(tau - float(robustness["tau_robust_ms"]))))
    basin_delta_db = float(robustness.get("basin_tolerance_db", 0.5))
    threshold = objective[robust_index] + basin_delta_db
    left = robust_index
    while left > 0 and objective[left - 1] <= threshold:
        left -= 1
    right = robust_index
    while right + 1 < tau.size and objective[right + 1] <= threshold:
        right += 1

    figure = go.Figure()
    figure.add_vrect(
        x0=float(tau[left]),
        x1=float(tau[right]),
        fillcolor="#22c55e",
        opacity=0.15,
        line_width=0,
        annotation_text=f"+{basin_delta_db:.2f} dB adaptive contiguous basin",
        annotation_position="top left",
    )
    figure.add_trace(
        go.Scatter(
            x=tau,
            y=objective,
            name="Raw f(tau)",
            line={"color": "#7dd3fc", "width": 1.5},
            hovertemplate="%{x:.2f} ms<br>f %{y:.2f} dB<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=tau,
            y=robust,
            name="Jitter-averaged f_robust(tau)",
            line={"color": "#c4b5fd", "width": 2.0},
            hovertemplate="%{x:.2f} ms<br>f_robust %{y:.2f} dB<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[tau[star_index]],
            y=[objective[star_index]],
            name="tau* (raw)",
            mode="markers",
            marker={"color": "#fbbf24", "size": 10, "symbol": "diamond"},
            hovertemplate="tau* %{x:.2f} ms<br>f %{y:.2f} dB<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[tau[robust_index]],
            y=[robust[robust_index]],
            name="tau_robust (reported delay)",
            mode="markers",
            marker={"color": "#86efac", "size": 10, "symbol": "circle"},
            hovertemplate="tau_robust %{x:.2f} ms<br>f_robust %{y:.2f} dB<extra></extra>",
        )
    )
    physical_tau = robustness.get("physical_tau_ms")
    if physical_tau is not None:
        figure.add_vline(
            x=float(physical_tau),
            line={"color": "#fb7185", "width": 1.5, "dash": "dash"},
            annotation_text="physical tau",
            annotation_position="bottom right",
        )
    symmetry = robustness.get("detrended_symmetry") or {}
    symmetry_axis = symmetry.get("axis_ms")
    symmetry_correlation = symmetry.get("correlation")
    symmetry_offset = symmetry.get("axis_offset_ms")
    if symmetry_axis is not None:
        symmetry_label = "detrended mirror axis"
        if symmetry_correlation is not None:
            symmetry_label += f" (r={float(symmetry_correlation):.2f})"
        if symmetry_offset is not None:
            symmetry_label += f"; offset {float(symmetry_offset):+.2f} ms"
        figure.add_vline(
            x=float(symmetry_axis),
            line={"color": "#94a3b8", "width": 1.2, "dash": "dot"},
            annotation_text=symmetry_label,
            annotation_position="top right",
        )
    figure.update_layout(
        title="Delay robustness (lower f is better)",
        xaxis={"title": "Delay applied to sub 2 (ms)"},
        yaxis={"title": "Objective f = -raw score (dB)"},
        margin={"l": 62, "r": 40, "t": 58, "b": 55},
        legend={"orientation": "h", "y": -0.24},
        template="plotly_dark",
        height=470,
    )
    return figure


def _overview_figure(
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
    mode: str,
    selected_keys: set[str],
    y_range: tuple[float, float] | None = None,
    room_modes: list[dict[str, Any]] | None = None,
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
    for trace in _room_mode_traces(room_modes, y_range, "vertical"):
        figure.add_trace(trace)
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
    room_modes: list[dict[str, Any]] | None = None,
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
    for trace in _room_mode_traces(room_modes, y_range, "vertical"):
        figure.add_trace(trace)
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
    room_modes: list[dict[str, Any]] | None = None,
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
    resolved_y_range = y_range
    if resolved_y_range is None:
        resolved_y_range = _finite_axis_range(
            [data[f"{prefix}excess_curve_ms"]],
            lower_limit=_EXCESS_GD_LOWER_LIMIT_MS,
            include_zero=True,
        )
    for trace in _room_mode_traces(room_modes, resolved_y_range, "vertical"):
        figure.add_trace(trace)
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


def _decay_figure(
    data: dict[str, Any],
    *,
    raw: bool = False,
    room_modes: list[dict[str, Any]] | None = None,
) -> go.Figure:
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
    times_ms = 1000.0 * np.asarray(data["decay_times"])
    x_span = (float(np.min(times_ms)), float(np.max(times_ms)))
    for trace in _room_mode_traces(room_modes, x_span, "horizontal"):
        figure.add_trace(trace)
    # fixedrange (not the blunter staticPlot config option) is what actually
    # keeps this from being nudged into an accidental zoom/pan; it does not
    # by itself disable hover or legend clicks, which room-mode toggling
    # needs -- see this figure's call site in build_report for when the
    # staticPlot config is skipped so those clicks can land.
    figure.update_xaxes(title_text="Time from sum peak (ms)", fixedrange=True)
    figure.update_yaxes(type="log", title_text="Frequency (Hz)", fixedrange=True)
    figure.update_layout(
        title=f"{label} CSD-style decay with zero-referenced excess-GD overlay",
        template="plotly_dark",
        height=520,
        legend={"orientation": "h", "y": -0.18},
        margin={"l": 66, "r": 70, "t": 72, "b": 82},
    )
    return figure


GATE_LABELS = {
    "gate_a_redundancy": "A · redundancy residual",
    "gate_b_ripple_correlation": "B · ripple correlation",
    "gate_c_physical_percentile": "C · physical-delay percentile",
    "basin_geometry": "Basin vs geometry",
    "gate_d_cancellation_deficit": "D · cancellation deficit",
    "gate_e_comb_signature": "E · comb signature",
    "gate_f_residual_notches": "F · residual notches",
    "gate_g_gain_asymmetry": "G · gain asymmetry",
    "gate_h_band_edge_stability": "H · band-edge stability",
    "gate_i_improvement_localization": "I · improvement localisation",
}

# What each gate is actually asking, in one sentence, so a fired reason can be
# understood from the table without opening the spec.
GATE_MEANINGS = {
    "gate_a_redundancy": (
        "How much of one position's response is just a delayed, rescaled copy "
        "of the other. A low residual means the two are near-duplicates, so "
        "pairing them buys nothing."
    ),
    "gate_b_ripple_correlation": (
        "Whether the two positions' peaks and dips fall at the same "
        "frequencies. Positive correlation means they reinforce the same room "
        "errors instead of filling each other's."
    ),
    "gate_c_physical_percentile": (
        "How well the pair does at its measured arrival alignment, ranked "
        "within its own delay landscape. A poor rank means the benefit rests "
        "on a delay far from where the geometry puts it."
    ),
    "basin_geometry": (
        "How many dB the pair loses when the listener moves, evaluated at the "
        "recommended delay. A large excursion penalty means razor-edge tuning."
    ),
    "gate_d_cancellation_deficit": (
        "Coherent sum level minus the matching incoherent power sum. Negative "
        "means the two subs are partly cancelling rather than adding."
    ),
    "gate_e_comb_signature": (
        "Periodic ripple spaced at 1/delay -- the fingerprint of a plain time "
        "offset rather than genuine modal complementarity."
    ),
    "gate_f_residual_notches": (
        "Deep, narrow nulls left in the summed response, too sharp to EQ and "
        "too deep to ignore."
    ),
    "gate_g_gain_asymmetry": (
        "How far apart the two subs' drive levels have to be. A large offset "
        "wastes one sub's headroom."
    ),
    "gate_h_band_edge_stability": (
        "How much this pair's score moves when the evaluation band shifts "
        "+/-1/6 octave, measured against how much every other pair moves."
    ),
    "gate_i_improvement_localization": (
        "What share of the pair's gain over its physical alignment comes from "
        "a single 1/6-octave region, once that gain is large enough to judge."
    ),
}


# Ranking-table column help. Keyed by the canonical concept; the raw and EQ'd
# variants of a column share one entry. Obvious columns (Pair) are omitted so
# the cursor only changes where there is something worth reading.
COLUMN_HELP = {
    "score": (
        "Usable-output score relative to the best pair, which sits at 0 dB. "
        "(1-w)x full-band SPL + w x low-end power - dip weight x worst smoothed "
        "dip, with the search's configured weights. Higher is better. A leading "
        "= marks a pair within its own score resolution of the 0 dB reference: "
        "the table still orders it, but the data does not support that order."
    ),
    "verdict_status": (
        "ACCEPT: every evaluated gate passed. CAUTION: at least one gate "
        "cautioned. REJECT: at least one rejected. Verdict is the primary sort "
        "key, so rejected pairs stay visible at the bottom whatever they score. "
        "A clean sheet is not validation - these are single-position "
        "disqualifiers only."
    ),
    "gate_reasons": (
        "Which gates cautioned or rejected this pair. Hover a cell to see what "
        "each one measured and the limit it was compared against."
    ),
    "redundancy_residual": GATE_MEANINGS["gate_a_redundancy"] + " Higher is better.",
    "ripple_correlation": GATE_MEANINGS["gate_b_ripple_correlation"] + " Lower is better.",
    "physical_percentile": GATE_MEANINGS["gate_c_physical_percentile"] + " Lower is better.",
    "gate_d_deficit": GATE_MEANINGS["gate_d_cancellation_deficit"] + " Higher is better.",
    "comb_index": GATE_MEANINGS["gate_e_comb_signature"] + " Lower is better.",
    "notch_count": GATE_MEANINGS["gate_f_residual_notches"] + " Lower is better.",
    "gain_asymmetry_db": GATE_MEANINGS["gate_g_gain_asymmetry"] + " Lower is better.",
    "band_edge_excess_spread_db": (
        GATE_MEANINGS["gate_h_band_edge_stability"] + " This column is the excess "
        "over the population median, so 0 means exactly as stable as its peers "
        "and negative means more stable. Lower is better."
    ),
    "localization_pct": GATE_MEANINGS["gate_i_improvement_localization"] + " Lower is better.",
    "fragility": (
        "f_robust(tau*) - f(tau*): how much worse the jitter-averaged objective "
        "is at the raw optimum than the raw objective there. A large value means "
        "the optimum only looks good if the timing is exact. Lower is better."
    ),
    "excursion_penalty_db": (
        "Worst degradation of the objective anywhere within +/-delta_tau_max/2 of "
        "the recommended delay - how many dB the pair loses when the listener "
        "moves by the configured --listener-movement. This drives the basin gate. "
        "Lower is better."
    ),
    "basin_w03": (
        "Width of the single contiguous delay interval containing tau* over which "
        "the raw objective stays within +0.3 dB of its minimum. Disconnected good "
        "regions do not count. A fixed-threshold diagnostic; the gate uses the "
        "excursion penalty instead. Wider is better."
    ),
    "worst_case_penalty_1": (
        "Worst degradation of the objective anywhere within +/-1 ms of the "
        "recommended delay, including interpolated interval edges. Same measure "
        "as the excursion penalty but at a fixed +/-1 ms instead of the "
        "geometry-derived window, so it is comparable across pairs whatever "
        "listener movement is configured. Lower is better."
    ),
    "geometric_pass": (
        "PASS when the excursion penalty stays within the basin tolerance for the "
        "configured listener movement. A FAIL proves the tuning is fragile; a PASS "
        "does not prove the pair is robust across seats, since this delay-only "
        "test cannot see the magnitude change from moving through the room's "
        "modal field."
    ),
    "physical_status": (
        "Whether the pair's measured arrival alignment is usable. OK: the raw "
        "optimum lies inside the physical-delay window. OUT: it lies outside, "
        "though the recommended delay was still constrained inside. N/A: no "
        "usable arrival timing for this pair. INVALID: the physical window does "
        "not intersect the delay scan."
    ),
    "polarity": "Polarity applied to the second sub of the pair: + normal, - inverted.",
    "delay_ms": (
        "Delay applied to the second sub at the recommended configuration, "
        "selected from the jitter-averaged objective inside the physical window."
    ),
    "gain_db": "Level applied to the second sub, relative to the first.",
    "headroom": (
        "Global attenuation applied to every compared response so the comparison "
        "is equal-drive: it removes positive relative pair gain and, post-EQ, the "
        "fitted bank's largest in-band boost. Already included in SPL, low-end "
        "power and the plotted magnitudes."
    ),
    "dip": (
        "Worst negative deviation from a one-third-octave Gaussian-smoothed "
        "version of the same equal-drive response. This is the only score term "
        "you cannot recover with the volume knob. Lower is better."
    ),
    "excess_gd": (
        "Energy-weighted mean of the denoised excess group delay across the "
        "scoring band, after the common delay is removed. A reported diagnostic "
        "and EQ-authority input, not a score term. Lower is better."
    ),
    "tail": (
        "Decay. Shows the loudest detected mode's level relative to direct sound "
        "in dB when this pair's modal fit is valid, otherwise the CSD envelope "
        "decay time in ms - ringing_ms saturates at 0 for every mode below the "
        "audibility margin, so the dB figure is preferred when available. Lower "
        "is better either way."
    ),
    "low_end_power": (
        "One-octave broad-trend pressure power weighted by the f^-4 excursion and "
        "amplifier cost of producing pressure low down, relative to the best pair. "
        "A score component. Higher is better."
    ),
    "spl": (
        "Mean in-band level relative to the best pair, at equal drive. A score "
        "component. Higher is better."
    ),
}


def _column_help(key: str, canonical: dict[str, str]) -> str:
    """Help text for one column key, resolving raw/EQ'd variants to one entry."""

    return COLUMN_HELP.get(canonical.get(key, key), "")


def _gate_evidence(gate_key: str, gate: dict[str, Any]) -> str:
    """The measured figure and the limit it was compared against."""

    def number(key: str, digits: int = 3) -> str | None:
        value = gate.get(key)
        return None if value is None else f"{float(value):.{digits}f}"

    pairs = {
        "gate_a_redundancy": ("residual", "reject_below", "residual %s (reject below %s)"),
        "gate_b_ripple_correlation": (
            "correlation",
            "reject_above",
            "correlation %s (reject above %s)",
        ),
        "gate_c_physical_percentile": (
            "percentile",
            "reject_above_percentile",
            "percentile %s (reject above %s)",
        ),
        "basin_geometry": (
            "excursion_penalty_db",
            "tolerance_db",
            "excursion penalty %s dB (tolerance %s dB)",
        ),
        "gate_d_cancellation_deficit": (
            "worst_deficit_db",
            "caution_below_db",
            "worst deficit %s dB (caution below %s dB)",
        ),
        "gate_e_comb_signature": (
            "comb_index",
            "caution_at_or_above",
            "comb index %s (caution at %s)",
        ),
        "gate_g_gain_asymmetry": (
            "gain_offset_db",
            "caution_above_db",
            "offset %s dB (caution above %s dB)",
        ),
        "gate_h_band_edge_stability": (
            "excess_spread_db",
            "reject_above_excess_db",
            "excess spread %s dB (reject above %s dB)",
        ),
        "gate_i_improvement_localization": (
            "fraction",
            "reject_above_fraction",
            "concentration %s (reject above %s)",
        ),
    }
    if gate_key == "gate_f_residual_notches":
        worst = gate.get("worst") or {}
        if not worst:
            return "no notch deeper than the limit"
        return (
            f"worst {float(worst['depth_db']):.1f} dB at "
            f"{float(worst['frequency_hz']):.1f} Hz, "
            f"{float(worst['width_octaves']):.3f} octave wide"
        )
    entry = pairs.get(gate_key)
    if entry is None:
        return ""
    measured_key, limit_key, template = entry
    measured, limit = number(measured_key), number(limit_key)
    if measured is None:
        return str(gate.get("detail") or "not evaluated")
    return template % (measured, limit if limit is not None else "—")


def _gate_tooltip(gate_key: str, gate: dict[str, Any], include_status: bool) -> str:
    """Plain-language explanation of one gate, for a title attribute."""

    status = str(gate.get("status", "not_run")).upper()
    head = GATE_LABELS.get(gate_key, gate_key)
    if include_status:
        head = f"{head} — {status}"
    parts = [head, GATE_MEANINGS.get(gate_key, "")]
    evidence = _gate_evidence(gate_key, gate)
    if evidence:
        parts.append(f"Measured: {evidence}.")
    detail = gate.get("detail")
    if detail and gate_key != "gate_f_residual_notches":
        parts.append(str(detail))
    return "\n".join(part for part in parts if part)


def _gate_value_html(value: Any) -> str:
    """One measured gate value, formatted for reading rather than for parsing.

    Gate blocks mix scalars, nested dicts (Gate H's three band-shift scores,
    Gate F's worst notch) and nulls, so a single JSON blob per gate is dense
    and hard to scan. Keys stay exactly as they appear in the result JSON so a
    figure in the report can be looked up there without translation.
    """

    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return html.escape(str(value))
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return html.escape(text if text not in {"", "-"} else "0")
    if isinstance(value, int):
        return html.escape(str(value))
    if isinstance(value, dict):
        return " · ".join(
            f"{html.escape(str(name))} {_gate_value_html(entry)}"
            for name, entry in sorted(value.items())
        )
    if isinstance(value, (list, tuple)):
        return " · ".join(_gate_value_html(entry) for entry in value)
    return html.escape(str(value))


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
    *,
    show_headroom: bool = True,
) -> str:
    eq = mode == "eq"
    score_key = "post_eq_relative_score_db" if eq else "relative_score_db"
    dip_key = "post_eq_dip_db" if eq else "dip_db"
    excess_key = "post_eq_excess_gd_ms" if eq else "excess_gd_ms"
    # "Tail" shows effective_tail_db/post_eq_effective_tail_db (the loudest
    # detected mode's level relative to direct sound, in dB) whenever the
    # source is modal, else effective_tail_ms/post_eq_effective_tail_ms (the
    # CSD-based envelope decay time, in ms) -- see engine.py's
    # settings.ranking.effective_tail documentation for why: ringing_ms
    # saturates at 0 for every mode below the audibility margin, which a
    # well-controlled room can do for every pair, so the dB figure (which
    # keeps varying below that floor) is preferred when it's available. Both
    # directions are "lower is better" (a shorter tail or a quieter mode), so
    # they share one colour-scaled column even though their units differ.
    tail_key = "post_eq_effective_tail_ms" if eq else "effective_tail_ms"
    tail_db_key = "post_eq_effective_tail_db" if eq else "effective_tail_db"
    tail_modal_key = "post_eq_effective_tail_is_modal" if eq else "effective_tail_is_modal"
    tail_display: dict[int, tuple[float | None, str]] = {}
    for pair in pairs:
        if pair.get(tail_modal_key) and pair.get(tail_db_key) is not None:
            tail_display[id(pair)] = (float(pair[tail_db_key]), "dB")
        else:
            raw_tail = pair.get(tail_key)
            tail_display[id(pair)] = (float(raw_tail) if raw_tail is not None else None, "ms")
    tie_key = "post_eq_score_ties_reference" if eq else "score_ties_reference"
    headroom_key = "post_eq_headroom_db" if eq else "headroom_db"
    low_end_power_key = (
        "post_eq_relative_low_end_power_db" if eq else "relative_low_end_power_db"
    )
    spl_key = "post_eq_relative_spl_db" if eq else "relative_spl_db"
    columns = [
        (score_key, "Score (dB)", "number"),
        ("pair", "Pair", "text"),
        ("verdict_status", "Verdict", "number"),
        ("gate_reasons", "Gate reasons", "text"),
        ("redundancy_residual", "A residual", "number"),
        ("ripple_correlation", "B ripple r", "number"),
        ("physical_percentile", "C physical (%)", "number"),
        ("gate_d_deficit", "D deficit (dB)", "number"),
        ("comb_index", "E comb", "number"),
        ("notch_count", "F notches", "number"),
        ("gain_asymmetry_db", "G gain (dB)", "number"),
        ("band_edge_excess_spread_db", "H edge excess (dB)", "number"),
        ("localization_pct", "I local (%)", "number"),
        ("fragility", "Fragility (dB)", "number"),
        ("excursion_penalty_db", "Excursion penalty (dB)", "number"),
        ("basin_w03", "Basin +0.3 (ms)", "number"),
        ("worst_case_penalty_1", "Worst penalty +/-1 ms (dB)", "number"),
        ("geometric_pass", "Basin vs geometry", "number"),
        ("physical_status", "Physical", "number"),
        ("polarity", "Pol 2", "number"),
        ("delay_ms", "Delay 2 (ms)", "number"),
        ("gain_db", "Gain 2 (dB)", "number"),
    ]
    if show_headroom:
        columns.append((headroom_key, "Headroom (dB)", "number"))
    columns.extend([
        (dip_key, "Residual dip (dB)", "number"),
        (excess_key, "Excess GD (ms)", "number"),
        (tail_key, "Tail", "number"),
        (low_end_power_key, "Low-end power (dB)", "number"),
        (spl_key, "Relative SPL (dB)", "number"),
    ])
    # Raw and EQ'd variants of a column share one help entry.
    canonical = {
        score_key: "score",
        headroom_key: "headroom",
        dip_key: "dip",
        excess_key: "excess_gd",
        tail_key: "tail",
        low_end_power_key: "low_end_power",
        spl_key: "spl",
    }

    def heading_cell(index: int, key: str, label: str, kind: str) -> str:
        help_text = _column_help(key, canonical)
        described = ' class="has-help"' if help_text else ""
        help_attribute = f' data-help="{html.escape(help_text)}"' if help_text else ""
        return (
            f'<th data-key="{key}" data-type="{kind}" '
            f'data-column-index="{index + 1}"{described}{help_attribute}>'
            f"{html.escape(label)}</th>"
        )

    heading = '<th class="selection-heading">Show</th>' + "".join(
        heading_cell(index, key, label, kind)
        for index, (key, label, kind) in enumerate(columns)
    )
    metric_directions = {
        score_key: "high",
        "verdict_status": "low",
        "redundancy_residual": "high",
        "ripple_correlation": "low",
        "physical_percentile": "low",
        "gate_d_deficit": "high",
        "comb_index": "low",
        "notch_count": "low",
        "gain_asymmetry_db": "low",
        "band_edge_excess_spread_db": "low",
        "localization_pct": "low",
        "fragility": "low",
        "excursion_penalty_db": "low",
        "basin_w03": "high",
        "worst_case_penalty_1": "low",
        "geometric_pass": "high",
        "physical_status": "low",
        dip_key: "low",
        excess_key: "low",
        tail_key: "low",
        low_end_power_key: "high",
        spl_key: "high",
    }

    verdict_rank = {"accept": 0.0, "caution": 1.0, "reject": 2.0}
    # Short forms for the narrow column; GATE_LABELS carries the full names the
    # tooltip and the per-pair sheet use.
    gate_labels = {
        key: label.split(" · ")[0] for key, label in GATE_LABELS.items()
    }

    def physical_status(pair: dict[str, Any]) -> tuple[str, str, float | None]:
        """Display text, best-first sort rank, and colour rank for physical status."""

        if not pair.get("optimized", True):
            gate_c = pair.get("gates", {}).get("gate_c_physical_percentile", {})
            if gate_c.get("status") == "reject" and gate_c.get("physical_tau_ms") is not None:
                return "INVALID", "3", 3.0
            return "N/A", "1", None
        if not pair.get("pair_valid", True):
            return "INVALID", "3", 3.0
        if pair.get("physical_tau") is None:
            # Missing arrival metadata is unknown, not good or bad. It sorts
            # between OK and known failures but remains neutral grey.
            return "N/A", "1", None
        if pair.get("non_physical_solution"):
            return "OUT", "2", 2.0
        return "OK", "0", 0.0

    def metric_value(pair: dict[str, Any], key: str) -> float | None:
        if key == tail_key:
            return tail_display[id(pair)][0]
        if key == "verdict_status":
            return verdict_rank.get(str(pair.get("verdict", "reject")))
        if key == "worst_case_penalty_1":
            value = pair.get("worst_case_penalty", {}).get("1.0")
        elif key == "gate_d_deficit":
            value = pair.get("cancellation_deficit_db")
            if value is None:
                value = pair.get("physical_cancellation_deficit_db")
        elif key == "localization_pct":
            fraction = pair.get("improvement_localization_fraction")
            value = 100.0 * float(fraction) if fraction is not None else None
        elif key == "geometric_pass":
            value = (
                1.0 if pair.get("geometric_pass") else 0.0
            ) if pair.get("optimized", True) else None
        elif key == "physical_status":
            return physical_status(pair)[2]
        else:
            value = pair.get(key)
        return float(value) if value is not None else None

    # Keep missing future metrics out of the colour-scaled range rather than
    # coercing them to numbers that could skew the best/worst endpoints.
    # Status columns use fixed ranges so an all-FAIL table cannot make FAIL
    # look green merely because it is the best value currently visible.
    fixed_metric_ranges = {
        "verdict_status": (0.0, 2.0),
        "geometric_pass": (0.0, 1.0),
        "physical_status": (0.0, 3.0),
    }
    metric_ranges: dict[str, tuple[float, float]] = {}
    for key in metric_directions:
        numeric = [
            value
            for pair in pairs
            if (value := metric_value(pair, key)) is not None
        ]
        metric_ranges[key] = fixed_metric_ranges.get(
            key,
            (min(numeric), max(numeric)) if numeric else (0.0, 0.0),
        )

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
        tail_value, tail_unit = tail_display[id(pair)]
        physical_text, physical_sort_value, _ = physical_status(pair)
        verdict = str(pair.get("verdict", "reject"))
        reason_tooltip = "\n\n".join(
            _gate_tooltip(
                str(reason.get("gate")),
                pair.get("gates", {}).get(str(reason.get("gate")), {}),
                include_status=True,
            )
            for reason in pair.get("reasons", [])
        ) or (
            "Every evaluated gate passed. A clean sheet is not validation: these "
            "are single-position disqualifiers only."
        )
        reason_text = ", ".join(
            f"{gate_labels.get(str(reason.get('gate')), str(reason.get('gate')))} "
            f"{str(reason.get('status', '')).upper()}"
            for reason in pair.get("reasons", [])
        ) or "none"
        tail_display_text = (
            (f"{tail_value:+.1f} dB" if tail_unit == "dB" else f"{tail_value:.1f} ms")
            if tail_value is not None
            else "—"
        )

        def number_value(
            key: str,
            format_spec: str,
            *,
            suffix: str = "",
        ) -> tuple[str, str]:
            value = metric_value(pair, key)
            if value is None:
                return "—", ""
            return f"{value:{format_spec}}{suffix}", str(value)

        # A pair inside its own resolution of the reference is flagged rather
        # than silently presented as ranked: the sort key is still the score, so
        # the row has a position, but that position is not evidence.
        score_text, score_sort = number_value(score_key, "+.2f")
        if pair.get(tie_key) and score_text != "—":
            score_text = f"= {score_text}"
        values: dict[str, tuple[str, str]] = {
            score_key: (score_text, score_sort),
            "pair": (
                f"{pair['first']} + {pair['second']}",
                f"{pair['first']:04d}-{pair['second']:04d}",
            ),
            "verdict_status": (verdict.upper(), str(verdict_rank.get(verdict, 2.0))),
            "gate_reasons": (reason_text, reason_text),
            "redundancy_residual": number_value("redundancy_residual", ".3f"),
            "ripple_correlation": number_value("ripple_correlation", "+.3f"),
            "physical_percentile": number_value("physical_percentile", ".1f", suffix="%"),
            "gate_d_deficit": number_value("gate_d_deficit", "+.2f"),
            "comb_index": number_value("comb_index", ".3f"),
            "notch_count": number_value("notch_count", ".0f"),
            "gain_asymmetry_db": number_value("gain_asymmetry_db", ".2f"),
            "band_edge_excess_spread_db": number_value(
                "band_edge_excess_spread_db", ".2f"
            ),
            "localization_pct": number_value("localization_pct", ".1f", suffix="%"),
            "fragility": number_value("fragility", ".2f"),
            "excursion_penalty_db": number_value("excursion_penalty_db", ".2f"),
            "basin_w03": number_value("basin_w03", ".2f"),
            "worst_case_penalty_1": number_value("worst_case_penalty_1", ".2f"),
            "geometric_pass": (
                ("PASS" if pair.get("geometric_pass") else "FAIL")
                if pair.get("optimized", True)
                else "—",
                ("1" if pair.get("geometric_pass") else "0")
                if pair.get("optimized", True)
                else "",
            ),
            "physical_status": (physical_text, physical_sort_value),
            "polarity": (
                ("+" if pair["polarity"] > 0 else "−")
                if pair.get("polarity") is not None
                else "—",
                str(pair["polarity"]) if pair.get("polarity") is not None else "",
            ),
            "delay_ms": (
                f"{pair['delay_ms']:+.3f}" if pair.get("delay_ms") is not None else "—",
                str(pair["delay_ms"]) if pair.get("delay_ms") is not None else "",
            ),
            "gain_db": (
                f"{pair['gain_db']:+.2f}" if pair.get("gain_db") is not None else "—",
                str(pair["gain_db"]) if pair.get("gain_db") is not None else "",
            ),
            headroom_key: number_value(headroom_key, "+.2f"),
            dip_key: number_value(dip_key, ".3f"),
            excess_key: number_value(excess_key, ".3f"),
            tail_key: (tail_display_text, "" if tail_value is None else str(tail_value)),
            low_end_power_key: number_value(low_end_power_key, "+.2f"),
            spl_key: number_value(spl_key, "+.2f"),
        }
        cells = []
        for key, _, _ in columns:
            is_metric = key in metric_directions
            numeric_value = metric_value(pair, key) if is_metric else None
            style = score_style(key, numeric_value) if is_metric else ""
            empty_class = " is-empty" if is_metric and numeric_value is None else ""
            style_attribute = f' style="{style}"' if style else ""
            help_attribute = (
                f' data-help="{html.escape(reason_tooltip)}"'
                if key == "gate_reasons"
                else ""
            )
            help_class = " has-help" if key == "gate_reasons" else ""
            cells.append(
                f'<td class="metric-cell{empty_class}{help_class}" '
                f'data-value="{html.escape(values[key][1])}"'
                f'{style_attribute}{help_attribute}>'
                f'{html.escape(values[key][0])}</td>'
            )
        if pair.get("optimized", True):
            checked = " checked" if key_value in selected_keys else ""
            checkbox = (
                '<td class="selection-cell"><input class="pair-select" type="checkbox" '
                f'data-mode="{mode}" data-pair-key="{key_value}"{checked} '
                f'aria-label="Show pair {pair["first"]}+{pair["second"]}"></td>'
            )
        else:
            checkbox = (
                '<td class="selection-cell"><input class="pair-select" type="checkbox" '
                f'data-mode="{mode}" data-pair-key="{key_value}" disabled '
                f'aria-label="Pair {pair["first"]}+{pair["second"]} was not optimized"></td>'
            )
        rejected_attributes = ' data-verdict="reject" hidden' if verdict == "reject" else ""
        rows.append(
            f'<tr data-pair-key="{key_value}"{rejected_attributes}>'
            f'{checkbox}{"".join(cells)}</tr>'
        )
    return (
        f'<table id="{table_id}" class="ranking-table"><thead><tr>{heading}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


_MODAL_PALETTE = (
    "#7dd3fc", "#c4b5fd", "#86efac", "#fca5a5", "#fcd34d", "#67e8f9", "#f0abfc", "#a5b4fc",
)


def _modal_mode_table(modes: list[dict[str, Any]], table_id: str) -> str:
    """Sortable per-mode table (reuses the ranking table's click-to-sort script)."""
    columns = [
        ("frequency_hz", "f (Hz)"),
        ("q", "Q"),
        ("level_db", "L (dB)"),
        ("t60_s", "T60 (s)"),
        ("t_audible_s", "t audible (s)"),
    ]
    heading = "".join(
        f'<th data-key="{key}" data-type="number" data-column-index="{index}">'
        f'{html.escape(label)}</th>'
        for index, (key, label) in enumerate(columns)
    )
    display = {
        "frequency_hz": lambda v: f"{v:.1f}",
        "q": lambda v: f"{v:.1f}",
        "level_db": lambda v: f"{v:+.1f}",
        "t60_s": lambda v: f"{v:.2f}",
        "t_audible_s": lambda v: f"{v:.2f}",
    }
    rows = []
    for mode in sorted(modes, key=lambda m: -float(m["q"])):
        cells = "".join(
            f'<td class="metric-cell" data-value="{mode[key]}">'
            f'{html.escape(display[key](mode[key]))}</td>'
            for key, _ in columns
        )
        rows.append(f"<tr>{cells}</tr>")
    return (
        f'<table id="{table_id}" class="ranking-table"><thead><tr>{heading}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def _modal_pole_map_figure(rows: list[tuple[dict[str, Any], dict[str, Any]]]) -> go.Figure:
    """f (x) vs Q (y), marker size ~ modal level, one colour per pair."""
    figure = go.Figure()
    high_q_threshold = 16.0
    for index, (pair, pair_modal) in enumerate(rows):
        modes = pair_modal.get("modes", [])
        if not modes:
            continue
        high_q_threshold = float(pair_modal.get("high_q_threshold", high_q_threshold))
        color = _MODAL_PALETTE[index % len(_MODAL_PALETTE)]
        sizes = [
            float(np.clip(10.0 + (float(mode["level_db"]) + 30.0) * 1.1, 6.0, 42.0))
            for mode in modes
        ]
        figure.add_trace(
            go.Scatter(
                x=[mode["frequency_hz"] for mode in modes],
                y=[mode["q"] for mode in modes],
                mode="markers",
                marker={"size": sizes, "color": color, "line": {"width": 1, "color": "#0f1b2d"}},
                name=f"{pair['first']}+{pair['second']}",
                customdata=[float(mode["level_db"]) for mode in modes],
                hovertemplate=(
                    "%{x:.1f} Hz<br>Q %{y:.1f}<br>L %{customdata:+.1f} dB<extra>"
                    f"{pair['first']}+{pair['second']}</extra>"
                ),
            )
        )
    figure.add_hline(
        y=high_q_threshold,
        line={"color": "#64748b", "width": 1, "dash": "dot"},
        annotation_text=f"Q = {high_q_threshold:g}",
        annotation_position="top left",
    )
    figure.update_xaxes(type="log", title_text="Frequency (Hz)")
    figure.update_yaxes(title_text="Q")
    figure.update_layout(
        title="Modal pole map (marker size ~ level relative to direct sound)",
        template="plotly_dark",
        height=430,
        legend={"orientation": "h", "y": -0.22},
        margin={"l": 62, "r": 24, "t": 52, "b": 78},
    )
    return figure


def _modal_invariance_figures(modal_signature: dict[str, Any]) -> tuple[go.Figure, go.Figure]:
    """Per-mode f_n/T60_n across every solo position, to check the joint fit's poles
    are genuinely room properties rather than an artifact of one measurement."""
    modes = modal_signature.get("modes", [])
    per_measurement = modal_signature.get("per_measurement", [])
    freq_figure = go.Figure()
    t60_figure = go.Figure()
    for mode_index, mode in enumerate(modes):
        color = _MODAL_PALETTE[mode_index % len(_MODAL_PALETTE)]
        label = f"{mode['frequency_hz']:.1f} Hz"
        titles: list[str] = []
        freqs: list[float] = []
        t60s: list[float] = []
        for entry in per_measurement:
            for candidate in entry.get("modes", []):
                if candidate.get("pooled_index") == mode_index:
                    titles.append(str(entry["title"]))
                    freqs.append(float(candidate["frequency_hz"]))
                    t60s.append(float(candidate["t60_s"]))
        marker = {"color": color, "size": 11}
        freq_figure.add_trace(
            go.Scatter(x=titles, y=freqs, mode="markers", name=label, marker=marker)
        )
        freq_figure.add_hline(y=float(mode["frequency_hz"]), line={"color": color, "width": 1, "dash": "dot"})
        t60_figure.add_trace(
            go.Scatter(x=titles, y=t60s, mode="markers", name=label, marker=marker)
        )
        t60_figure.add_hline(y=float(mode["t60_s"]), line={"color": color, "width": 1, "dash": "dot"})
    freq_figure.update_layout(
        title="Pole frequency invariance across solo positions",
        template="plotly_dark",
        height=360,
        yaxis={"title": "Frequency (Hz)"},
        margin={"l": 62, "r": 24, "t": 52, "b": 60},
    )
    t60_figure.update_layout(
        title="T60 invariance across solo positions",
        template="plotly_dark",
        height=360,
        yaxis={"title": "T60 (s)"},
        margin={"l": 62, "r": 24, "t": 52, "b": 60},
    )
    return freq_figure, t60_figure


def build_report(
    cache_dir: Path,
    results_path: Path,
    output_path: Path,
    top: int = 5,
    limit: int = 15,
    raw: bool = False,
    room_dimensions_cm: tuple[float, float, float] | None = None,
    report_title: str = "subpair report",
) -> Path:
    if limit < 1:
        raise ReportError("Report result limit must be at least 1")
    report_title = report_title.strip()
    if not report_title:
        raise ReportError("Report title must not be empty")
    results = load_results(results_path)
    measurements, _ = load_cache(cache_dir)
    if len(measurements) != int(results.get("measurement_count", -1)):
        raise ReportError("Cache measurement count does not match the search results")
    settings = results["settings"]
    band = tuple(float(value) for value in settings["band_hz"])
    room_modes = (
        room_mode_frequencies(room_dimensions_cm, band[1])
        if room_dimensions_cm is not None
        else None
    )
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
    eq_possible = eq_options.max_filters > 0
    raw = raw or not eq_possible
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
        "effective_tail_ms",
        "post_eq_effective_tail_ms",
        "effective_tail_db",
        "post_eq_effective_tail_db",
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
        "robustness",
        "fragility",
        "basin_tolerance_db",
        "basin_tolerance_ms",
        "excursion_half_width_ms",
        "excursion_penalty_db",
        "basin_w03",
        "worst_case_penalty",
        "geometric_pass",
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
    if int(results.get("format_version", 0)) < 22:
        raise ReportError(
            "Search results predate the modal-aware effective tail metric; "
            "run 'subpair search' again"
        )
    if int(results.get("format_version", 0)) < 23:
        raise ReportError(
            "Search results predate the dB-based modal ringing margin; "
            "run 'subpair search' again"
        )
    if int(results.get("format_version", 0)) < 24:
        raise ReportError(
            "Search results predate basin robustness scoring; run 'subpair search' again"
        )
    if int(results.get("format_version", 0)) < 25:
        raise ReportError(
            "Search results predate disqualifier gates; run 'subpair search' again"
        )
    required_pair_fields = {
        "rank",
        "eq_rank",
        "optimized",
        "verdict",
        "reasons",
        "gates",
    }
    if any(
        not required_pair_fields.issubset(pair)
        or (
            pair["optimized"]
            and not required_ranking_fields.issubset(pair)
        )
        for pair in results["pairs"]
    ):
        raise ReportError(
            "Search results contain incomplete gate/ranking data; run 'subpair search' again"
        )
    context = AnalysisContext(measurements, band, int(settings["ppo"]))
    mode = "raw" if raw else "eq"
    mode_label = "Raw" if raw else "EQ’d"
    rank_key = "rank" if raw else "eq_rank"
    score_key = "relative_score_db" if raw else "post_eq_relative_score_db"
    pairs = sorted(results["pairs"], key=lambda pair: int(pair[rank_key]))[:limit]
    score_settings = settings.get("ranking", {}).get("score", {})
    score_low_end_weight = float(score_settings.get("low_end_weight", 0.5))
    score_dip_weight = float(score_settings.get("dip_weight", 1.0))

    def pair_key(pair: dict[str, Any]) -> str:
        return f"{int(pair['first'])}-{int(pair['second'])}"

    optimized_pairs = [pair for pair in pairs if pair["optimized"]]
    default_selectable_pairs = [
        pair for pair in optimized_pairs if pair.get("verdict") != "reject"
    ]
    default_count = max(0, min(top, len(default_selectable_pairs)))
    default_keys = {
        pair_key(pair) for pair in default_selectable_pairs[:default_count]
    }
    initial_active_key = (
        pair_key(default_selectable_pairs[0]) if default_count else None
    )
    diagnostic_by_key: dict[str, dict[str, Any]] = {}
    for pair in optimized_pairs:
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

    overview = [
        (pair, diagnostic_by_key[pair_key(pair)]) for pair in optimized_pairs
    ]
    magnitude_range, excess_range = _selected_axis_ranges(
        overview,
        raw=raw,
        selected_keys=default_keys,
    )
    eq_notes = ""
    if not raw:
        eq_notes = (
            f"EQ: {html.escape(eq_options.target)} target, "
            f"{eq_range[0]:g}–{eq_range[1]:g} Hz, "
            f"{eq_options.correction_slope_db_per_octave:g} dB/oct curtain, "
            f"max boost {eq_options.max_boost_db:g} dB, up to "
            f"{eq_options.max_filters} EQ bands; excess-GD guarded."
        )
        if eq_options.low_shelf:
            eq_notes += (
                " Automatic low shelf enabled: its corner and gain/cut "
                "compete with PK filters and consume one EQ-band slot "
                "when selected."
            )
    score_formula_note = (
        f"{'Raw score' if raw else 'EQ’d score'}: {(1.0 - score_low_end_weight):g} × "
        f"full-band equal-drive SPL + {score_low_end_weight:g} × excursion-weighted "
        f"low-end power − {score_dip_weight:g} × worst dip below the one-third-octave "
        "smoothed response. Higher is better; the best pair is 0 dB."
    )
    headroom_note = (
        "Headroom is the negative global gain which removes positive pair "
        "gain"
        + ("—and, post-EQ, the fitted response’s maximum boost—" if eq_possible else ", ")
        + "so every pair uses the same maximum driver drive. It is applied "
        "to the magnitude comparisons, final summed response, scoring "
        "inputs, low-end power, and Relative SPL."
    )
    any_modal_tail = any(
        pair.get("effective_tail_is_modal") or pair.get("post_eq_effective_tail_is_modal")
        for pair in optimized_pairs
    )
    tail_note = (
        "Tail is this pair's loudest detected room mode's level relative to "
        "direct sound, in dB (marked “(modal)” in the pair summary; less "
        "negative is closer to audible) when this pair's own --modal fit "
        "succeeded, else the original CSD-based 1/3-octave envelope decay "
        "time in ms. The dB figure is shown instead of the modal fit's own "
        "ringing time (worst-case time for any detected mode to fall below "
        "the audibility margin) because that time saturates at 0 for every "
        "mode below the margin — which a well-controlled room can do for "
        "every pair — while the dB figure keeps varying below that floor. "
        "Both are diagnostics."
        + (
            " See the “Modal analysis” section below for per-mode detail."
            if any_modal_tail
            else ""
        )
    )
    robustness_settings = settings.get("robustness", {})
    listener_movement_cm = 100.0 * float(
        robustness_settings.get("listener_movement_m", 0.25)
    )
    gain_jitter_sigma_db = float(
        robustness_settings.get("gain_jitter_sigma_db", 0.5)
    )
    physical_delay_window_ms = float(
        robustness_settings.get("physical_delay_window_ms", 1.5)
    )
    gate_thresholds = settings.get("gates", {}).get("thresholds", {})
    robustness_foundations_note = (
        "<strong>Delay-robustness basis.</strong> These are raw, pre-EQ diagnostics "
        "for the polarity and gain shown in the row. <strong>f(τ)</strong> is the "
        "negative raw usable-output score as delay τ is swept, so lower is better. "
        "<strong>τ*</strong> is the delay at the minimum of raw f. "
        "<strong>f_robust(τ)</strong> Gaussian-averages f over timing and gain "
        f"uncertainty: the timing σ is half each pair’s geometric delay-excursion "
        f"bound, and gain σ is {gain_jitter_sigma_db:g} dB (approximately "
        f"±{2.0 * gain_jitter_sigma_db:g} dB at ±2σ). The reported robust delay "
        "minimizes f_robust inside the measured physical-delay window when that "
        "window is available."
    )
    robustness_columns_note = (
        "<strong>Robustness columns.</strong> <strong>Fragility</strong> is "
        "f_robust(τ*) − f(τ*): the score penalty created by jitter at the raw "
        "optimum; lower is better. <strong>Basin +0.3</strong> is the width of the "
        "single contiguous delay interval containing τ* whose raw f stays within "
        "+0.3 dB of its minimum; disconnected good regions do not count, and wider "
        "is better. <strong>Excursion penalty</strong> is the largest raw-f "
        "degradation over ±Δτ_max/2 around the <em>recommended</em> delay, and it "
        "drives the geometry gate: reject above "
        f"{float(gate_thresholds.get('basin_tolerance_db', 0.5)):g} dB. It is an "
        "absolute dB figure on purpose — a tolerance scaled to each pair’s own "
        "objective range normalises away the delay-insensitivity the gate is "
        "testing for, and can fail a pair whose score barely moves with delay at "
        "all. <strong>Worst penalty ±1 ms</strong> is the same degradation "
        "measured over a fixed ±1 ms around the recommended delay rather than "
        "the geometry-derived window, so it stays comparable across pairs "
        "whatever listener movement is configured; lower is better. Competing minima, shown in each pair summary, counts "
        "distinct local minima within +0.3 dB of the best raw minimum."
    )
    robustness_status_note = (
        "<strong>Geometry and physical status.</strong> The geometric delay-excursion "
        f"bound models up to {listener_movement_cm:g} cm of listener movement from "
        "configured source/listener coordinates; without complete coordinates it "
        "uses the conservative 2d/c bound. <strong>Basin vs geometry</strong> is "
        "PASS when the adaptive basin is at least as wide as that bound. "
        "<strong>Physical</strong> compares τ* with the loopback-referenced arrival "
        "difference (first arrival − second arrival, because delay is applied to "
        f"sub 2): OK is within ±{physical_delay_window_ms:g} ms, OUT is outside, "
        "INVALID marks an arrival-delay outlier or a physical window outside the "
        "scan, and N/A means arrival metadata is unavailable."
    )
    robustness_graph_note = (
        "<strong>Robustness graph.</strong> The blue curve is raw f(τ), the purple "
        "curve is jitter-averaged f_robust(τ), the yellow diamond is τ*, and the "
        "green circle is the reported robust delay. The green band is the contiguous "
        "adaptive basin; the dashed red line is the arrival-derived physical delay, "
        "and the dotted grey line is the best-fit mirror axis after removing the "
        "broad f(τ) envelope. Mirror correlation and axis offset are visual timing/"
        "redundancy diagnostics only, never a gate. Lower curves and a broad, shallow basin are preferable. "
        "Several similarly deep valleys indicate competing alignment solutions. "
        "Hover the curves and markers for exact delay/objective values."
    )
    table_colour_note = (
        "<strong>Table colors.</strong> Colored metric cells compare the rows shown "
        "in this report: green with an inset outline is best and red is worst. "
        "Fragility, worst penalty, residual dip, excess GD, and Tail prefer lower values; "
        "Score, Basin width, low-end power, and Relative SPL prefer higher values. "
        "For the gate columns, A residual and D deficit prefer higher values, while "
        "B correlation, C percentile, E comb, F notch count, G gain offset, H edge "
        "spread, and I localisation prefer lower values. "
        "PASS/OK use fixed categorical scales so a table full of failures cannot "
        "appear green; unavailable values remain grey. Click any heading to sort "
        "ascending, then click it again for descending order."
    )
    gate_pipeline_note = (
        "<strong>Verdict pipeline.</strong> ACCEPT means every evaluated gate passed; "
        "CAUTION means no hard rejection but at least one caution; REJECT means one "
        "or more hard failures. A/B run before the delay optimiser, followed by the "
        "cheap physical-delay C screen. A hard failure there stops optimisation, "
        "but the row and its measured reasons remain visible. Basin and D–I run on "
        "the chosen configuration. Verdict sorts before score."
    )
    gate_ab_note = (
        "<strong>A — redundancy residual.</strong> Fits measurement B as a complex "
        "scale and delay of A over the scoring band. Lower means the placements carry "
        "less independent spatial information; reject below "
        f"{float(gate_thresholds.get('redundancy_reject', 0.50)):g}, caution below "
        f"{float(gate_thresholds.get('redundancy_caution', 0.60)):g}. "
        "<strong>B — ripple correlation.</strong> Correlates each solo magnitude "
        "after subtracting its one-octave trend. Positive correlation means their "
        "errors reinforce; reject above "
        f"{float(gate_thresholds.get('ripple_correlation_reject', 0.30)):g}. "
        f"Values below {float(gate_thresholds.get('ripple_complementary', -0.10)):g} "
        "are labelled complementary but do not improve rank."
    )
    gate_cd_note = (
        "<strong>C — physical-delay percentile.</strong> Scores the normal-polarity, "
        "equal-gain sum at the header-derived arrival alignment and locates it in that "
        "pair’s delay landscape; reject above the "
        f"{float(gate_thresholds.get('physical_percentile_reject', 75.0)):g}th "
        "percentile. The absolute f(physical τ) − f(τ*) gap is also reported. "
        "<strong>D — cancellation deficit.</strong> Coherent mean level minus the "
        "matching incoherent power sum is evaluated at physical and chosen delay; "
        "negative values mean flatness was bought by cancellation. Reject below "
        f"{float(gate_thresholds.get('cancellation_deficit_reject_db', -3.0)):g} dB "
        "and caution below "
        f"{float(gate_thresholds.get('cancellation_deficit_caution_db', -1.0)):g} dB."
    )
    gate_ef_note = (
        "<strong>E — comb signature.</strong> Detrends the chosen sum on a uniform "
        "linear-frequency grid and measures normalized autocorrelation at 1/|τ| and "
        "its harmonics; larger values are more comb-like. Caution/reject thresholds "
        f"are {float(gate_thresholds.get('comb_index_caution', 0.40)):g}/"
        f"{float(gate_thresholds.get('comb_index_reject', 0.65)):g}. "
        "<strong>F — residual notches.</strong> Rejects a local null deeper than "
        f"{float(gate_thresholds.get('notch_depth_reject_db', 8.0)):g} dB below its "
        "one-octave trend when its −3 dB width is no more than "
        f"{float(gate_thresholds.get('notch_max_width_octaves', 1.0 / 6.0)):g} octave."
    )
    gate_ghi_note = (
        "<strong>G — gain asymmetry.</strong> Cautions when the chosen offset exceeds "
        f"{float(gate_thresholds.get('gain_asymmetry_caution_db', 4.0)):g} dB; the "
        "report records boundary/headroom information, while actual hardware limits "
        "remain unknown without amplifier/driver data. <strong>H — band-edge "
        "stability.</strong> Shifts the whole scoring window down and up by 1/6 octave "
        "and rejects when this pair’s score spread exceeds the median spread across "
        "every scored pair by more than "
        f"{float(gate_thresholds.get('band_edge_excess_spread_reject_db', 1.0)):g} dB. "
        "The raw spread is nearly identical for every pair — the subs roll off below "
        "the band and low-end power weights f⁻⁴, so an up-shifted band always scores "
        "higher — and that common-mode term belongs to the score, not to any pair. "
        "<strong>I — improvement localisation.</strong> Compares detrended ripple "
        "against the physical-alignment response (equal gain, better polarity) and "
        "reports the largest share of all positive improvement concentrated in a "
        "sliding 1/6-octave region, alongside the total score change; reject above "
        f"{100.0 * float(gate_thresholds.get('localization_fraction_reject', 0.5)):g}%, "
        "but only once the mean improvement reaches "
        f"{float(gate_thresholds.get('localization_min_mean_improvement_db', 0.25)):g} dB "
        "per bin — below that the share is a ratio of two noise-level sums."
    )
    detail_sections = []
    for pair in optimized_pairs:
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
        if pair.get("effective_tail_is_modal"):
            raw_tail_text = f"tail {pair['effective_tail_db']:+.1f} dB (modal)"
        else:
            raw_tail_text = f"tail {pair['effective_tail_ms']:.1f} ms"
        if pair.get("post_eq_effective_tail_is_modal"):
            post_eq_tail_text = f"tail {pair['post_eq_effective_tail_db']:+.1f} dB (modal)"
        else:
            post_eq_tail_text = f"tail {pair['post_eq_effective_tail_ms']:.1f} ms"
        metric_summary = (
            f"Raw score {pair['relative_score_db']:+.2f} dB · "
            f"residual dip {pair['dip_db']:.3f} dB · "
            f"excess GD {pair['excess_gd_ms']:.3f} ms · "
            f"peak {pair['excess_gd_peak_ms']:.2f} ms · "
            f"{raw_tail_text}"
            if raw
            else (
                f"EQ’d score {pair['post_eq_relative_score_db']:+.2f} dB · "
                f"residual dip {pair['post_eq_dip_db']:.3f} dB · "
                f"excess GD {pair['post_eq_excess_gd_ms']:.3f} ms · "
                f"peak {pair['post_eq_excess_gd_peak_ms']:.2f} ms · "
                f"{post_eq_tail_text}"
            )
        )
        robustness = pair["robustness"]
        physical_text = (
            f"{float(robustness['physical_tau_ms']):+.3f} ms"
            if robustness.get("physical_tau_ms") is not None
            else "unavailable"
        )
        geometry_text = (
            "conservative 2d/c bound"
            if robustness.get("geometry_conservative_bound")
            else "configured coordinates"
        )
        if robustness.get("arrival_delay_repaired"):
            physical_warning = (
                " <strong>ARRIVAL TIMING REPAIRED — delay window widened, "
                "Gate C advisory</strong>"
            )
        elif robustness.get("measurement_delay_outlier"):
            physical_warning = (
                " <strong>ARRIVAL-DELAY OUTLIER: PHYSICAL TIMING DISCARDED</strong>"
            )
        elif not robustness.get("physical_window_in_scan", True):
            physical_warning = " <strong>INVALID: PHYSICAL WINDOW OUTSIDE SCAN</strong>"
        elif pair.get("non_physical_solution"):
            physical_warning = " <strong>NON-PHYSICAL RAW OPTIMUM</strong>"
        else:
            physical_warning = ""
        robustness_summary = (
            f"Raw tau* {pair['tau_star']:+.3f} ms · robust tau {pair['tau_robust']:+.3f} ms · "
            f"f(tau*) {pair['f_tau_star']:.2f} dB · "
            f"f_robust(tau_robust) {pair['f_robust_tau_robust']:.2f} dB · "
            f"fragility {pair['fragility']:.2f} dB · +0.3 dB basin {pair['basin_w03']:.2f} ms · "
            f"+{pair['basin_tolerance_db']:.2f} dB basin at the recommended delay "
            f"{pair['basin_tolerance_ms']:.2f} ms · "
            f"excursion penalty (+/-{pair['excursion_half_width_ms']:.2f} ms) "
            f"{pair['excursion_penalty_db']:.2f} dB · "
            f"worst penalty (+/-1 ms) {pair['worst_case_penalty']['1.0']:.2f} dB · "
            f"{pair['n_competing']} competing minima · "
            f"physical tau {physical_text} · geometric excursion "
            f"{robustness['delta_tau_max_ms']:.2f} ms ({geometry_text}) · "
            f"basin {'PASS' if pair['geometric_pass'] else 'FAIL'}"
        )
        gate_rows = []
        gate_names = GATE_LABELS
        for gate_key, gate_name in gate_names.items():
            gate = pair["gates"][gate_key]
            measured = {
                key: value
                for key, value in gate.items()
                if key not in {"status", "stage", "offenders"}
            }
            status = str(gate.get("status", "not_run"))
            terms = "".join(
                f"<dt>{html.escape(str(name))}</dt><dd>{_gate_value_html(value)}</dd>"
                for name, value in sorted(measured.items())
            )
            tooltip = html.escape(_gate_tooltip(gate_key, gate, include_status=False))
            gate_rows.append(
                f'<li class="gate-{html.escape(status)} has-help" data-help="{tooltip}">'
                f'<strong>{html.escape(gate_name)}: '
                f'{html.escape(status.upper())}</strong>'
                f'<p class="gate-meaning">{html.escape(GATE_MEANINGS.get(gate_key, ""))}</p>'
                + (f"<dl>{terms}</dl>" if terms else "")
                + "</li>"
            )
        counts = Counter(
            str(pair["gates"][key].get("status", "not_run")) for key in gate_names
        )
        tally = " · ".join(
            f"{counts[name]} {name}"
            for name in ("reject", "caution", "pass", "not_run")
            if counts[name]
        )
        gate_sheet_html = (
            '<details class="gate-sheet"><summary><strong>Verdict: '
            f'{html.escape(pair["verdict"].upper())}</strong> — {html.escape(tally)}'
            "</summary>"
            '<p class="note">A clean sheet is not validation; these are single-position '
            'disqualifiers only.</p><ul>'
            + "".join(gate_rows)
            + "</ul></details>"
        )
        peq_html = ""
        if not raw:
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
        modal_html = ""
        pair_modal = pair.get("modal")
        if pair_modal and pair_modal.get("valid") and pair_modal.get("modes"):
            robustness = pair_modal.get("robustness") or {}
            robustness_note = (
                f" · n_highQ stable in {robustness['fraction_stable'] * 100.0:.0f}% of a "
                f"±{robustness['delay_jitter_ms']:.2f} ms / ±{robustness['gain_jitter_db']:.0f} dB "
                "neighbourhood"
                if robustness.get("valid")
                else ""
            )
            modal_html = (
                '<div class="modal-block"><h3>Modal resonance (diagnostic only)</h3>'
                f'<p class="note">n_highQ (Q &gt; {pair_modal["high_q_threshold"]:g}, '
                f'L &gt; {pair_modal["primary_gate_db"]:+g} dB) = {pair_modal["n_high_q"]}'
                + (
                    f' · stored energy {pair_modal["sum_modal_energy_db"]:+.1f} dB'
                    if pair_modal.get("sum_modal_energy_db") is not None
                    else ""
                )
                + robustness_note
                + "</p>"
                + f'<div class="table-wrap">{_modal_mode_table(pair_modal["modes"], f"modal-{key}")}</div>'
                + "</div>"
            )
        detail_sections.append(
            f"""
            <section class="{detail_class}" data-pair-key="{key}"
              data-pair-label="{pair['first']}+{pair['second']}"
              data-rank="{pair[rank_key]}"{axis_attributes}>
              <h2>{html.escape(pair['verdict'].upper())} · {mode_label} score {pair[score_key]:+.2f} dB:
                {html.escape(pair['first_name'])} + {html.escape(pair['second_name'])}</h2>
              <p class="configuration">Sub 2: {'normal' if pair['polarity'] > 0 else 'inverted'},
                robust delay {pair['delay_ms']:+.3f} ms, gain {pair['gain_db']:+.2f} dB,
                headroom {pair['post_eq_headroom_db' if not raw else 'headroom_db']:+.2f} dB{
                  f'<br>{physical_warning.strip()}' if physical_warning else ''
                }</p>
              <details class="measured-card">
                <summary><strong>Measured values</strong> — score components, delay
                  robustness and physical timing</summary>
                <p class="configuration">{metric_summary} · score resolution
                  {pair['score_resolution_db']:.2f} dB</p>
                <p class="configuration">{robustness_summary}</p>
              </details>
              {gate_sheet_html}
              {_plot_html(
                  _magnitude_figure(pair, data, raw=raw, y_range=magnitude_range, room_modes=room_modes),
                  f'magnitude-{key}',
              )}
              {_plot_html(
                  _decay_figure(data, raw=raw, room_modes=room_modes),
                  f'decay-{key}',
                  static=not room_modes,
              )}
              {_plot_html(_robustness_figure(pair), f'robustness-{key}')}
              {_plot_html(
                  _excess_figure(data, raw=raw, y_range=excess_range, room_modes=room_modes),
                  f'excess-{key}',
              )}
              {peq_html}
              {modal_html}
            </section>
            """.strip()
        )

    default_json = json.dumps(
        [pair_key(pair) for pair in default_selectable_pairs[:default_count]],
        separators=(",", ":"),
    )
    copy_peq_script = "" if raw else """
function copyPeq(button) {
  navigator.clipboard.writeText(button.parentElement.querySelector('pre').innerText);
  const old=button.innerText; button.innerText='Copied'; setTimeout(()=>button.innerText=old,900);
}
""".strip()

    modal_signature = results.get("modal_signature")
    modal_section_html = ""
    if modal_signature and modal_signature.get("valid"):
        modal_rows = [
            (pair, pair["modal"])
            for pair in optimized_pairs
            if pair.get("modal", {}).get("valid") and pair["modal"].get("modes")
        ]
        pole_map_html = (
            _plot_html(_modal_pole_map_figure(modal_rows), "modal-pole-map")
            if modal_rows
            else "<p class=\"note\">No displayed pair has a gated high-Q mode.</p>"
        )
        freq_figure, t60_figure = _modal_invariance_figures(modal_signature)
        discard_pct = float(modal_signature.get("discard_fraction", 0.0)) * 100.0
        modal_warnings = "".join(
            f"<p class=\"note\">{html.escape(str(warning))}</p>"
            for warning in modal_signature.get("warnings", [])
        )
        modal_section_html = f"""
        <details class="modal-section" open>
        <summary>Modal analysis (matrix-pencil pole estimation, diagnostic only)</summary>
        <p class="note">{len(modal_signature.get('modes', []))} room mode(s) retained jointly
          across every solo position ({discard_pct:.0f}% of candidate poles discarded as
          noise/order-inconsistent). Modal metrics do not affect ranking unless
          <code>--modal-tiebreak</code> was enabled at search time; see the settings JSON below.</p>
        {modal_warnings}
        {pole_map_html}
        <div class="modal-figures">
          {_plot_html(freq_figure, "modal-invariance-frequency")}
          {_plot_html(t60_figure, "modal-invariance-t60")}
        </div>
        </details>
        """.strip()

    timing = results.get("arrival_timing") or {}
    timing_rows = timing.get("measurements") or []
    repaired_any = any(row.get("repaired") for row in timing_rows)
    if timing_rows:
        def _cell(value: object, digits: int = 3) -> str:
            return "—" if value is None else f"{float(value):.{digits}f}"

        body_rows = "".join(
            "<tr class=\"{cls}\"><td>{pos}</td><td>{title}</td><td>{rep}</td>"
            "<td>{onset}</td><td>{lag}</td><td>{dev}</td><td>{res}</td></tr>".format(
                cls="timing-repaired" if row.get("repaired") else "",
                pos=int(row.get("position", 0)),
                title=html.escape(str(row.get("title", ""))),
                rep=_cell(row.get("reported_ms")),
                onset=_cell(row.get("onset_ms")),
                lag=_cell(row.get("peak_minus_onset_ms")),
                dev=_cell(row.get("lag_deviation_ms")),
                res=(
                    f"<strong>{_cell(row.get('resolved_ms'))} (repaired)</strong>"
                    if row.get("repaired")
                    else "as reported"
                ),
            )
            for row in timing_rows
        )
        band_hz = timing.get("onset_band_hz") or [0.0, 0.0]
        timing_warning_html = "".join(
            f'<p class="warning"><strong>Warning:</strong> {html.escape(str(text))}</p>'
            for text in timing.get("warnings", [])
        )
        repaired_count = sum(1 for row in timing_rows if row.get("repaired"))
        timing_section = f"""
        <details class="card timing-card">
        <summary><strong>Arrival timing</strong> — {
            f'{repaired_count} repaired' if repaired_any else 'all peaks consistent'
        }</summary>
        {timing_warning_html}
        <p class="note">REW reports arrival delay as the position of the largest sample in the
        impulse response. Across a subwoofer's two or three octaves that impulse is a slow
        oscillatory blob, so wherever a room mode rings hard a later half-cycle can outgrow the
        direct arrival and the pick jumps a whole cycle. The leading edge does not jump, so a
        measurement whose peak-minus-onset lag departs from the cache median by more than
        {float(timing.get('slip_tolerance_ms', 0.0)):g} ms is rebuilt from its own onset plus the
        median lag ({_cell(timing.get('median_peak_minus_onset_ms'))} ms), which keeps that
        position's real distance instead of flattening it to the median arrival. Onsets are taken
        over {band_hz[0]:g}–{band_hz[1]:g} Hz at {float(timing.get('onset_threshold_db', 0.0)):g} dB.
        Only differences between arrivals are used downstream, so a bias common to every onset
        cancels and is left uncorrected.</p>
        <p class="note">A repaired arrival still aims the delay search, on a window widened to
        reflect that it is a reconstruction, and its pairs' Gate C is capped at caution: a poor
        timing pick changes how exact the reported delay figure is, never which pairs are
        recommended.</p>
        <table class="timing-table"><thead><tr>
        <th>Pos</th><th>Title</th><th>REW peak (ms)</th><th>Onset (ms)</th>
        <th>Peak−onset (ms)</th><th>Deviation (ms)</th><th>Used</th>
        </tr></thead><tbody>{body_rows}</tbody></table>
        </details>
        """.strip()
    else:
        timing_section = ""

    resolutions = [
        float(pair["score_resolution_db"])
        for pair in pairs
        if pair.get("score_resolution_db") is not None
    ]
    tied = [
        f"{pair['first']}+{pair['second']}"
        for pair in pairs
        if pair.get("score_ties_reference")
    ]
    margin = float(
        (settings.get("ranking", {}).get("score_tie_margin_db") or 0.0)
    )
    if resolutions:
        resolution_note = (
            '<p class="note"><strong>Score resolution.</strong> The smallest score '
            f"difference these pairs can be ordered by runs {min(resolutions):.2f}–"
            f"{max(resolutions):.2f} dB: how far the score moves between adjacent "
            "points of the delay/gain grid, plus each pair's excess over the "
            "population median in the band-edge shift"
            + (f", plus the {margin:g} dB tie margin" if margin else "")
            + ". That is a floor — microphone repositioning, level drift and "
            "measurement noise all add to it and none are visible in a single "
            "cached sweep"
            + (
                "; <code>--score-tie-margin</code> adds a flat allowance for them"
                if not margin
                else ""
            )
            + ". Rows marked <code>=</code> are within their own resolution of the "
            "0 dB reference"
            + (f" ({', '.join(tied)})" if tied else "")
            + " and are not meaningfully ordered against it.</p>"
        )
    else:
        resolution_note = ""

    settings_json = html.escape(json.dumps(settings, sort_keys=True, indent=2))
    escaped_report_title = html.escape(report_title)
    inline_logo, favicon_href = _brand_assets()
    grain_texture = _svg_data_uri(_GRAIN_SVG)
    # Arrival warnings are rendered inside the Arrival timing card next to the
    # table that explains them, so they are not repeated at the top of the page.
    timing_warnings = set(timing.get("warnings", []))
    warning_html = "".join(
        f'<p class="warning"><strong>Warning:</strong> {html.escape(str(warning))}</p>'
        for warning in results.get("warnings", [])
        if warning not in timing_warnings
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escaped_report_title}</title>
<link rel="icon" type="image/svg+xml" href="{favicon_href}">
<style>
:root {{ color-scheme: dark; --bg:#07111f; --card:#0f1b2d; --line:#26364d; --text:#e5edf8; --muted:#9fb0c7; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; min-height:100vh; isolation:isolate; color:var(--text); font:15px/1.5 ui-sans-serif,system-ui,sans-serif;
  background:radial-gradient(circle at 8% -8%,rgba(56,189,248,.16),transparent 34rem),
    radial-gradient(circle at 96% 16%,rgba(139,92,246,.1),transparent 38rem),
    linear-gradient(145deg,#07111f 0%,#0a1628 48%,#050b14 100%); background-attachment:fixed; }}
body::before {{ content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background-image:url("{grain_texture}"); background-size:180px 180px; opacity:.055; mix-blend-mode:soft-light; }}
main {{ position:relative; z-index:1; width:min(1500px,96vw); margin:0 auto; padding:36px 0 80px; }}
.brand {{ display:flex; align-items:center; gap:14px; margin:0 0 4px; }}
.brand-mark {{ display:block; flex:0 0 auto; width:48px; height:48px; filter:drop-shadow(0 8px 18px rgba(56,189,248,.12)); }}
h1 {{ font-size:2.2rem; line-height:1.1; margin:0; }} h2 {{ margin-top:0; }}
.lede,.configuration,.note {{ color:var(--muted); }}
.warning {{ color:#fecaca; border-left:3px solid #fb7185; padding-left:10px; }}
.timing-card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 16px; margin:14px 0; }}
.timing-card summary {{ cursor:pointer; }}
.timing-table {{ border-collapse:collapse; margin-top:10px; font-variant-numeric:tabular-nums; }}
.timing-table th, .timing-table td {{ border-bottom:1px solid var(--line); padding:4px 12px 4px 0; text-align:right; }}
.timing-table th:nth-child(2), .timing-table td:nth-child(2) {{ text-align:left; }}
.timing-table tr.timing-repaired {{ color:#fed7aa; }}
.chart-tabs,.pair-tabs {{ display:flex; flex-wrap:wrap; gap:6px; margin:14px 0; }}
.chart-tab,.pair-tab {{ border:1px solid #3b506d; border-radius:7px; padding:7px 13px; color:var(--muted); background:#101e31; cursor:pointer; font-weight:650; }}
.chart-tab.active,.pair-tab.active {{ color:#07111f; border-color:#7dd3fc; background:#7dd3fc; }}
.pair-tabs {{ min-height:39px; margin:14px 0 2px; }}
.empty-selection {{ align-self:center; color:var(--muted); }}
.overview-panels,#pair-details {{ position:relative; }}
.overview-panel,.pair-detail {{ width:100%; }}
.overview-panel.is-inactive,.pair-detail.is-inactive {{ position:absolute; inset:0; visibility:hidden; pointer-events:none; }}
.table-controls {{ display:flex; align-items:center; flex-wrap:wrap; gap:8px; margin:14px 0; }}
.table-controls button {{ border:1px solid #3b506d; border-radius:7px; padding:7px 13px; color:var(--muted); background:#101e31; cursor:pointer; font-weight:650; }}
.table-controls button:hover {{ background:#1b2d48; color:var(--text); }}
.reject-toggle {{ display:flex; align-items:center; gap:7px; margin-left:4px; padding:7px 2px; color:var(--muted); cursor:pointer; font-weight:650; }}
.reject-toggle input {{ width:17px; height:17px; margin:0; accent-color:#7dd3fc; cursor:pointer; }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:12px; background:var(--card); }}
table {{ width:100%; border-collapse:collapse; white-space:nowrap; }}
th,td {{ padding:10px 13px; text-align:right; border-bottom:1px solid var(--line); }}
th:nth-child(3),td:nth-child(3),th:nth-child(5),td:nth-child(5) {{ text-align:left; }}
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
.gate-sheet {{ margin:16px 0; padding:16px; background:#091322; border-radius:9px; }}
.gate-sheet > summary {{ cursor:pointer; font-size:1.05rem; }}
.measured-card {{ margin:12px 0; padding:12px 16px; background:#091322; border-radius:9px; }}
.measured-card > summary {{ cursor:pointer; }}
.measured-card p {{ margin:10px 0 0; font-size:.9rem; }}
.gate-meaning {{ margin:6px 0 0; color:var(--muted); font-size:.86rem; }}
.has-help {{ cursor:help; }}
.ranking-table th.has-help {{ text-decoration:underline dotted var(--muted); text-underline-offset:3px; }}
.hover-help[hidden] {{ display:none; }}
.hover-help {{ position:fixed; z-index:60; max-width:min(420px,86vw); padding:9px 11px;
  background:#0b1728; color:var(--text); border:1px solid var(--line); border-radius:7px;
  box-shadow:0 6px 20px rgba(0,0,0,.5); font-size:.82rem; line-height:1.45;
  white-space:pre-line; pointer-events:none; }}
.gate-sheet ul {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:8px; padding:0; list-style:none; }}
.gate-sheet li {{ padding:10px; border:1px solid var(--line); border-left-width:4px; border-radius:7px; overflow-wrap:anywhere; }}
.gate-sheet li span {{ color:var(--muted); font-size:.86rem; }}
.gate-sheet dl {{ display:grid; grid-template-columns:auto 1fr; gap:2px 10px; margin:8px 0 0; font-size:.86rem; }}
.gate-sheet dt {{ color:var(--muted); font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
.gate-sheet dd {{ margin:0; font-variant-numeric:tabular-nums; }}
.gate-pass {{ border-left-color:#22c55e !important; }}
.gate-caution {{ border-left-color:#f59e0b !important; }}
.gate-reject {{ border-left-color:#ef4444 !important; }}
.gate-not_run {{ border-left-color:#64748b !important; }}
.modal-block {{ margin-top:18px; padding:16px; background:#091322; border-radius:9px; }}
.modal-block h3 {{ margin:0 0 10px; }}
.modal-section .table-wrap {{ margin-top:12px; }}
.modal-figures {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:12px; }}
@media (max-width:900px) {{ .modal-figures {{ grid-template-columns:1fr; }} }}
details {{ margin:22px 0; }} details pre {{ overflow:auto; color:var(--muted); }}
.glossary p {{ margin:0 0 10px; color:var(--muted); }} .glossary p:last-child {{ margin-bottom:0; }}
</style>
<script>{get_plotlyjs()}</script>
</head>
<body><main>
<header class="brand">{inline_logo}<h1>{escaped_report_title}</h1></header>
<p class="lede">{results['measurement_count']} positions · {band[0]:g}–{band[1]:g} Hz · {settings['ppo']} points/octave · {mode_label} usable-output score · showing up to {limit} pairs</p>
<p class="note">Check table rows to choose comparison pairs. The top {default_count} start selected; the pair tabs below the table open one full diagnostic at a time. Hotkeys 1–9 open the first nine selected tabs.</p>
{warning_html}
{timing_section}
{f'<p class="note">Room {room_dimensions_cm[0]:g}×{room_dimensions_cm[1]:g}×{room_dimensions_cm[2]:g} cm: theoretical rigid-box mode frequencies are overlaid on frequency charts (vertical) and CSD heatmaps (horizontal) — solid axial, dashed tangential, dotted oblique. Click a &ldquo;Room mode: …&rdquo; legend entry to hide/show that type; a purely geometric reference, not the measured poles from <code>--modal</code>.</p>' if room_modes is not None else ''}
<div class="chart-tabs" role="tablist" aria-label="{mode_label} overview chart">
  <button class="chart-tab active" data-overview-view="magnitude" role="tab" aria-selected="true" onclick="setOverviewView('magnitude')">Magnitude</button>
  <button class="chart-tab" data-overview-view="excess" role="tab" aria-selected="false" onclick="setOverviewView('excess')">Excess GD</button>
</div>
<div class="overview-panels">
  <div class="overview-panel" data-overview-panel data-overview-view="magnitude">
    {_plot_html(
        _overview_figure(overview, mode, default_keys, magnitude_range, room_modes),
        f'selected-pairs-magnitude-{mode}',
    )}
  </div>
  <div class="overview-panel is-inactive" data-overview-panel data-overview-view="excess">
    {_plot_html(
        _overview_excess_figure(overview, mode, default_keys, excess_range, room_modes),
        f'selected-pairs-excess-{mode}',
    )}
  </div>
</div>
<p class="note">Verdict is primary: ACCEPT rows sort before CAUTION, and REJECT rows always sort last regardless of score. Rejected rows are hidden by default; use <strong>Show rejected</strong> to reveal them. Early A/B/C rejects are intentionally not optimized and cannot be selected. Score remains unchanged and 0 dB marks the highest numeric score among optimized pairs. Click a heading to sort; colored metric cells run green (best) to red (worst), while unavailable values are grey.</p>
{resolution_note}
<div class="table-controls">
  <button type="button" onclick="selectTopN(0)">Clear</button>
  <button type="button" onclick="selectTopN(3)">Top 3</button>
  <button type="button" onclick="selectTopN(5)">Top 5</button>
  <label class="reject-toggle"><input id="show-rejected" type="checkbox" onchange="setRejectedVisibility(this.checked)">Show rejected</label>
</div>
<div class="table-wrap">{_ranking_table(pairs, mode, f'ranking-{mode}', default_keys, show_headroom=eq_possible)}</div>
<div class="pair-tabs" data-pair-tabs role="tablist" aria-label="Selected {mode_label} pairs"></div>
<div id="pair-details">{''.join(detail_sections)}</div>
<details><summary>Score &amp; metric notes</summary><div class="glossary">
<p>{score_formula_note}</p>
<p>{headroom_note}</p>
<p>Low-end power weights the broad response through 100 Hz by the amplifier/excursion cost of producing pressure at each frequency (+12.04 dB per octave downward). Excess GD and tail remain diagnostics; they do not alter Score.</p>
<p>{tail_note}</p>
<p>{robustness_foundations_note}</p>
<p>{robustness_columns_note}</p>
<p>{robustness_status_note}</p>
<p>{robustness_graph_note}</p>
<p>{table_colour_note}</p>
<p>{gate_pipeline_note}</p>
<p>{gate_ab_note}</p>
<p>{gate_cd_note}</p>
<p>{gate_ef_note}</p>
<p>{gate_ghi_note}</p>
<p>Delay fragility is a disqualifier, not a certificate. A narrow basin proves the timing solution cannot survive the configured listener excursion. A wide basin only shows timing tolerance: it does not model how each sub’s magnitude response changes as the microphone moves through the modal pressure field. Validate a listening area with multi-position measurements.</p>
{f'<p>{eq_notes}</p>' if eq_notes else ''}
<p>CSD overlay (in each pair's excess-GD and decay charts): excess GD with common delay removed; a vertical line is frequency-independent delay.</p>
</div></details>
{modal_section_html}
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
      const aMissing=av==='', bMissing=bv==='';
      if(aMissing||bMissing) return aMissing===bMissing?0:(aMissing?1:-1);
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
      if(!trace.meta||trace.meta.pair_key===undefined) return;
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
function selectTopN(n) {{
  const table=document.getElementById('ranking-'+reportMode);
  const rows=Array.from(table.querySelectorAll('tbody tr'));
  selectedPairs.clear();
  let selectedCount=0;
  rows.forEach(row=>{{
    const checkbox=row.querySelector('.pair-select');
    const select=!row.hidden&&!checkbox.disabled&&selectedCount<n;
    checkbox.checked=select;
    if(select) {{ selectedPairs.add(row.dataset.pairKey); selectedCount+=1; }}
  }});
  renderPairTabs();
  updateOverview();
  renderActiveDetail();
}}
function setRejectedVisibility(show) {{
  const table=document.getElementById('ranking-'+reportMode);
  table.querySelectorAll('tbody tr[data-verdict="reject"]').forEach(row=>{{
    row.hidden=!show;
    if(!show) {{
      const checkbox=row.querySelector('.pair-select');
      checkbox.checked=false;
      selectedPairs.delete(row.dataset.pairKey);
    }}
  }});
  renderPairTabs();
  updateOverview();
  renderActiveDetail();
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
// Hover help. A single element parented to <body> rather than a CSS ::after on
// each target: the ranking table scrolls inside its own overflow container, and
// a pseudo-element tooltip would be clipped at that container's edge. Shown on
// pointerover with no delay, and positioned to stay inside the viewport.
(function(){{
  const tip=document.createElement('div');
  tip.className='hover-help';
  tip.hidden=true;
  document.body.appendChild(tip);
  let current=null;
  function place(target){{
    const box=target.getBoundingClientRect();
    tip.style.left='0px'; tip.style.top='0px';
    const size=tip.getBoundingClientRect();
    const margin=8;
    let left=box.left+box.width/2-size.width/2;
    left=Math.max(margin,Math.min(left,window.innerWidth-size.width-margin));
    let top=box.bottom+6;
    if(top+size.height>window.innerHeight-margin) top=box.top-size.height-6;
    const lowest=Math.max(margin,window.innerHeight-size.height-margin);
    top=Math.max(margin,Math.min(top,lowest));
    tip.style.left=left+'px';
    tip.style.top=top+'px';
  }}
  function hide(){{ current=null; tip.hidden=true; }}
  document.addEventListener('pointerover',event=>{{
    const node=event.target;
    const target=node&&node.closest?node.closest('[data-help]'):null;
    if(target===current) return;
    if(!target){{ hide(); return; }}
    current=target;
    tip.textContent=target.dataset.help;
    tip.hidden=false;
    place(target);
  }});
  document.addEventListener('pointerdown',hide);
  window.addEventListener('scroll',hide,true);
  window.addEventListener('blur',hide);
}})();
renderPairTabs();
setOverviewView('magnitude');
renderActiveDetail();
updateSharedYAxisRanges();
</script></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path
