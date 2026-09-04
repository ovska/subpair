"""Command-line interface for subpair."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

from .api import RewApiError, RewClient
from .cache import CacheError, write_cache
from .engine import GateThresholds, SearchOptions, run_search
from .html_report import ReportError, build_report
from .verification import VerificationError, run_verification


DEFAULT_CACHE = Path(".subpair-cache")
DEFAULT_REW_URL = "http://127.0.0.1:4735"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _at_least_two(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("must be at least 2")
    return parsed


def _bounded_float(low: float, high: float):
    def parse(value: str) -> float:
        parsed = float(value)
        if not low <= parsed <= high:
            raise argparse.ArgumentTypeError(f"must be between {low:g} and {high:g}")
        return parsed

    return parse


def _bounded_int(low: int, high: int):
    def parse(value: str) -> int:
        parsed = int(value)
        if not low <= parsed <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}")
        return parsed

    return parse


def _results_path(cache: Path, explicit: Path | None) -> Path:
    return explicit if explicit is not None else cache / "search-results.json"


def _parse_indices(value: str) -> list[int]:
    try:
        result = [int(token.strip()) for token in value.split(",") if token.strip()]
    except ValueError as exc:
        raise ValueError("--indices must be a comma-separated list of integers") from exc
    if len(result) < 2 or any(index < 1 for index in result) or len(set(result)) != len(result):
        raise ValueError("--indices must contain at least 2 distinct positive indices")
    return result


def _parse_room_dimensions(value: str) -> tuple[float, float, float]:
    parts = value.lower().split("x")
    usage = "must be LENGTHxWIDTHxHEIGHT in cm, e.g. 345x274x248"
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(usage)
    try:
        length, width, height = (float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(usage) from exc
    if not all(math.isfinite(dimension) and dimension > 0.0 for dimension in (length, width, height)):
        raise argparse.ArgumentTypeError("room dimensions must be positive")
    return length, width, height


def _parse_geometry_config(
    path: Path,
) -> tuple[
    tuple[float, float, float] | None,
    dict[int, tuple[float, float, float]] | None,
    tuple[float, float, float] | None,
]:
    """Read listener/sub coordinates in metres from a small JSON config."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read geometry config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Geometry config must be a JSON object")

    def coordinate(raw: object, label: str) -> tuple[float, float, float]:
        if not isinstance(raw, list) or len(raw) != 3:
            raise ValueError(f"{label} must be a three-number [x, y, z] array in metres")
        result = tuple(float(item) for item in raw)
        if not all(math.isfinite(item) for item in result):
            raise ValueError(f"{label} coordinates must be finite")
        return result  # type: ignore[return-value]

    listener_raw = value.get("listening_position_m", value.get("listener_position_m"))
    listener = coordinate(listener_raw, "listening_position_m") if listener_raw is not None else None
    subs_raw = value.get("sub_positions_m")
    subs: dict[int, tuple[float, float, float]] | None = None
    if subs_raw is not None:
        subs = {}
        if isinstance(subs_raw, list):
            for position, raw in enumerate(subs_raw, start=1):
                subs[position] = coordinate(raw, f"sub_positions_m[{position - 1}]")
        elif isinstance(subs_raw, dict):
            for key, raw in subs_raw.items():
                try:
                    position = int(key)
                except (TypeError, ValueError) as exc:
                    raise ValueError("sub_positions_m object keys must be 1-based positions") from exc
                if position < 1:
                    raise ValueError("sub_positions_m object keys must be positive")
                subs[position] = coordinate(raw, f"sub_positions_m[{key!r}]")
        else:
            raise ValueError("sub_positions_m must be an array or object")
    room_raw = value.get("room_dimensions_m")
    room = coordinate(room_raw, "room_dimensions_m") if room_raw is not None else None
    if room is not None and any(item <= 0.0 for item in room):
        raise ValueError("room_dimensions_m values must be positive")
    return listener, subs, room


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subpair",
        description="Find optimal two-subwoofer placements from time-referenced REW measurements.",
    )
    parser.add_argument("--version", action="version", version="subpair 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="cache loaded REW impulse responses")
    fetch.add_argument("--url", default=DEFAULT_REW_URL, help="REW API root URL")
    fetch.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="cache directory")
    selection = fetch.add_mutually_exclusive_group()
    selection.add_argument("--count", type=_at_least_two, help="use the first N loaded measurements")
    selection.add_argument("--indices", help="comma-separated REW indices to use")
    fetch.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout in seconds")

    search = commands.add_parser("search", help="enumerate and rank all cached pairs")
    search.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="cache directory")
    search.add_argument("--results", type=Path, help="search JSON output path")
    search.add_argument("--band", nargs=2, type=float, metavar=("LOW_HZ", "HIGH_HZ"), default=(25.0, 150.0))
    search.add_argument(
        "--delay-range",
        nargs=3,
        type=float,
        metavar=("MIN_MS", "MAX_MS", "STEP_MS"),
        default=(-10.0, 10.0, 0.05),
    )
    search.add_argument(
        "--geometry-config",
        "--geometry",
        dest="geometry_config",
        type=Path,
        help=(
            "JSON geometry in metres: listening_position_m, sub_positions_m "
            "(1-based array/object), and optional room_dimensions_m"
        ),
    )
    search.add_argument(
        "--listener-movement",
        type=_bounded_float(0.001, 5.0),
        default=0.25,
        metavar="METRES",
        help="listener displacement used for geometric timing jitter (default: 0.25)",
    )
    search.add_argument(
        "--speed-of-sound",
        type=_bounded_float(250.0, 400.0),
        default=343.0,
        metavar="M_PER_S",
        help="speed of sound used for geometry and path checks (default: 343)",
    )
    search.add_argument(
        "--physical-delay-window",
        type=_bounded_float(0.05, 20.0),
        default=1.5,
        metavar="MS",
        help="search window around measured relative arrival delay (default: 1.5 ms)",
    )
    _add_gate_arguments(search)
    search.add_argument(
        "--gain-range",
        nargs=3,
        type=float,
        metavar=("MIN_DB", "MAX_DB", "STEP_DB"),
        default=(-3.0, 3.0, 0.5),
    )
    search.add_argument("--ppo", type=_positive_int, default=48, help=argparse.SUPPRESS)
    search.add_argument(
        "--eq-target",
        choices=("trend", "flat", "dsp"),
        default="trend",
        help=(
            "EQ target: broad trend, aggressive flat response, or 'dsp' "
            "(flat-response alias for an external-DSP workflow)"
        ),
    )
    search.add_argument(
        "--aggressive-correction",
        action="store_true",
        help="alias for --eq-target flat",
    )
    search.add_argument(
        "--eq-range",
        nargs=2,
        type=float,
        metavar=("LOW_HZ", "HIGH_HZ"),
        help="frequency range in which EQ band centres may be fitted (default: analysis band)",
    )
    search.add_argument(
        "--eq-range-slope",
        type=_bounded_float(0.0, 48.0),
        default=48.0,
        metavar="DB_PER_OCT",
        help="correction curtain outside --eq-range, 0 is hard (default: 48)",
    )
    search.add_argument(
        "--max-boost",
        type=_bounded_float(0.0, 12.0),
        default=0.0,
        metavar="DB",
        help="maximum combined EQ boost, 0..12 dB (default: 0)",
    )
    search.add_argument(
        "--max-cut",
        type=_bounded_float(0.0, 30.0),
        default=18.0,
        metavar="DB",
        help="maximum single-filter EQ cut, 0..30 dB (default: 18)",
    )
    search.add_argument(
        "--eq-bands",
        type=_bounded_int(0, 16),
        default=7,
        metavar="COUNT",
        help="maximum automatic EQ band count (PK plus optional LS), 0..16 (default: 7)",
    )
    search.add_argument(
        "--score-low-end-weight",
        type=_bounded_float(0.0, 1.0),
        default=0.5,
        metavar="WEIGHT",
        help=(
            "score output blend: 0 is full-band SPL only, 1 is excursion-weighted "
            "low-end power only (default: 0.5)"
        ),
    )
    search.add_argument(
        "--score-dip-weight",
        type=_bounded_float(0.0, 4.0),
        default=1.0,
        metavar="WEIGHT",
        help=(
            "score penalty per dB below the 1/3-octave smoothed response, "
            "0..4 (default: 1)"
        ),
    )
    _add_shelf_arguments(search)
    search.add_argument(
        "--modal",
        choices=("on", "off"),
        default="off",
        help=(
            "compute parametric modal decomposition (matrix-pencil pole "
            "estimation, jointly across every solo measurement): per-pair "
            "high-Q resonance metrics reported as diagnostics only, off by "
            "default (default: off). Still experimental and not fully "
            "tested against real captures -- treat its output with "
            "skepticism, especially with few measurements, where joint "
            "pole persistence requires near-unanimous agreement"
        ),
    )
    search.add_argument(
        "--modal-tiebreak",
        choices=("on", "off"),
        default="off",
        help=(
            "after the primary usable-output score, break ties using "
            "(n_highQ, sum_modal_energy_db), both lower-is-better; requires "
            "--modal on (default: off)"
        ),
    )
    search.add_argument("--top", type=_positive_int, default=10, help="rows to print")

    report = commands.add_parser("report", help="write the self-contained HTML report")
    report.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="cache directory")
    report.add_argument("--results", type=Path, help="search JSON input path")
    report.add_argument("--output", type=Path, default=Path("subpair-report.html"))
    report.add_argument(
        "--top",
        type=_positive_int,
        default=5,
        help="pairs selected initially",
    )
    report.add_argument(
        "--limit",
        type=_positive_int,
        default=15,
        metavar="COUNT",
        help="maximum ranked pairs shown (default: 15)",
    )
    report.add_argument(
        "--raw",
        action="store_true",
        help="show raw results instead of the default EQ'd results",
    )
    report.add_argument(
        "--room",
        type=_parse_room_dimensions,
        metavar="LxWxH",
        help=(
            "room dimensions in cm, e.g. 345x274x248; overlays theoretical "
            "rigid-box mode frequencies on frequency charts (vertical) and "
            "CSD heatmaps (horizontal), toggleable per mode type via the "
            "legend"
        ),
    )

    verify = commands.add_parser("verify", help="compare one physical sum with a prediction")
    verify.add_argument("--url", default=DEFAULT_REW_URL, help="REW API root URL")
    verify.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="cache directory")
    verify.add_argument("--results", type=Path, help="search JSON input path")
    verify.add_argument("--rank", type=_positive_int, default=1, help="ranked configuration")
    verify.add_argument("--measurement", help="new REW measurement index or UUID")
    verify.add_argument("--band", nargs=2, type=float, metavar=("LOW_HZ", "HIGH_HZ"))
    verify.add_argument("--keep-level", action="store_true", help="do not remove a constant level offset")
    verify.add_argument("--output", type=Path, default=Path("subpair-verification.html"))
    return parser


def _add_shelf_arguments(subparser: argparse.ArgumentParser) -> None:
    """Enable or disable the automatic low-shelf EQ candidate."""

    subparser.add_argument(
        "--low-shelf",
        choices=("on", "off"),
        default="on",
        help=(
            "allow one automatically fitted low-shelf EQ band; its corner and "
            "gain are chosen by the fitter and count toward --eq-bands "
            "(default: on)"
        ),
    )


def _add_gate_arguments(subparser: argparse.ArgumentParser) -> None:
    """Expose every verdict threshold while keeping defaults in GateThresholds."""

    defaults = GateThresholds()
    group = subparser.add_argument_group("disqualifier gates")
    specifications = (
        ("--gate-redundancy-reject", "gate_redundancy_reject", defaults.redundancy_reject),
        ("--gate-redundancy-caution", "gate_redundancy_caution", defaults.redundancy_caution),
        ("--gate-ripple-reject", "gate_ripple_reject", defaults.ripple_correlation_reject),
        ("--gate-ripple-complementary", "gate_ripple_complementary", defaults.ripple_complementary),
        ("--gate-physical-percentile", "gate_physical_percentile", defaults.physical_percentile_reject),
        ("--gate-cancellation-reject", "gate_cancellation_reject", defaults.cancellation_deficit_reject_db),
        ("--gate-cancellation-caution", "gate_cancellation_caution", defaults.cancellation_deficit_caution_db),
        ("--gate-comb-reject", "gate_comb_reject", defaults.comb_index_reject),
        ("--gate-comb-caution", "gate_comb_caution", defaults.comb_index_caution),
        ("--gate-notch-depth", "gate_notch_depth", defaults.notch_depth_reject_db),
        ("--gate-notch-width", "gate_notch_width", defaults.notch_max_width_octaves),
        ("--gate-gain-asymmetry", "gate_gain_asymmetry", defaults.gain_asymmetry_caution_db),
        ("--gate-band-edge-excess-spread", "gate_band_edge_excess_spread", defaults.band_edge_excess_spread_reject_db),
        ("--gate-localization-fraction", "gate_localization_fraction", defaults.localization_fraction_reject),
        ("--gate-localization-min-improvement", "gate_localization_min_improvement", defaults.localization_min_mean_improvement_db),
        ("--gate-basin-tolerance", "gate_basin_tolerance", defaults.basin_tolerance_db),
    )
    for flag, destination, default in specifications:
        group.add_argument(
            flag,
            dest=destination,
            type=float,
            default=default,
            metavar="VALUE",
            help=f"default: {default:g}",
        )


def _fetch(args: argparse.Namespace) -> int:
    client = RewClient(args.url, timeout=args.timeout)
    routes = client.discover()
    summaries = client.list_measurements()
    if not summaries:
        raise RewApiError("REW has no loaded measurements")
    print(f"Discovered OpenAPI: {routes.spec_url}")
    print(f"Read-only routes: {routes.measurements_path}, {routes.impulse_path}")
    print("\nLoaded REW measurements:")
    print(f"{'Index':>5}  {'Sample rate':>12}  Name")
    for summary in summaries:
        rate = client.measurement_sample_rate(summary)
        rate_text = f"{rate:g} Hz" if rate is not None else "(from IR)"
        print(f"{summary['_index']:>5}  {rate_text:>12}  {client.measurement_title(summary)}")

    if args.indices:
        indices = _parse_indices(args.indices)
    elif args.count is not None:
        if args.count > len(summaries):
            raise RewApiError(
                f"--count {args.count} requested, but REW has only {len(summaries)} measurements"
            )
        indices = list(range(1, args.count + 1))
    else:
        indices = list(range(1, len(summaries) + 1))
    if len(indices) < 2:
        raise RewApiError(f"At least 2 measurements are required, selected {len(indices)}")
    if max(indices) > len(summaries):
        raise RewApiError(
            f"Selected index {max(indices)}, but REW has only {len(summaries)} measurements"
        )

    rows = []
    for ordinal, index in enumerate(indices, start=1):
        summary = summaries[index - 1]
        print(
            f"Fetching {ordinal}/{len(indices)}: #{index} "
            f"{client.measurement_title(summary)} ...",
            flush=True,
        )
        impulse, metadata = client.fetch_impulse(index)
        summary_arrival_delay = client.measurement_arrival_delay_seconds(summary)
        if summary_arrival_delay is not None:
            metadata["arrival_delay_seconds"] = summary_arrival_delay
        rows.append(
            {
                "source_index": index,
                "title": client.measurement_title(summary),
                "uuid": client.measurement_uuid(summary),
                "sample_rate": metadata["sample_rate"],
                "start_time_seconds": metadata["start_time_seconds"],
                "impulse": impulse,
                "metadata": {"summary": summary, "impulse": metadata},
            }
        )
    manifest = write_cache(
        args.cache,
        rows,
        api={
            "root_url": client.root_url,
            "openapi_url": routes.spec_url,
            "measurements_path": routes.measurements_path,
            "impulse_path": routes.impulse_path,
        },
    )
    print(f"Cached {len(rows)} responses at {manifest.resolve()}")
    return 0


def _format_tail(row: dict, eq: bool) -> str:
    """This pair's 'Tail' cell: dB (worst mode's level vs. direct sound) when
    the source is modal, else ms (CSD envelope decay time). ringing_ms
    saturates at 0 for every mode below the audibility margin -- a
    well-controlled room can do that for every pair -- so the dB figure,
    which keeps varying below that floor, is shown instead whenever it's
    available; see engine.py's settings.ranking.effective_tail.
    """
    is_modal = row["post_eq_effective_tail_is_modal" if eq else "effective_tail_is_modal"]
    if is_modal:
        value = row["post_eq_effective_tail_db" if eq else "effective_tail_db"]
        return f"{value:>+6.1f}dB"
    value = row["post_eq_effective_tail_ms" if eq else "effective_tail_ms"]
    return f"{value:>6.1f}ms"


def _print_ranking(result: dict, top: int) -> None:
    for warning in result.get("warnings", []):
        print(f"WARNING: {warning}", file=sys.stderr)

    def print_mode(rows: list[dict], eq: bool) -> None:
        label = "EQ'd" if eq else "Raw"
        print(f"\n{label} ranking")
        print(
            "Verdict  Score dB  Pair     Pol  Robust ms    Raw* ms  Gain dB  Frag dB  Basin03  "
            "Worst1 f  Geo  Phys  Headroom  Dip dB  Excess ms  Excess95 ms  Peak ms  "
            "    Tail  LE power  Rel SPL  A-res  B-corr  C-pct  D-def"
        )
        for row in rows[:top]:
            pair = f"{row['first']}+{row['second']}"
            verdict = str(row.get("verdict", "accept")).upper()
            if not row.get("optimized", True):
                def optional(value: object, digits: int = 2) -> str:
                    return "—" if value is None else f"{float(value):.{digits}f}"

                print(
                    f"{verdict:<7}  {'—':>8}  {pair:<7}  "
                    f"{'not optimized after gate rejection':<105}  "
                    f"{optional(row.get('redundancy_residual'), 3):>5}  "
                    f"{optional(row.get('ripple_correlation'), 3):>6}  "
                    f"{optional(row.get('physical_percentile'), 1):>5}  "
                    f"{optional(row.get('physical_cancellation_deficit_db')):>5}"
                )
                continue
            polarity = "+" if row["polarity"] > 0 else "-"
            physical_status = (
                "N/A"
                if not row.get("physical_constraint_available", row.get("physical_tau") is not None)
                else ("OUT" if row["non_physical_solution"] else "OK")
            )
            print(
                f"{verdict:<7}  "
                f"{row['post_eq_relative_score_db' if eq else 'relative_score_db']:>+8.2f}  "
                f"{pair:<7}  {polarity:>3}  "
                f"{row['delay_ms']:>+9.3f}  {row['tau_star']:>+9.3f}  "
                f"{row['gain_db']:>+7.2f}  "
                f"{row['fragility']:>7.2f}  {row['basin_w03']:>7.2f}  "
                f"{row['worst_case']['1.0']:>8.2f}  "
                f"{'PASS' if row['geometric_pass'] else 'FAIL':>4}  "
                f"{physical_status:>4}  "
                f"{row['post_eq_headroom_db' if eq else 'headroom_db']:>+8.2f}  "
                f"{row['post_eq_dip_db' if eq else 'dip_db']:>6.3f}  "
                f"{row['post_eq_excess_gd_ms' if eq else 'excess_gd_ms']:>9.3f}  "
                f"{row['post_eq_excess_gd_tail_ms' if eq else 'excess_gd_tail_ms']:>11.3f}  "
                f"{row['post_eq_excess_gd_peak_ms' if eq else 'excess_gd_peak_ms']:>7.3f}  "
                f"{_format_tail(row, eq):>8}  "
                f"{row['post_eq_relative_low_end_power_db' if eq else 'relative_low_end_power_db']:>+8.2f}  "
                f"{row['post_eq_relative_spl_db' if eq else 'relative_spl_db']:>+7.2f}  "
                f"{row['redundancy_residual']:>5.3f}  "
                f"{row['ripple_correlation']:>+6.3f}  "
                f"{row['physical_percentile'] if row['physical_percentile'] is not None else math.nan:>5.1f}  "
                f"{row['cancellation_deficit_db']:>+5.2f}"
            )

    print_mode(result["pairs"], eq=False)
    print_mode(sorted(result["pairs"], key=lambda row: row["eq_rank"]), eq=True)


def _search(args: argparse.Namespace) -> int:
    output = _results_path(args.cache, args.results)
    listener_position = None
    sub_positions = None
    room_dimensions = None
    if args.geometry_config is not None:
        listener_position, sub_positions, room_dimensions = _parse_geometry_config(
            args.geometry_config
        )
    options = SearchOptions(
        band=tuple(args.band),
        delay_range_ms=tuple(args.delay_range),
        gain_range_db=tuple(args.gain_range),
        ppo=args.ppo,
        eq_target="flat" if args.aggressive_correction else args.eq_target,
        eq_range_hz=tuple(args.eq_range) if args.eq_range else None,
        eq_range_slope_db_per_octave=args.eq_range_slope,
        max_boost_db=args.max_boost,
        max_cut_db=args.max_cut,
        eq_bands=args.eq_bands,
        score_low_end_weight=args.score_low_end_weight,
        score_dip_weight=args.score_dip_weight,
        low_shelf=args.low_shelf == "on",
        modal=args.modal == "on",
        modal_tiebreak=args.modal_tiebreak == "on",
        listener_position_m=listener_position,
        sub_positions_m=sub_positions,
        room_dimensions_m=room_dimensions,
        listener_movement_m=args.listener_movement,
        speed_of_sound_m_per_s=args.speed_of_sound,
        physical_delay_window_ms=args.physical_delay_window,
        gate_thresholds=GateThresholds(
            redundancy_reject=args.gate_redundancy_reject,
            redundancy_caution=args.gate_redundancy_caution,
            ripple_correlation_reject=args.gate_ripple_reject,
            ripple_complementary=args.gate_ripple_complementary,
            physical_percentile_reject=args.gate_physical_percentile,
            cancellation_deficit_reject_db=args.gate_cancellation_reject,
            cancellation_deficit_caution_db=args.gate_cancellation_caution,
            comb_index_reject=args.gate_comb_reject,
            comb_index_caution=args.gate_comb_caution,
            notch_depth_reject_db=args.gate_notch_depth,
            notch_max_width_octaves=args.gate_notch_width,
            gain_asymmetry_caution_db=args.gate_gain_asymmetry,
            band_edge_excess_spread_reject_db=args.gate_band_edge_excess_spread,
            localization_fraction_reject=args.gate_localization_fraction,
            localization_min_mean_improvement_db=args.gate_localization_min_improvement,
            basin_tolerance_db=args.gate_basin_tolerance,
        ),
    )

    def progress(done: int, total: int, pair: str) -> None:
        print(f"\rEvaluated pair {done}/{total} ({pair})", end="", flush=True)

    result = run_search(args.cache, output, options, progress=progress)
    print()
    _print_ranking(result, args.top)
    print(f"\nWrote complete ranking to {output.resolve()}")
    return 0


def _report(args: argparse.Namespace) -> int:
    results = _results_path(args.cache, args.results)
    output = build_report(
        args.cache,
        results,
        args.output,
        top=args.top,
        limit=args.limit,
        raw=args.raw,
        room_dimensions_cm=args.room,
    )
    print(f"Wrote self-contained report to {output.resolve()}")
    return 0


def _verify(args: argparse.Namespace) -> int:
    results = _results_path(args.cache, args.results)
    value = run_verification(
        args.cache,
        results,
        args.output,
        args.url,
        rank=args.rank,
        measurement_id=args.measurement,
        keep_level=args.keep_level,
        band_override=tuple(args.band) if args.band else None,
    )
    print(
        f"Measurement #{value['measurement_index']} {value['measurement_name']}: "
        f"max in-band deviation {value['max_deviation_db']:.3f} dB"
    )
    print(f"Wrote verification overlay to {Path(value['output']).resolve()}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            return _fetch(args)
        if args.command == "search":
            return _search(args)
        if args.command == "report":
            return _report(args)
        if args.command == "verify":
            return _verify(args)
        parser.error(f"unknown command {args.command!r}")
    except (RewApiError, CacheError, ReportError, VerificationError, ValueError) as exc:
        print(f"subpair: error: {exc}", file=sys.stderr)
        return 2
    return 2
