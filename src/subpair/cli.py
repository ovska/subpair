"""Command-line interface for subpair."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .api import RewApiError, RewClient
from .cache import CacheError, write_cache
from .dsp import ShelfOptions
from .engine import SearchOptions, run_search
from .html_report import ReportError, build_report
from .verification import VerificationError, run_verification


DEFAULT_CACHE = Path(".subpair-cache")
DEFAULT_REW_URL = "http://127.0.0.1:4735"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _at_least_three(value: str) -> int:
    parsed = int(value)
    if parsed < 3:
        raise argparse.ArgumentTypeError("must be at least 3")
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
    if len(result) < 3 or any(index < 1 for index in result) or len(set(result)) != len(result):
        raise ValueError("--indices must contain at least 3 distinct positive indices")
    return result


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
    selection.add_argument("--count", type=_at_least_three, help="use the first N loaded measurements")
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
        default=(-10.0, 10.0, 0.1),
    )
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
            "PEQ target: broad trend, aggressive flat response, or 'dsp' "
            "(flat response; ranking barely penalises minimum-phase dips, "
            "for placements an external DSP will correct)"
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
        help="frequency range in which PEQ centres may be fitted (default: analysis band)",
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
        help="maximum combined PEQ boost, 0..12 dB (default: 0)",
    )
    search.add_argument(
        "--max-cut",
        type=_bounded_float(0.0, 30.0),
        default=18.0,
        metavar="DB",
        help="maximum single-filter PEQ cut, 0..30 dB (default: 18)",
    )
    search.add_argument(
        "--eq-bands",
        type=_bounded_int(0, 16),
        default=7,
        metavar="COUNT",
        help="maximum PEQ band count, 0..16 (default: 7)",
    )
    search.add_argument(
        "--tie-tolerance-db",
        type=_bounded_float(0.0, 3.0),
        default=0.0,
        metavar="DB",
        help=(
            "treat null-score differences below this as ties before falling "
            "back to excess-GD/tail, 0..3 dB (default: 0, strict lexicographic)"
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
        help="pairs selected initially in each ranking mode",
    )
    report.add_argument(
        "--limit",
        type=_positive_int,
        default=15,
        metavar="COUNT",
        help="maximum ranked pairs shown per mode (default: 15)",
    )
    _add_shelf_arguments(report)

    verify = commands.add_parser("verify", help="compare one physical sum with a prediction")
    verify.add_argument("--url", default=DEFAULT_REW_URL, help="REW API root URL")
    verify.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="cache directory")
    verify.add_argument("--results", type=Path, help="search JSON input path")
    verify.add_argument("--rank", type=_positive_int, default=1, help="ranked configuration")
    verify.add_argument("--measurement", help="new REW measurement index or UUID")
    verify.add_argument("--band", nargs=2, type=float, metavar=("LOW_HZ", "HIGH_HZ"))
    verify.add_argument("--keep-level", action="store_true", help="do not remove a constant level offset")
    verify.add_argument("--output", type=Path, default=Path("subpair-verification.html"))
    _add_shelf_arguments(verify)
    return parser


def _add_shelf_arguments(subparser: argparse.ArgumentParser) -> None:
    """A fixed, broad low-shelf tonal control, independent of the fitted PEQ bank.

    Shared by ``report``/``verify`` only: it never reaches ``search`` or any
    ranking key, so it cannot change which placement wins (see
    ``dsp.ShelfOptions``).
    """
    subparser.add_argument(
        "--low-shelf-freq",
        type=_bounded_float(1.0, 20000.0),
        default=None,
        metavar="HZ",
        help="low-shelf corner frequency; required if --low-shelf-gain is nonzero",
    )
    subparser.add_argument(
        "--low-shelf-gain",
        type=_bounded_float(-15.0, 15.0),
        default=0.0,
        metavar="DB",
        help="low-shelf boost/cut, -15..15 dB (default: 0, disabled)",
    )
    subparser.add_argument(
        "--low-shelf-slope",
        type=_bounded_float(0.1, 1.0),
        default=1.0,
        help=argparse.SUPPRESS,
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
    if len(indices) < 3:
        raise RewApiError(f"At least 3 measurements are required, selected {len(indices)}")
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


def _print_ranking(result: dict, top: int) -> None:
    def print_mode(rows: list[dict], eq: bool) -> None:
        label = "EQ'd" if eq else "Raw"
        print(f"\n{label} ranking")
        print(
            "Rank  Pair     Pol   Delay ms  Gain dB  Null dB  Excess ms  "
            "Excess95 ms  Tail ms  Ext Hz  Rel SPL"
        )
        for row in rows[:top]:
            pair = f"{row['first']}+{row['second']}"
            polarity = "+" if row["polarity"] > 0 else "-"
            print(
                f"{row['eq_rank' if eq else 'rank']:>4}  {pair:<7}  {polarity:>3}  "
                f"{row['delay_ms']:>+9.3f}  {row['gain_db']:>+7.2f}  "
                f"{row['post_eq_null_score_db' if eq else 'null_score_db']:>7.3f}  "
                f"{row['post_eq_excess_gd_ms' if eq else 'excess_gd_ms']:>9.3f}  "
                f"{row['post_eq_excess_gd_tail_ms' if eq else 'excess_gd_tail_ms']:>11.3f}  "
                f"{row['post_eq_tail_ms' if eq else 'raw_tail_ms']:>7.1f}  "
                f"{row['post_eq_low_end_extension_hz' if eq else 'low_end_extension_hz']:>6.1f}  "
                f"{row['post_eq_relative_spl_db' if eq else 'relative_spl_db']:>+7.2f}"
            )

    print_mode(result["pairs"], eq=False)
    print_mode(sorted(result["pairs"], key=lambda row: row["eq_rank"]), eq=True)


def _search(args: argparse.Namespace) -> int:
    output = _results_path(args.cache, args.results)
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
        tie_tolerance_db=args.tie_tolerance_db,
    )

    def progress(done: int, total: int, pair: str) -> None:
        print(f"\rScored pair {done}/{total} ({pair})", end="", flush=True)

    result = run_search(args.cache, output, options, progress=progress)
    print()
    _print_ranking(result, args.top)
    print(f"\nWrote complete ranking to {output.resolve()}")
    return 0


def _shelf_options(args: argparse.Namespace) -> ShelfOptions:
    return ShelfOptions(
        freq_hz=args.low_shelf_freq,
        gain_db=args.low_shelf_gain,
        slope=args.low_shelf_slope,
    )


def _report(args: argparse.Namespace) -> int:
    results = _results_path(args.cache, args.results)
    output = build_report(
        args.cache,
        results,
        args.output,
        top=args.top,
        limit=args.limit,
        shelf=_shelf_options(args),
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
        shelf=_shelf_options(args),
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
