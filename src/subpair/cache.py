"""On-disk measurement cache with strict shape invariants."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class CacheError(RuntimeError):
    pass


@dataclass(frozen=True)
class CachedMeasurement:
    position: int
    source_index: int
    title: str
    uuid: str
    sample_rate: float
    start_time_seconds: float
    impulse: np.ndarray
    metadata: dict[str, Any]
    path: Path


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_cache(
    cache_dir: Path,
    rows: Iterable[dict[str, Any]],
    api: dict[str, Any],
) -> Path:
    """Validate all responses first, then replace the cache files atomically."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    materialised = list(rows)
    if len(materialised) < 2:
        raise CacheError(f"At least 2 measurements are required, got {len(materialised)}")
    rates = {float(row["sample_rate"]) for row in materialised}
    lengths = {int(np.asarray(row["impulse"]).size) for row in materialised}
    if len(rates) != 1:
        detail = ", ".join(
            f"#{row['source_index']}={row['sample_rate']:g} Hz" for row in materialised
        )
        raise CacheError(f"Mismatched sample rates; refusing to resample: {detail}")
    if len(lengths) != 1:
        detail = ", ".join(
            f"#{row['source_index']}={np.asarray(row['impulse']).size} samples"
            for row in materialised
        )
        raise CacheError(f"Mismatched response lengths; refusing to zero-pad: {detail}")
    for row in materialised:
        impulse = np.asarray(row["impulse"], dtype=np.float64)
        if not np.isfinite(impulse).all():
            raise CacheError(
                f"Measurement #{row['source_index']} contains NaN or infinite IR samples"
            )
        if not np.any(impulse):
            raise CacheError(f"Measurement #{row['source_index']} has an all-zero impulse response")

    manifest_rows: list[dict[str, Any]] = []
    temporary_paths: list[tuple[Path, Path]] = []
    try:
        for position, row in enumerate(materialised, start=1):
            destination = cache_dir / f"measurement-{position:03d}.npz"
            fd, temporary_name = tempfile.mkstemp(
                prefix=destination.name + ".", suffix=".npz", dir=cache_dir
            )
            os.close(fd)
            temporary = Path(temporary_name)
            metadata = dict(row.get("metadata", {}))
            compact = {
                "position": position,
                "source_index": int(row["source_index"]),
                "title": str(row["title"]),
                "uuid": str(row.get("uuid", "")),
                "sample_rate": float(row["sample_rate"]),
                "start_time_seconds": float(row.get("start_time_seconds", 0.0)),
                "metadata": metadata,
            }
            np.savez_compressed(
                temporary,
                impulse=np.asarray(row["impulse"], dtype=np.float64),
                metadata_json=np.asarray(json.dumps(compact, sort_keys=True)),
            )
            temporary_paths.append((temporary, destination))
            manifest_rows.append({**compact, "file": destination.name})
        for temporary, destination in temporary_paths:
            os.replace(temporary, destination)
        manifest = {
            "format_version": 1,
            "sample_rate": next(iter(rates)),
            "length": next(iter(lengths)),
            "measurements": manifest_rows,
            "api": api,
        }
        manifest_path = cache_dir / "manifest.json"
        _atomic_json(manifest_path, manifest)
        return manifest_path
    finally:
        for temporary, _ in temporary_paths:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def load_cache(cache_dir: Path) -> tuple[list[CachedMeasurement], dict[str, Any]]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise CacheError(f"Cache manifest not found: {manifest_path}; run 'subpair fetch' first")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheError(f"Cannot read cache manifest {manifest_path}: {exc}") from exc
    rows: list[CachedMeasurement] = []
    for entry in manifest.get("measurements", []):
        path = cache_dir / entry["file"]
        try:
            with np.load(path, allow_pickle=False) as archive:
                impulse = np.asarray(archive["impulse"], dtype=np.float64)
                embedded = json.loads(str(archive["metadata_json"].item()))
        except Exception as exc:
            raise CacheError(f"Cannot read cached measurement {path}: {exc}") from exc
        rows.append(
            CachedMeasurement(
                position=int(embedded["position"]),
                source_index=int(embedded["source_index"]),
                title=str(embedded["title"]),
                uuid=str(embedded.get("uuid", "")),
                sample_rate=float(embedded["sample_rate"]),
                start_time_seconds=float(embedded.get("start_time_seconds", 0.0)),
                impulse=impulse,
                metadata=dict(embedded.get("metadata", {})),
                path=path,
            )
        )
    if len(rows) < 2:
        raise CacheError(f"Cache contains {len(rows)} measurements; at least 2 are required")
    rates = {row.sample_rate for row in rows}
    lengths = {row.impulse.size for row in rows}
    if len(rates) != 1 or len(lengths) != 1:
        raise CacheError("Cache invariant failed: measurement rates or lengths differ")
    expected_rate = float(manifest.get("sample_rate", next(iter(rates))))
    expected_length = int(manifest.get("length", next(iter(lengths))))
    if rates != {expected_rate} or lengths != {expected_length}:
        raise CacheError("Cache files do not match manifest sample rate/length")
    return rows, manifest


def write_json(path: Path, value: Any) -> None:
    _atomic_json(path, value)
