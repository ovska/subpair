"""Small read-only client for REW's beta, self-documented HTTP API."""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np


class RewApiError(RuntimeError):
    """Raised when REW cannot be queried without guessing its API shape."""


def _normal_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _field(obj: dict[str, Any], *names: str, default: Any = None) -> Any:
    wanted = {_normal_key(name) for name in names}
    for key, value in obj.items():
        if _normal_key(str(key)) in wanted:
            return value
    return default


def decode_float32_array(value: Any) -> np.ndarray:
    """Decode a REW array (big-endian float32 Base64, or a JSON list)."""
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float64)
    if not isinstance(value, str):
        raise RewApiError(f"REW array has unsupported type {type(value).__name__}")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise RewApiError("REW returned invalid Base64 array data") from exc
    if len(raw) % 4:
        raise RewApiError(
            f"REW float array has {len(raw)} bytes, which is not divisible by 4"
        )
    return np.frombuffer(raw, dtype=">f4").astype(np.float64, copy=False)


@dataclass(frozen=True)
class ApiRoutes:
    spec_url: str
    measurements_path: str
    impulse_path: str
    impulse_parameters: frozenset[str]


class RewClient:
    """Read-only REW client which resolves routes from the advertised OpenAPI."""

    def __init__(self, root_url: str = "http://127.0.0.1:4735", timeout: float = 15.0):
        self.root_url = root_url.rstrip("/") + "/"
        self.timeout = timeout
        self.spec: dict[str, Any] = {}
        self.routes: ApiRoutes | None = None

    def _get_bytes(self, url: str) -> tuple[bytes, str]:
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json, text/html;q=0.9, */*;q=0.1"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read(), response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RewApiError(f"GET {url} failed: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RewApiError(
                f"Cannot reach REW at {url}: {exc.reason}. Is its API server running?"
            ) from exc

    def _get_json(self, url: str) -> Any:
        body, _ = self._get_bytes(url)
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RewApiError(f"GET {url} did not return JSON") from exc

    @staticmethod
    def _spec_candidates(root_url: str, body: bytes) -> list[str]:
        text = body.decode("utf-8", "replace")
        found: list[str] = []
        patterns = (
            r"\burl\s*:\s*['\"]([^'\"]+\.(?:json|yaml))['\"]",
            r"['\"]([^'\"]*(?:openapi|swagger|doc)[^'\"]*\.json)['\"]",
            r"(?:href|src)\s*=\s*['\"]([^'\"]+\.json)['\"]",
        )
        for pattern in patterns:
            for match in re.findall(pattern, text, flags=re.IGNORECASE):
                url = urllib.parse.urljoin(root_url, match)
                if url not in found:
                    found.append(url)
        return found

    @staticmethod
    def _is_openapi(value: Any) -> bool:
        return (
            isinstance(value, dict)
            and isinstance(value.get("paths"), dict)
            and ("openapi" in value or "swagger" in value)
        )

    def discover(self) -> ApiRoutes:
        """Load Swagger/OpenAPI from REW's root and semantically select GET routes."""
        root_body, root_type = self._get_bytes(self.root_url)
        candidates: list[tuple[str, dict[str, Any]]] = []
        if "json" in root_type.lower() or root_body.lstrip().startswith(b"{"):
            try:
                root_json = json.loads(root_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                root_json = None
            if self._is_openapi(root_json):
                candidates.append((self.root_url, root_json))

        for spec_url in self._spec_candidates(self.root_url, root_body):
            if not spec_url.lower().endswith(".json"):
                continue
            try:
                value = self._get_json(spec_url)
            except RewApiError:
                continue
            if self._is_openapi(value):
                candidates.append((spec_url, value))
                break

        # Older Swagger UI versions do not expose the URL in their static HTML.
        # This is a compatibility *probe*, never an assumed operation route: the
        # returned document and every operation below are validated before use.
        if not candidates:
            compatibility_url = urllib.parse.urljoin(self.root_url, "doc.json")
            try:
                value = self._get_json(compatibility_url)
            except RewApiError as exc:
                raise RewApiError(
                    "REW's root did not advertise a readable JSON OpenAPI document, "
                    "and the validated compatibility probe failed"
                ) from exc
            if not self._is_openapi(value):
                raise RewApiError(
                    f"{compatibility_url} is not an OpenAPI document with paths"
                )
            candidates.append((compatibility_url, value))

        spec_url, spec = candidates[0]
        paths = spec["paths"]

        def resolve_reference(value: Any) -> Any:
            if not isinstance(value, dict) or not isinstance(value.get("$ref"), str):
                return value
            reference = value["$ref"]
            if not reference.startswith("#/"):
                return value
            current: Any = spec
            for token in reference[2:].split("/"):
                token = token.replace("~1", "/").replace("~0", "~")
                if not isinstance(current, dict) or token not in current:
                    return value
                current = current[token]
            return current

        def get_operations() -> list[tuple[str, dict[str, Any], str]]:
            operations: list[tuple[str, dict[str, Any], str]] = []
            for path, item in paths.items():
                if not isinstance(item, dict) or not isinstance(item.get("get"), dict):
                    continue
                op = item["get"]
                text = " ".join(
                    str(op.get(key, "")) for key in ("summary", "description", "operationId")
                ).lower()
                operations.append((path, op, text))
            return operations

        operations = get_operations()
        list_candidates: list[tuple[int, str, dict[str, Any]]] = []
        ir_candidates: list[tuple[int, str, dict[str, Any]]] = []
        for path, operation, text in operations:
            low = path.lower()
            placeholders = re.findall(r"\{[^}]+\}", path)
            if not placeholders and "measurement" in (low + " " + text):
                score = 0
                score += 8 if low.rstrip("/").endswith("measurements") else 0
                score += 3 if "list" in text or "summar" in text else 0
                score += 2 if "measurement" in text else 0
                list_candidates.append((score, path, operation))
            if placeholders and "impulse" in (low + " " + text) and "response" in (
                low + " " + text
            ):
                score = 0
                score += 8 if "impulse-response" in low else 0
                score += 4 if "measurement" in low else 0
                score -= 5 if "filter" in low or "/eq/" in low else 0
                ir_candidates.append((score, path, operation))

        if not list_candidates or not ir_candidates:
            advertised = ", ".join(path for path, _, _ in operations)
            raise RewApiError(
                "OpenAPI discovery could not identify read-only measurement-list and "
                f"impulse-response GET routes. Advertised GET paths: {advertised}"
            )
        list_candidates.sort(key=lambda row: (-row[0], row[1]))
        ir_candidates.sort(key=lambda row: (-row[0], row[1]))
        measurement_path = list_candidates[0][1]
        impulse_path, impulse_operation = ir_candidates[0][1:]

        parameters: set[str] = set()
        item_parameters = paths.get(impulse_path, {}).get("parameters", [])
        for unresolved in [*item_parameters, *impulse_operation.get("parameters", [])]:
            parameter = resolve_reference(unresolved)
            if isinstance(parameter, dict) and isinstance(parameter.get("name"), str):
                parameters.add(parameter["name"])

        self.spec = spec
        self.routes = ApiRoutes(
            spec_url=spec_url,
            measurements_path=measurement_path,
            impulse_path=impulse_path,
            impulse_parameters=frozenset(parameters),
        )
        return self.routes

    def _api_url(self, path: str, query: dict[str, Any] | None = None) -> str:
        url = urllib.parse.urljoin(self.root_url, path.lstrip("/"))
        if query:
            url += "?" + urllib.parse.urlencode(query)
        return url

    @staticmethod
    def _coerce_measurements(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
            return list(payload)
        if isinstance(payload, dict):
            nested = _field(payload, "measurements", "measurementSummaries")
            if isinstance(nested, list) and all(isinstance(item, dict) for item in nested):
                return list(nested)
            numeric = []
            for key, value in payload.items():
                if str(key).isdigit() and isinstance(value, dict):
                    numeric.append((int(key), value))
            if numeric:
                return [value for _, value in sorted(numeric)]
            # Some REW builds return a UUID-to-summary mapping.
            values = list(payload.values())
            if values and all(isinstance(value, dict) for value in values):
                if all(_field(value, "title", "name") is not None for value in values):
                    return values
        raise RewApiError("REW measurement-list response has an unrecognised JSON shape")

    def list_measurements(self) -> list[dict[str, Any]]:
        routes = self.routes or self.discover()
        payload = self._get_json(self._api_url(routes.measurements_path))
        measurements = self._coerce_measurements(payload)
        result: list[dict[str, Any]] = []
        for index, summary in enumerate(measurements, start=1):
            item = dict(summary)
            item["_index"] = index
            result.append(item)
        return result

    @staticmethod
    def measurement_title(summary: dict[str, Any]) -> str:
        return str(_field(summary, "title", "name", default="(untitled)"))

    @staticmethod
    def measurement_uuid(summary: dict[str, Any]) -> str:
        return str(_field(summary, "uuid", "id", default=""))

    @staticmethod
    def measurement_sample_rate(summary: dict[str, Any]) -> float | None:
        value = _field(summary, "sampleRate", "samplerate")
        return None if value is None else float(value)

    @staticmethod
    def measurement_arrival_delay_seconds(payload: dict[str, Any]) -> float | None:
        """Extract REW's loopback-referenced arrival delay from nested metadata.

        Beta API builds have used several spellings. Explicit unit-bearing
        fields are preferred. REW's API emits a bare numeric
        ``delay``/``arrivalDelay`` in seconds; strings may carry an explicit
        ``ms`` or ``s`` suffix as they do in exported notes.
        """

        millisecond_keys = {
            "arrivaldelayms",
            "delayms",
            "estimateddelayms",
            "acousticdelayms",
            "irpeakdelayms",
            "timingdelayms",
        }
        second_keys = {
            "arrivaldelayseconds",
            "delayseconds",
            "estimateddelayseconds",
            "acousticdelayseconds",
            "irpeakdelayseconds",
            "timeofirpeakseconds",
        }
        bare_keys = {"arrivaldelay", "delay", "estimateddelay", "acousticdelay"}

        def numeric(value: Any, scale: float) -> float | None:
            if isinstance(value, bool):
                return None
            if isinstance(value, (int, float)):
                result = float(value) * scale
                return result if np.isfinite(result) else None
            if isinstance(value, str):
                match = re.fullmatch(
                    r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(ms|s)?\s*",
                    value,
                    flags=re.IGNORECASE,
                )
                if match:
                    parsed = float(match.group(1))
                    unit = (match.group(2) or "").lower()
                    if unit == "ms":
                        return parsed / 1000.0
                    if unit == "s":
                        return parsed
                    return parsed * scale
            return None

        def walk(value: Any) -> float | None:
            if not isinstance(value, dict):
                return None
            for keys, scale in ((second_keys, 1.0), (millisecond_keys, 1e-3), (bare_keys, 1.0)):
                for key, item in value.items():
                    if _normal_key(str(key)) in keys:
                        parsed = numeric(item, scale)
                        if parsed is not None:
                            return parsed
            for item in value.values():
                if isinstance(item, dict):
                    parsed = walk(item)
                    if parsed is not None:
                        return parsed
            return None

        return walk(payload)

    @staticmethod
    def _find_ir_data(payload: dict[str, Any]) -> Any:
        direct = _field(
            payload,
            "data",
            "responseData",
            "impulseResponse",
            "impulse",
            "samples",
        )
        if isinstance(direct, (str, list)):
            return direct
        for value in payload.values():
            if isinstance(value, dict):
                found = RewClient._find_ir_data(value)
                if found is not None:
                    return found
        return None

    def fetch_impulse(self, measurement_id: int | str) -> tuple[np.ndarray, dict[str, Any]]:
        routes = self.routes or self.discover()
        path = re.sub(
            r"\{[^}]+\}",
            urllib.parse.quote(str(measurement_id), safe=""),
            routes.impulse_path,
            count=1,
        )
        if re.search(r"\{[^}]+\}", path):
            raise RewApiError(
                f"Impulse route {routes.impulse_path!r} has unsupported extra path parameters"
            )
        query: dict[str, str] = {}
        canonical = {_normal_key(name): name for name in routes.impulse_parameters}
        if "normalised" in canonical:
            query[canonical["normalised"]] = "false"
        elif "normalized" in canonical:
            query[canonical["normalized"]] = "false"
        if "windowed" in canonical:
            query[canonical["windowed"]] = "false"
        payload = self._get_json(self._api_url(path, query))
        if not isinstance(payload, dict):
            raise RewApiError("REW impulse-response response is not a JSON object")
        data = self._find_ir_data(payload)
        if data is None:
            raise RewApiError(
                "REW impulse-response JSON did not contain a recognised sample array"
            )
        impulse = decode_float32_array(data)
        sample_rate = _field(payload, "sampleRate", "samplerate")
        sample_interval = _field(payload, "sampleInterval", "sampleIntervalSeconds")
        if sample_rate is None and sample_interval is not None:
            sample_rate = 1.0 / float(sample_interval)
        if sample_rate is None:
            raise RewApiError("REW impulse response does not advertise a sample rate")
        start_time = _field(
            payload,
            "startTime",
            "startTimeSeconds",
            "timeOfIRStartSeconds",
            default=0.0,
        )
        metadata = {
            "sample_rate": float(sample_rate),
            "start_time_seconds": float(start_time),
            "arrival_delay_seconds": self.measurement_arrival_delay_seconds(payload),
            "timing_reference": str(
                _field(payload, "timingReference", "timingReferenceDescription", default="")
            ),
            "api_payload": payload,
        }

        def redact(value: Any) -> Any:
            # Do not duplicate a potentially very large Base64/numeric array in
            # cache metadata, including when a beta build nests its data model.
            if value is data:
                return f"<sample array: {impulse.size} float32 values>"
            if isinstance(value, str) and len(value) > 1024:
                return f"<large string: {len(value)} characters>"
            if isinstance(value, dict):
                return {key: redact(item) for key, item in value.items()}
            if isinstance(value, list):
                return [redact(item) for item in value]
            return value

        metadata["api_payload"] = redact(metadata["api_payload"])
        return impulse, metadata
