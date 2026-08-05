from __future__ import annotations

import base64
import json
import unittest
import urllib.parse

import numpy as np

from subpair.api import RewClient, decode_float32_array


class _FakeClient(RewClient):
    impulse = np.asarray([0.125, 0.25, 0.375, 0.5], dtype=">f4").tobytes()

    def _get_bytes(self, url: str):
        path = urllib.parse.urlsplit(url).path
        if path == "/":
            return b'<script>SwaggerUIBundle({url: "/openapi.json"})</script>', "text/html"
        if path == "/openapi.json":
            spec = {
                "openapi": "3.0.0",
                "paths": {
                    "/v7/things": {
                        "get": {"summary": "List measurement summaries"}
                    },
                    "/v7/things/{uuid}/raw": {
                        "get": {
                            "summary": "Get measurement impulse response",
                            "parameters": [
                                {"name": "normalised", "in": "query"},
                                {"name": "windowed", "in": "query"},
                            ],
                        }
                    },
                },
            }
            return json.dumps(spec).encode(), "application/json"
        if path == "/v7/things":
            rows = [
                {"title": "One", "uuid": "u1", "sampleRate": 48000},
                {"title": "Two", "uuid": "u2", "sampleRate": 48000},
                {"title": "Three", "uuid": "u3", "sampleRate": 48000},
            ]
            return json.dumps(rows).encode(), "application/json"
        if path == "/v7/things/1/raw":
            payload = {
                "sampleRate": 48000,
                "startTime": -0.01,
                "timingReference": "Loopback",
                "data": base64.b64encode(self.impulse).decode(),
            }
            return json.dumps(payload).encode(), "application/json"
        raise AssertionError(f"Unexpected URL: {url}")


class ApiTests(unittest.TestCase):
    def test_big_endian_example(self):
        encoded = "PgAAAD6AAAA+wAAAPwAAAA=="
        np.testing.assert_allclose(decode_float32_array(encoded), [0.125, 0.25, 0.375, 0.5])

    def test_discovers_nonstandard_routes_from_root(self):
        client = _FakeClient("http://rew.test:4735")
        routes = client.discover()
        self.assertEqual(routes.measurements_path, "/v7/things")
        self.assertEqual(routes.impulse_path, "/v7/things/{uuid}/raw")
        rows = client.list_measurements()
        self.assertEqual(rows[0]["_index"], 1)
        impulse, metadata = client.fetch_impulse(1)
        np.testing.assert_allclose(impulse, [0.125, 0.25, 0.375, 0.5])
        self.assertEqual(metadata["sample_rate"], 48000)


if __name__ == "__main__":
    unittest.main()
