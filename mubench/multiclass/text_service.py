#!/usr/bin/env python3
"""
Phase-J `text` service --- memory + serialisation workload.

Each request builds a nested JSON-able object holding ``size`` bytes
of random payload, then serialises it via ``json.dumps``. Memory
allocation + Python object construction + serialisation latency all
scale with the payload size. The size is parametric via ``?size=N``,
defaulting to TEXT_DEFAULT_SIZE (250 bytes). The 250 -> 5000 byte
shift is the AWARE "text size" stress (USENIX ATC '23, sec 2.3).

Endpoints
---------
GET  /            small landing
GET  /healthz     200 OK
GET  /text        build + serialise a payload of ``size`` bytes,
                  return the serialised JSON (so wrk also pays the
                  bandwidth tax in the response body)
GET  /metrics     JSON: requests served, total serialise ms, uptime

Environment
-----------
TEXT_SERVICE_PORT      default 8080
TEXT_DEFAULT_SIZE      default 250 (payload bytes when ?size= omitted)
TEXT_MAX_SIZE          default 65536 (safety cap)
TEXT_WORK_MULTIPLIER   default 1 (number of serialise-rounds per request)
"""

import base64
import json
import os
import string
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

PORT = int(os.environ.get("TEXT_SERVICE_PORT", "8080"))
DEFAULT_SIZE = int(os.environ.get("TEXT_DEFAULT_SIZE", "250"))
MAX_SIZE = int(os.environ.get("TEXT_MAX_SIZE", "65536"))
WORK_MULTIPLIER = max(1, int(os.environ.get("TEXT_WORK_MULTIPLIER", "1")))
START_TS = time.time()

_LOCK = threading.Lock()
_STATS = {"requests": 0, "total_serialise_ms": 0.0}

# Fixed alphabet so payload content is deterministic-ish but boring.
_PAYLOAD_POOL = (string.ascii_letters + string.digits) * 4


def _build_blob(size: int):
    """Build a nested dict roughly ``size`` bytes when JSON-serialised.

    The structure is a list of small records, each ~64 bytes serialised,
    so memory and serialise cost both scale linearly with ``size``.
    """
    size = max(16, min(int(size), MAX_SIZE))
    n_records = max(1, size // 64)
    records = []
    for i in range(n_records):
        records.append({
            "i": i,
            "tag": _PAYLOAD_POOL[i % len(_PAYLOAD_POOL):
                                 (i % len(_PAYLOAD_POOL)) + 24],
            "score": (i * 31 + 7) % 997,
            "flag": (i & 1) == 0,
        })
    return {"size": size, "n_records": n_records, "records": records}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args, **kwargs):
        return

    def _send(self, code, body=b"", headers=None):
        self.send_response(code)
        if headers:
            for k, v in headers.items():
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query or "")

        if path == "/":
            self._send(
                200,
                b"text OK -- GET /text?size=N for an N-byte JSON payload\n",
                {"Content-Type": "text/plain"},
            )
            return
        if path == "/healthz":
            self._send(200, b"ok\n", {"Content-Type": "text/plain"})
            return
        if path == "/metrics":
            with _LOCK:
                payload = json.dumps({
                    "requests": _STATS["requests"],
                    "total_serialise_ms": round(_STATS["total_serialise_ms"], 3),
                    "uptime_s": int(time.time() - START_TS),
                    "default_size": DEFAULT_SIZE,
                    "max_size": MAX_SIZE,
                    "work_multiplier": WORK_MULTIPLIER,
                }).encode()
            self._send(200, payload, {"Content-Type": "application/json"})
            return
        if path == "/text":
            try:
                size = int(query.get("size", [DEFAULT_SIZE])[0])
            except (TypeError, ValueError):
                size = DEFAULT_SIZE
            t0 = time.perf_counter()
            body_bytes = b""
            for _i in range(WORK_MULTIPLIER):
                blob = _build_blob(size)
                serialised = json.dumps(blob).encode()
                # Also do a base64 round-trip so we touch the bytes again
                # (text-processing workloads typically do encode+decode).
                _ = base64.b64encode(serialised)
                body_bytes = serialised
            dt_ms = (time.perf_counter() - t0) * 1000.0
            with _LOCK:
                _STATS["requests"] += 1
                _STATS["total_serialise_ms"] += dt_ms
            self._send(200, body_bytes, {"Content-Type": "application/json"})
            return
        self._send(404, b"unknown path\n", {"Content-Type": "text/plain"})


def main() -> None:
    print(
        f"text listening on :{PORT} "
        f"(default_size={DEFAULT_SIZE}, max_size={MAX_SIZE}, "
        f"work_multiplier={WORK_MULTIPLIER})",
        flush=True,
    )
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
