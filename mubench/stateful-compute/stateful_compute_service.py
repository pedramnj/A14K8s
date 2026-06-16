#!/usr/bin/env python3
"""Phase-O stateful + CPU-bound workload (session-compute).

Fuses two earlier muBench workloads:

  * Phase-I ``session-cache`` --- per-pod in-memory session STORE. State
    lives *inside* the pod, so the advisor's state detector classifies
    this deployment ``stateful`` and the LLM emits VPA (vertical) the
    same way it did in v23 (10/10).
  * Phase-J ``compute`` --- an O(n^4) ``numpy.linalg.eigh`` payload whose
    cost scales with a ``?size=N`` query string. A 50 -> 150 size shift
    is ~27x more CPU per request, the AWARE (USENIX ATC '23, sec 2.3/5.1)
    input-size stress that breaks a threshold/histogram autoscaler
    calibrated on the small payload.

Every /compute request does BOTH: it runs the eigh payload AND mutates
per-pod session state (writes a session blob, bumps a per-pod counter).
So the "stateful" label is honest --- a reviewer reading this sees the
STORE grow with traffic --- while the compute payload makes the pod
CPU-bound (CPU >= 80%), which guarantees the advisor's force-VPA path
fires (the thread-bound override only skips VPA when CPU < 70%).

Endpoints
---------
GET  /                landing
GET  /healthz         200 OK (liveness/readiness)
GET  /compute         eigh payload + state write; query: size=N
GET  /allocate        pure state write (session-cache compatible)
GET  /sessions/<key>  read a stored blob
GET  /metrics         JSON: requests, sessions held, total compute ms

Environment
-----------
SESSION_COMPUTE_PORT     default 8080
COMPUTE_DEFAULT_SIZE     default 50   (matrix dim when ?size= omitted)
COMPUTE_MAX_SIZE         default 256  (safety cap)
COMPUTE_WORK_MULTIPLIER  default 1    (scales iters; VM-side tuning)
SESSION_BLOB_BYTES       default 8192 (per-session in-pod state size)
SESSION_MAX_HELD         default 4096 (rolling cap so the pod won't OOM)
"""

import json
import os
import random
import string
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

PORT = int(os.environ.get("SESSION_COMPUTE_PORT", "8080"))
DEFAULT_SIZE = int(os.environ.get("COMPUTE_DEFAULT_SIZE", "50"))
MAX_SIZE = int(os.environ.get("COMPUTE_MAX_SIZE", "256"))
WORK_MULTIPLIER = max(1, int(os.environ.get("COMPUTE_WORK_MULTIPLIER", "1")))
BLOB_SIZE = int(os.environ.get("SESSION_BLOB_BYTES", "8192"))
MAX_HELD = int(os.environ.get("SESSION_MAX_HELD", "4096"))
START_TS = time.time()

_LOCK = threading.Lock()
STORE = {}                       # per-pod session state (the "stateful" part)
_STATS = {"requests": 0, "total_compute_ms": 0.0}
_KEY_ALPHABET = string.ascii_lowercase + string.digits


def _random_key(prefix="s"):
    return prefix + "".join(random.choices(_KEY_ALPHABET, k=10))


def _store_session():
    """Write one session blob into per-pod state, rolling-capped."""
    key = _random_key()
    blob = b"X" * BLOB_SIZE
    with _LOCK:
        STORE[key] = blob
        if len(STORE) > MAX_HELD:           # drop oldest, never OOM
            STORE.pop(next(iter(STORE)))
    return key


def _do_multiply(n: int) -> float:
    """CPU-bound numerical work that scales with size.

    iters = (n // 10) * WORK_MULTIPLIER eigendecompositions, each O(n^3),
    so total work is O(n^4). Inherits the Phase-J/compute calibration:
    at cpu_limit=300m under wrk c=16, size=50 sits under a 500ms SLA and
    a size shift toward 150 crosses it.
    """
    n = max(1, min(int(n), MAX_SIZE))
    a = np.random.random((n, n))
    sym = a + a.T
    total = 0.0
    iters = max(1, (n // 10) * WORK_MULTIPLIER)
    for _i in range(iters):
        eigs, _ = np.linalg.eigh(sym)
        total += float(eigs.sum())
    return total


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
            self._send(200,
                       b"session-compute OK -- GET /compute?size=N "
                       b"(eigh + per-pod state)\n",
                       {"Content-Type": "text/plain"})
            return
        if path == "/healthz":
            self._send(200, b"ok\n", {"Content-Type": "text/plain"})
            return
        if path == "/metrics":
            with _LOCK:
                payload = json.dumps({
                    "requests": _STATS["requests"],
                    "sessions_held": len(STORE),
                    "total_compute_ms": round(_STATS["total_compute_ms"], 3),
                    "uptime_s": int(time.time() - START_TS),
                    "default_size": DEFAULT_SIZE,
                    "blob_size": BLOB_SIZE,
                }).encode()
            self._send(200, payload, {"Content-Type": "application/json"})
            return
        if path == "/compute":
            try:
                size = int(query.get("size", [DEFAULT_SIZE])[0])
            except (TypeError, ValueError):
                size = DEFAULT_SIZE
            t0 = time.perf_counter()
            checksum = _do_multiply(size)          # CPU-bound payload
            key = _store_session()                 # per-pod state mutation
            dt_ms = (time.perf_counter() - t0) * 1000.0
            with _LOCK:
                _STATS["requests"] += 1
                _STATS["total_compute_ms"] += dt_ms
            body = json.dumps({
                "size": size,
                "checksum": checksum,
                "session": key,
                "elapsed_ms": round(dt_ms, 3),
            }).encode()
            self._send(200, body, {"Content-Type": "application/json"})
            return
        if path == "/allocate":
            key = _store_session()
            self._send(201, (key + "\n").encode(),
                       {"Content-Type": "text/plain"})
            return
        if path.startswith("/sessions/"):
            key = path[len("/sessions/"):]
            with _LOCK:
                val = STORE.get(key)
            if val is None:
                self._send(404, b"not found\n", {"Content-Type": "text/plain"})
            else:
                self._send(200, val, {"Content-Type": "application/octet-stream"})
            return
        self._send(404, b"unknown path\n", {"Content-Type": "text/plain"})


def main() -> None:
    print(f"session-compute listening on :{PORT} "
          f"(default_size={DEFAULT_SIZE}, blob_size={BLOB_SIZE})", flush=True)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
