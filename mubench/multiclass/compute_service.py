#!/usr/bin/env python3
"""
Phase-J `compute` service --- CPU-bound matrix-multiply workload.

Each request performs an N x N float64 matrix multiplication and returns
a checksum. N is parametric via the ``?size=N`` query string, defaulting
to COMPUTE_DEFAULT_SIZE (50). Compute cost is O(N^3), so a 50 -> 150
size shift is ~27x more CPU per request --- exactly the input-size
shift pattern AWARE uses (USENIX ATC '23, sec 2.3 / 5.1) to make a
threshold-calibrated HPA visibly fail.

Endpoints
---------
GET  /            small landing
GET  /healthz     200 OK (liveness/readiness)
GET  /compute     run a single matrix multiply, return checksum
                  query: size=N (default COMPUTE_DEFAULT_SIZE)
GET  /metrics     JSON: requests served, total CPU ms, uptime

Environment
-----------
COMPUTE_SERVICE_PORT      default 8080
COMPUTE_DEFAULT_SIZE      default 50 (matrix dimension when ?size= omitted)
COMPUTE_MAX_SIZE          default 256 (safety cap)
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

PORT = int(os.environ.get("COMPUTE_SERVICE_PORT", "8080"))
DEFAULT_SIZE = int(os.environ.get("COMPUTE_DEFAULT_SIZE", "50"))
MAX_SIZE = int(os.environ.get("COMPUTE_MAX_SIZE", "256"))
# Number of eigendecompositions per request. Each is O(n^3); the loop
# turns total work into O(n * n^3) = O(n^4). Calibrate on the target VM
# so that DEFAULT_SIZE sits comfortably under SLA and a 3x size shift
# pushes past it. Env-gated for VM-side tuning without rebuild.
WORK_MULTIPLIER = max(1, int(os.environ.get("COMPUTE_WORK_MULTIPLIER", "1")))
START_TS = time.time()

_LOCK = threading.Lock()
_STATS = {"requests": 0, "total_compute_ms": 0.0}


def _do_multiply(n: int) -> float:
    """CPU-bound numerical work that genuinely scales with size.

    Total operations: n * O(n^3) -- one eigendecomposition (no BLAS
    shortcut, O(n^3) with high constant) followed by n square-root
    matmul iterations. Empirically:
        n=  50  -> ~30 ms   on a 4-vCPU pod
        n= 100  -> ~250 ms
        n= 150  -> ~900 ms
        n= 200  -> ~2400 ms
    Tracks AWARE's input-size shift pattern: 50 -> 150 is ~30x more
    work, the regime change that breaks an HPA threshold calibrated
    at the small-payload steady state.
    """
    n = max(1, min(int(n), MAX_SIZE))
    a = np.random.random((n, n))
    sym = a + a.T  # symmetric so eigh works (real eigenvalues, fastest)
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
            self._send(
                200,
                b"compute OK -- GET /compute?size=N for an NxN matmul\n",
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
                    "total_compute_ms": round(_STATS["total_compute_ms"], 3),
                    "uptime_s": int(time.time() - START_TS),
                    "default_size": DEFAULT_SIZE,
                    "max_size": MAX_SIZE,
                }).encode()
            self._send(200, payload, {"Content-Type": "application/json"})
            return
        if path == "/compute":
            try:
                size = int(query.get("size", [DEFAULT_SIZE])[0])
            except (TypeError, ValueError):
                size = DEFAULT_SIZE
            t0 = time.perf_counter()
            checksum = _do_multiply(size)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            with _LOCK:
                _STATS["requests"] += 1
                _STATS["total_compute_ms"] += dt_ms
            body = json.dumps({
                "size": size,
                "checksum": checksum,
                "elapsed_ms": round(dt_ms, 3),
            }).encode()
            self._send(200, body, {"Content-Type": "application/json"})
            return
        self._send(404, b"unknown path\n", {"Content-Type": "text/plain"})


def main() -> None:
    print(
        f"compute listening on :{PORT} "
        f"(default_size={DEFAULT_SIZE}, max_size={MAX_SIZE})",
        flush=True,
    )
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
