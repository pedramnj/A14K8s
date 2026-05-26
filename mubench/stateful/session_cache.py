#!/usr/bin/env python3
"""
Phase-I stateful workload: a per-pod in-memory session cache.

Each pod keeps its own dict; there is no Redis, DB, file persistence,
or sidecar -- state lives entirely inside the pod. Hitting the
``/allocate`` endpoint stores a fixed-size blob under a fresh random
key, so memory grows linearly with request count *for that specific
pod*. Horizontal scaling does not help much (new pods start empty),
which is why this workload is the right shape to exercise the LLM's
VPA auto-route.

Endpoints
---------
GET    /                     small landing page; does not allocate
GET    /healthz              200 OK (used by liveness/readiness probes)
GET    /allocate             create a new session under a random key,
                             store a SESSION_BLOB_BYTES-sized blob,
                             return the key as plain text. This is
                             the endpoint wrk targets.
POST   /sessions/<key>       allocate under an explicit key
GET    /sessions/<key>       fetch the blob (200) or 404
DELETE /sessions/<key>       drop the key (returns 204)
GET    /metrics              JSON: sessions, est_bytes, uptime_s

Environment
-----------
SESSION_CACHE_PORT   default 8080
SESSION_BLOB_BYTES   default 8192 (8 KiB per session)
"""

import json
import os
import random
import string
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("SESSION_CACHE_PORT", "8080"))
BLOB_SIZE = int(os.environ.get("SESSION_BLOB_BYTES", "8192"))
START_TS = time.time()
LOCK = threading.Lock()
STORE: "dict[str, bytes]" = {}

_KEY_ALPHABET = string.ascii_lowercase + string.digits


def _random_key(prefix: str = "s") -> str:
    return prefix + "".join(random.choices(_KEY_ALPHABET, k=10))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args, **kwargs):
        # Silence the default per-request access log -- wrk has its own stats.
        return

    def _send(self, code: int, body: bytes = b"", headers=None):
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
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(
                200,
                b"session-cache OK -- use /allocate to add a session\n",
                {"Content-Type": "text/plain"},
            )
        elif path == "/healthz":
            self._send(200, b"ok\n", {"Content-Type": "text/plain"})
        elif path == "/metrics":
            with LOCK:
                count = len(STORE)
            payload = json.dumps({
                "sessions": count,
                "estimated_bytes": count * BLOB_SIZE,
                "uptime_s": int(time.time() - START_TS),
                "blob_size": BLOB_SIZE,
            }).encode()
            self._send(200, payload, {"Content-Type": "application/json"})
        elif path == "/allocate":
            key = _random_key()
            blob = b"X" * BLOB_SIZE
            with LOCK:
                STORE[key] = blob
            self._send(
                201,
                (key + "\n").encode(),
                {"Content-Type": "text/plain"},
            )
        elif path.startswith("/sessions/"):
            key = path[len("/sessions/"):]
            with LOCK:
                val = STORE.get(key)
            if val is None:
                self._send(404, b"not found\n", {"Content-Type": "text/plain"})
            else:
                self._send(
                    200, val, {"Content-Type": "application/octet-stream"}
                )
        else:
            self._send(404, b"unknown path\n", {"Content-Type": "text/plain"})

    def do_POST(self):
        if self.path.startswith("/sessions/"):
            key = self.path[len("/sessions/"):]
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length > 0:
                try:
                    self.rfile.read(length)
                except Exception:
                    pass
            blob = b"X" * BLOB_SIZE
            with LOCK:
                STORE[key] = blob
            self._send(201, b"stored\n", {"Content-Type": "text/plain"})
        else:
            self._send(404, b"unknown path\n", {"Content-Type": "text/plain"})

    def do_DELETE(self):
        if self.path.startswith("/sessions/"):
            key = self.path[len("/sessions/"):]
            with LOCK:
                STORE.pop(key, None)
            self._send(204)
        else:
            self._send(404, b"unknown path\n", {"Content-Type": "text/plain"})


def main() -> None:
    print(
        f"session-cache listening on :{PORT} (blob_size={BLOB_SIZE} bytes)",
        flush=True,
    )
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("session-cache shutting down", flush=True)
        server.server_close()


if __name__ == "__main__":
    main()
