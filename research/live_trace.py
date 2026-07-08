#!/usr/bin/env python3
"""live_trace.py — 24-hour diurnal AutoSage trace against DSB Hotel Reservation.

Runs a passive, long-form trace where the AutoSage continuous daemon operates
against the DSB Hotel Reservation substrate under a realistic sinusoidal load
profile with four business-hour peaks per day. Emits one JSON with a full
tick history so a reviewer can reproduce fig_live_trace.png byte-for-byte
without re-running the trace.

Not part of the eval harness — this is a companion tool for the artifact
bundle. Runs on the same VM as `run_comparison_eval.py`; assumes the DSB
substrate is already deployed.

Usage:
    python3 research/live_trace.py --duration-h 24 --results live_trace_1day.json
    python3 research/live_trace.py --duration-h 1 --results smoke_trace.json  # smoke

Load profile (per hour t in [0, 24)):
    R(t) = R_MIN + (R_MAX - R_MIN) * (0.5 + 0.5 * cos(2*pi * 4 * (t/24) - pi))
    → 4 peaks per day, symmetric business-hour envelope.

Trace record (one per 30 s tick):
    { "wall_time": iso8601, "hour_of_day": float,
      "wrk_rate": int, "cpu_millis": {svc: float},
      "replicas": {svc: int}, "autosage_action": {svc: str},
      "autosage_target": {svc: int} }
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Config ───────────────────────────────────────────────────────────────────
DSB_APP_SERVICES = ["frontend", "search", "geo", "rate", "profile"]
FRONTEND_PORT = 5000
NAMESPACE = "default"

R_MIN = 100          # requests/sec at the diurnal low
R_MAX = 1500         # requests/sec at each of the four peaks
TICK_S = 30          # sampling period
WRK_LUA = os.path.join(_REPO, "dsb-hotel", "wrk2",
                       "mixed-workload_type_1.lua")


# ── kubectl helpers (subset copy from run_comparison_eval to keep this
#    script standalone-runnable) ────────────────────────────────────────────
def kubectl(*args, timeout: int = 30) -> str:
    r = subprocess.run(["kubectl"] + list(args),
                       capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()


def get_replica_count(svc: str) -> int:
    out = kubectl("get", "deployment", svc,
                  "-o", "jsonpath={.status.readyReplicas}")
    try:
        return int(out) if out and out != "null" else 0
    except ValueError:
        return 0


def get_cpu_millis(svc: str) -> float:
    out = kubectl("top", "pods", "-l", f"io.kompose.service={svc}",
                  "--no-headers")
    vals = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                vals.append(int(parts[1].rstrip("m")))
            except ValueError:
                pass
    return sum(vals) / len(vals) if vals else 0.0


def frontend_ip() -> str | None:
    ip = kubectl("get", "svc", "frontend",
                 "-o", "jsonpath={.spec.clusterIP}")
    return ip if ip else None


# ── Diurnal rate profile ──────────────────────────────────────────────────────
def rate_at(hour_of_day: float,
            r_min: int = R_MIN, r_max: int = R_MAX) -> int:
    """Sinusoid with 4 daily peaks (period = 6 h)."""
    t = (hour_of_day / 24.0) * 2 * math.pi * 4
    envelope = 0.5 + 0.5 * math.cos(t - math.pi)
    return int(r_min + (r_max - r_min) * envelope)


# ── AutoSage tick ─────────────────────────────────────────────────────────────
def import_advisor():
    """Late import so this script can be inspected without the runtime env."""
    sys.path.insert(0, _REPO)
    from llm_autoscaling_advisor import LLMAutoscalingAdvisor  # noqa: E402
    from autoscaling_engine import HorizontalPodAutoscaler   # noqa: E402
    return LLMAutoscalingAdvisor(), HorizontalPodAutoscaler()


def autosage_pick(advisor, hpa_manager, svc: str,
                  cpu_pct: float, replicas: int) -> dict:
    """Ask the advisor for a decision on this service. Non-blocking-ish
    (~1.5 s per call on local Qwen). Falls back to a stub if the advisor
    call fails so the trace never crashes on transient LLM issues.
    """
    try:
        rec_result = advisor.get_intelligent_recommendation(
            deployment_name=svc,
            namespace=NAMESPACE,
            current_metrics={
                "cpu_usage": cpu_pct,
                "memory_usage": 30.0,
                "pod_count": replicas,
                "running_pod_count": replicas,
            },
            forecast={
                "cpu": {"current": cpu_pct, "peak": cpu_pct * 1.5,
                        "trend": "stable",
                        "predictions": [cpu_pct] * 6},
                "memory": {"current": 30.0, "peak": 38.0,
                           "trend": "stable",
                           "predictions": [30.0] * 6},
            },
            hpa_status={"exists": True, "current_replicas": replicas,
                        "desired_replicas": replicas,
                        "target_cpu": 70, "target_memory": 80,
                        "scaling_status": "steady"},
            vpa_status={"exists": False},
            current_resources={
                "cpu_request": "150m", "cpu_limit": "500m",
                "memory_request": "128Mi", "memory_limit": "256Mi",
            },
            current_replicas=replicas,
            min_replicas=2, max_replicas=4,
            hpa_manager=hpa_manager,
        )
        rec = rec_result.get("recommendation", {}) or {}
        return {
            "action": rec.get("action", "maintain"),
            "scaling_type": rec.get("scaling_type", "hpa"),
            "target_replicas": rec.get("target_replicas"),
            "target_cpu": rec.get("target_cpu"),
            "target_memory": rec.get("target_memory"),
            "llm_model": rec_result.get("llm_model", "?"),
        }
    except Exception as e:
        return {"action": "error", "error": str(e)[:200]}


# ── Load driver (wrk2) ────────────────────────────────────────────────────────
def start_wrk_at_rate(target_ip: str, rate: int, duration_s: int):
    """Fire off a bounded wrk2 run for `duration_s` at `rate` req/s."""
    wrk_bin = subprocess.run(["which", "wrk2"], capture_output=True,
                             text=True).stdout.strip() or "/usr/local/bin/wrk2"
    cmd = [
        wrk_bin, "-t", "4", "-c", "48", "-d", f"{duration_s}s",
        "-R", str(rate), "-L", "-s", WRK_LUA,
        f"http://{target_ip}:{FRONTEND_PORT}",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)


# ── Main loop ─────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration-h", type=float, default=24.0,
                    help="Wall-clock duration in hours (default 24).")
    ap.add_argument("--tick-s", type=int, default=TICK_S,
                    help="Sampling period in seconds (default 30).")
    ap.add_argument("--results", default="live_trace.json",
                    help="Output JSON path.")
    ap.add_argument("--r-min", type=int, default=R_MIN)
    ap.add_argument("--r-max", type=int, default=R_MAX)
    ap.add_argument("--dry-run-no-llm", action="store_true",
                    help="Skip advisor calls; just log CPU/replicas + wrk rate. "
                         "Useful for verifying the load profile without paying "
                         "LLM latency.")
    args = ap.parse_args()

    fip = frontend_ip()
    if not fip:
        print("[fatal] frontend service ClusterIP not resolvable — is the "
              "DSB substrate deployed? Try `kubectl apply -f "
              "dsb-hotel/kubernetes/` first.", file=sys.stderr)
        return 1

    if not os.path.exists(WRK_LUA):
        print(f"[fatal] wrk2 lua script not found: {WRK_LUA}", file=sys.stderr)
        return 1

    advisor, hpa_manager = (None, None)
    if not args.dry_run_no_llm:
        advisor, hpa_manager = import_advisor()

    total_s = int(args.duration_h * 3600)
    print(f"  [live-trace] frontend={fip}  duration={args.duration_h}h  "
          f"tick={args.tick_s}s  ticks={total_s // args.tick_s}")

    records: list[dict] = []
    t0 = time.time()
    current_wrk = None
    current_wrk_rate = None
    tick_idx = 0

    try:
        while (time.time() - t0) < total_s:
            elapsed_s = time.time() - t0
            hour_of_day = (elapsed_s / 3600.0) % 24.0
            rate = rate_at(hour_of_day, args.r_min, args.r_max)

            # Restart wrk2 if the target rate has drifted > 10 %.
            if (current_wrk is None or current_wrk.poll() is not None
                    or current_wrk_rate is None
                    or abs(rate - current_wrk_rate) / max(current_wrk_rate, 1)
                    > 0.10):
                if current_wrk and current_wrk.poll() is None:
                    current_wrk.terminate()
                    try:
                        current_wrk.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        current_wrk.kill()
                # Kick off a fresh wrk run scoped to the next 60 s (long
                # enough to cover a few ticks; when we drift we'll restart).
                current_wrk = start_wrk_at_rate(fip, rate,
                                                 duration_s=max(60, args.tick_s * 2))
                current_wrk_rate = rate
                print(f"  [t={elapsed_s:>6.0f}s] "
                      f"hour={hour_of_day:.2f}  rate={rate} req/s  "
                      f"(wrk PID={current_wrk.pid})")

            # Sample cluster + advisor.
            record = {
                "wall_time": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime()),
                "elapsed_s": round(elapsed_s, 1),
                "hour_of_day": round(hour_of_day, 3),
                "wrk_rate": rate,
                "cpu_millis": {},
                "replicas": {},
                "autosage": {},
            }
            for svc in DSB_APP_SERVICES:
                record["cpu_millis"][svc] = round(get_cpu_millis(svc), 1)
                record["replicas"][svc] = get_replica_count(svc)
                if not args.dry_run_no_llm:
                    cpu_pct = round(record["cpu_millis"][svc] / 5.0, 1)  # rough
                    record["autosage"][svc] = autosage_pick(
                        advisor, hpa_manager, svc,
                        cpu_pct=cpu_pct, replicas=record["replicas"][svc])
            records.append(record)
            tick_idx += 1

            # Periodic checkpoint to disk so a mid-trace crash doesn't lose
            # everything.
            if tick_idx % 40 == 0:
                with open(args.results, "w") as f:
                    json.dump({"records": records,
                               "config": {
                                   "duration_h": args.duration_h,
                                   "tick_s": args.tick_s,
                                   "r_min": args.r_min,
                                   "r_max": args.r_max,
                               }}, f, indent=2)
                print(f"  [checkpoint] {tick_idx} ticks -> {args.results}")

            # Sleep to the next tick boundary.
            next_tick = t0 + (tick_idx + 1) * args.tick_s
            slack = next_tick - time.time()
            if slack > 0:
                time.sleep(slack)
    except KeyboardInterrupt:
        print("\n  [live-trace] interrupted — flushing partial trace.")
    finally:
        if current_wrk and current_wrk.poll() is None:
            current_wrk.terminate()

    with open(args.results, "w") as f:
        json.dump({
            "records": records,
            "config": {
                "duration_h": args.duration_h,
                "tick_s": args.tick_s,
                "r_min": args.r_min,
                "r_max": args.r_max,
                "services": DSB_APP_SERVICES,
            },
        }, f, indent=2)
    print(f"  [live-trace] wrote {args.results}  ({len(records)} ticks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
