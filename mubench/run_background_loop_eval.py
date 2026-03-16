#!/usr/bin/env python3
"""
AutoSage Background Loop Evaluation
====================================
Simulates the production 5-minute predictive autoscaling loop against a live
muBench workload. Directly calls predict_and_scale() for each service on a
timed interval — the same code path the Flask background thread executes.

Usage (on CrownLabs VM):
  source ~/ai4k8s/venv/bin/activate
  # Default: 3 cycles × 300 s interval = ~16 min total
  python3 -u ~/ai4k8s/mubench/run_background_loop_eval.py

  # Faster: override interval to 120 s for ~8 min total
  LOOP_INTERVAL_S=120 python3 -u ~/ai4k8s/mubench/run_background_loop_eval.py

Results saved to: /tmp/background_loop_results.json
"""

import sys, os, time, subprocess, json, logging

# ── Path & .env setup ────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

_env_path = os.path.join(_ROOT, '.env')
if os.path.exists(_env_path):
    for _line in open(_env_path):
        _line = _line.strip()
        if '=' in _line and not _line.startswith('#'):
            _k, _v = _line.split('=', 1)
            os.environ.setdefault(_k.strip(), _v.strip())

logging.disable(logging.WARNING)

from predictive_autoscaler import PredictiveAutoscaler
from autoscaling_engine import HorizontalPodAutoscaler
from vpa_engine import VerticalPodAutoscaler
from predictive_monitoring import PredictiveMonitoringSystem

# ── Constants ────────────────────────────────────────────────────────────────
NAMESPACE       = "default"
SERVICES        = ["ingest", "process", "analyze"]
HPA_MIN         = 2
HPA_MAX         = 4
WRK_THREADS     = 4
WRK_CONNECTIONS = 48
WRK_DURATION    = "1200s"          # 20 minutes, covers all cycles
LOOP_INTERVAL_S = int(os.getenv("LOOP_INTERVAL_S", "300"))
N_CYCLES        = 3
RESULTS_PATH    = "/tmp/background_loop_results.json"

# ── Helpers ───────────────────────────────────────────────────────────────────
def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)

def get_cluster_ip(svc, ns=NAMESPACE):
    r = run(["kubectl", "get", "svc", svc, "-n", ns,
             "-o", "jsonpath={.spec.clusterIP}"])
    return r.stdout.strip() if r.returncode == 0 else None

def get_replicas(svc, ns=NAMESPACE):
    r = run(["kubectl", "get", "deploy", svc, "-n", ns,
             "-o", "jsonpath={.status.readyReplicas}"])
    try:
        return int(r.stdout.strip() or "0")
    except ValueError:
        return 0

def get_cpu_pct(svc, ns=NAMESPACE):
    """Returns per-pod average CPU% based on kubectl top pods."""
    r = run(["kubectl", "top", "pods", "-n", ns,
             "--no-headers", "-l", f"app={svc}"])
    if r.returncode != 0 or not r.stdout.strip():
        return None
    total_m, count = 0, 0
    for line in r.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            cpu_str = parts[1].rstrip('m')
            try:
                total_m += int(cpu_str)
                count += 1
            except ValueError:
                pass
    if count == 0:
        return None
    cpu_limit = {"ingest": 500, "process": 500, "analyze": 300}.get(svc, 500)
    return round((total_m / count) / cpu_limit * 100, 1)

def print_banner(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print_banner("AutoSage Background Loop Evaluation")
    print(f"Services  : {SERVICES}")
    print(f"Interval  : {LOOP_INTERVAL_S} s  |  Cycles: {N_CYCLES}")
    print(f"Load      : wrk {WRK_CONNECTIONS}c × {WRK_DURATION}")

    # Locate ingest ClusterIP for wrk
    ingest_ip = get_cluster_ip("ingest")
    if not ingest_ip:
        print("ERROR: cannot resolve ingest ClusterIP")
        sys.exit(1)
    print(f"Ingest IP : {ingest_ip}")

    # Initialise autoscaling stack (same as production path)
    print("\n[init] Loading AutoscalingIntegration stack...")
    hpa_manager = HorizontalPodAutoscaler()
    vpa_manager = VerticalPodAutoscaler()
    monitoring   = PredictiveMonitoringSystem()
    autoscaler   = PredictiveAutoscaler(
        monitoring, hpa_manager, vpa_manager=vpa_manager, use_llm=True
    )
    print("[init] Stack ready")

    # Start wrk load
    print(f"\n[load] Starting wrk {WRK_CONNECTIONS}c against {ingest_ip}:8080/api/v1 ...")
    wrk_proc = subprocess.Popen(
        ["/usr/bin/wrk", "-t", str(WRK_THREADS), "-c", str(WRK_CONNECTIONS),
         "-d", WRK_DURATION, f"http://{ingest_ip}:8080/api/v1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    print(f"[load] wrk PID={wrk_proc.pid}")

    # Wait for load to settle and first metrics to appear
    print(f"[load] Waiting 30 s for load to ramp up...")
    time.sleep(30)

    results = {
        "config": {
            "services": SERVICES,
            "loop_interval_s": LOOP_INTERVAL_S,
            "n_cycles": N_CYCLES,
            "wrk_connections": WRK_CONNECTIONS,
            "hpa_min": HPA_MIN,
            "hpa_max": HPA_MAX,
        },
        "cycles": []
    }

    # ── Main loop ─────────────────────────────────────────────────────────────
    for cycle in range(1, N_CYCLES + 1):
        cycle_start = time.time()
        print_banner(f"Cycle {cycle}/{N_CYCLES}  (T+{int(time.time() - (cycle_start - 30))} s from load start)")

        cycle_data = {
            "cycle": cycle,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "decisions": []
        }

        for svc in SERVICES:
            replicas = get_replicas(svc)
            cpu_pct  = get_cpu_pct(svc)
            print(f"\n  [{svc}] replicas={replicas}  cpu≈{cpu_pct}%")
            print(f"  [{svc}] Calling predict_and_scale ...")

            t0 = time.time()
            try:
                result = autoscaler.predict_and_scale(
                    svc, NAMESPACE, min_replicas=HPA_MIN, max_replicas=HPA_MAX
                )
            except Exception as e:
                print(f"  [{svc}] ERROR: {e}")
                cycle_data["decisions"].append({
                    "service": svc, "error": str(e),
                    "cpu_pct": cpu_pct, "replicas": replicas
                })
                continue

            elapsed = round(time.time() - t0, 1)
            action          = result.get('action', 'unknown')
            target_replicas = result.get('target_replicas', replicas)
            confidence      = result.get('confidence')
            timing          = result.get('timing_breakdown', {})
            llm_s           = timing.get('llm_inference_s') or timing.get('recommendation_s', elapsed)
            mcda            = result.get('mcda_validation') or result.get('mcda', {})
            mcda_agreement  = mcda.get('agreement', 'unknown') if mcda else 'unknown'
            mcda_gap        = mcda.get('score_difference', mcda.get('score_gap', None)) if mcda else None
            actuated        = result.get('actuated', False)

            print(f"  [{svc}] action={action}  target={target_replicas}  conf={confidence}")
            print(f"  [{svc}] mcda={mcda_agreement}  gap={mcda_gap}  llm={llm_s:.1f}s  actuated={actuated}")

            cycle_data["decisions"].append({
                "service":        svc,
                "cpu_pct":        cpu_pct,
                "replicas":       replicas,
                "action":         action,
                "target_replicas":target_replicas,
                "confidence":     confidence,
                "mcda_agreement": mcda_agreement,
                "mcda_score_gap": mcda_gap,
                "actuated":       actuated,
                "llm_inference_s":round(llm_s, 2) if llm_s else None,
                "total_s":        elapsed,
                "timing_breakdown": timing,
            })

        results["cycles"].append(cycle_data)

        # Save intermediate results after each cycle
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[cycle {cycle}] Results saved to {RESULTS_PATH}")

        # Wait for next cycle (skip wait after last cycle)
        if cycle < N_CYCLES:
            wait = max(0, LOOP_INTERVAL_S - int(time.time() - cycle_start))
            print(f"[cycle {cycle}] Waiting {wait} s for next cycle ...")
            time.sleep(wait)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print_banner("Stopping load")
    wrk_proc.terminate()
    wrk_proc.wait()
    print("[load] wrk stopped")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_banner("Summary")
    for cycle_data in results["cycles"]:
        print(f"\nCycle {cycle_data['cycle']}  ({cycle_data['timestamp']})")
        print(f"  {'Service':<10} {'CPU%':>6} {'Replicas':>9} {'Action':<12} {'Target':>7} {'MCDA':<10} {'LLM(s)':>7}")
        print(f"  {'-'*65}")
        for d in cycle_data["decisions"]:
            if "error" in d:
                print(f"  {d['service']:<10}  ERROR: {d['error']}")
                continue
            print(f"  {d['service']:<10} {str(d['cpu_pct'] or '?'):>6} {str(d['replicas']):>9} "
                  f"{d['action']:<12} {str(d['target_replicas']):>7} "
                  f"{d['mcda_agreement']:<10} {str(d['llm_inference_s'] or '?'):>7}")

    print(f"\nFull results: {RESULTS_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
