#!/usr/bin/env python3
"""
muBench HPA/VPA vs AutoSage comparison.
Runs on the CrownLabs VM directly (not via SSH - this script runs ON the VM).
Usage: source ~/ai4k8s/venv/bin/activate && python3 ~/ai4k8s/mubench/run_comparison_eval.py

Compares three autoscaling approaches against the muBench ingest→process→analyze chain:
  1. Native HPA  – reactive cpu-target=70%, min=2, max=4
  2. Native VPA  – attempt to create VPA object; mark N/A if controller absent
  3. AutoSage    – LLM (Qwen3.5-2B) + MCDA (TOPSIS) predictive advisor

Metrics collected per method:
  - provisioning_latency_s   (time to create the control-plane object)
  - first_scale_latency_s    (seconds from load start to first replica increase)
  - peak_replicas            (max observed replicas during window)
  - p95_latency_s            (95th-percentile response time from 20 probe requests at T+90s)
  - sla_violation_rate       (fraction of probes exceeding SLA_THRESHOLD_S=2.0s)
  - cost_proxy               (avg_replicas × cpu_request_millicores / 1000)
  - recommendation_latency_s (AutoSage only – wall-clock LLM+MCDA time)
"""

import sys, os, time, subprocess, json, math, statistics, logging, tempfile, threading

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

# Suppress library chatter so our output stays readable
logging.disable(logging.WARNING)

from llm_autoscaling_advisor import LLMAutoscalingAdvisor
from autoscaling_engine import HorizontalPodAutoscaler
from vpa_engine import VerticalPodAutoscaler

# ── Constants ────────────────────────────────────────────────────────────────
NAMESPACE        = "default"

# Phase-I: workload configurations. Selected via --workload flag below.
# Each entry holds everything that differs between the CPU-bound chain
# (the original ingest→process→analyze workload) and a stateful workload
# that exercises the LLM's VPA auto-route.
WORKLOAD_CFGS = {
    "cpu": {
        "deployment":     "ingest",
        "services":       ["ingest", "process", "analyze"],
        "cpu_limits":     {"ingest": 500, "process": 500, "analyze": 300},
        "wrk_path":       "/api/v1",
        "svc_name":       "ingest",
        "manifest_dir":   os.path.join(_ROOT, "mubench", "k8s-manifests"),
    },
    "stateful": {
        "deployment":     "session-cache",
        "services":       ["session-cache"],
        "cpu_limits":     {"session-cache": 300},
        "wrk_path":       "/allocate",
        "svc_name":       "session-cache",
        "manifest_dir":   os.path.join(_ROOT, "mubench", "k8s-manifests-stateful"),
    },
    "multiclass": {
        # Phase J: AWARE-style multi-class workload. Currently ships with
        # `compute` (CPU-bound, parametric NxN eigendecomposition).
        #
        # The harness uses a single-URL wrk for the trial window, so we
        # aim it at size=100 (the post-shift heavy payload) for the
        # whole 120s window. That puts the system in the regime an
        # AWARE-style payload shift would create: heavy enough that an
        # HPA calibrated at "size=50 looks fine" cannot keep the
        # workload under the 500ms SLA even after scaling to max=4
        # replicas (Python ThreadingHTTPServer is GIL-bound, so more
        # pods help proportionally less than the math would suggest).
        # The 4-phase payload-shift loadgen YAML (k8s-manifests-multiclass/
        # loadgen.yaml) is kept for ad-hoc out-of-band stress tests but
        # is not what the eval harness drives.
        "deployment":     "compute",
        "services":       ["compute"],
        "cpu_limits":     {"compute": 300},
        "wrk_path":       "/compute?size=100",
        "svc_name":       "compute",
        "manifest_dir":   os.path.join(_ROOT, "mubench", "k8s-manifests-multiclass"),
    },
}
# Default is "cpu" for backward compatibility with every prior eval (v3-v8).
# Argparse at the bottom of the file overrides this when --workload is passed.
WORKLOAD         = os.environ.get("WORKLOAD", "cpu")
_cfg             = WORKLOAD_CFGS[WORKLOAD]
DEPLOYMENT       = _cfg["deployment"]
SERVICES         = list(_cfg["services"])
CPU_REQUEST_M    = 125             # millicores per pod (requests)
CPU_LIMITS       = dict(_cfg["cpu_limits"])
WRK_PATH         = _cfg["wrk_path"]
SVC_NAME         = _cfg["svc_name"]
MANIFEST_DIR     = _cfg["manifest_dir"]
HPA_MIN          = 2
HPA_MAX          = 4
HPA_CPU_TARGET   = 70              # %
WRK_THREADS      = 4
WRK_CONNECTIONS  = int(os.environ.get("WRK_CONNECTIONS", "48"))
WRK_DURATION     = os.environ.get("WRK_DURATION", "120s")
WARMUP_S         = 15
WINDOW_S         = 120
PROBE_AT_S       = 90              # seconds into load window to probe latency
COOLDOWN_S       = 30
# Phase K: env-gated so v13 can run with VPA_POLL_WINDOW=120 to match the
# other methods' trial window. Default 300 preserves v3-v12 reproducibility.
VPA_POLL_WINDOW  = int(os.environ.get("VPA_POLL_WINDOW", "300"))

# Phase K: multi-tick AutoSage. When enabled, the harness fires the advisor
# on AUTOSAGE_TICK_INTERVAL_S cadence inside the trial window. The first
# tick does a full LLM+MCDA call; subsequent ticks reuse the cached LLM
# recommendation and re-run MCDA validation against fresh metrics so the
# decision can adapt without paying a 90-second LLM call per tick.
# Default OFF for v3-v12 reproducibility.
AUTOSAGE_MULTI_TICK_ENABLED = os.environ.get(
    "AUTOSAGE_MULTI_TICK_ENABLED", "0").strip().lower() in {"1", "true", "yes"}
AUTOSAGE_TICK_INTERVAL_S = int(os.environ.get("AUTOSAGE_TICK_INTERVAL_S", "30"))
AUTOSAGE_MAX_TICKS = int(os.environ.get("AUTOSAGE_MAX_TICKS", "5"))
# Phase K post-mortem: with N=3 the per-method CIs are ±14-26 pp,
# so individual trial outliers swing the means by 15+ pp (e.g. HPA's
# v12→v13 mean changed 20 pp mostly because trial 1 went from 55% to
# 15% SLA). Env-gate so v14+ can run N=10 to tighten the picture
# without editing the file.
N_RUNS           = int(os.environ.get("N_RUNS", "3"))
PROBE_REQUESTS   = 20
# Phase J: SLA threshold is env-configurable so the multiclass workload
# can run at 500ms (matching DeathStarBench / AWARE practice) while the
# CPU chain stays at 2.0s for v3-v8 reproducibility.
SLA_THRESHOLD_S  = float(os.environ.get("SLA_THRESHOLD_S", "2.0"))
RESULTS_PATH     = os.environ.get("RESULTS_PATH", "/tmp/comparison_results.json")
PROBE_IMAGE      = "curlimages/curl:8.9.1"

# ── kubectl helper ────────────────────────────────────────────────────────────
def kubectl(*args, silent=False, timeout=60) -> str:
    r = subprocess.run(["kubectl"] + list(args),
                       capture_output=True, text=True, timeout=timeout)
    if not silent and r.returncode != 0:
        print(f"  [kubectl warn] {' '.join(args[:4])}: {r.stderr.strip()[:200]}")
    return r.stdout.strip()

def get_replica_count(deployment: str) -> int:
    # Try readyReplicas first; fall back to spec.replicas when pods are initializing
    out = kubectl("get", "deployment", deployment,
                  "-o", "jsonpath={.status.readyReplicas}", silent=True)
    try:
        if out and out != 'null':
            return int(out)
    except (ValueError, TypeError):
        pass
    out2 = kubectl("get", "deployment", deployment,
                   "-o", "jsonpath={.spec.replicas}", silent=True)
    try:
        return int(out2)
    except (ValueError, TypeError):
        return 0

def get_cpu_millis(svc: str) -> float:
    """Return mean CPU usage in millicores across pods for a service."""
    out = kubectl("top", "pods", "-l", f"app={svc}", "--no-headers", silent=True)
    vals = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                vals.append(int(parts[1].replace("m", "")))
            except ValueError:
                pass
    return sum(vals) / len(vals) if vals else 0.0

def get_svc_cluster_ip(name: str) -> str:
    return kubectl("get", "svc", name, "-o", "jsonpath={.spec.clusterIP}")

def reset_replicas(deployment: str, count: int = HPA_MIN):
    kubectl("scale", "deployment", deployment, f"--replicas={count}", silent=True)
    # Wait up to 20s for stabilisation
    for _ in range(20):
        if get_replica_count(deployment) == count:
            break
        time.sleep(1)

def delete_hpa_if_exists(name: str):
    kubectl("delete", "hpa", name, "-n", NAMESPACE, "--ignore-not-found", silent=True)

def delete_vpa_if_exists(name: str):
    kubectl("delete", "vpa", name, "-n", NAMESPACE, "--ignore-not-found", silent=True)

# ── Load generator ────────────────────────────────────────────────────────────
def start_wrk(target_ip: str):
    """Launch wrk in background; return Popen handle and start timestamp.

    Uses WRK_PATH (workload-specific) so this works for both the CPU chain
    (/api/v1 on ingest) and the stateful workload (/allocate on session-cache).
    """
    proc = subprocess.Popen(
        ["/usr/bin/wrk", "-t", str(WRK_THREADS), "-c", str(WRK_CONNECTIONS),
         "-d", WRK_DURATION, f"http://{target_ip}:8080{WRK_PATH}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return proc, time.time()

# ── Latency probe ─────────────────────────────────────────────────────────────
def probe_latency() -> dict:
    """
    Run PROBE_REQUESTS curl calls inside the cluster via a ephemeral kubectl pod.
    Returns p95 latency (s) and SLA violation rate.
    """
    probe_name = f"probe-{int(time.time())}"
    curl_loop = (
        f"for i in $(seq 1 {PROBE_REQUESTS}); do "
        f"curl -s -o /dev/null -w '%{{time_total}} %{{http_code}}\\n' "
        f"http://{SVC_NAME}.{NAMESPACE}.svc.cluster.local:8080{WRK_PATH} "
        f"|| echo '5.0000 0'; done"
    )
    cmd = [
        "kubectl", "run", probe_name,
        "--rm", "-i", "--restart=Never",
        f"--image={PROBE_IMAGE}",
        "--command", "--",
        "sh", "-c", curl_loop,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        latencies = []
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 1:
                try:
                    latencies.append(float(parts[0]))
                except ValueError:
                    pass
        if not latencies:
            return {"p95_s": None, "sla_violation_rate": None, "n_probes": 0}
        latencies.sort()
        p95_idx = int(math.ceil(0.95 * len(latencies))) - 1
        p95 = latencies[max(p95_idx, 0)]
        violations = sum(1 for l in latencies if l > SLA_THRESHOLD_S)
        return {
            "p95_s": round(p95, 4),
            "sla_violation_rate": round(violations / len(latencies), 4),
            "n_probes": len(latencies),
        }
    except subprocess.TimeoutExpired:
        kubectl("delete", "pod", probe_name, "--ignore-not-found",
                "--force", "--grace-period=0", silent=True, timeout=60)
        return {"p95_s": None, "sla_violation_rate": None, "n_probes": 0}
    except Exception as e:
        print(f"  [probe error] {e}")
        return {"p95_s": None, "sla_violation_rate": None, "n_probes": 0}

# ── Replica watcher ───────────────────────────────────────────────────────────
def watch_replicas(deployment: str, initial: int, duration: float,
                   first_scale_event: threading.Event, results_holder: list):
    """
    Poll replica count every 5s for `duration` seconds.
    Sets first_scale_event when replicas > initial, records peak and avg.
    """
    counts = [initial]
    first_time = None
    start = time.time()
    while time.time() - start < duration:
        n = get_replica_count(deployment)
        counts.append(n)
        if n > initial and first_time is None:
            first_time = time.time() - start
            first_scale_event.set()
        time.sleep(5)
    results_holder.append({
        "peak_replicas": max(counts),
        "avg_replicas": round(sum(counts) / len(counts), 2),
        "first_scale_latency_s": round(first_time, 1) if first_time is not None else None,
    })

# ── Statistics helpers ────────────────────────────────────────────────────────
def mean_ci(values: list) -> tuple:
    """Return (mean, half-CI at 95%) for a list of floats. None values skipped."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    m = statistics.mean(vals)
    s = statistics.stdev(vals)
    # t-value for 95% CI with small n (use 4.303 for n=2, 3.182 for n=3)
    t = {1: 12.706, 2: 4.303, 3: 3.182}.get(len(vals), 2.776)
    half_ci = t * s / math.sqrt(len(vals))
    return round(m, 4), round(half_ci, 4)

# ── Build live metrics context ────────────────────────────────────────────────
def collect_live_metrics(svc: str, cpu_limit_m: int, force_rising: bool = False) -> tuple:
    """Return (current_metrics dict, forecast dict) built from live kubectl top.

    force_rising=True: always set trend=rapidly_increasing with 2.5× peak multiplier.
    Used for AutoSage predictive trials so the LLM evaluates rising-load scenarios.
    """
    cpu_m = get_cpu_millis(svc)
    cpu_pct = round(cpu_m / cpu_limit_m * 100, 1) if cpu_limit_m else 50.0
    replicas = get_replica_count(svc)

    if force_rising:
        trend = "rapidly_increasing"
        preds = [round(min(cpu_pct * (1 + 0.35 * i), 350.0), 1) for i in range(6)]
        peak  = round(min(cpu_pct * 2.5, 350.0), 1)
    else:
        # Simple linear forecast: assume CPU grows 30% per step under sustained load
        preds = [round(min(cpu_pct * (1 + 0.30 * i), 350.0), 1) for i in range(6)]
        peak  = max(preds)
        trend = "rapidly_increasing" if cpu_pct > 60 else "stable"

    current_metrics = {
        "cpu_usage": cpu_pct,
        "memory_usage": 30.0,
        "pod_count": replicas,
        "running_pod_count": replicas,
    }
    forecast = {
        "cpu": {
            "current": cpu_pct,
            "peak": peak,
            "trend": trend,
            "predictions": preds,
        },
        "memory": {
            "current": 30.0,
            "peak": 38.0,
            "trend": "stable",
            "predictions": [30.0] * 6,
        },
    }
    return current_metrics, forecast

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1 — Native HPA
# ══════════════════════════════════════════════════════════════════════════════
def run_hpa_trial(ingest_ip: str, hpa_manager: HorizontalPodAutoscaler,
                  run_idx: int) -> dict:
    print(f"\n  [HPA run {run_idx+1}/{N_RUNS}] resetting replicas → {HPA_MIN} …")
    # Ensure maxReplicas=4 on the existing HPA (restore from any previous pin)
    kubectl("patch", "hpa", DEPLOYMENT, "-n", NAMESPACE,
            "--type=merge", f'--patch={{"spec":{{"maxReplicas":{HPA_MAX}}}}}', silent=True)
    reset_replicas(DEPLOYMENT)

    # HPA already exists in cluster (named same as deployment); measure patch latency
    t_create = time.time()
    # Ensure HPA min/max/target are correct
    kubectl("patch", "hpa", DEPLOYMENT, "-n", NAMESPACE,
            "--type=merge",
            f'--patch={{"spec":{{"minReplicas":{HPA_MIN},"maxReplicas":{HPA_MAX},"targetCPUUtilizationPercentage":{HPA_CPU_TARGET}}}}}',
            silent=True)
    provisioning_latency = time.time() - t_create

    print(f"  [HPA] HPA verified/patched in {provisioning_latency*1000:.1f}ms  — warmup {WARMUP_S}s …")
    time.sleep(WARMUP_S)

    # Start load and watch replicas
    initial_replicas = get_replica_count(DEPLOYMENT)
    wrk, t_load = start_wrk(ingest_ip)
    print(f"  [HPA] wrk PID={wrk.pid}  — monitoring {WINDOW_S}s …")

    scale_event = threading.Event()
    watcher_results = []
    watcher = threading.Thread(
        target=watch_replicas,
        args=(DEPLOYMENT, initial_replicas, WINDOW_S, scale_event, watcher_results),
        daemon=True,
    )
    watcher.start()

    # Wait for probe point
    time.sleep(PROBE_AT_S)
    print(f"  [HPA] T+{PROBE_AT_S}s — running latency probe …")
    probe = probe_latency()

    watcher.join(timeout=WINDOW_S - PROBE_AT_S + 10)
    wrk.wait(timeout=30)

    watch = watcher_results[0] if watcher_results else {}
    first_scale = watch.get("first_scale_latency_s")
    if first_scale is not None:
        first_scale += WARMUP_S   # offset: we started timing at load start

    cost_proxy = round(
        watch.get("avg_replicas", HPA_MIN) * CPU_REQUEST_M / 1000, 4
    )

    return {
        "provisioning_latency_s": round(provisioning_latency, 3),
        "first_scale_latency_s": first_scale,
        "peak_replicas": watch.get("peak_replicas"),
        "p95_latency_s": probe.get("p95_s"),
        "sla_violation_rate": probe.get("sla_violation_rate"),
        "cost_proxy": cost_proxy,
        "n_probes": probe.get("n_probes"),
    }

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2 — Native VPA
# ══════════════════════════════════════════════════════════════════════════════
def run_vpa_trial(ingest_ip: str, vpa_manager: VerticalPodAutoscaler,
                  run_idx: int) -> dict:
    print(f"\n  [VPA run {run_idx+1}/{N_RUNS}] checking VPA controller availability …")
    avail = vpa_manager.check_vpa_available()
    if not avail.get("available"):
        print(f"  [VPA] controller not available — marking N/A")
        return {"na": True, "reason": avail.get("error", "VPA CRD/controller absent")}

    delete_vpa_if_exists(f"{DEPLOYMENT}-vpa")
    reset_replicas(DEPLOYMENT)
    time.sleep(WARMUP_S)

    # Provision VPA
    t_create = time.time()
    result = vpa_manager.create_vpa(
        deployment_name=DEPLOYMENT,
        namespace=NAMESPACE,
        min_cpu="100m",
        max_cpu="1000m",
        min_memory="128Mi",
        max_memory="512Mi",
        update_mode="Auto",
    )
    provisioning_latency = time.time() - t_create

    if not result.get("success"):
        print(f"  [VPA] create failed: {result.get('error')}")
        return {"na": True, "reason": result.get("error")}

    print(f"  [VPA] VPA created in {provisioning_latency:.2f}s  — waiting for first recommendation …")

    # Start load
    wrk, t_load = start_wrk(ingest_ip)
    print(f"  [VPA] wrk PID={wrk.pid}")

    # Poll for first recommendation (up to VPA_POLL_WINDOW seconds)
    first_rec_latency = None
    for elapsed in range(0, VPA_POLL_WINDOW, 5):
        time.sleep(5)
        vpa_info = vpa_manager.get_vpa(f"{DEPLOYMENT}-vpa", NAMESPACE)
        if vpa_info.get("success"):
            status = vpa_info.get("result", {})
            if isinstance(status, dict):
                recs = status.get("status", {}).get("recommendation", {})
                if recs:
                    first_rec_latency = elapsed + 5
                    print(f"  [VPA] first recommendation at T+{first_rec_latency}s")
                    break

    time.sleep(max(0, PROBE_AT_S - (first_rec_latency or 0)))
    probe = probe_latency()

    wrk.wait(timeout=30)
    delete_vpa_if_exists(f"{DEPLOYMENT}-vpa")

    return {
        "provisioning_latency_s": round(provisioning_latency, 3),
        "first_recommendation_latency_s": first_rec_latency,
        "p95_latency_s": probe.get("p95_s"),
        "sla_violation_rate": probe.get("sla_violation_rate"),
        "n_probes": probe.get("n_probes"),
    }

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 3 — AutoSage (LLM + MCDA)
# ══════════════════════════════════════════════════════════════════════════════
def _autosage_actuate(rec: dict, initial_replicas: int) -> bool:
    """Apply the recommendation against the cluster.

    Phase-K extraction: single source of truth for the actuation logic
    so single-tick and multi-tick paths cannot diverge. Returns True iff
    something was actuated.
    """
    action = rec.get("action", "maintain")
    if action != "scale_up":
        return False
    scaling_type = rec.get("scaling_type", "hpa").lower()
    target_replicas = rec.get("target_replicas")
    actuated = False
    if scaling_type in ("hpa", "both") and target_replicas and target_replicas > initial_replicas:
        kubectl("scale", "deployment", DEPLOYMENT,
                f"--replicas={min(target_replicas, HPA_MAX)}", silent=True)
        print(f"  [AutoSage] scaled deployment/{DEPLOYMENT} to {target_replicas} replicas")
        actuated = True
    if scaling_type in ("vpa", "both") and rec.get("target_cpu") and rec.get("target_memory"):
        new_cpu = str(rec.get("target_cpu"))
        new_mem = str(rec.get("target_memory"))
        kubectl("set", "resources", f"deployment/{DEPLOYMENT}",
                f"--requests=cpu={new_cpu},memory={new_mem}",
                "-n", NAMESPACE, silent=True)
        print(f"  [AutoSage] set resources on deployment/{DEPLOYMENT}: "
              f"cpu={new_cpu}, mem={new_mem}")
        actuated = True
    return actuated


def _autosage_tick(advisor: 'LLMAutoscalingAdvisor', hpa_manager,
                   initial_replicas: int, cached_rec: dict, tick_idx: int) -> dict:
    """Run one advisor tick: collect metrics, get recommendation, actuate.

    Phase K (multi-tick AutoSage). If ``cached_rec`` is None, this is the
    first tick of the trial -- do a full LLM+MCDA call (~90s on Qwen,
    ~1s on Groq). If ``cached_rec`` is set, this is a subsequent tick --
    re-use the cached LLM recommendation but re-run MCDA validation
    against fresh metrics so the per-tick decision can adapt without
    paying another LLM call.
    """
    t_metrics = time.time()
    current_metrics, forecast = collect_live_metrics(
        DEPLOYMENT, CPU_LIMITS[DEPLOYMENT], force_rising=True)
    metrics_s = round(time.time() - t_metrics, 3)
    cpu_pct = current_metrics.get("cpu_usage", 0.0)

    if cached_rec is None:
        # First tick: full LLM + MCDA call.
        print(f"  [AutoSage] tick {tick_idx}: CPU={cpu_pct:.1f}%  firing LLM+MCDA …")
        t_rec = time.time()
        rec_result = advisor.get_intelligent_recommendation(
            deployment_name=DEPLOYMENT,
            namespace=NAMESPACE,
            current_metrics=current_metrics,
            forecast=forecast,
            hpa_status={
                "exists": False,
                "current_replicas": get_replica_count(DEPLOYMENT) or initial_replicas,
                "desired_replicas": initial_replicas,
                "target_cpu": HPA_CPU_TARGET,
                "target_memory": 80,
                "scaling_status": "absent",
            },
            vpa_status={"exists": False},
            current_resources={
                "cpu_request": f"{CPU_REQUEST_M}m",
                "cpu_limit": f"{CPU_LIMITS[DEPLOYMENT]}m",
                "memory_request": "128Mi",
                "memory_limit": "256Mi",
            },
            current_replicas=get_replica_count(DEPLOYMENT) or initial_replicas,
            min_replicas=HPA_MIN,
            max_replicas=HPA_MAX,
            hpa_manager=hpa_manager,
        )
        rec_latency = round(time.time() - t_rec, 2)
        rec = rec_result.get("recommendation", {})
        llm_model = rec_result.get("llm_model", "?")
    else:
        # Subsequent tick: keep the LLM's pick, re-validate via MCDA with fresh metrics.
        rec = dict(cached_rec)
        rec_latency = 0.0
        llm_model = "(cached)"
        try:
            mcda = getattr(advisor, "mcda_optimizer", None)
            if mcda is not None:
                current_reps_now = get_replica_count(DEPLOYMENT) or initial_replicas
                validation = mcda.validate_llm_decision(
                    llm_action=rec.get("action", "maintain"),
                    llm_target=rec.get("target_replicas") or current_reps_now,
                    current_replicas=current_reps_now,
                    min_replicas=HPA_MIN,
                    max_replicas=HPA_MAX,
                    metrics={
                        "cpu_percent": cpu_pct,
                        "memory_percent": current_metrics.get("memory_usage", 0),
                    },
                    forecast={
                        "predicted_cpu": forecast.get("cpu", {}).get("predictions", [cpu_pct] * 6) if isinstance(forecast, dict) else [cpu_pct] * 6,
                        "predicted_memory": forecast.get("memory", {}).get("predictions", [current_metrics.get("memory_usage", 0)] * 6) if isinstance(forecast, dict) else [current_metrics.get("memory_usage", 0)] * 6,
                        "cpu_trend": "stable",
                    },
                    agreement_threshold=getattr(advisor, "mcda_agreement_threshold", 0.20),
                )
                rec["mcda_validation"] = {
                    "agreement": validation.get("agreement"),
                    "llm_score": validation.get("llm_score"),
                    "mcda_score": validation.get("mcda_score"),
                    "mcda_target": validation.get("mcda_target"),
                    "score_difference": validation.get("score_difference"),
                    "should_override": validation.get("should_override"),
                    "validation_note": validation.get("validation_note"),
                }
                # If MCDA overrides, swap the action to its preferred choice.
                if validation.get("should_override"):
                    rec["target_replicas"] = validation.get("mcda_target")
                    rec["action"] = validation.get("mcda_action", rec.get("action", "maintain"))
                    rec["scaling_type"] = validation.get("mcda_scaling_type", rec.get("scaling_type", "hpa"))
        except Exception as e:  # noqa: BLE001
            print(f"  [AutoSage] tick {tick_idx}: MCDA re-validation failed (non-fatal): {e}")

    mcda = rec.get("mcda_validation", {}) or {}
    print(f"  [AutoSage] tick {tick_idx}: action={rec.get('action')} "
          f"type={rec.get('scaling_type')} target_reps={rec.get('target_replicas')} "
          f"cpu={rec.get('target_cpu')} mem={rec.get('target_memory')} "
          f"gap={mcda.get('score_difference', 0):.4f} "
          f"override={mcda.get('should_override', False)} "
          f"({rec_latency:.2f}s, {llm_model})")

    t_actuate = time.time()
    actuated = _autosage_actuate(rec, initial_replicas)
    actuation_s = round(time.time() - t_actuate, 3)

    return {
        "tick_idx": tick_idx,
        "cpu_pct": cpu_pct,
        "action": rec.get("action", "maintain"),
        "scaling_type": rec.get("scaling_type", "hpa"),
        "target_replicas": rec.get("target_replicas"),
        "target_cpu": rec.get("target_cpu"),
        "target_memory": rec.get("target_memory"),
        "confidence": rec.get("confidence", 0.0),
        "actuated": actuated,
        "mcda_agreement": mcda.get("agreement", "N/A"),
        "mcda_score_gap": round(mcda.get("score_difference", 0) or 0, 4),
        "mcda_override": mcda.get("should_override", False),
        "rec_latency_s": rec_latency,
        "metrics_s": metrics_s,
        "actuation_s": actuation_s,
        "llm_model": llm_model,
        "_raw_rec": rec,  # cached for subsequent ticks
    }


def run_autosage_trial(ingest_ip: str, advisor: LLMAutoscalingAdvisor,
                       hpa_manager: HorizontalPodAutoscaler, run_idx: int) -> dict:
    print(f"\n  [AutoSage run {run_idx+1}/{N_RUNS}] resetting replicas → {HPA_MIN} …")
    reset_replicas(DEPLOYMENT)
    # Phase I: reset per-pod resources to the manifest baseline between
    # trials so each AutoSage run starts from the same point. Otherwise
    # a VPA actuation from trial N drifts the resource floor for trial N+1.
    kubectl("set", "resources", f"deployment/{DEPLOYMENT}",
            f"--requests=cpu={CPU_REQUEST_M}m,memory=128Mi",
            "-n", NAMESPACE, silent=True)
    # Phase-J task 1: DELETE the HPA for the duration of the AutoSage
    # trial instead of pinning maxReplicas=2. The original pinning
    # silently reconciled AutoSage's kubectl scale calls back to 2
    # within ~15s, so AutoSage's decisions never actuated in the cluster
    # state and peak_replicas was always 2 regardless of what it chose
    # (Phase-I caveat sec:caveat, and Phase-J v11 sec:phasej-v11 saw it
    # again). Deletion matches the AutoScaleAI trial pattern and lets
    # AutoSage's kubectl scale lands stick. HPA is reapplied from
    # manifest at trial end.
    for svc in SERVICES:
        delete_hpa_if_exists(svc)
    print(f"  [AutoSage] HPA deleted for trial window  — warmup {WARMUP_S}s …")
    time.sleep(WARMUP_S)

    initial_replicas = get_replica_count(DEPLOYMENT)

    # Start load
    wrk, t_load = start_wrk(ingest_ip)
    print(f"  [AutoSage] wrk PID={wrk.pid}  — collecting metrics then firing advisor …")

    # Watch replicas in background
    scale_event = threading.Event()
    watcher_results = []
    watcher = threading.Thread(
        target=watch_replicas,
        args=(DEPLOYMENT, initial_replicas, WINDOW_S, scale_event, watcher_results),
        daemon=True,
    )
    watcher.start()

    # Wait 30s so there's real CPU signal before calling the advisor
    time.sleep(30)

    # Phase-K: optional multi-tick path. Tick 1 is the same single-fire
    # advisor call we have always done; subsequent ticks (only when
    # AUTOSAGE_MULTI_TICK_ENABLED) re-evaluate MCDA against fresh metrics
    # using the cached LLM recommendation, so the decision adapts during
    # the trial without paying another 90s LLM call per tick.
    tick_history = []
    cached_rec = None
    first_tick = _autosage_tick(advisor, hpa_manager, initial_replicas,
                                 cached_rec=None, tick_idx=1)
    tick_history.append(first_tick)
    cached_rec = first_tick["_raw_rec"]

    # Trial-end deadline: leave 30s of headroom before the probe so the
    # rollout from any actuation has a chance to land.
    trial_deadline = t_load + WINDOW_S - 30

    if AUTOSAGE_MULTI_TICK_ENABLED:
        next_tick_at = time.time() + AUTOSAGE_TICK_INTERVAL_S
        while (time.time() < trial_deadline
               and len(tick_history) < AUTOSAGE_MAX_TICKS):
            wait = next_tick_at - time.time()
            if wait > 0:
                time.sleep(wait)
            tick = _autosage_tick(advisor, hpa_manager, initial_replicas,
                                   cached_rec=cached_rec,
                                   tick_idx=len(tick_history) + 1)
            tick_history.append(tick)
            # Use the latest decision as the cache base, so an MCDA override
            # propagates forward instead of being lost on the next tick.
            cached_rec = tick["_raw_rec"]
            next_tick_at = time.time() + AUTOSAGE_TICK_INTERVAL_S

    # The "final" trial-level decision is whatever the last tick produced.
    last_tick = tick_history[-1]
    action = last_tick["action"]
    scaling_type = last_tick["scaling_type"]
    target_replicas = last_tick["target_replicas"]
    confidence = last_tick.get("confidence", 0.0) or 0.0
    actuated = any(t["actuated"] for t in tick_history)
    metrics_collection_s = sum(t["metrics_s"] for t in tick_history)
    actuation_s = sum(t["actuation_s"] for t in tick_history)
    recommendation_latency_s = sum(t["rec_latency_s"] for t in tick_history)
    mcda = (last_tick["_raw_rec"] or {}).get("mcda_validation", {}) or {}
    rec = last_tick["_raw_rec"] or {}

    # Probe latency. Reserve enough time for any final VPA rollout.
    probe_wait_target = max(0, PROBE_AT_S - 30 - recommendation_latency_s)
    if probe_wait_target > 0 and time.time() - t_load < PROBE_AT_S:
        gap = PROBE_AT_S - (time.time() - t_load)
        if gap > 0:
            time.sleep(min(gap, probe_wait_target))
    if actuated and scaling_type in ("vpa", "both"):
        print(f"  [AutoSage] waiting for VPA rollout to finish …")
        kubectl("rollout", "status", f"deployment/{DEPLOYMENT}",
                "-n", NAMESPACE, "--timeout=60s", silent=True)
    print(f"  [AutoSage] running latency probe …")
    probe = probe_latency()

    watcher.join(timeout=max(5, WINDOW_S - (time.time() - t_load) + 10))
    wrk.wait(timeout=30)

    # Phase-J task 1: re-apply HPA manifest (was deleted at trial start).
    # Restores the workload's HPA cleanly for the next method's trial.
    kubectl("apply", "-f", HPA_MANIFEST_PATH, silent=True)
    print(f"  [AutoSage] HPA re-applied from {os.path.basename(HPA_MANIFEST_PATH)}")

    watch = watcher_results[0] if watcher_results else {}
    first_scale = watch.get("first_scale_latency_s")

    cost_proxy = round(
        watch.get("avg_replicas", initial_replicas) * CPU_REQUEST_M / 1000, 4
    )

    return {
        "recommendation_latency_s": recommendation_latency_s,
        "action": action,
        "scaling_type": scaling_type,
        "target_replicas": target_replicas,
        # Phase I: capture the LLM's resource targets so downstream
        # analysis can compare LLM vs MCDA picks across both axes.
        "target_cpu": rec.get("target_cpu"),
        "target_memory": rec.get("target_memory"),
        "confidence": round(confidence, 3),
        "actuated": actuated,
        "mcda_agreement": mcda.get("agreement", "N/A"),
        "mcda_score_gap": round(
            mcda.get("score_difference", mcda.get("score_gap", 0)), 4
        ),
        "mcda_override": mcda.get("should_override", False),
        # Phase I: persist MCDA's per-axis pick from the unified pool.
        # mcda_target stays for the replica count; the new fields cover
        # the vertical axis. mcda_scaling_type tells you which axis won.
        "mcda_target": mcda.get("mcda_target"),
        "mcda_target_cpu_m": mcda.get("mcda_target_cpu_m"),
        "mcda_target_memory_mi": mcda.get("mcda_target_memory_mi"),
        "mcda_scaling_type": mcda.get("mcda_scaling_type"),
        "mcda_llm_score": mcda.get("llm_score"),
        "mcda_mcda_score": mcda.get("mcda_score"),
        "mcda_dominance_margin": mcda.get("dominance_margin"),
        "first_scale_latency_s": first_scale,
        "peak_replicas": watch.get("peak_replicas"),
        "p95_latency_s": probe.get("p95_s"),
        "sla_violation_rate": probe.get("sla_violation_rate"),
        "cost_proxy": cost_proxy,
        "n_probes": probe.get("n_probes"),
        "timing_breakdown": {
            "metrics_collection_s": metrics_collection_s,
            "recommendation_s": recommendation_latency_s,
            "actuation_s": actuation_s,
        },
        "llm_model": last_tick.get("llm_model", "?"),
        # Phase-K: multi-tick history. n_ticks=1 when AUTOSAGE_MULTI_TICK_ENABLED
        # is off (back-compat with v3-v12); the per-tick records let the eval
        # show mid-trial decision changes.
        "n_ticks": len(tick_history),
        "tick_history": [
            {
                "tick_idx": t["tick_idx"],
                "cpu_pct": t["cpu_pct"],
                "action": t["action"],
                "scaling_type": t["scaling_type"],
                "target_replicas": t["target_replicas"],
                "target_cpu": t["target_cpu"],
                "target_memory": t["target_memory"],
                "actuated": t["actuated"],
                "mcda_agreement": t["mcda_agreement"],
                "mcda_score_gap": t["mcda_score_gap"],
                "mcda_override": t["mcda_override"],
                "rec_latency_s": t["rec_latency_s"],
                "llm_model": t["llm_model"],
            }
            for t in tick_history
        ],
    }

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 4 — AutoScaleAI (PPO RL baseline)
# Vendored at baselines/autoscaleai with two upstream bug fixes and a
# production inference daemon (runner.py). See baselines/autoscaleai/RUNNER_README.md
# for build/deploy details. The image is pre-built on the VM as
# localhost/autoscaleai:v1 and imported into k3s containerd.
# ══════════════════════════════════════════════════════════════════════════════
AUTOSCALEAI_MANIFESTS_DIR = os.path.join(_ROOT, "baselines", "autoscaleai", "k8s-manifests")
# HPA manifest path follows the selected workload — restored after each
# AutoScaleAI trial finishes so the cluster ends up back at native-HPA.
HPA_MANIFEST_PATH         = os.path.join(MANIFEST_DIR, "hpa.yaml")

def _wait_for_pod_ready(label: str, namespace: str = NAMESPACE, timeout_s: int = 90) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        out = kubectl("get", "pods", "-n", namespace, "-l", label,
                      "-o", "jsonpath={.items[0].status.containerStatuses[0].ready}",
                      silent=True)
        if out.strip().lower() == "true":
            return True
        time.sleep(2)
    return False


def run_autoscaleai_trial(ingest_ip: str, run_idx: int) -> dict:
    print(f"\n  [AutoScaleAI run {run_idx+1}/{N_RUNS}] tearing down conflicting controllers …")
    # AutoScaleAI is the sole controller during this trial. HPAs would fight
    # the RL agent on every tick, so we delete them and restore at the end.
    for svc in SERVICES:
        delete_hpa_if_exists(svc)
        delete_vpa_if_exists(f"{svc}-vpa")
    for svc in SERVICES:
        reset_replicas(svc)

    # Apply manifests — provisioning latency = "kubectl apply" → "controller Ready"
    print(f"  [AutoScaleAI] applying RL controller manifests …")
    t_create = time.time()
    out = subprocess.run(
        ["kubectl", "apply", "-k", AUTOSCALEAI_MANIFESTS_DIR],
        capture_output=True, text=True, timeout=60,
    )
    if out.returncode != 0:
        return {"error": f"apply failed: {out.stderr.strip()[:200]}"}

    if not _wait_for_pod_ready("app=autoscaleai", timeout_s=90):
        kubectl("delete", "-k", AUTOSCALEAI_MANIFESTS_DIR, silent=True)
        kubectl("apply", "-f", HPA_MANIFEST_PATH, silent=True)
        return {"error": "controller did not become Ready within 90s"}
    provisioning_latency = time.time() - t_create

    print(f"  [AutoScaleAI] controller Ready in {provisioning_latency:.1f}s  — warmup {WARMUP_S}s …")
    time.sleep(WARMUP_S)

    initial_replicas = get_replica_count(DEPLOYMENT)
    wrk, t_load = start_wrk(ingest_ip)
    print(f"  [AutoScaleAI] wrk PID={wrk.pid}  — monitoring {WINDOW_S}s …")

    scale_event = threading.Event()
    watcher_results = []
    watcher = threading.Thread(
        target=watch_replicas,
        args=(DEPLOYMENT, initial_replicas, WINDOW_S, scale_event, watcher_results),
        daemon=True,
    )
    watcher.start()

    time.sleep(PROBE_AT_S)
    print(f"  [AutoScaleAI] T+{PROBE_AT_S}s — running latency probe …")
    probe = probe_latency()

    watcher.join(timeout=WINDOW_S - PROBE_AT_S + 10)
    try:
        wrk.wait(timeout=30)
    except subprocess.TimeoutExpired:
        wrk.kill()

    # Always tear down the RL controller and restore HPAs, even on errors above.
    print(f"  [AutoScaleAI] tearing down RL controller, restoring HPAs …")
    kubectl("delete", "-k", AUTOSCALEAI_MANIFESTS_DIR, silent=True)
    kubectl("apply", "-f", HPA_MANIFEST_PATH, silent=True)

    watch = watcher_results[0] if watcher_results else {}
    first_scale = watch.get("first_scale_latency_s")
    if first_scale is not None:
        first_scale += WARMUP_S

    cost_proxy = round(
        watch.get("avg_replicas", HPA_MIN) * CPU_REQUEST_M / 1000, 4
    )

    return {
        "provisioning_latency_s": round(provisioning_latency, 3),
        "first_scale_latency_s": first_scale,
        "peak_replicas": watch.get("peak_replicas"),
        "p95_latency_s": probe.get("p95_s"),
        "sla_violation_rate": probe.get("sla_violation_rate"),
        "cost_proxy": cost_proxy,
        "n_probes": probe.get("n_probes"),
    }

# ── Results aggregation ───────────────────────────────────────────────────────
def aggregate(trials: list, fields: list) -> dict:
    agg = {}
    for f in fields:
        vals = [t.get(f) for t in trials if isinstance(t.get(f), (int, float))]
        m, ci = mean_ci(vals)
        agg[f] = {"mean": m, "half_ci_95": ci, "values": vals}
    return agg

# ── Formatting helpers ────────────────────────────────────────────────────────
def fmt(val, ci=None, unit=""):
    if val is None:
        return "N/A"
    s = f"{val:.3f}{unit}"
    if ci is not None and ci > 0:
        s += f" ± {ci:.3f}"
    return s

def print_section(title: str):
    bar = "═" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)

def print_results_table(all_results: dict):
    print_section("COMPARISON RESULTS SUMMARY")

    metrics = [
        ("provisioning_latency_s",  "Provisioning latency (s)"),
        ("first_scale_latency_s",   "First scale latency (s)"),
        ("peak_replicas",           "Peak replicas"),
        ("p95_latency_s",           "p95 response latency (s)"),
        ("sla_violation_rate",      "SLA violation rate"),
        ("cost_proxy",              "Cost proxy (replicas×req/1000)"),
        ("recommendation_latency_s","AutoSage rec. latency (s)"),
    ]

    col_w = 28
    name_w = 36
    header = f"  {'Metric':<{name_w}}" + "".join(f"{'Method':<{col_w}}" for _ in all_results)
    # header row with method names
    meth_names = list(all_results.keys())
    print(f"\n  {'Metric':<{name_w}}" + "".join(f"{m:<{col_w}}" for m in meth_names))
    print("  " + "-" * (name_w + col_w * len(meth_names)))

    for key, label in metrics:
        row = f"  {label:<{name_w}}"
        for method, data in all_results.items():
            agg = data.get("aggregate", {})
            if data.get("na"):
                row += f"{'N/A (no controller)':<{col_w}}"
            elif key in agg:
                m = agg[key]["mean"]
                ci = agg[key]["half_ci_95"]
                row += f"{fmt(m, ci):<{col_w}}"
            else:
                row += f"{'—':<{col_w}}"
        print(row)

    # AutoSage timing breakdown
    if "AutoSage" in all_results and not all_results["AutoSage"].get("na"):
        print_section("AutoSage Timing Breakdown (mean across runs)")
        for trial in all_results["AutoSage"].get("trials", []):
            tb = trial.get("timing_breakdown", {})
            if tb:
                print(f"  metrics_collection={tb.get('metrics_collection_s','?'):.3f}s  "
                      f"recommendation={tb.get('recommendation_s','?'):.1f}s  "
                      f"actuation={tb.get('actuation_s','?'):.3f}s")
                break   # one representative line is enough

    # Per-run detail for AutoSage. Phase I: VPA decisions have
    # target_replicas=None but target_cpu/target_memory set, so format
    # defensively (no `{None:<3}` blowups) and show resource targets too.
    if "AutoSage" in all_results and not all_results["AutoSage"].get("na"):
        print_section("AutoSage Per-Run Details")
        for i, t in enumerate(all_results["AutoSage"].get("trials", []), 1):
            tr = t.get("target_replicas")
            tr_str = str(tr) if tr is not None else "—"
            tc = t.get("target_cpu") or "—"
            tm = t.get("target_memory") or "—"
            print(f"  Run {i}: action={t.get('action','?'):<12} "
                  f"type={t.get('scaling_type','?'):<6} "
                  f"replicas={tr_str:<3} cpu={tc:<8} mem={tm:<8} "
                  f"conf={t.get('confidence',0):.2f}  "
                  f"rec_latency={t.get('recommendation_latency_s',0):.1f}s  "
                  f"model={t.get('llm_model','?')}")
            print(f"         MCDA agreement={t.get('mcda_agreement','?')}  "
                  f"gap={t.get('mcda_score_gap',0):.4f}  "
                  f"override={t.get('mcda_override',False)}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print_section("muBench HPA / VPA / AutoSage / AutoScaleAI Comparison Evaluation")
    print(f"  Workload        : {WORKLOAD}  (deployment={DEPLOYMENT}, "
          f"svc={SVC_NAME}, wrk_path={WRK_PATH})")
    print(f"  Runs per method : {N_RUNS}")
    print(f"  Warmup          : {WARMUP_S}s")
    print(f"  Load window     : {WINDOW_S}s  ({WRK_CONNECTIONS} connections)")
    print(f"  Probe at        : T+{PROBE_AT_S}s")
    print(f"  Cooldown        : {COOLDOWN_S}s")
    print(f"  SLA threshold   : {SLA_THRESHOLD_S}s")
    print(f"  Results path    : {RESULTS_PATH}")

    hpa_manager = HorizontalPodAutoscaler()
    vpa_manager = VerticalPodAutoscaler()
    advisor = LLMAutoscalingAdvisor()

    ingest_ip = get_svc_cluster_ip(SVC_NAME)
    if not ingest_ip:
        print(f"\n[FATAL] Could not resolve ClusterIP for '{SVC_NAME}' service. "
              f"Is the '{WORKLOAD}' workload deployed? Aborting.")
        sys.exit(1)
    print(f"\n  {SVC_NAME} ClusterIP: {ingest_ip}")

    all_results = {}

    # ── Method 1: HPA ────────────────────────────────────────────────────────
    print_section("METHOD 1 — Native HPA")
    hpa_trials = []
    for i in range(N_RUNS):
        trial = run_hpa_trial(ingest_ip, hpa_manager, i)
        hpa_trials.append(trial)
        if i < N_RUNS - 1:
            print(f"  cooldown {COOLDOWN_S}s …")
            time.sleep(COOLDOWN_S)

    hpa_agg = aggregate(
        [t for t in hpa_trials if not t.get("error")],
        ["provisioning_latency_s", "first_scale_latency_s", "peak_replicas",
         "p95_latency_s", "sla_violation_rate", "cost_proxy"],
    )
    all_results["HPA"] = {"trials": hpa_trials, "aggregate": hpa_agg}

    # ── Method 2: VPA ────────────────────────────────────────────────────────
    print_section("METHOD 2 — Native VPA")
    vpa_trials = []
    for i in range(N_RUNS):
        trial = run_vpa_trial(ingest_ip, vpa_manager, i)
        vpa_trials.append(trial)
        if trial.get("na"):
            # No point retrying if controller is absent
            break
        if i < N_RUNS - 1:
            print(f"  cooldown {COOLDOWN_S}s …")
            time.sleep(COOLDOWN_S)

    vpa_na = all(t.get("na") for t in vpa_trials)
    if vpa_na:
        all_results["VPA"] = {
            "na": True,
            "reason": vpa_trials[0].get("reason", "VPA controller absent"),
            "trials": vpa_trials,
        }
        print(f"  VPA marked N/A: {all_results['VPA']['reason']}")
    else:
        vpa_agg = aggregate(
            [t for t in vpa_trials if not t.get("na") and not t.get("error")],
            ["provisioning_latency_s", "first_recommendation_latency_s",
             "p95_latency_s", "sla_violation_rate"],
        )
        all_results["VPA"] = {"trials": vpa_trials, "aggregate": vpa_agg}

    print(f"  cooldown {COOLDOWN_S}s …")
    time.sleep(COOLDOWN_S)

    # ── Method 3: AutoSage ───────────────────────────────────────────────────
    print_section("METHOD 3 — AutoSage (LLM + MCDA)")
    autosage_trials = []
    for i in range(N_RUNS):
        trial = run_autosage_trial(ingest_ip, advisor, hpa_manager, i)
        autosage_trials.append(trial)
        if i < N_RUNS - 1:
            print(f"  cooldown {COOLDOWN_S}s …")
            time.sleep(COOLDOWN_S)

    autosage_agg = aggregate(
        autosage_trials,
        ["recommendation_latency_s", "first_scale_latency_s", "peak_replicas",
         "p95_latency_s", "sla_violation_rate", "cost_proxy"],
    )
    all_results["AutoSage"] = {"trials": autosage_trials, "aggregate": autosage_agg}

    print(f"  cooldown {COOLDOWN_S}s …")
    time.sleep(COOLDOWN_S)

    # ── Method 4: AutoScaleAI (PPO RL baseline) ──────────────────────────────
    print_section("METHOD 4 — AutoScaleAI (PPO RL baseline)")
    autoscaleai_trials = []
    for i in range(N_RUNS):
        trial = run_autoscaleai_trial(ingest_ip, i)
        autoscaleai_trials.append(trial)
        if trial.get("error"):
            print(f"  [AutoScaleAI] trial {i+1} errored: {trial['error']}")
        if i < N_RUNS - 1:
            print(f"  cooldown {COOLDOWN_S}s …")
            time.sleep(COOLDOWN_S)

    autoscaleai_agg = aggregate(
        [t for t in autoscaleai_trials if not t.get("error")],
        ["provisioning_latency_s", "first_scale_latency_s", "peak_replicas",
         "p95_latency_s", "sla_violation_rate", "cost_proxy"],
    )
    all_results["AutoScaleAI"] = {"trials": autoscaleai_trials, "aggregate": autoscaleai_agg}

    # ── Save JSON FIRST (before any pretty-printing) ─────────────────────────
    # If a downstream print blows up, the eval data is still on disk.
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "n_runs": N_RUNS,
            "warmup_s": WARMUP_S,
            "window_s": WINDOW_S,
            "wrk_connections": WRK_CONNECTIONS,
            "probe_at_s": PROBE_AT_S,
            "sla_threshold_s": SLA_THRESHOLD_S,
            "hpa_min": HPA_MIN,
            "hpa_max": HPA_MAX,
            "hpa_cpu_target_pct": HPA_CPU_TARGET,
            "workload": WORKLOAD,
            "deployment": DEPLOYMENT,
        },
        "results": {
            method: {
                "na": data.get("na", False),
                "na_reason": data.get("reason"),
                "trials": data.get("trials", []),
                "aggregate": data.get("aggregate", {}),
            }
            for method, data in all_results.items()
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_PATH}")

    # ── Print table (best-effort; results are already persisted) ─────────────
    try:
        print_results_table(all_results)
    except Exception as e:
        print(f"\n  [print warning] pretty-print failed (results already saved): {e}")

    # ── Mean ± CI summary ────────────────────────────────────────────────────
    print_section("MEAN ± 95% CI SUMMARY")
    summary_keys = [
        ("provisioning_latency_s",  "Provisioning latency (s)"),
        ("first_scale_latency_s",   "First scale latency (s)"),
        ("peak_replicas",           "Peak replicas"),
        ("p95_latency_s",           "p95 latency (s)"),
        ("sla_violation_rate",      "SLA violation rate"),
        ("cost_proxy",              "Cost proxy"),
        ("recommendation_latency_s","AutoSage rec. latency (s)"),
    ]
    for method, data in all_results.items():
        print(f"\n  {method}:")
        if data.get("na"):
            print(f"    N/A — {data.get('reason', 'controller absent')}")
            continue
        agg = data.get("aggregate", {})
        for key, label in summary_keys:
            if key not in agg:
                continue
            m = agg[key]["mean"]
            ci = agg[key]["half_ci_95"]
            vals = agg[key]["values"]
            if m is None:
                continue
            print(f"    {label:<38} {fmt(m, ci)}   (n={len(vals)}: {vals})")

    print("\n  Done.")

def _apply_workload(name: str) -> None:
    """Rebind module-level constants when --workload changes the selection.

    Importing this script picks up the default workload (env var WORKLOAD
    or "cpu"). When the user passes --workload from the CLI we rebind the
    globals so every downstream reference (DEPLOYMENT, SERVICES, WRK_PATH,
    SVC_NAME, MANIFEST_DIR, HPA_MANIFEST_PATH, CPU_LIMITS) sees the new
    workload's values.
    """
    global WORKLOAD, DEPLOYMENT, SERVICES, CPU_LIMITS
    global WRK_PATH, SVC_NAME, MANIFEST_DIR, HPA_MANIFEST_PATH
    if name not in WORKLOAD_CFGS:
        print(f"[FATAL] unknown workload '{name}'. "
              f"Known: {sorted(WORKLOAD_CFGS.keys())}")
        sys.exit(2)
    cfg = WORKLOAD_CFGS[name]
    WORKLOAD = name
    DEPLOYMENT = cfg["deployment"]
    SERVICES = list(cfg["services"])
    CPU_LIMITS = dict(cfg["cpu_limits"])
    WRK_PATH = cfg["wrk_path"]
    SVC_NAME = cfg["svc_name"]
    MANIFEST_DIR = cfg["manifest_dir"]
    HPA_MANIFEST_PATH = os.path.join(MANIFEST_DIR, "hpa.yaml")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "muBench autoscaler comparison eval. Pick --workload cpu for the "
            "ingest→process→analyze CPU chain (default; reproduces v3-v8) or "
            "--workload stateful for the Phase-I session-cache workload."
        )
    )
    parser.add_argument(
        "--workload",
        choices=sorted(WORKLOAD_CFGS.keys()),
        default=os.environ.get("WORKLOAD", "cpu"),
        help="Workload selection. Default: cpu (or $WORKLOAD env var).",
    )
    args = parser.parse_args()
    _apply_workload(args.workload)
    main()
