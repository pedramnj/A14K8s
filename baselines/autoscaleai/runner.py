"""Production inference daemon for the AutoScaleAI baseline.

Loads the PPO policy we trained on a muBench-shaped simulator
(`baselines/autoscaleai/training/`) and uses it to scale Kubernetes
deployments based on live Prometheus metrics.

Each TICK_SECONDS, for each TARGET_DEPLOYMENT:
  - query Prometheus for total per-pod CPU rate (cores)
  - read the current replica count via the K8s API
  - build the 3-d observation `[cpu_util, replicas_norm, request_rate_norm]`
    in the order matching `training/mubench_env.py`
  - call `model.predict(obs, deterministic=True)` -> {0=down, 1=noop, 2=up}
  - patch `deployments/scale` if the action moves the count, clamped to
    [MIN_REPLICAS, MAX_REPLICAS]

The runner exposes a /healthz + /metrics HTTP endpoint on HEALTH_PORT for
liveness probes and observability.

Configuration is via environment variables (see Config.from_env). The
daemon is single-threaded and stateless across restarts.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import numpy as np
from stable_baselines3 import PPO

# Vendored modules — we keep upstream's collector and scaler (with the
# two upstream bug fixes) but replace its agent with the sb3-trained policy.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collector.metrics import PrometheusCollector
from executor.scaler import K8sScaler

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s autoscaleai %(message)s",
)
log = logging.getLogger("autoscaleai")


@dataclass
class Config:
    prometheus_url: str
    namespace: str
    deployments: list[str]
    min_replicas: int
    max_replicas: int
    tick_seconds: float
    model_path: str
    state_dim: int
    action_dim: int
    cpu_request_millicores: int   # used to normalise rate -> utilisation
    rps_proxy_max_cores: float    # state[0] saturates here
    health_port: int

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            prometheus_url=os.getenv(
                "PROMETHEUS_URL",
                "http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090",
            ),
            namespace=os.getenv("TARGET_NAMESPACE", "default"),
            deployments=[d.strip() for d in os.getenv(
                "TARGET_DEPLOYMENTS", "ingest,process,analyze"
            ).split(",") if d.strip()],
            min_replicas=int(os.getenv("MIN_REPLICAS", "2")),
            max_replicas=int(os.getenv("MAX_REPLICAS", "4")),
            tick_seconds=float(os.getenv("TICK_SECONDS", "10")),
            model_path=os.getenv(
                "MODEL_PATH", "/app/mubench_ppo_v1.zip"
            ),
            state_dim=int(os.getenv("STATE_DIM", "3")),
            action_dim=int(os.getenv("ACTION_DIM", "3")),
            cpu_request_millicores=int(os.getenv("CPU_REQUEST_MILLICORES", "125")),
            rps_proxy_max_cores=float(os.getenv("RPS_PROXY_MAX_CORES", "2.0")),
            health_port=int(os.getenv("HEALTH_PORT", "8081")),
        )


class HealthHandler(BaseHTTPRequestHandler):
    last_tick_ts = 0.0
    decisions = 0
    last_decision = {}

    def log_message(self, fmt, *args):  # silence default access log
        return

    def do_GET(self):
        if self.path == "/healthz":
            ok = (time.time() - HealthHandler.last_tick_ts) < 60
            self.send_response(200 if ok else 503)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok\n" if ok else b"stale\n")
        elif self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            payload = {
                "decisions_total": HealthHandler.decisions,
                "last_decision": HealthHandler.last_decision,
                "seconds_since_last_tick": round(time.time() - HealthHandler.last_tick_ts, 2),
            }
            self.wfile.write(json.dumps(payload).encode())
        else:
            self.send_response(404)
            self.end_headers()


def serve_health(port: int) -> None:
    httpd = HTTPServer(("0.0.0.0", port), HealthHandler)
    log.info("health server listening on :%d", port)
    httpd.serve_forever()


def get_replica_count(scaler: K8sScaler, namespace: str, name: str) -> int:
    resp = scaler.apps.read_namespaced_deployment_scale(name=name, namespace=namespace)
    return int(resp.spec.replicas or 0)


def build_state(cpu_cores: float, replicas: int, cfg: Config) -> np.ndarray:
    """Map real cluster signals onto the trained policy's 3-d observation.

    Order matches `training/mubench_env.py:MubenchScalingEnv._obs`:
        s[0] = cpu_util_per_pod   in [0, 1]
        s[1] = replicas_norm      in [0, 1]
        s[2] = rps_norm           in [0, 1]

    Notes:
      - cpu_util is computed against the per-pod CPU *request* (125 m), so
        a value of 1.0 means each pod is using its full request
        allotment — past that point K8s starts throttling. This matches
        the simulator's "rho = arrival/capacity" intuition.
      - rps_norm is approximated by total CPU rate (cores) clipped to
        RPS_PROXY_MAX_CORES. We don't have a first-class request-rate
        metric on muBench without instrumenting the services, and CPU
        rate is a reasonable proxy because all three services are
        compute-bound.
    """
    request_cores = cfg.cpu_request_millicores / 1000.0
    capacity_cores = max(1e-6, replicas * request_cores)
    cpu_util_per_pod = min(1.0, cpu_cores / capacity_cores)
    replicas_norm = replicas / max(1, cfg.max_replicas)
    rps_norm = min(1.0, cpu_cores / max(1e-6, cfg.rps_proxy_max_cores))
    return np.array([cpu_util_per_pod, replicas_norm, rps_norm], dtype=np.float32)


ACTION_NAMES = {0: "scale_down", 1: "noop", 2: "scale_up"}


def decide_target(action: int, current: int, cfg: Config) -> int:
    if action == 0:
        return max(cfg.min_replicas, current - 1)
    if action == 2:
        return min(cfg.max_replicas, current + 1)
    return current


def control_loop(cfg: Config) -> None:
    log.info("starting control loop: %s", cfg)
    collector = PrometheusCollector(base_url=cfg.prometheus_url, mode="real")
    scaler = K8sScaler(in_cluster=True, mode="real")
    # CPU device — the testbed has no GPU. sb3 picks the device on load
    # but we pin to CPU explicitly so a CUDA-built torch wheel doesn't
    # try to allocate on a GPU that isn't there.
    model = PPO.load(cfg.model_path, device="cpu")
    log.info("loaded sb3 PPO model from %s", cfg.model_path)

    while True:
        tick_start = time.time()
        for dep in cfg.deployments:
            try:
                cpu_cores = collector.get_cpu_usage(namespace=cfg.namespace, deployment=dep)
                current = get_replica_count(scaler, cfg.namespace, dep)
                state = build_state(cpu_cores, current, cfg)
                # deterministic=True so two pods seeing the same metrics
                # would always emit the same action. Stochastic during
                # training, deterministic during inference.
                action_arr, _ = model.predict(state, deterministic=True)
                action = int(action_arr)
                target = decide_target(action, current, cfg)

                decision = {
                    "deployment": dep,
                    "cpu_cores": round(cpu_cores, 4),
                    "current_replicas": current,
                    "state": state.tolist(),
                    "action": ACTION_NAMES[action],
                    "target_replicas": target,
                    "ts": int(time.time()),
                }
                log.info("decision: %s", json.dumps(decision))
                HealthHandler.last_decision = decision
                HealthHandler.decisions += 1

                if target != current:
                    scaler.scale_deployment(cfg.namespace, dep, target)
            except Exception as e:  # keep the loop alive
                log.exception("error processing %s: %s", dep, e)

        HealthHandler.last_tick_ts = time.time()
        elapsed = time.time() - tick_start
        time.sleep(max(0.0, cfg.tick_seconds - elapsed))


def main() -> int:
    cfg = Config.from_env()
    Thread(target=serve_health, args=(cfg.health_port,), daemon=True).start()
    control_loop(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
