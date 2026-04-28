# AutoScaleAI Baseline — Build & Deploy

Vendored copy of [shaktijit-r/AutoScaleAI](https://github.com/shaktijit-r/AutoScaleAI)
with two upstream bug fixes and a production inference daemon (`runner.py`)
added on top. Used as the RL baseline in our paper revision against HPA, VPA
and Ai4k8s.

## What the upstream gives us
- `agent/` — Actor-Critic MLP policy network and a thin `AutoScaleAgent` wrapper
- `collector/metrics.py` — Prometheus client (we fixed the value extraction)
- `executor/scaler.py` — Kubernetes scaler (we fixed the early-return bug)
- `env/simulated_env.py` — RPS/replica/cpu_util simulator used to train the model
- `autoscale_agent.pt` — pretrained policy checkpoint (state_dim=3, action_dim=3)

## What we add
- `runner.py` — single-pod inference daemon: queries Prometheus, reads replica
  counts from the K8s API, runs `agent.act(state)`, patches `deployments/scale`.
  Healthcheck on `:8081/healthz`, JSON metrics on `/metrics`.
- `Dockerfile` — CPU-only PyTorch image, ~250 MB final size.
- `requirements-runner.txt` — minimal runtime deps (torch CPU, numpy, requests, kubernetes).
- `k8s-manifests/` — ServiceAccount + Role + RoleBinding (default namespace
  scope, only `deployments/scale`) and the controller Deployment.

## Build (on the CrownLabs VM)

```bash
cd ~/ai4k8s/baselines/autoscaleai
buildah bud -t localhost/autoscaleai:v1 .
buildah push localhost/autoscaleai:v1 oci-archive:/tmp/autoscaleai-v1.tar
sudo k3s ctr images import /tmp/autoscaleai-v1.tar
```

The image is named `localhost/autoscaleai:v1` and imported directly into
containerd — no external registry. The k8s manifest uses
`imagePullPolicy: Never` so the kubelet serves the local copy.

## Deploy

```bash
kubectl apply -k k8s-manifests/
kubectl wait --for=condition=Ready pod -l app=autoscaleai --timeout=60s
kubectl logs -f deploy/autoscaleai-controller
```

A successful tick logs lines like:
```
2026-04-28 22:31:00 INFO autoscaleai decision: {"deployment":"ingest","cpu_cores":0.012,"current_replicas":2,"state":[0.006,0.5,0.048],"action":"noop","target_replicas":2,"ts":1745881860}
```

## Teardown

```bash
kubectl delete -k k8s-manifests/
```

## Configuration

Everything is environment-variable driven; see the `Config.from_env()`
method in `runner.py` for the full list. Key knobs the eval harness
overrides:

| Variable | Default | What it controls |
|---|---|---|
| `TARGET_DEPLOYMENTS` | `ingest,process,analyze` | Comma-separated deployment names to scale |
| `MIN_REPLICAS` | `2` | Lower clamp |
| `MAX_REPLICAS` | `4` | Upper clamp (matches our HPA cap) |
| `TICK_SECONDS` | `10` | Decision interval |
| `CPU_REQUEST_MILLICORES` | `125` | Per-pod CPU request, used to normalise utilisation |
| `RPS_PROXY_MAX_CORES` | `2.0` | Saturation point for `state[0]` |

## Honest notes for the paper

- The bundled `autoscale_agent.pt` was trained on the upstream simulator
  (`env/simulated_env.py`), not on muBench traces. We deploy it as-is —
  matching the "trained on simulator, deployed without retraining" pattern
  KIScaler explicitly markets.
- The state vector mapping in `runner.py:build_state` is best-effort:
  the simulator's `rps_norm` is approximated by total CPU rate (cores).
  We do not have first-class HTTP request-rate metrics on the muBench
  services without instrumenting them.
- The action space is ±1 replica per tick, same as the simulator. This
  is intentionally less aggressive than HPA's instant jump to the target.
