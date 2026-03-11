# muBench — Heavy Workload Integration for AutoSage

Deploys a 3-service CPU-heavy microservice chain on the CrownLabs k3s cluster
so that AutoSage (with Qwen3.5-2B via Ollama) can monitor and make real scaling
decisions against a realistic workload.

## Topology

```
[client]
   │
   ▼  HTTP /api/v1
┌─────────┐  image_processing  ┌─────────┐  matrix_compute  ┌─────────┐  video_frames
│ ingest  │ ─────────────────► │ process │ ───────────────► │ analyze │
│ 500m CPU│                    │ 500m CPU│                   │ 300m CPU│ ← bottleneck
└─────────┘                    └─────────┘                   └─────────┘
```

Each service runs the muBench `msvcbench/microservice:latest` image executing a
custom numpy-based heavy workload:

| Service  | Function          | What it simulates                        | CPU limit |
|----------|-------------------|------------------------------------------|-----------|
| `ingest` | image_processing  | Decode → blur convolution → hist-eq      | 500m      |
| `process`| matrix_compute    | YOLO-like 6-layer neural net forward pass| 500m      |
| `analyze`| video_frames      | Motion detect → optical flow → FFT       | 300m ← tight |

`analyze` has the tightest CPU limit and the most sustained per-request work,
so it saturates first under load — Qwen3.5-2B should recommend scaling it before
the other two.

## Prerequisites

- `kubectl` pointing at the k3s cluster
- `metrics-server` running (required for HPA): `kubectl top pods`
- AutoSage running (`ai4k8s-web.service`) with Ollama/Qwen3.5-2B active

## Deploy

```bash
# 1. Apply ConfigMaps (workmodel + functions)
kubectl apply -f mubench/k8s-manifests/configmaps.yaml

# 2. Deploy the 3 services
kubectl apply -f mubench/k8s-manifests/deployments.yaml

# 3. Expose via ClusterIP Services
kubectl apply -f mubench/k8s-manifests/services.yaml

# 4. Create HPAs (70% CPU threshold, 2–8 replicas)
kubectl apply -f mubench/k8s-manifests/hpa.yaml

# 5. Wait for pods to be ready (initContainer installs numpy first)
kubectl wait --for=condition=ready pod -l app.kubernetes.io/part-of=mubench \
  --timeout=120s -n default
```

## Generate Load

```bash
# One-shot phased load generator (warmup → sustained → burst → cooldown, ~10min)
kubectl apply -f mubench/k8s-manifests/loadgen.yaml

# Watch it run
kubectl logs -f mubench-loadgen

# Or quick manual test
kubectl run curl-test --rm -i --restart=Never --image=curlimages/curl \
  -- curl -s http://ingest:8080/api/v1
```

## Watch AutoSage React

```bash
# CPU usage per pod (update every 5s)
watch -n5 kubectl top pods -l app.kubernetes.io/part-of=mubench

# HPA status — replica count changes here
watch -n10 kubectl get hpa

# AutoSage LLM advisor log — shows Qwen3.5 decisions
ssh crownlabs "sudo journalctl -u ai4k8s-web.service -f" | grep -E "GPT OSS|scale|replicas|qwen"

# AutoSage monitoring dashboard
open http://ai4k8s.online/monitoring/default
```

## Enable AutoSage Predictive Autoscaling

After load starts, register the muBench deployments with AutoSage's predictive loop:

```bash
for svc in ingest process analyze; do
  curl -s -X POST http://localhost:5003/autoscaling/predictive/enable \
    -H "Content-Type: application/json" \
    -d "{\"deployment\": \"$svc\", \"namespace\": \"default\", \"min_replicas\": 2, \"max_replicas\": 8}"
done
```

AutoSage runs every 5 minutes. After ~10 minutes of sustained load you should see:
- Qwen3.5-2B recommending `scale_up` on `analyze` first
- MCDA optimizer confirming or overriding the LLM decision
- Replica counts increasing in `kubectl get hpa`

## Tuning Workload Intensity

Edit `workmodel.json` or the ConfigMap to change per-call work:

| Parameter | Default | Heavy |
|-----------|---------|-------|
| `image_processing.iterations` | 6 | 3 (light) |
| `matrix_compute.num_layers` | 6 | 4 (light) |
| `matrix_compute.hidden_size` | 512 | 256 (light) |
| `matrix_compute.input_size` | 1024 | 512 (light) |
| `video_frames.num_frames` | 32 | 16 (light) |
| `video_frames.frame_width` | 640 | 320 (light) |

After editing `workmodel.json`, update the ConfigMap and restart pods:
```bash
kubectl create configmap mubench-workmodel \
  --from-file=workmodel.json=mubench/workmodel.json \
  -n default --dry-run=client -o yaml | kubectl apply -f -

kubectl rollout restart deployment/ingest deployment/process deployment/analyze
```

## Cleanup

```bash
kubectl delete -f mubench/k8s-manifests/
# or individually:
kubectl delete deployment ingest process analyze -n default
kubectl delete service    ingest process analyze -n default
kubectl delete hpa        ingest process analyze -n default
kubectl delete configmap  mubench-workmodel mubench-functions -n default
kubectl delete pod        mubench-loadgen -n default --ignore-not-found
```
