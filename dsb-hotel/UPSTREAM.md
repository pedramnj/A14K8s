# DeathStarBench Hotel Reservation — Upstream Attribution

## Source

- **Repository**: https://github.com/delimitrou/DeathStarBench
- **Path**: `hotelReservation/`
- **Commit SHA**: `6ecb09706140f8730b5385c08f1386c654c3c526`
- **Branch**: `master`
- **Licence**: Apache 2.0

This directory is a minimal, resource-tuned subset of the upstream DSB
Hotel Reservation benchmark, used as the Phase R evaluation substrate
for AutoSage. It is not a fork or a hard copy of the upstream code —
it contains only the Kubernetes manifests and the wrk2 load script,
adjusted to fit a single 4 vCPU / 16 GiB VM. The DSB Go binaries and
gRPC service mesh come from the upstream container image
`deathstarbench/hotel-reservation:latest` unchanged.

## Files included

### `kubernetes/`
- `deployments.yaml` — 11 Deployments, consolidated from upstream
  `kubernetes/{consul,frontend,search,geo,rate,profile}/*.yaml` and
  the per-service Mongo/memcached deployments.
- `services.yaml` — 11 ClusterIP Services, same source paths.
- `hpa.yaml` — 5 HorizontalPodAutoscalers for the app services only
  (upstream ships no HPAs).

### `wrk2/`
- `mixed-workload_type_1.lua` — copied from upstream
  `wrk2/scripts/hotel-reservation/mixed-workload_type_1.lua` with one
  minimal patch: the top-level `local url = "http://localhost:5000"`
  is set to `""` (empty string). Rationale documented inline in the
  file; short version: upstream prepends `http://localhost:5000` to
  every path passed to `wrk.format(method, path)`, producing HTTP
  absolute-form request URIs. That's fine when wrk2 runs on the same
  host as the frontend; on our k8s cluster the frontend is a
  ClusterIP service, so an absolute URI with `localhost:5000` in the
  Host header misroutes. Empty prefix means `wrk.format` composes the
  request relative to the target URL passed on the wrk2 command line
  (`http://frontend:5000`), which is what we want.
- `heavy-search.lua` — new file (not upstream). Skews the request mix
  to 100 % `/hotels` searches with widened lat/lon windows so the
  fan-out through search → geo/rate/profile saturates. Used by the
  Phase Q.5-style shift oscillation to simulate an AWARE-style input
  shift without modifying DSB app code.
- `light-search.lua` — new file (Phase R.5, not upstream). Companion
  to heavy-search.lua with a narrow ±0.05° lat/lon window (~5 km
  radius). memcached-geo caches every result after one iteration so
  per-request cost drops to a memcached lookup. Used in oscillating
  `DSB_SHIFT_PHASES=heavy-search.lua:60,light-search.lua:60,…`
  schedules that break VPA's histogram-based recommender
  (VPA averages the two regimes into a mid-value that fits neither).

## Modifications applied to the upstream manifests

The 11 Deployments and 11 Services in `kubernetes/` differ from
upstream in the following ways (documented here so reviewers can
diff back to the source):

### Common to every object
- Added `app.kubernetes.io/part-of: dsb-hotel` label for
  `kubectl -l app.kubernetes.io/part-of=dsb-hotel` selection.
- Added `managed-by: ai4k8s` label on Deployments.
- Removed the auto-generated `kompose.cmd` / `kompose.version` /
  `sidecar.istio.io/*` annotations (noise from the Docker-Compose ↔
  Kubernetes conversion tool; nothing consumes them without Istio).
- Removed the `creationTimestamp: null` and
  `strategy: {}` / `status: {}` boilerplate.
- Namespace explicitly set to `default` (upstream inherits).

### Per-Deployment resource envelope
Upstream defaults are `cpu.requests: 100m`, `cpu.limits: 1000m` on
every container. That envelope assumes a beefy multi-node cluster.
On our 4-vCPU CrownLabs VM the total upstream request budget
(11 × 100 m = 1.1 vCPU) leaves enough for Kubelet and Qwen, but the
1 vCPU per-pod limit lets a single service burst-consume the whole
node and mask cross-service back-pressure. We trim as follows:

| Pod                | requests CPU | limits CPU | requests memory | limits memory |
|--------------------|:------------:|:----------:|:---------------:|:-------------:|
| consul             | 100 m        | 500 m      | 128 Mi          | 256 Mi        |
| mongodb-{geo,rate,profile} (each) | 100 m | 500 m | 256 Mi | 512 Mi        |
| memcached-{rate,profile} (each)   | 50 m  | 200 m | 64 Mi  | 192 Mi        |
| frontend           | 100 m        | 500 m      | 128 Mi          | 256 Mi        |
| search             | 150 m        | 400 m      | 192 Mi          | 384 Mi        |
| geo                | 150 m        | 300 m      | 128 Mi          | 256 Mi        |
| rate               | 150 m        | 300 m      | 128 Mi          | 256 Mi        |
| profile            | 150 m        | 300 m      | 256 Mi          | 512 Mi        |
| **total requests** | **1300 m**   | –          | **~2.7 GiB**    | –             |
| **total limits**   | –            | **4600 m** | –               | **~4.5 GiB**  |

The `.limits` sum (4.6 vCPU) intentionally exceeds the 4-vCPU node
capacity — CPU limits are throttling caps, not reservations, and this
lets the harness observe cross-service queueing effects when
saturation lands.

Memory `.requests` were added for every pod (upstream sets none,
which puts Kubernetes' default of 0 on the pod and can cause
OOM-driven evictions to appear as flakes rather than as real signal).

### Autoscaling annotations (5 app services only)
Added on `frontend`, `search`, `geo`, `rate`, `profile`:
```yaml
annotations:
  autosage.ai4k8s/monitor: "true"
```
The AutoSage predictive autoscaler picks up any Deployment carrying
this annotation. Backing stores (Mongo / memcached / Consul) are
deliberately **not** annotated — they are fixed infrastructure, not
autoscaled by this benchmark.

### Persistent volumes → emptyDir
Upstream ships a `PersistentVolume` + `PersistentVolumeClaim` pair
for each Mongo (`geo-persistent-volume.yaml`, `geo-pvc.yaml`, etc.)
using `hostPath: /data/volumes/geo-pv` — which requires a specific
host directory to pre-exist on the node. We swap those for
`emptyDir: {}` volumes. Trial data does not need to survive pod
restart for the benchmark (the wrk2 workload re-seeds all
collections at the start of each trial via `./setup.py`). The
`emptyDir` volume lives for the pod lifetime, which is what we want.

### Image pinning
- `hashicorp/consul:latest` → `hashicorp/consul:1.15`
- `mongo:4.4.6` → `mongo:4.4.6` (already versioned upstream, kept)
- `memcached` (no tag) → `memcached:1.6-alpine`
- `deathstarbench/hotel-reservation:latest` → **still `:latest`**
  (upstream publishes only a rolling tag; Phase W will pin to a
  resolved `@sha256:<digest>`).

### Consul port trim
Upstream Service exposes 8300 / 8400 / 8500 / 8600 (UDP-53). We keep
all four — DSB relies on both the gRPC-style port (8300) and the
HTTP port (8500). Nothing dropped.

## Nothing else is changed

- Service names match upstream (`frontend`, `search`, `geo`, `rate`,
  `profile`, `consul`, `mongodb-geo`, `mongodb-rate`,
  `mongodb-profile`, `memcached-rate`, `memcached-profile`).
- Container ports match upstream (5000 / 8082 / 8083 / 8084 / 8081
  for app services; 27017 for Mongo; 11211 for memcached; 8300 /
  8400 / 8500 / 8600 for Consul).
- Container entrypoints are absolute paths (`/go/bin/frontend`,
  `/go/bin/search`, `/go/bin/geo`, `/go/bin/rate`, `/go/bin/profile`)
  instead of upstream's `./frontend` etc. Upstream assumes WORKDIR is
  set to where the binaries live; the current
  `deathstarbench/hotel-reservation:latest` image (built 2024-06-27)
  keeps WORKDIR at `/workspace` (which contains the Go source, not the
  binaries) and installs the binaries into `/go/bin/`. Using the
  absolute path removes the WORKDIR dependency.
- The DSB `config.json` referenced by each Go binary is baked into
  the upstream image at build time and is not overridden.
- The upstream wrk2 workload script is used unchanged for the
  steady-state (v35/v36) versions.

## Deployment

On the CrownLabs k3s node:
```bash
kubectl apply -f dsb-hotel/kubernetes/
```
All 11 pods should reach `Ready` within 3 minutes. Consul must be
`Running` before the 5 app services register — the harness in
`mubench/run_comparison_eval.py` waits on this before starting wrk2.

Verification:
```bash
kubectl get pods -l app.kubernetes.io/part-of=dsb-hotel
# Should show 11 pods, all Running

kubectl port-forward svc/frontend 5000:5000 &
curl "http://localhost:5000/hotels?inDate=2015-04-09&outDate=2015-04-10&lat=37.7&lon=-122.4"
# Should return a JSON hotel list
```

Once verified, feed the substrate to the eval harness:
```bash
python3 mubench/run_comparison_eval.py --workload dsb_hotel
```
