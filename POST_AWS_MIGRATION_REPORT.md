## Post-AWS Migration Report: CrownLabs & FABRIC Integration with AI4K8s

### 1. Context and AWS Decommissioning

- **Initial environment**: AI4K8s was originally deployed on **AWS EKS**, with autoscaling experiments (HPA, predictive, scheduled) and AI-driven monitoring.
- **Issue**: Unexpected AWS costs (EC2, EKS, NAT Gateway, storage, etc.) made continued use of AWS unsuitable for a university thesis environment.
- **Actions taken**:
  - Systematically **enumerated and terminated** all chargeable resources across relevant regions.
  - Verified that the EKS cluster and all supporting infrastructure were deleted.
  - Confirmed via billing console that no new AWS usage was accruing.
  - Updated AI4K8s monitoring and autoscaling code paths so that when the EKS cluster disappeared, the UI no longer showed stale “healthy” metrics but instead:
    - Marked the cluster as **disconnected**.
    - Surfaced clear, user-facing errors and stopped generating synthetic demo data for a dead cluster.

At this point, AWS was fully decommissioned, and the project shifted focus to **non‑AWS environments**: the **AMD HPC cluster**, **FABRIC**, and **CrownLabs**.

---

### 2. AMD HPC: Hosting AI4K8s via Cloudflare Tunnel

#### 2.1 Runtime model

- **Application host**: AMD HPC login environment (`login1.hpcfund`).
- **Web app**: `ai_kubernetes_web_app.py` run under a user-level systemd service:
  - Service: `ai4k8s-web.service`
  - Bind address: `0.0.0.0:5003`
  - Environment: production Flask, with MCP tools and Groq-based AI enabled.
- **Public exposure**: **Cloudflare Tunnel** (no inbound ports, outbound-only):
  - Service: `cloudflared.service` (user systemd).
  - Tunnel configuration in Cloudflare Zero Trust:
    - Hostname: `ai4k8s.online`
    - Service: `http://127.0.0.1:5003`
  - Result: HTTPS access to the AMD-hosted AI4K8s app without violating HPC networking policy.

#### 2.2 Reliability fixes

- Implemented and enabled three user-level systemd services:
  - `ai4k8s-web.service` – Flask web app.
  - `cloudflared.service` – Cloudflare Tunnel.
  - `mcp-http.service` – MCP HTTP server (`uvicorn mcp_http_server:app --port 5002`).
- Ensured all three are **enabled** and **auto-start on login**:
  - `systemctl --user enable ai4k8s-web.service cloudflared.service mcp-http.service`
  - Verified they are consistently **ACTIVE (running)**.
- Resolved:
  - **Error 1033 (Cloudflare Tunnel)** by starting both Flask and cloudflared and confirming `https://ai4k8s.online/` returns HTTP 200.
  - **MCP connection errors** by starting `mcp-http.service` so the AI layer can load ~20 MCP tools and stop falling back to “regex-only” processing.

---

### 3. FABRIC Work (Slice JSON & Approach)

Although CrownLabs became the primary experimental cluster, there was partial work on **FABRIC**:

- Designed a **slice topology JSON** for a simple, single-node VM slice:
  - One Ubuntu VM node with 4 vCPUs, 8 GiB RAM, and 50 GiB disk.
  - One FABNetv4 network service to provide a routable IPv4 address.
- Clarified portal workflow:
  - Project selection, node/component additions (VM only, no SmartNIC/GPU for this phase), and FABNetv4 service configuration.
- Decided to postpone full FABRIC deployment until the CrownLabs path was working end-to-end with AI4K8s, to avoid splitting effort across two experimental environments.

FABRIC remains an extension point for future work (e.g., multi-site experiments or advanced networking), but the **primary post-AWS environment is CrownLabs + AMD HPC**.

---

### 4. CrownLabs Environment: VM and k3s Cluster

#### 4.1 CrownLabs VM

- **Access model**:
  - Bastion: `ssh.crownlabs.polito.it` with user `bastion`.
  - Internal VM IP: `10.108.203.201`, user `crownlabs`.
  - SSH key: `~/.ssh/crownlabs_key` (also installed on AMD HPC for tunneling).
- **Direct command-line access**:
  - From Mac:
    - `ssh crownlabs-bastion` (via `~/.ssh/config`) to reach bastion.
    - `ssh -J bastion@ssh.crownlabs.polito.it crownlabs@10.108.203.201` to reach the VM.

#### 4.2 k3s installation

On the CrownLabs VM (`crownlabs@xubuntu-base`):

- Installed k3s with a single control-plane node:

  ```bash
  curl -sfL https://get.k3s.io | sh -s - --write-kubeconfig-mode 644
  ```

- Verified:
  - `sudo k3s kubectl get nodes -o wide` → 1 node `xubuntu-base` in `Ready` state.
  - System pods (coredns, traefik, metrics-server, etc.) all Running.
- Kubeconfig:
  - Located at `/etc/rancher/k3s/k3s.yaml`.
  - Initially had `server: https://127.0.0.1:6443` (suitable for in-VM usage).

---

### 5. Integrating CrownLabs k3s with AI4K8s on AMD HPC

#### 5.1 Kubeconfig transfer

From Mac → CrownLabs VM → AMD HPC:

- Extracted `/etc/rancher/k3s/k3s.yaml` from the VM.
- Stored on AMD HPC as `~/crownlabs-k3s.yaml`.
- For use via an SSH tunnel, changed the `server:` line to:

```yaml
server: https://127.0.0.1:16443
```

This allows AMD HPC to connect to the k3s API over a local port forward.

#### 5.2 SSH tunnel from AMD HPC to CrownLabs VM

On AMD HPC:

- Installed `~/.ssh/crownlabs_key`.
- Configured SSH to reach bastion and VM:

```sshconfig
Host crownlabs-bastion
    HostName ssh.crownlabs.polito.it
    User bastion
    IdentityFile ~/.ssh/crownlabs_key
    IdentitiesOnly yes

Host crownlabs
    HostName 10.108.203.201
    User crownlabs
    ProxyJump crownlabs-bastion
    IdentityFile ~/.ssh/crownlabs_key
    IdentitiesOnly yes
```

- Established a **local port forward** inside a `screen` session:

```bash
screen -dmS crownlabs-tunnel \
  ssh -i ~/.ssh/crownlabs_key \
      -o IdentitiesOnly=yes \
      -J bastion@ssh.crownlabs.polito.it \
      -L 16443:127.0.0.1:6443 \
      -N crownlabs@10.108.203.201
```

- Verified:
  - `ss -ltnp` showed `127.0.0.1:16443` listening.
  - `KUBECONFIG=~/crownlabs-k3s.yaml kubectl get nodes -o wide` returned the CrownLabs k3s node.

#### 5.3 Registering the CrownLabs cluster in AI4K8s

Within the AI4K8s SQLite DB on AMD HPC (`~/ai4k8s/instance/ai4k8s.db`), programmatically:

- Located user `pedramnj`.
- Created a `Server` entry:
  - `name`: **CrownLabs k3s Cluster**
  - `server_type`: `kubernetes`
  - `connection_string`: `crownlabs-vm:10.108.203.201`
  - `kubeconfig`: contents of `~/crownlabs-k3s.yaml`
  - `status`: `active`

Result: the cluster appears in the AI4K8s dashboard under user `pedramnj` and is selectable for chat, monitoring, and autoscaling.

---

### 6. Fixing Monitoring & Connectivity States

#### 6.1 Dashboard status checks

- `ai_kubernetes_web_app.py`:
  - Updated `/dashboard` to run a **quick `kubectl cluster-info`** per server using a temp kubeconfig file.
  - If the command fails with common network errors (e.g., `no such host`, `connection refused`, `timeout`, `name resolution`), the server:
    - `status` → `error`
    - `connection_error` recorded in DB.
  - For the CrownLabs cluster, with the tunnel and kubeconfig correctly configured, `cluster-info` succeeds and the status becomes **active**.

#### 6.2 AI monitoring integration

- `AIMonitoringIntegration` and `KubernetesMetricsCollector` were updated to:
  - Prefer **real metrics** when available.
  - Only declare **“Cluster Disconnected”** when actual connection errors occur.
  - Clear `last_analysis` cache on disconnection to avoid stale metrics.
- A key bug was fixed:
  - A helper `_patch_kubeconfig_for_container` used to **rewrite `server: https://127.0.0.1:...` to `host.docker.internal`**, which is correct when AI4K8s runs inside Docker, but **wrong on AMD HPC**.
  - This caused requests to go to `host.docker.internal:16443`, which cannot be resolved from HPC, producing disconnect states even though the tunnel and k3s were healthy.
  - The function now:
    - Detects container vs host (checks `/.dockerenv` / `DOCKER_CONTAINER`).
    - **Skips rewriting** the server URL on bare-metal/HPC (keeps `127.0.0.1`).
- After this fix:
  - `get_current_analysis()` for the CrownLabs cluster shows live metrics:
    - CPU ≈ 2–4%
    - Memory ≈ 19%
    - Pod count (including `nginx-test` deployment).
  - The monitoring UI transitions from **“Cluster Disconnected”** to an active health score, trends, and forecasts.

---

### 7. Workload Deployment for Testing (CrownLabs k3s)

To exercise monitoring and autoscaling logic:

- Deployed a test **nginx** application into namespace `ai4k8s-test` on the CrownLabs cluster:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-test
  namespace: ai4k8s-test
spec:
  replicas: 5
  selector:
    matchLabels:
      app: nginx-test
  template:
    metadata:
      labels:
        app: nginx-test
    spec:
      containers:
      - name: nginx
        image: nginx:1.25-alpine
        ports:
        - containerPort: 80
        resources:
          requests:
            cpu: \"50m\"
            memory: \"64Mi\"
          limits:
            cpu: \"200m\"
            memory: \"256Mi\"
---
apiVersion: v1
kind: Service
metadata:
  name: nginx-test
  namespace: ai4k8s-test
spec:
  selector:
    app: nginx-test
  ports:
  - port: 80
    targetPort: 80
    protocol: TCP
    name: http
```

- Verified:
  - `kubectl get pods -n ai4k8s-test -o wide` → 5 Running pods on `xubuntu-base`.
- These pods:
  - Contribute to `pod_count` and resource usage seen by the monitoring stack.
  - Provide a non-trivial, but still safe, workload for future autoscaling experiments.

---

### 8. UI/UX Adjustments Related to the New Environment

Several UI polish changes were made to better reflect the new, multi-cluster, non-AWS setup:

- **Dashboard (`dashboard.html`)**
  - Unified primary actions per cluster:
    - `View Details` – primary.
    - `AI Monitoring` – primary (same styling as View Details and Autoscaling).
    - `Autoscaling` – primary.
    - `AI Chat` – secondary.
- **Cluster detail page (`server_detail.html`)**
  - Quick Actions:
    - **Start AI Chat** – primary.
    - **AI Monitoring** – secondary.
    - **Autoscaling** – added as a secondary quick action.
    - **Test Connection**, **Delete Cluster** – additional management actions.
  - Removed a redundant top-right AI Chat button in the header (to avoid duplication with Quick Actions).
- **Home page (`index.html`)**
  - Logged-in CTA changed from **“Go to Dashboard”** (self-link) to:
    - **“Go to Clusters”**, linking directly to the `/dashboard` clusters view.

These UI changes align the navigation and emphasize that the key entry point after login is the **Clusters** dashboard, regardless of whether the underlying cluster is AWS, CrownLabs, or another environment.

---

### 9. Current Status and Next Steps

**Current status:**

- AWS infrastructure: fully decommissioned, no active billing.
- AMD HPC:
  - Hosting AI4K8s web app, MCP HTTP server, and Cloudflare Tunnel.
  - All three services are enabled and running under user systemd.
- CrownLabs:
  - Single-node k3s cluster (Ubuntu 20.04 VM) with working metrics-server.
  - SSH tunneling from AMD HPC provides secure API access.
  - AI4K8s sees the CrownLabs cluster as “CrownLabs k3s Cluster” and can:
    - Monitor health and metrics.
    - Run AI chat against it (via kubectl/MCP).
    - Prepare for autoscaling experiments.
- FABRIC:
  - Topology and token concepts explored; slice JSON draft prepared.
  - Full deployment deferred until after stabilizing the CrownLabs setup.

**Potential next steps:**

1. **Autoscaling experiments on CrownLabs k3s**:
   - Create HPAs for the `nginx-test` deployment (e.g., CPU-based, custom metrics).
   - Exercise AI4K8s autoscaling reports and controls against this environment.
2. **End-to-end benchmark and anomaly scenarios**:
   - Generate load on `nginx-test` to drive observable CPU/memory changes.
   - Validate AI monitoring (forecasts, anomaly detection, and RAG recommendations) under real load.
3. **Optional FABRIC follow-up**:
   - Provision a FABRIC slice with similar k3s setup.
   - Compare behavior (latency, resource characteristics) vs CrownLabs, using the same AI4K8s frontend on AMD HPC.

This report captures the transition from AWS to **HPC + CrownLabs** (and partially FABRIC), ensuring the AI4K8s thesis platform remains **fully operational, cost-controlled, and suitable for further experimental work**.


