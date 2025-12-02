## AI4K8s ➜ AMD HPC Cluster Transfer Report

### Cluster Overview
- **Login host**: `login1.hpcfund`
- **User home**: `/home1/pedramnj`
- **Workspace**: `/work1/alessiosacco/pedramnj/ai4k8s`
- **Scheduler**: SLURM
- **Python**: `/usr/bin/python3` (3.9.21)

### What We Transferred (clean rsync)
- **Top-level app**: `ai_kubernetes_web_app.py`, `ai_processor.py`, `kubernetes_mcp_server.py`, `mcp_client.py`, `simple_kubectl_executor.py`, `predictive_monitoring.py`
- **Web UI**: `templates/`, `static/`
- **Client**: `client/` (excluding local `.venv`)
- **Data/DB**: `instance/ai4k8s.db`
- **Docs/Configs**: `README.md`, `requirements.txt`, `docker-compose.yml`, `mcp-bridge-deployment.yaml`, `web-app-iframe-solution.yaml`, reports
- **Integrations**: `netpress-integration/`

Command used:

```bash
rsync -avz --progress --partial --inplace \
  --exclude='client/.venv' --exclude='__pycache__' --exclude='.git' \
  --exclude='backup' --exclude='thesis_reports/figures' --exclude='*.pyc' \
  --exclude='.DS_Store' . pedramnj@hpcfund.amd.com:/work1/alessiosacco/pedramnj/ai4k8s/
```

### Python Environment
- Created venv: `venv` (Python 3.9)
- Upgraded `pip` to 25.2
- Installed core deps: Flask, SQLAlchemy, Werkzeug, requests, python-dotenv, groq, anthropic, paramiko
- Installed AI/ML deps: numpy, pandas, scikit-learn, kubernetes

Note: `mcp` (PyPI) is not available for Python 3.9; MCP usage is disabled at import time. The app handles missing MCP server gracefully and continues with AI features.

### Tests Executed
- Syntax check: `python -m py_compile` on key modules — ✅
- Imports smoke test: Flask/Requests/Anthropic and app modules — ✅
- Functional script: `test_kubectl_formatting.py` (generated HTML outputs) — ✅

### SLURM Usage (recap)
- **Submit**: `sbatch job.sh`
- **Monitor**: `squeue -u $USER`, `sacct -j <jobid>`
- **Partitions**: `devel` for short interactive tests; production partitions for longer runs
- **Logs**: `%j` in SBATCH expands to job ID for `.out`/`.err`

### Validation Summary
- **Transfer integrity**: ✅ Complete (fresh rsync after cleanup)
- **Environment ready**: ✅ Virtualenv with core deps installed
- **App smoke tests**: ✅ Passed (MCP skipped)
- **Action items**: ⚠️ MCP requires Python ≥3.10 or an alternative distribution approach

### Recommended Next Steps
- If MCP server access is required on HPC: use Python 3.10+ (module or Conda) and install a compatible `mcp` implementation, or vendor the needed client/server bits.
- Optionally create a SLURM job wrapper to launch the Flask app on a compute node and port-forward via SSH for UI access.

## How I Brought the Website Online on the HPC 

Here is exactly what I did, in order, to get `ai4k8s.online` serving the Flask app from the AMD HPC cluster without using the VPS.

1) Clean transfer and app runtime on HPC
- I cleaned the target directory and re-synced the full project from my local machine:
  ```bash
  rsync -avz --progress --partial --inplace \
    --exclude='client/.venv' --exclude='__pycache__' --exclude='.git' \
    --exclude='backup' --exclude='thesis_reports/figures' --exclude='*.pyc' \
    --exclude='.DS_Store' . pedramnj@hpcfund.amd.com:/work1/alessiosacco/pedramnj/ai4k8s/
  ```
- On the HPC, I created and activated a virtualenv (Python 3.9), upgraded pip, and installed app deps (skipping MCP for now due to Python 3.9):
  ```bash
  cd /work1/alessiosacco/pedramnj/ai4k8s
  python3 -m venv venv && source venv/bin/activate
  pip install --upgrade pip
  pip install Flask==2.3.3 Flask-SQLAlchemy==3.0.5 Werkzeug==2.3.7 \
              requests==2.31.0 python-dotenv==1.0.0 groq>=0.4.0 \
              anthropic>=0.68.0 paramiko>=3.0.0 \
              numpy>=1.24.0 pandas>=2.0.0 scikit-learn>=1.3.0 kubernetes>=28.1.0
  ```
- I installed gunicorn and started the Flask app bound to localhost:
  ```bash
  pip install gunicorn
  nohup bash -lc 'source venv/bin/activate && \
    exec gunicorn -w 2 -b 127.0.0.1:5003 ai_kubernetes_web_app:app' \
    >/tmp/ai4k8s_gunicorn.out 2>&1 &
  ```
- I verified it was listening and serving locally:
  ```bash
  ss -ltnp | grep 5003
  curl -s http://127.0.0.1:5003 | head -5
  ```

2) Remove VPS dependency and expose via Cloudflare Tunnel
- Because running a public web server on HPC login nodes isn’t permitted, I used an outbound-only Cloudflare Tunnel (no root required).
- I downloaded a user-space `cloudflared` binary on the HPC:
  ```bash
  mkdir -p /work1/alessiosacco/pedramnj/ai4k8s/bin
  cd /work1/alessiosacco/pedramnj/ai4k8s/bin
  curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
  chmod +x cloudflared
  ./cloudflared --version
  ```
- Named tunnel with token: the dashboard provided me a tunnel run token. I launched the connector (no sudo, no service install):
  ```bash
  nohup ./cloudflared tunnel run --token '<REDACTED_TUNNEL_TOKEN>' \
    >/tmp/cloudflared_service.log 2>&1 &
  tail -n 20 /tmp/cloudflared_service.log
  ```
  This produced a working connector with tunnel ID similar to `e3b5...d276`.

3) Cloudflare Zero Trust configuration (in dashboard)
- In Zero Trust → Networks → Tunnels → selected the `ai4k8s` tunnel created via token.
- I removed any old DNS A record for `ai4k8s.online` (to avoid conflict).
- In the tunnel configuration, I added a Published Application Route:
  - Hostname: `ai4k8s.online` (root, no subdomain)
  - Service Type: HTTP
  - URL: `http://127.0.0.1:5003` (important: HTTP, not HTTPS)
  Cloudflare created/managed the CNAME automatically.

4) Fixes and gotchas I handled along the way
- Initial attempts failed due to origin protocol mismatch (`https://127.0.0.1:5003`). I corrected the service to `http://127.0.0.1:5003`.
- QUIC connection warnings appeared in logs; these were transient. The tunnel still served traffic over Cloudflare’s edge.
- The CLI `tunnel route dns` required an origin certificate (`cert.pem`). I avoided that by running the tunnel with a dashboard-issued token instead of relying on `origincert`.
- DNS conflict error (“A, AAAA, or CNAME exists”) was resolved by deleting the old A record for the apex and letting the tunnel own the hostname via a CNAME.

5) Validation
- From the HPC, I confirmed the domain returned 200 and served the app over HTTPS:
  ```bash
  curl -I https://ai4k8s.online | sed -n '1,5p'
  # Expected: HTTP/2 200 with server: cloudflare
  ```
- I double-checked the tunnel logs to ensure the ingress pointed to HTTP on localhost and the connector stayed registered.

6) Current runtime model
- App: `gunicorn` on `127.0.0.1:5003` inside the user venv.
- Public exposure: Cloudflare Tunnel connector (user-space) with dashboard-managed DNS and TLS.
- No public ports or root privileges required on the HPC; all traffic is egress-only from the HPC to Cloudflare.

7) Follow-ups
- Make the tunnel more resilient on the login node (e.g., a user-level supervisor or a SLURM job wrapper with restart) without violating HPC policy.
- Proceed to set up a Python 3.11 environment (via Conda or module) to enable the MCP server/client stack and integrate it back into the app.


