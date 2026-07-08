<div align="center">

# AutoSage

**An agentic AI framework for predictive monitoring and multi-criteria autoscaling in Kubernetes clusters.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Kubernetes](https://img.shields.io/badge/kubernetes-k3s%20%7C%20eks%20%7C%20gke-326ce5?logo=kubernetes&logoColor=white)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)

</div>

---

## What it does

AutoSage decides when and how to scale a Kubernetes deployment by combining
three cooperating layers:

1. A **deterministic enforcement layer** that blocks physically-impossible or
   operationally-unsafe scaling actions before any model is consulted.
2. A **language-model reasoning layer** that reads live cluster metrics, a
   short-horizon forecast, and workload metadata, and emits a plain-English
   scaling recommendation.
3. A **TOPSIS multi-criteria validator** that scores every candidate action
   across cost, performance, stability, forecast alignment, and response
   time, and confirms or overrides the LLM's pick.

A continuous daemon runs the three-layer pipeline every 30 s for the full
deployment lifetime, matching the control cadence of the native Kubernetes
Vertical Pod Autoscaler.

Every scaling decision is logged with a full, human-readable trace ---
enforcement verdict, LLM reasoning, and per-criterion TOPSIS scores ---
making the system's behaviour fully auditable.

## Highlights

- **Workload-class-correct decisions**: the LLM picks horizontal scaling on
  stateless workloads and vertical scaling on stateful ones on every trial
  across a 99-decision evaluation --- matching Kubernetes convention
  without a single override from the TOPSIS validator.
- **28–47% SLA-normalised cost reduction** versus native HPA across both
  workload classes, robust to input amplitude, rollout strategy, and
  workload oscillation.
- **Local-first LLM cascade**: a 2.7 GB quantised `qwen3.5:2b` model runs
  entirely on-premises via Ollama. A resilient fallback chain
  (local → Groq cloud → deterministic regex) keeps end-to-end decision
  latency under 2 s under CPU saturation.
- **Model Context Protocol server**: every `kubectl` operation flows
  through a localhost-only FastAPI MCP tool layer, so cluster access is
  isolated behind a structured API.

## Architecture

```
Browser ── Flask + SocketIO (:5003)
               ├── AIProcessor            (NLP + tool use)
               ├── AutoscalingIntegration ── LLMAdvisor
               │                         └── MCDAOptimizer  (TOPSIS)
               └── AIMonitoringIntegration
                       ├── K8sMetricsCollector
                       ├── PredictiveMonitoring  (ARIMA + bootstrap UQ)
                       └── KubernetesRAG         (BM25 over K8s KB)
                                 │
                                 ▼
                          MCP server (:5002)
                                 │
                                 ▼
                            k3s / k8s cluster
```

## Repository layout

```
.
├── ai_kubernetes_web_app.py      # Flask + SocketIO entry point
├── autoscaling_integration.py    # Top-level autoscaling orchestrator
├── predictive_autoscaler.py      # LLM + MCDA + enforcement pipeline
├── llm_autoscaling_advisor.py    # LLM cascade (Qwen → Groq → regex)
├── mcda_optimizer.py             # TOPSIS validator
├── autoscaling_engine.py         # HPA scaling actions
├── vpa_engine.py                 # VPA sizing actions
├── scheduled_autoscaler.py       # Cron-style scheduled actions
├── predictive_monitoring.py      # ARIMA + bootstrap prediction intervals
├── uncertainty_quantifier.py     # Platt-scaling + calibrated anomaly
├── k8s_metrics_collector.py      # Multi-source metrics pipeline
├── mcp_http_server.py            # FastAPI MCP tool server
├── mcp_client.py                 # MCP client used by the web app
├── kubernetes_mcp_server.py      # MCP tool implementations
├── kubernetes_rag.py             # BM25 keyword retrieval over the KB
├── ai_processor.py               # Natural-language command router
├── ai_monitoring_integration.py  # Monitoring loop + ring buffer
├── config.py                     # Runtime config
├── logging_utils.py              # Structured logging
├── simple_kubectl_executor.py    # Fallback kubectl wrapper
├── scaling_decision.py           # Decision dataclass + serialisation
├── requirements.txt
│
├── templates/                    # Jinja2 templates
├── static/                       # Front-end assets
├── kb_kubernetes/                # BM25 knowledge base
├── client/                       # Standalone MCP client SDK
├── baselines/                    # RL baseline (AutoScaleAI) for comparison
├── mubench/                      # Microservice benchmark harness
└── research/                     # Offline analysis + evaluation scripts
```

## Quick start

### Requirements

- Python 3.11+
- A Kubernetes cluster with the metrics-server enabled (`k3s` is the
  reference target)
- [Ollama](https://ollama.com) with `qwen3.5:2b` pulled locally, or a
  Groq API key for cloud fallback

### Local setup

```bash
git clone https://github.com/pedramnj/A14K8s.git autosage && cd autosage
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Copy the environment template and fill in the credentials you have
cp .env.example .env

# Optional: pull the local model
ollama pull qwen3.5:2b

# Start the web app
python ai_kubernetes_web_app.py
```

The web UI is served at `http://localhost:5003`.

### Configuration

All runtime knobs are set via environment variables (see `.env.example`).
The key ones:

| Variable | Purpose | Default |
|---|---|---|
| `GPT_OSS_API_BASE` | Ollama endpoint. Point at `http://disabled:1/v1` to force the Groq fallback. | `http://localhost:11434/v1` |
| `GPT_OSS_MODEL` | Local model tag pulled into Ollama | `qwen3.5:2b` |
| `GROQ_API_KEY` | Cloud-fallback API key | (required if local is unreachable) |
| `AUTOSAGE_CONTINUOUS_DAEMON_ENABLED` | Enable the continuous daemon | `0` |
| `AUTOSAGE_DAEMON_TICK_S` | Daemon tick period in seconds | `30` |
| `AUTOSAGE_VPA_REQUEST_MULTIPLIER` | Multiplier applied to LLM CPU/memory picks before actuation | `1.0` |
| `AUTOSAGE_VPA_SET_LIMITS` | Also raise container `limits` when actuating VPA targets | `0` |

## Reproducibility

The evaluation harness lives in `mubench/`; the RL baseline in `baselines/`;
offline analysis and plotting scripts in `research/`. Each has its own
README with a `python -m` invocation and expected outputs.

## License

MIT --- see [`LICENSE`](LICENSE).
