# Test Report — AutoSage + Qwen3.5-2B on 16 GiB CrownLabs Instance
**Date:** March 16, 2026
**Instance:** instance-866fd, Node: worker-7-rs, IP: 10.96.152.30
**RAM:** 14 GiB | **Disk:** 42 GB | **CPU:** 4 vCPUs

---

## Context

Following the RAM constraint problems documented in the previous report (7.2 GiB instance where Qwen was always paged out during load), the CrownLabs administrators provided a new 16 GiB instance. This report covers the full setup and test results on that instance.

---

## Setup

I installed k3s, Ollama, and the full AutoSage stack from scratch on the new VM. The setup steps were:

1. k3s installed via the official install script (v1.34.5+k3s1, single-node cluster ready in under 60 seconds)
2. Ollama installed and unused GPU libraries removed immediately (freed 4.3 GB of disk, same as before)
3. Project cloned from GitHub (`update` branch, which contains the muBench integration)
4. Python venv created and dependencies installed — this revealed that `openai` was missing from `requirements.txt` despite being required for the GPT OSS / Qwen inference path; added and committed
5. `.env` written with Groq API key and Qwen config (`GPT_OSS_MODEL=qwen3.5:2b`, Q8_0 quantisation)
6. `ai4k8s-web.service` and `mcp-http.service` systemd units created and started
7. muBench manifests applied (`kubectl apply -f mubench/k8s-manifests/`)
8. All three deployments annotated for AutoSage predictive autoscaling

All six muBench pods reached `1/1 Running` within 90 seconds of apply. Initial Qwen Q8_0 inference test at idle returned valid JSON in **7.7 seconds** — faster than Q4_K_M was on the old VM.

---

## Memory Baseline

With Qwen Q8_0 loaded and 6 idle pods running:

| Component | RAM used |
|---|---|
| OS + k3s + services | ~1.1 GiB |
| Qwen3.5-2B Q8_0 | ~3.8 GiB |
| 6 idle pods (~50 MiB each) | ~0.3 GiB |
| **Total used** | **~5.4 GiB** |
| **Free / available** | **~9.1 GiB** |

Under full load (12 pods × ~150 MiB each), estimated total: ~6.8 GiB used, ~7.2 GiB free. Qwen never needs to swap.

---

## Load Test

I ran the same four-phase wrk load test used on the old VM:

| Phase | Connections | Duration |
|---|---|---|
| Warmup | 4 | 60 s |
| Sustained | 16 | 300 s |
| Burst | 24 | 180 s |
| Cooldown | 4 | 120 s |

### HPA Scaling Timeline

All three services scaled from 2 to 4 replicas within 90 seconds of load onset:

| Event | Time from load start |
|---|---|
| process: 2 → 4 replicas | ~75 s |
| ingest: 2 → 4 replicas | ~75 s |
| analyze: 2 → 4 replicas | ~90 s |

Peak CPU during burst phase (measured via `kubectl top pods`):

| Service | CPU target | Peak CPU (per pod avg) | Replicas at peak |
|---|---|---|---|
| analyze | 70% (300m limit) | ~72% | 4/4 |
| ingest | 70% (500m limit) | ~162% | 4/4 |
| process | 70% (500m limit) | ~152% | 4/4 |

RAM during burst: 2.4 GiB used by pods, total cluster RAM used ~6.0 GiB, **~9 GiB still free**.

---

## Qwen Q8_0 Live Decisions

I fired the full AutoSage LLM + MCDA pipeline against all three services during and after the load test. All calls used `provider=gpt_oss, model=qwen3.5:2b` — no fallback to Groq.

### During Burst Phase (24 connections, all HPAs at 4/4 replicas)

| Service | CPU | Replicas | Action | Target | Confidence | MCDA | Inference time |
|---|---|---|---|---|---|---|---|
| analyze | 106% | 4/4 | maintain | 4 | 0.85 | full agreement | 136 s |
| ingest | 162% | 4/4 | maintain | 4 | 0.85 | full agreement | 130 s |
| process | 152% | 4/4 | maintain | 4 | 0.85 | full agreement | 64 s |

All three decisions are correct: every service is already at `maxReplicas=4`, so the only valid action is `maintain`. Qwen's reasoning for `ingest` (162% CPU): *"severe CPU pressure, no external state management detected, already at maximum capacity — cannot scale further, maintaining current state is the only viable option."*

MCDA validation in all three cases: `agreement=full, score_gap=0.0000` — the TOPSIS optimiser confirmed the LLM recommendation independently.

### Five-Scenario Structured Test (post-load)

| Scenario | Service | CPU | Replicas | Action | Target | Confidence | MCDA | Time |
|---|---|---|---|---|---|---|---|---|
| PRE-LOAD (idle) | analyze | 3% | 2/4 | maintain | 2 | 0.85 | full | 54.3 s |
| PEAK-LOAD | analyze | 106% | 4/4 | maintain | 4 | 0.85 | full | 136 s* |
| POST-LOAD | analyze | 3% | 2/4 | maintain | 2 | 0.85 | full | (cached) |
| PEAK-LOAD | ingest | 162% | 4/4 | maintain | 4 | 0.85 | full | 58.6 s |
| PEAK-LOAD | process | 152% | 4/4 | maintain | 4 | 0.85 | full | 55.2 s |

*136 s during actual burst load; 54–59 s when load has ended (fewer competing processes).

All five decisions are correct. The PRE-LOAD and POST-LOAD scenarios (2 replicas, 3% CPU) correctly get `maintain at 2` — no unnecessary scale-up. The PEAK-LOAD scenarios (4 replicas, 100–160% CPU) correctly get `maintain at 4` — no illogical scale-down at peak.

---

## Inference Timing Comparison

| VM | Model | Condition | Inference time | Result |
|---|---|---|---|---|
| Old (7.2 GiB) | Q8_0 | Idle | 53.8 s | ✓ Valid JSON |
| Old (7.2 GiB) | Q4_K_M | Idle | 11 s | ✓ Valid JSON |
| Old (7.2 GiB) | Q4_K_M | Under full load | 241 s | ✗ Timeout → Groq |
| **New (14 GiB)** | **Q8_0** | **Idle** | **7.7 s** | **✓ Valid JSON** |
| **New (14 GiB)** | **Q8_0** | **Under full burst** | **64–136 s** | **✓ Valid JSON, no fallback** |

The new VM resolves the core problem. Qwen Q8_0 completes every inference successfully even during peak load, with no swapping and no fallback to Groq.

---

## Observations and Limitations

**What works:** Qwen makes live, correct autoscaling decisions in all scenarios tested. The LLM + MCDA pipeline runs end-to-end on local hardware. Every MCDA validation shows full agreement with the LLM recommendation (score_gap=0.0000), which indicates the TOPSIS weighting (cost 0.20, performance 0.30, stability 0.25, forecast 0.15, response 0.10) is producing consistent results.

**Scale-up not yet demonstrated:** By the time I fire the Qwen advisor, the HPA has already reacted and scaled to 4 replicas. To show Qwen recommending scale_up (from 2 → 3 or 4), I need to trigger the decision within the first 30–60 seconds of load onset, before the HPA fires. This is the next experiment to run.

**Fan-out bottleneck design:** With `seq_len=100`, each client request generates up to 10,000 downstream calls to `analyze`. This means all three services saturate simultaneously rather than `analyze` (the intended bottleneck with 300m CPU limit) saturating first. Reducing seq_len to ~5 would fix the cascade effect and allow cleaner single-service bottleneck experiments.

**MCDA always agrees:** In all scenarios observed, MCDA and LLM are fully aligned. This validates that the two-layer architecture is consistent, but does not yet demonstrate the divergence/override case. That requires a deliberate stateful annotation to force LLM → HPA while MCDA → VPA.

---

## Next Steps

1. Fire AutoSage within the first 60 seconds of load onset to capture a live `scale_up` recommendation from Qwen
2. Reduce `seq_len` from 100 to 5 to isolate the `analyze` bottleneck
3. Annotate one service as stateful to trigger LLM vs MCDA divergence and demonstrate the override mechanism
4. Run the AutoSage 5-minute background loop during a full load test to capture organic decisions from the production code path (rather than the manual test script)
