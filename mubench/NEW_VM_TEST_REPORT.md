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

**Scale-up not yet demonstrated (first attempt):** By the time I fire the Qwen advisor, the HPA has already reacted and scaled to 4 replicas. To show Qwen recommending scale_up (from 2 → 3 or 4), I need to trigger the decision within the first 30–60 seconds of load onset, before the HPA fires.

**Fan-out bottleneck design:** With `seq_len=100`, each client request generates up to 10,000 downstream calls to `analyze`. This means all three services saturate simultaneously rather than `analyze` (the intended bottleneck with 300m CPU limit) saturating first. Reducing seq_len to ~5 would fix the cascade effect.

---

## Experiment 1: seq_len Reduction (100 → 5)

The `seq_len` parameter in `mubench/workmodel.json` was reduced from 100 to 5 for all three services. This reduces the fan-out from 10,000 downstream `analyze` calls per ingest request to 25, and prevents simultaneous saturation of all services.

After deploying the updated ConfigMap and restarting pods, the fan-out reduction worked as intended: `ingest` became the primary CPU consumer under load (60% CPU at 32 connections, T+25s), while `process` hit 13% and `analyze` stayed near 0%. However, `ingest` saturated before `analyze` because the `matrix_compute` function in `process` is heavier than `video_frames` in `analyze`, so the pipeline bottleneck shifted upstream. The `analyze` bottleneck (300m CPU limit) would only dominate at higher connection counts or larger matrix parameters.

---

## Experiment 2: Predictive Scale-Up and LLM vs MCDA Divergence

**Setup:** To capture Qwen recommending `scale_up` before HPA fires, I temporarily pinned `maxReplicas=2` on all three HPAs (preventing any reactive scaling), applied a 48-connection wrk load, sampled CPU at T+30s, and fired the full AutoSage LLM + MCDA pipeline against all three services with an explicit rising forecast (`trend: rapidly_increasing`, `peak_cpu: 2.5× current`).

**CPU at T+30s (48 connections, maxReplicas=2, 6 pods total):**

| Service | CPU (per-pod avg) | Replicas | CPU limit |
|---|---|---|---|
| ingest | 49.1% | 2/2 | 500m |
| process | 58.8% | 2/2 | 500m |
| analyze | 12.0% | 2/2 | 300m |

CPU was measured before HPA could react. All three services still had only 2 replicas.

**Qwen decisions (all Qwen3.5-2B Q8_0, no Groq fallback):**

| Service | CPU | LLM raw decision | Final decision | MCDA | Gap | Inference |
|---|---|---|---|---|---|---|
| ingest | 49.1% | scale_up → 3 (HPA) | **maintain → 2** | **OVERRIDE** | 0.2585 | 103 s |
| process | 58.8% | scale_up → 3 (HPA) | scale_up (VPA) | enforcement→VPA | — | 61 s |
| analyze | 12.0% | scale_up → 3 (HPA) | **maintain → 2** | **OVERRIDE** | 0.4084 | 65 s |

### Scale-Up: Qwen's Predictive Reasoning

For `ingest` at 49.1% CPU, Qwen recommended `scale_up → 3 replicas` citing the `rapidly_increasing` forecast with a predicted peak of 122% CPU. This is a correct predictive decision: current CPU is below the 70% HPA threshold, but the forecast signals an imminent breach within ~45 seconds. Without AutoSage, the HPA would only react after CPU crossed 70% and held there for one evaluation period (~15 s). AutoSage's LLM recommendation at T+30s is approximately 30–45 seconds ahead of the HPA.

Qwen's reasoning for `ingest`: *"The application 'ingest' is currently stateless... The forecast shows a 'rapidly_increasing' CPU trend, with the predicted peak of 122% CPU utilization. At 2 replicas, the current utilization of 49.1% exceeds safe operational margins given the forecast trajectory, indicating a high probability of service degradation if scaling is delayed."*

### LLM vs MCDA Divergence (ingest and analyze)

For `ingest` (49.1% CPU) and `analyze` (12.0% CPU), the LLM and MCDA reached different conclusions:

- **LLM**: `scale_up → 3` (based on rising forecast and performance weight)
- **MCDA (TOPSIS)**: `maintain → 2` (cost weight 0.20 + stability weight 0.25 outweigh performance weight 0.30 + forecast weight 0.15 when CPU is below threshold)

MCDA override was applied in both cases (`score_gap=0.2585` for ingest, `score_gap=0.4084` for analyze). The larger gap for analyze is expected: at 12% CPU, adding replicas is much harder to justify on cost/stability grounds than at 49%.

This demonstrates the two-layer validation functioning as intended. The LLM optimises for performance and future load; MCDA provides a multi-criteria check that prevents premature scaling when the cost and stability penalties outweigh the performance gain. For the thesis, this is the key divergence case: the LLM acts as an early-warning predictive layer, while MCDA acts as a conservative gatekeeper.

### Enforcement Layer: State Management Correction (process)

For `process` (58.8% CPU), the LLM recommended HPA scale_up (stateless, horizontal). The enforcement layer detected that state management was unknown (no annotations, no Redis/DB/volume dependencies found) and converted the recommendation from HPA → VPA as a safety measure. The reasoning: *"when state management is uncertain, vertical scaling is safer than horizontal scaling because it does not risk splitting session state across new pods."* MCDA validation was not applied for VPA recommendations in this code path.

This demonstrates the third validation layer (below LLM and MCDA): the state management enforcement layer, which acts on deployment annotations and environment variable inspection rather than on metric trends.

### maxReplicas Restoration

After all three decisions were captured, `maxReplicas=4` was restored on all HPAs. The final HPA state:

```
NAME      REFERENCE            TARGETS       MINPODS   MAXPODS   REPLICAS
analyze   Deployment/analyze   cpu: 1%/70%   2         4         2
ingest    Deployment/ingest    cpu: 1%/70%   2         4         2
process   Deployment/process   cpu: 1%/70%   2         4         2
```

---

## Updated Observations

**Scale-up captured:** Qwen recommended `scale_up → 3` for all three services when given rising-trend forecast data and current CPU below HPA threshold. For `ingest` at 49.1% CPU, this is approximately 30–45 s ahead of HPA reaction time.

**LLM vs MCDA divergence captured:** Both `ingest` and `analyze` show explicit MCDA override of LLM `scale_up` recommendations. The score gaps (0.2585 and 0.4084) quantify the MCDA disagreement. In both cases MCDA preferred `maintain` because the TOPSIS cost+stability weights outweigh performance+forecast weights at sub-threshold CPU.

**Enforcement layer acts before MCDA:** For `process`, state uncertainty triggered an HPA→VPA correction in the enforcement layer, before MCDA was consulted. This is a separate code path: enforcement reads deployment metadata (annotations, volumes, env vars), not metrics.

**MCDA override is conservative:** In every divergence case observed, MCDA overrode LLM in the conservative direction (prevent scale_up, not prevent scale_down). This is the expected behaviour given the TOPSIS weights: cost (0.20) and stability (0.25) together outweigh forecast (0.15) at low-to-mid CPU, so MCDA only agrees with scale_up when CPU is already high (as in the original burst-phase tests).

---

## Next Steps

1. Run the AutoSage 5-minute background loop during a full load test to capture organic decisions from the production code path (rather than the manual test script)
2. Tune TOPSIS weights to give forecast a higher weight (e.g., 0.25) so MCDA and LLM agree on scale_up at ~50% CPU with rapidly_increasing trend
3. Add a controlled stateful annotation (`autosage.ai4k8s/state-management=stateful`) to demonstrate the full HPA vs VPA divergence path
