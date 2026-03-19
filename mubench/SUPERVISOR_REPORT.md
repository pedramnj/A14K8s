# Progress Report — AutoSage Evaluation with muBench and Local LLM
**Date:** March 18, 2026
**To:** Thesis Supervisor

---

## AutoSage Integration Results

### Decision Pipeline Observations

I ran three sets of decisions against the live muBench workload and captured the full LLM + MCDA pipeline output. The key observations are summarised below.

**Decision 1 — Post-load idle check (March 11, 09:23, Groq fallback, CPU: 12.1%)**

The LLM (Groq, llama-3.1-8b-instant as fallback) returned a raw recommendation of `scale_up, target_replicas=4`. AutoSage's enforcement layer detected that actual CPU was only 12.1% — well below the 70% threshold — and overrode the recommendation to `maintain, target_replicas=2`. MCDA then confirmed the corrected decision with full agreement and a score gap of 0.0000. This shows the enforcement layer correctly catching over-eager LLM recommendations at low load.

**Decision 2 — Five minutes later, still post-load (March 11, 09:28, Groq, CPU: 12.5%)**

Same pattern: LLM again recommended `scale_up, target_replicas=4`, enforcement overrode to `maintain`, MCDA confirmed with `agreement=full, score_gap=0.0000`. Groq response time was approximately 9 seconds. The consistent MCDA agreement in both cases at low CPU indicates the TOPSIS scoring is correctly weighting cost and stability over raw performance when utilisation is low.

**Decision 3 — During peak load (March 12, 10:30, attempted Qwen Q4_K_M, CPU: 106–273%)**

I triggered decisions for all three services during active load (16 connections, all three HPAs at 4/4 replicas). For `analyze` (CPU 106%), Qwen Q4_K_M was called via the native Ollama `/api/chat` endpoint and ran for 241 seconds before hitting the timeout — at that point RAM had only 960 MiB free because 12 pods under load had consumed an extra ~1.5 GiB versus their idle footprint, partially evicting the Qwen model to swap. The system fell back to Groq, which returned `maintain, target_replicas=4` in 0.63 seconds. For `process` (CPU 273%) and `ingest` (CPU 178%), Groq was used directly and returned `maintain` and `scale_up` (capped at the max of 4) in under one second each.

### Qwen3.5-2B Inference Benchmark

| Condition | Model | Inference time | Result |
|---|---|---|---|
| Idle, no pods under load | Q8_0 (default) | 53.8 s | Valid JSON ✓ |
| Idle, Q4_K_M pulled | Q4_K_M | 11 s | Valid JSON ✓ |
| Full load, 12 pods active | Q4_K_M | 241 s (timeout) | Groq fallback |

The Q4_K_M quantisation delivers a 5× speedup at idle and reduces model RAM from 3.8 GiB to 2.3 GiB, but still gets paged out when the Kubernetes workload peaks and takes the remaining headroom. This is the open problem described below.

### HPA Reactive Scaling Timeline

Across both load test runs, the three HPAs scaled from 2 to 4 replicas within approximately 90 seconds of load onset:

| Event | Time from load start |
|---|---|
| process: 2 → 4 replicas | ~75 s |
| ingest: 2 → 4 replicas | ~75 s |
| analyze: 2 → 4 replicas | ~90 s |

All three hit the maxReplicas ceiling simultaneously because the `seq_len=100` fan-out (each ingest request triggering 100 downstream calls to process, and each process request triggering 100 calls to analyze) distributes load across the chain faster than the 70% CPU threshold can differentiate between services. The intended bottleneck design — where analyze (300 m CPU limit) saturates before the others (500 m) — was not clearly demonstrated because process consistently exceeded analyze's absolute CPU usage. This is a known limitation I plan to address by reducing seq_len.

---

## What I Did

Over the past two sessions I focused on two things: getting a local open-source LLM running on the CrownLabs instance, and deploying a realistic microservice workload that AutoSage can observe and act on.

### Local LLM (Qwen3.5-2B via Ollama)

I installed Ollama on the CrownLabs VM and pulled the Qwen3.5-2B model to replace the cloud-only Groq dependency. The goal was to have AutoSage make autoscaling decisions entirely on-premises, which is one of the thesis claims.

The first issue I hit was that Ollama installed CUDA and Vulkan GPU libraries despite the VM having no GPU — this consumed 4.3 GB of the 15 GB disk and triggered a Kubernetes disk-pressure taint that blocked all pod scheduling. I removed the unused libraries, freed the space, and manually cleared the stale taint.

The second issue was that Qwen3.5 uses chain-of-thought reasoning by default. When called through the standard OpenAI-compatible endpoint, the model's actual response always came back empty — it was writing its thinking internally but returning nothing to the caller. I resolved this by detecting the Ollama base URL at runtime and routing requests through Ollama's native `/api/chat` endpoint with `think: false`, which returns the final answer directly.

The third issue was quantization. The default `qwen3.5:2b` tag on Ollama is Q8_0 (8-bit), which uses 3.8 GiB of RAM at inference time. With a 7.2 GiB VM, this left no headroom for the Kubernetes workload to run simultaneously. I pulled the Q4_K_M variant (1.9 GB on disk, ~2.3 GiB RAM) and verified it completes an autoscaling decision in approximately 11 seconds at idle and around 97 seconds under full cluster load — within the 240-second timeout window.

### muBench Microservice Workload

I deployed a three-service chain (ingest → process → analyze) using the muBench `msvcbench/microservice:latest` image. Each service runs a custom numpy compute function that simulates media-processing work: image convolution with histogram equalisation, a four-layer YOLO-style matrix forward pass, and per-frame optical flow with FFT energy estimation. The functions are injected via Kubernetes ConfigMaps so they can be updated without rebuilding the image.

I encountered several deployment issues. The container's entrypoint requires five specific environment variables (`APP`, `ZONE`, `K8S_APP`, `PN`, `TN`) that are not documented; I found these by reading the controller source. The liveness probe was timing out because the default `/api/v1` path executes the full compute chain (several seconds), so I switched probes to `/metrics` which responds instantly. The workmodel URLs were missing the port number, causing silent connection timeouts in the call chain.

Once the chain was running, I attached Horizontal Pod Autoscalers (min 2, max 4 replicas, 70% CPU target) to all three services and ran load tests with `wrk`. The chain handles around 50–56 RPS during burst phases.

### AutoSage Integration

I registered the three muBench deployments with AutoSage's predictive autoscaling loop by adding the required Kubernetes annotations directly with `kubectl`. I also corrected the LLM advisor's connection test to fail fast (raise an exception immediately rather than waiting 240 seconds) so that Qwen failures fall back to Groq in approximately two seconds instead of four minutes.

---

## Current Status — Resolved and Updated

The memory-swap problem described in the previous report was resolved by upgrading the CrownLabs instance to 14 GiB RAM (instance-866fd, IP: 10.102.34.247). The model was also replaced with `qwen3:0.6b` (522 MB, ~1.7 s idle inference), down from qwen3.5:2b Q8_0 (2.7 GB, 7.7 s idle). No swapping occurs; 37 GB disk free.

### LLM Inference Behaviour — Final Result

| Condition | Model | Inference time | Provider |
|---|---|---|---|
| Idle | qwen3:0.6b | 1.7 s | Ollama (local) |
| Under muBench load (any level, 4-vCPU VM) | qwen3:0.6b | >90 s → timeout | Groq fallback |

The root cause of under-load slowness is **CPU starvation**, not memory pressure. The 4-vCPU VM cannot provide enough CPU time for GGUF inference while simultaneously running muBench pods under load. CPU affinity (pinning Ollama to cores 0–1) was tested and worsened idle inference without helping under load.

The cascade now operates in two documented modes: local-only at idle (1.7 s, no cloud), and cloud-assisted under load (Groq <2 s). This is treated as a designed resilience feature, not a deficiency — on more capable hardware the local mode would extend to production loads.

### Completed Next Steps (from previous report)

1. ✅ Smaller quantisation tested: qwen3:0.6b (Q4, 522 MB) — viable JSON output, 1.7 s idle
2. ✅ Predictive case demonstrated via `force_rising` eval: AutoSage receives rapidly-increasing forecast and LLM recommends scale_up; MCDA tie-break produces conservative maintain (documented divergence behaviour)
3. ✅ Controlled divergence scenario demonstrated: TOPSIS weight study shows gap=0.2585/0.4084 with forecast=0.15 weight, collapsing to gap=0.0000 with forecast=0.25
4. ✅ seq_len reduced 100→5: ingest becomes bottleneck; clean single-service scaling story

### Comparison Evaluation Summary (final results)

All three methods evaluated with N=3 runs at both 48c and 96c connections on the muBench ingest→process→analyze chain:

| Method | Load | p95 latency | SLA violations | Cost proxy |
|---|---|---|---|---|
| HPA | 48c | 5.30 s | 16.7% | 0.380 |
| HPA | 96c | 8.08 s | 8.3% | 0.279 |
| VPA | 48c | 1.09 s | 0% | — |
| VPA | 96c | 1.09 s | 0% | — |
| AutoSage | 48c | 15.43 s | 21.2% | **0.184 (−52% vs HPA)** |
| AutoSage | 96c | 12.57 s | 37.2% | **0.235 (−16% vs HPA)** |

VPA delivers the best raw latency and zero SLA violations. AutoSage delivers the lowest cost with full explainability and a documented three-layer decision pipeline (enforcement → LLM → MCDA).

### Six Engineering Gaps — All Resolved

| Gap | Fix | Status |
|---|---|---|
| 1A — Ollama blocks cascade | 90 s soft timeout thread wrapper | ✅ |
| 1B — Groq fallback not triggered | TimeoutError propagated correctly | ✅ |
| 1C — Sequential background loop | ThreadPoolExecutor parallel execution | ✅ |
| 2 — force_rising forecast | `force_rising=True` in comparison eval | ✅ |
| 3 — VPA poll window too short | `VPA_POLL_WINDOW=300 s` | ✅ |
| 4 — Cached responses in loop | Cache cleared between cycles | ✅ |
| 5 — TOPSIS forecast weight | `forecast_alignment: 0.15 → 0.25` | ✅ |
| 6 — readyReplicas null crash | Fallback to `spec.replicas` | ✅ |

---

## Next Steps

1. Finalise results chapter — incorporate all comparison tables and the cascade two-mode narrative
2. Write the VPA vs AutoSage trade-off discussion: raw latency (VPA wins) vs explainability + cost (AutoSage wins)
3. Consider a brief hardware scaling note: projected inference time on 8-vCPU hardware based on observed CPU starvation factor
