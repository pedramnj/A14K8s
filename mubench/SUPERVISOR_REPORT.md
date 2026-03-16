# Progress Report — AutoSage Evaluation with muBench and Local LLM
**Date:** March 12, 2026
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

## Current Status and Open Problem

The main open problem is that Qwen Q4_K_M still times out when inference runs concurrently with a full-load test. At peak, the 12 pods (3 services × 4 replicas) inflate memory from their idle ~50 MiB to ~150–200 MiB each, consuming an extra ~1.5–1.8 GiB. This pushes the Qwen model out of active RAM and into swap, which causes inference to exceed the 240-second timeout.

The practical consequence is that during a live load test, Groq (cloud) is used as the fallback rather than Qwen (local). Qwen does work correctly at idle and at light load.

I contacted the CrownLabs administrators and they suggested trying more aggressive quantization. Q2_K (~1.3 GiB RAM) is the next option to test, though the tag appears to not yet be available in the Ollama registry for this model. Alternatively, I am considering restructuring the experiment so that AutoSage makes its predictive decision *before* the load peak arrives — which is actually the correct behaviour for a predictive system and aligns better with the thesis claim.

---

## Next Steps

1. Test whether a Q3_K_M or Q2_K quantization is available and viable for JSON output quality
2. Redesign the experiment to demonstrate the predictive case: AutoSage fires at rising CPU (~50%) and recommends scale_up *before* HPA would trigger at 70%
3. Add a controlled divergence scenario where LLM and MCDA disagree, to show the two-layer validation is non-trivial
4. Reduce the fan-out ratio (seq_len from 100 to ~5) so the `analyze` bottleneck saturates before `process` and `ingest`, creating a cleaner single-service scaling story
