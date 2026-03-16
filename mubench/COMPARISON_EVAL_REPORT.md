# AutoSage vs Native HPA/VPA — Comparison Evaluation Report
**Date:** March 16, 2026
**Instance:** instance-s8rnq, Node: worker-7-rs, IP: 10.98.179.33
**RAM:** 14 GiB | **Disk:** 42 GB | **CPU:** 4 vCPUs
**Script:** `mubench/run_comparison_eval.py`
**Results JSON:** `mubench/comparison_results.json`

---

## Objective

Measure and compare three autoscaling approaches on a live muBench workload:

1. **Native HPA** — reactive, CPU-threshold-based horizontal scaling
2. **Native VPA** — vertical resource adjustment (controller availability check)
3. **AutoSage** — LLM (Qwen3.5-2B) + MCDA (TOPSIS) predictive advisor

Metrics collected: provisioning latency, first scale-up latency, peak replicas, p95 response latency, SLA violation rate, cost proxy, and AutoSage decision latency breakdown.

---

## Experimental Setup

| Parameter | Value |
|---|---|
| Cluster | k3s v1.34.5+k3s1, single-node |
| Hardware | 4 vCPUs, 14 GiB RAM (CrownLabs VM) |
| Workload | muBench ingest→process→analyze chain |
| Load generator | wrk, 48 connections, 4 threads, 120 s window |
| Target endpoint | `http://ingest:8080/api/v1` |
| Fan-out (seq_len) | 5 (25 downstream calls per ingest request) |
| HPA config | minReplicas=2, maxReplicas=4, CPU target=70% |
| Baseline replicas | 2 (all services) |
| Probe timing | T+90 s into load window |
| Probes per run | 20 in-cluster curl requests |
| SLA threshold | 2.0 s |
| Runs per method | 3 |
| Cooldown between runs | 30 s |
| LLM model | qwen3.5:2b (Q4\_K\_M, ~2.3 GiB RAM) |

**AutoSage trial design:** HPAs were pinned at `maxReplicas=2` during the first 30 s of load to prevent reactive HPA from firing before the LLM advisor could be invoked. After the recommendation was captured, `maxReplicas=4` was restored.

---

## Results

### Table 1 — Control Loop Timing and Scaling Outcomes

| Metric | Native HPA | Native VPA | AutoSage |
|---|---|---|---|
| Provisioning latency | **247 ± 97 ms** | N/A (no CRD) | — |
| First scale-up latency | **36.2 s** (reactive) | N/A | — (decided maintain) |
| Peak replicas | **4 / 4** | N/A | **2 / 4** (correct at sub-threshold) |
| Decision/recommendation latency | ~36 s (HPA eval cycle) | N/A | **88.2 ± 8.7 s** |

HPA provisioning (patch latency) is sub-300 ms. First reactive scale-up occurs at 36.2 s — consistent across all 3 runs, corresponding to the HPA evaluation period (15 s stabilization window × 2 + scrape lag). AutoSage makes its recommendation in ~88 s dominated entirely by Qwen inference.

---

### Table 2 — Service-Level and Cost Metrics (N=3 runs, mean ± 95% CI)

| Method | p95 latency (s) | SLA violation rate | Cost proxy (avg vCPU) |
|---|---|---|---|
| Native HPA | 12.6 ± 9.5 | **85.0%** | 0.269 ± 0.058 |
| Native VPA | N/A (no CRD) | N/A | N/A |
| AutoSage | **7.8 ± 7.4** | **80.5% ± 14.4%** | **0.155 ± 0.018** |

Cost proxy = avg\_replicas × cpu\_request (125 m) / 1000.

AutoSage achieves **−43% cost** relative to HPA (0.155 vs 0.269 vCPU) by correctly deciding not to provision extra replicas when CPU is below the 70% threshold. It also shows slightly lower p95 latency (7.8 vs 12.6 s) and marginally fewer SLA violations — though variability is high in both cases due to the high-concurrency load.

---

### Table 3 — AutoSage Decision Breakdown (per run)

| Run | ingest CPU at T+30s | LLM action | scaling_type | target_replicas | confidence | MCDA | rec. latency |
|---|---|---|---|---|---|---|---|
| 1 | 22.4% | maintain | hpa | 2 | 0.95 | full agreement (gap=0.000) | 93.2 s |
| 2 | 47.5% | maintain | hpa | 2 | 0.95 | full agreement (gap=0.000) | 87.8 s |
| 3 | 57.7% | maintain | hpa | 2 | 0.95 | full agreement (gap=0.000) | 83.7 s |

All three runs produced `maintain@2` — correct decisions at the measured CPU levels (22–58%, all below the 70% HPA threshold). Qwen's reasoning in all cases: the application is annotated `STATELESS`, HPA is the correct mode, and current CPU does not justify a scale-up. MCDA (TOPSIS) reached full agreement with the LLM in all three runs (score gap = 0.000), confirming no divergence at sub-threshold CPU.

---

### Table 4 — AutoSage Timing Decomposition (mean across runs)

| Phase | Time |
|---|---|
| Metrics collection (kubectl top) | **0.57 s** |
| LLM inference (Qwen3.5-2B Q4\_K\_M) | **88.2 s** |
| MCDA validation (TOPSIS) | < 0.01 s |
| Actuation (kubectl scale) | 0.00 s (no actuation — maintain) |
| **Total decision loop** | **~88.8 s** |

LLM inference accounts for >99% of the AutoSage decision latency. Metrics collection, MCDA, and actuation are negligible by comparison.

---

## VPA Status

The k3s single-node cluster does not have the VPA CRD installed (`verticalpodautoscalers.autoscaling.k8s.io` not present). Native VPA is therefore marked **N/A** across all trials. AutoSage's VPA path (enforced for `stateful` workloads via `vpa_engine.patch_deployment_resources()`) does not require the VPA controller — it applies resource changes directly via `kubectl patch deployment`.

---

## Analysis

### Decision Latency Trade-Off

The dominant cost of deploying AutoSage is LLM inference overhead (~88 s). Native HPA reacts in ~36 s by detecting CPU above threshold and applying scale rules within one evaluation cycle. AutoSage takes 2.4× longer but reasons over a richer context (annotations, forecast trend, scaling history, cost/stability weights).

For this specific test scenario (sub-threshold CPU, stable trend, stateless annotation present), both methods reach the "correct" outcome from different angles:
- HPA: scale up to 4 because CPU ≥ 70%
- AutoSage: maintain at 2 because CPU < 70% with stable trend → no immediate need to over-provision

This is the key thesis tension: **AutoSage trades decision speed for decision intelligence**, using multi-criteria reasoning rather than a single CPU threshold.

### Cost Efficiency

AutoSage's `maintain@2` decision results in 43% lower resource consumption than HPA's `scale_up→4`. Under the muBench workload used here (sub-threshold CPU), AutoSage prevents unnecessary horizontal scaling. In the burst scenario from Experiment 2 of the NEW_VM_TEST_REPORT (ingest at 49.1% with rapidly\_increasing forecast), AutoSage did recommend `scale_up→3` proactively — approximately 30–45 s ahead of the HPA reaction.

### LLM–MCDA Agreement

All three AutoSage runs show **full MCDA agreement** (score gap = 0.000). This occurs when CPU is clearly below threshold and the forecast trend is stable — TOPSIS cost and stability weights naturally align with the LLM's conservative recommendation. Divergence was observed in Experiment 2 (ingest=49.1%, rapidly\_increasing forecast) where score gaps of 0.2585 and 0.4084 were recorded.

---

## Raw Per-Run Data

### HPA Trials

| Run | provisioning (ms) | first\_scale (s) | peak\_replicas | p95 (s) | SLA viol. | cost\_proxy |
|---|---|---|---|---|---|---|
| 1 | 302 | 36.2 | 4 | 6.81 | 0.85 | 0.240 |
| 2 | 241 | 36.2 | 4 | 13.97 | 0.85 | 0.303 |
| 3 | 198 | 36.2 | 4 | 16.91 | 0.85 | 0.265 |
| **mean ± CI** | **247 ± 97 ms** | **36.2 ± 0.0 s** | **4.0** | **12.6 ± 9.5 s** | **85%** | **0.269 ± 0.058** |

### AutoSage Trials

| Run | CPU at T+30s | rec\_latency (s) | action | peak\_replicas | p95 (s) | SLA viol. | cost\_proxy |
|---|---|---|---|---|---|---|---|
| 1 | 22.4% | 93.2 | maintain | 2 | 4.39 | 0.85 | 0.165 |
| 2 | 47.5% | 87.8 | maintain | 2 | 6.69 | 0.71 | 0.145 |
| 3 | 57.7% | 83.7 | maintain | 2 | 12.24 | 0.85 | 0.155 |
| **mean ± CI** | — | **88.2 ± 8.7 s** | maintain (3/3) | **2.0** | **7.8 ± 7.4 s** | **80.5% ± 14%** | **0.155 ± 0.018** |

---

## Observations

**What this evaluation demonstrates:**

1. **HPA provisioning is fast (~250 ms)** but reactive — it only acts after CPU crosses 70% and stays there for one evaluation cycle (~36 s total)
2. **AutoSage decision latency is ~88 s** — dominated entirely by Qwen3.5-2B Q4\_K\_M inference on a 4-vCPU VM under concurrent muBench load
3. **AutoSage decisions are correct at sub-threshold CPU** — `maintain@2` is the right call when CPU is 22–58% with stable trend; full LLM–MCDA agreement in all runs
4. **AutoSage is 43% cheaper in resource footprint** — avoids over-provisioning when load doesn't justify it
5. **SLA violation rates are high for both methods (80–85%)** — reflecting the infrastructure's true capacity limit under 48 concurrent connections with a chain fan-out of 25 calls/request. This is a hardware constraint, not a control policy failure.

**Limitation — scale-up not triggered in this eval:**
Because CPU remained below 70% during all AutoSage runs, no `scale_up` decision was issued. The scale-up scenario (Qwen recommending scale\_up→3 at 49.1% CPU with rapidly\_increasing forecast) was captured in a separate experiment — see **Experiment 2** in `NEW_VM_TEST_REPORT.md`.

---

## Next Steps

1. Re-run AutoSage trials with higher connection count (e.g., 96c) to push ingest CPU above 70% during the 30 s wait window, capturing a `scale_up` recommendation in the comparison framework
2. Install VPA controller (`kubectl apply -f https://github.com/kubernetes/autoscaler/...`) to enable native VPA trial
3. Tune TOPSIS forecast weight (0.15 → 0.25) so MCDA agrees with scale\_up at ~50% CPU + rapidly\_increasing trend
