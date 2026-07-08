# AutoSage — Artifact Evaluation Guide

This document walks a reviewer through reproducing the headline claims of the
AutoSage evaluation on a single machine. Expected wall-clock time: **~1 h**
for the shipped `reproduce-*` targets on a 4 vCPU / 16 GiB VM. Longer if
you rerun the full 10-trial matrix.

## What we claim

1. **Workload-class-correct decisions.** AutoSage's LLM picks horizontal
   scaling on stateless workloads and vertical scaling on stateful ones on
   every trial across a 99-decision evaluation.
2. **28–47 % SLA-normalised cost reduction** versus native HPA on both
   workload classes.
3. **Continuous daemon closes the p95 latency gap** to native VPA on
   stateful workloads from 3.04× to 1.45×.
4. **Multi-tier chain generalisation** (Phase R): the workload-class
   heuristic and cost-frontier result carry over to the DeathStarBench
   Hotel Reservation 5-service chain.

Every raw JSON that backs these numbers is in `mubench/` and
`thesis_reports/`; every plot is regenerable from those JSONs.

## System requirements

- **CPU / RAM**: 4 vCPU / 16 GiB minimum. 8 vCPU / 32 GiB removes CPU
  contention on Phase R.
- **OS**: Ubuntu 22.04 / Debian 12 (any Linux with `apt` + Docker will do).
- **Kubernetes**: k3s v1.28+ with the metrics-server addon (default for
  k3s installer).
- **Python**: 3.11+.
- **LLM**: Ollama with `qwen3.5:2b` pulled locally (~3.8 GiB resident),
  OR a Groq API key (cloud fallback, ~1 s per decision).
- **Load generators**: `wrk` on `$PATH` for muBench workloads, `wrk2`
  for the DSB Hotel Reservation Phase R workload.

## One-time setup

```bash
# 1. Install k3s (with metrics-server enabled by default)
curl -sfL https://get.k3s.io | sh -s - --write-kubeconfig-mode 644

# 2. Install Ollama + pull the local model
sudo apt-get install -y zstd
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.5:2b

# 3. Clone AutoSage + Python deps
git clone https://github.com/pedramnj/A14K8s.git ~/autosage
cd ~/autosage
git checkout update
python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt

# 4. .env file (Ollama endpoint + optional Groq key)
cat > .env <<'EOF'
GPT_OSS_API_BASE=http://localhost:11434/v1
GPT_OSS_MODEL=qwen3.5:2b
# Optional: uncomment to enable Groq cloud fallback
# GROQ_API_KEY=your_groq_key_here
EOF

# 5. Load generators
sudo apt-get install -y wrk
# wrk2 needs to be built from source:
git clone https://github.com/giltene/wrk2.git /tmp/wrk2
make -C /tmp/wrk2
sudo install /tmp/wrk2/wrk /usr/local/bin/wrk2
```

## Reproducing the three headline claims

Every command below is invoked from the repo root with the Python venv
active. The Makefile wraps the raw eval-harness invocations with the
correct env-vars.

### Claim 1 + 2 — Workload-class convergence + cost frontier (Phase I)

```bash
make reproduce-phase-i
```

Deploys the session-cache stateful workload and runs one 3-trial pass of
HPA / VPA / AutoSage. Expected: AutoSage picks VPA on every AutoSage
trial (workload-class check), and its cost proxy is 25–40 % below HPA's
avg replicas × CPU request. Output JSON:
`mubench/comparison_results_reproduce_phase_i.json`.

Run `python3 research/stats_rigor.py --workload stateful` to get the
bootstrap CIs and Wilcoxon p-values on the same data alongside the
full Phase A–Q historical archive.

### Claim 3 — Continuous daemon closes p95 gap (Phase P)

```bash
make reproduce-phase-p
```

Deploys the stateful-compute session-compute workload and runs the
continuous-daemon AutoSage variant against HPA + native VPA. Expected:
AutoSage p95 ~1.0–1.5 s (native VPA ~0.60 s, HPA ~1.5–2.0 s) and the
same 28–47 % cost reduction. Output JSON:
`mubench/comparison_results_reproduce_phase_p.json`.

### Claim 4 — DSB Hotel Reservation multi-tier chain (Phase R)

```bash
make reproduce-phase-r
```

Deploys the 5-service DSB Hotel Reservation chain (11 pods total —
frontend, search, geo, rate, profile + 3 Mongo + 2 memcached + Consul)
per the modified upstream manifests in `dsb-hotel/kubernetes/`. Runs
HPA / VPA / AutoSage across all 5 app services concurrently. Expected:
AutoSage picks HPA on frontend/search/geo/rate and VPA on profile with
≥ 90 % agreement across ticks; cross-service cost proxy stays 20–35 %
below HPA. Output JSON:
`mubench/comparison_results_reproduce_phase_r.json`.

## Statistical rigor pass

Regenerate the bootstrap 95 % CI + Wilcoxon signed-rank + Cliff's delta
table for every metric and every version in the archive:

```bash
make stats
```

Emits `thesis_reports/stats_rigor.tex` (LaTeX longtable) and
`thesis_reports/stats_rigor.json` (machine-readable dump). `make
stats-quick` restricts to versions with ≥ 5 trials per method.

## Live 24-hour trace

A 24-hour diurnal workload trace against the DSB Hotel Reservation
substrate lives on Zenodo (see below). To regenerate it locally:

```bash
python3 research/live_trace.py --duration-h 24 \
    --results live_trace_1day.json
```

The script sinusoidally varies wrk2's `-R` rate through four peaks over
24 h and samples the AutoSage daemon's cluster state every 30 s. Output
`live_trace_1day.json` (~3–5 MB) is the direct input to
`fig_live_trace.png`.

## Data archive

- Every `comparison_results_v*.json` from Phases A–Q shipped in
  `mubench/` and `thesis_reports/`.
- Every plot generation script in `research/`.
- Every wrk / wrk2 lua script in `mubench/k8s-manifests-*` and
  `dsb-hotel/wrk2/`.
- The tarballed raw data + one-week live trace are on Zenodo. The DOI
  will be filled in on the last commit before the artifact evaluation
  submission.

  **DOI**: `10.5281/zenodo.<TBD>`   *(placeholder — final upload lands
  on the ARTIFACT-BUNDLE tag; the DOI is minted at that point and the
  README updated in the same commit).*

## Image pinning

Container images are pinned to versioned tags in the shipped manifests
(`mongo:4.4.6`, `memcached:1.6-alpine`, `hashicorp/consul:1.15`,
`msvcbench/microservice:latest` — the last of which will be replaced
with a `@sha256:<digest>` on the ARTIFACT-BUNDLE tag). Ollama pulls
`qwen3.5:2b` which is a rolling tag; for exact reproducibility the
model manifest SHA is captured in `Ollama-manifest.txt` alongside the
Zenodo tarball.

## Reviewer feedback

If a `reproduce-*` target fails on your machine we would very much like
to know. File an issue on
https://github.com/pedramnj/A14K8s/issues with the command, the
observed error, and the trailing 50 lines of `kubectl describe pods`
and `kubectl top pods`.
