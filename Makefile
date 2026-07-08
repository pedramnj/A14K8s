# AutoSage — reproducibility Makefile
# ─────────────────────────────────────────────────────────────────────────────
# Targets meant for the ACM/USENIX artifact-evaluation reviewer or anyone
# rebuilding a specific figure from the paper. Each `reproduce-*` target
# runs a scoped subset of the full evaluation (3 trials per method) so it
# completes in ~15 min on a single 4-vCPU / 16 GiB VM.
#
# Prerequisites:
#   - k3s (with metrics-server) running locally, `kubectl` on $PATH
#   - Python 3.11+, deps installed via `pip install -r requirements.txt`
#   - Ollama + qwen3.5:2b (`ollama pull qwen3.5:2b`) OR a Groq API key
#   - `wrk` on $PATH (Debian: `apt install wrk`)
#   - wrk2 on $PATH for `reproduce-phase-r` (build from
#     https://github.com/giltene/wrk2)
#
# All targets assume they are invoked from the repo root and hit whatever
# cluster $KUBECONFIG points at.

PYTHON ?= python3
N_RUNS ?= 3

.PHONY: help figures stats stats-quick \
        reproduce-phase-i reproduce-phase-p reproduce-phase-r \
        clean-results

help:
	@echo "AutoSage — reproducibility targets"
	@echo ""
	@echo "  make figures            regenerate every figure from shipped JSONs"
	@echo "  make stats              full stats-rigor pass (bootstrap + Wilcoxon"
	@echo "                          + Cliff's delta) into thesis_reports/"
	@echo "  make stats-quick        stats-rigor filtered to N_RUNS >= 5"
	@echo ""
	@echo "  make reproduce-phase-i  session-cache stateful workload,"
	@echo "                          Phase I baseline (~15 min, N_RUNS=$(N_RUNS))"
	@echo "  make reproduce-phase-p  continuous-daemon AutoSage on stateful-"
	@echo "                          compute (Phase P headline, ~20 min)"
	@echo "  make reproduce-phase-r  DSB Hotel Reservation (5 services + 3"
	@echo "                          Mongo + 2 memcached + Consul, ~25 min)"
	@echo ""
	@echo "  make clean-results      remove local comparison_results_v*.json"
	@echo ""
	@echo "Env knobs: PYTHON=$(PYTHON)  N_RUNS=$(N_RUNS)"

# ── Figures + stats ──────────────────────────────────────────────────────────
figures:
	$(PYTHON) -m research.generate_thesis_evaluation_plots

stats:
	$(PYTHON) -m research.stats_rigor

stats-quick:
	$(PYTHON) -m research.stats_rigor --min-runs 5

# ── Phase I — session-cache stateful workload ────────────────────────────────
reproduce-phase-i:
	kubectl apply -f mubench/k8s-manifests-stateful/
	kubectl wait --for=condition=Ready pod -l app=session-cache \
	        --timeout=180s
	N_RUNS=$(N_RUNS) WORKLOAD=stateful \
	    RESULTS_PATH=mubench/comparison_results_reproduce_phase_i.json \
	    $(PYTHON) mubench/run_comparison_eval.py --workload stateful
	@echo ""
	@echo "  Phase I reproduction saved to"
	@echo "  mubench/comparison_results_reproduce_phase_i.json"

# ── Phase P — continuous-loop AutoSage daemon ────────────────────────────────
reproduce-phase-p:
	kubectl apply -f mubench/k8s-manifests-stateful-compute/
	kubectl wait --for=condition=Ready pod -l app=session-compute \
	        --timeout=180s
	N_RUNS=$(N_RUNS) WORKLOAD=stateful_compute \
	    AUTOSAGE_CONTINUOUS_DAEMON_ENABLED=1 \
	    AUTOSAGE_DAEMON_TICK_S=30 \
	    AUTOSAGE_VPA_REQUEST_MULTIPLIER=2.5 \
	    AUTOSAGE_VPA_SET_LIMITS=1 \
	    RESULTS_PATH=mubench/comparison_results_reproduce_phase_p.json \
	    $(PYTHON) mubench/run_comparison_eval.py --workload stateful_compute
	@echo ""
	@echo "  Phase P reproduction saved to"
	@echo "  mubench/comparison_results_reproduce_phase_p.json"

# ── Phase R — DSB Hotel Reservation multi-tier chain ─────────────────────────
reproduce-phase-r:
	kubectl apply -f dsb-hotel/kubernetes/
	kubectl wait --for=condition=Ready pod \
	        -l app.kubernetes.io/part-of=dsb-hotel --timeout=300s
	N_RUNS=$(N_RUNS) WORKLOAD=dsb_hotel \
	    RESULTS_PATH=mubench/comparison_results_reproduce_phase_r.json \
	    $(PYTHON) mubench/run_comparison_eval.py --workload dsb_hotel
	@echo ""
	@echo "  Phase R reproduction saved to"
	@echo "  mubench/comparison_results_reproduce_phase_r.json"

# ── Cleanup ──────────────────────────────────────────────────────────────────
clean-results:
	rm -f mubench/comparison_results_reproduce_phase_*.json
	rm -f mubench/comparison_results_v*.json
