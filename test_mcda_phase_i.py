#!/usr/bin/env python3
"""
Phase-I sanity check for the unified MCDA pool.
================================================

Runs three scenarios that exercise the new vertical pool, the unified
TOPSIS gap, and the LLM-vs-MCDA agreement decision for VPA picks.

Run with:
    python3 test_mcda_phase_i.py
    MCDA_COST_EXPONENT=0.5 python3 test_mcda_phase_i.py
    MCDA_VPA_RESTART_PENALTY=0.1 python3 test_mcda_phase_i.py

The script prints PASS/FAIL per scenario and exits 0 only if all pass.
"""
import os
import sys

from mcda_optimizer import (
    MCDAAutoscalingOptimizer,
    MCDA_COST_EXPONENT,
    MCDA_VPA_RESTART_PENALTY,
    VPA_MULTIPLIER_GRID,
)


def banner(s):
    print("\n" + "=" * 70)
    print(s)
    print("=" * 70)


def horizontal_only_reproduces_phase_h():
    """No resources supplied → 3 alternatives, identical to Phase-H behaviour."""
    banner("Scenario 1: horizontal-only (Phase-H reproduction)")
    opt = MCDAAutoscalingOptimizer(profile="balanced")
    r = opt.optimize(
        current_replicas=2, min_replicas=2, max_replicas=4,
        metrics={"cpu_percent": 55, "memory_percent": 50},
        forecast={
            "predicted_cpu": [55, 58, 60, 62, 60, 58],
            "predicted_memory": [50] * 6,
            "cpu_trend": "stable",
        },
    )
    print(f"  best={r.best_alternative} type={r.scaling_type} "
          f"score={r.mcda_score} evaluated={r.alternatives_evaluated}")
    ok = (r.alternatives_evaluated == 3 and r.scaling_type == "hpa")
    print(f"  {'PASS' if ok else 'FAIL'}: expected 3 alternatives, hpa winner")
    return ok


def mixed_pool_runs_topsis():
    """Resources supplied → 18 alternatives (3 horiz + 15 vert). VPA is in the ranking."""
    banner("Scenario 2: mixed pool, moderate load")
    opt = MCDAAutoscalingOptimizer(profile="balanced")
    r = opt.optimize(
        current_replicas=2, min_replicas=2, max_replicas=4,
        metrics={"cpu_percent": 55, "memory_percent": 70},
        forecast={
            "predicted_cpu": [55, 57, 58, 59, 58, 57],
            "predicted_memory": [70, 73, 76, 79, 81, 82],
            "cpu_trend": "stable",
        },
        current_cpu_m=200, current_memory_mi=128,
        enable_vpa=True,
    )
    print(f"  best={r.best_alternative} type={r.scaling_type} "
          f"score={r.mcda_score} evaluated={r.alternatives_evaluated}")
    print(f"  target_replicas={r.target_replicas} "
          f"target_cpu_m={r.target_cpu_m} target_memory_mi={r.target_memory_mi}")
    print(f"  dominance_margin={r.dominance_margin}")
    # 3 horizontal + (4*4 - 1) = 18 vertical-or-horizontal alternatives
    expected = 3 + len(VPA_MULTIPLIER_GRID) ** 2 - 1
    vpa_in_ranking = any(
        r.criteria_scores[name]["scaling_type"] == "vpa"
        for name, _ in r.ranking
    )
    ok = (r.alternatives_evaluated == expected and vpa_in_ranking)
    print(f"  {'PASS' if ok else 'FAIL'}: expected {expected} alternatives "
          f"and at least one vpa entry in the ranking")
    return ok


def llm_vpa_pick_off_grid_is_scored():
    """LLM picks an off-grid VPA target → that exact entry shows up in the ranking with a real score."""
    banner("Scenario 3: LLM picks off-grid VPA, validate")
    opt = MCDAAutoscalingOptimizer(profile="balanced")
    validation = opt.validate_llm_decision(
        llm_action="scale_up",
        llm_target=2,
        current_replicas=2, min_replicas=2, max_replicas=4,
        metrics={"cpu_percent": 55, "memory_percent": 70},
        forecast={
            "predicted_cpu": [55, 57, 58, 59, 58, 57],
            "predicted_memory": [70, 73, 76, 79, 81, 82],
            "cpu_trend": "stable",
        },
        agreement_threshold=0.20,
        llm_scaling_type="vpa",
        llm_target_cpu_m=247, llm_target_memory_mi=192,
        current_cpu_m=200, current_memory_mi=128,
    )
    print(f"  agreement={validation['agreement']} "
          f"should_override={validation['should_override']}")
    print(f"  llm_score={validation['llm_score']} "
          f"mcda_score={validation['mcda_score']} "
          f"gap={validation['score_difference']}")
    print(f"  mcda_pick: type={validation['mcda_scaling_type']} "
          f"reps={validation['mcda_target']} "
          f"cpu={validation['mcda_target_cpu_m']}m "
          f"mem={validation['mcda_target_memory_mi']}Mi")
    # The LLM pick must have been injected and got a non-zero score
    llm_alt_in_ranking = any(
        name.startswith("llm_vpa_cpu247m_mem192Mi")
        for name, _ in validation["mcda_ranking"]
    )
    ok = (
        validation["llm_score"] > 0.0
        and llm_alt_in_ranking
        and validation["agreement"] in {"full", "partial", "disagree"}
    )
    print(f"  {'PASS' if ok else 'FAIL'}: LLM off-grid VPA pick injected and scored")
    return ok


def restart_penalty_actually_penalises():
    """Cranking the restart penalty up should make MCDA prefer horizontal."""
    banner("Scenario 4: restart penalty bites")
    # Need to import after env mutation so the module constant is re-read.
    # We bake in a deliberately small penalty first, then a deliberately huge
    # one, by reimporting the module each time.
    os.environ["MCDA_VPA_RESTART_PENALTY"] = "0.0"
    sys.modules.pop("mcda_optimizer", None)
    import mcda_optimizer as low
    opt_low = low.MCDAAutoscalingOptimizer(profile="balanced")
    r_low = opt_low.optimize(
        current_replicas=2, min_replicas=2, max_replicas=4,
        metrics={"cpu_percent": 30, "memory_percent": 88},
        forecast={
            "predicted_cpu": [30, 32, 33, 33, 32, 31],
            "predicted_memory": [88, 90, 92, 94, 95, 96],
            "cpu_trend": "stable",
        },
        current_cpu_m=200, current_memory_mi=128,
    )

    os.environ["MCDA_VPA_RESTART_PENALTY"] = "0.9"
    sys.modules.pop("mcda_optimizer", None)
    import mcda_optimizer as high
    opt_high = high.MCDAAutoscalingOptimizer(profile="balanced")
    r_high = opt_high.optimize(
        current_replicas=2, min_replicas=2, max_replicas=4,
        metrics={"cpu_percent": 30, "memory_percent": 88},
        forecast={
            "predicted_cpu": [30, 32, 33, 33, 32, 31],
            "predicted_memory": [88, 90, 92, 94, 95, 96],
            "cpu_trend": "stable",
        },
        current_cpu_m=200, current_memory_mi=128,
    )
    print(f"  penalty=0.0  best={r_low.best_alternative} type={r_low.scaling_type}")
    print(f"  penalty=0.9  best={r_high.best_alternative} type={r_high.scaling_type}")
    # Restoring the default for downstream tests
    os.environ["MCDA_VPA_RESTART_PENALTY"] = str(MCDA_VPA_RESTART_PENALTY)
    sys.modules.pop("mcda_optimizer", None)
    # With penalty=0.0 a VPA pick should be at least competitive; with 0.9
    # we expect HPA to take over (the vertical pool is essentially eliminated).
    ok = (r_high.scaling_type == "hpa")
    print(f"  {'PASS' if ok else 'FAIL'}: high restart-penalty pushes winner to HPA")
    return ok


def main():
    print(f"MCDA_COST_EXPONENT       = {MCDA_COST_EXPONENT}")
    print(f"MCDA_VPA_RESTART_PENALTY = {MCDA_VPA_RESTART_PENALTY}")
    print(f"VPA_MULTIPLIER_GRID      = {VPA_MULTIPLIER_GRID}")
    results = [
        horizontal_only_reproduces_phase_h(),
        mixed_pool_runs_topsis(),
        llm_vpa_pick_off_grid_is_scored(),
        restart_penalty_actually_penalises(),
    ]
    banner(f"RESULTS: {sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
