import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from simulate_forecast_workloads import SCENARIOS, _gen_stream

# =========================
# GLOBAL STYLE (paper-ready)
# =========================
DPI = 300
FONT_SIZE_LABEL = 10
FONT_SIZE_TICKS = 9
FONT_SIZE_LEGEND = 8.5
GRID_STYLE = ":"
GRID_ALPHA = 0.6
GRID_LINEWIDTH = 1.2
LINEWIDTH = 2.2
BAR_EDGEWIDTH = 1.2


ACTIONS = ["scale_up", "maintain", "scale_down"]
MIN_REPLICAS = 2
MAX_REPLICAS = 8


def oracle_action(cpu_next: float, ram_next: float) -> str:
    if cpu_next > 62.0 or ram_next > 66.0:
        return "scale_up"
    if cpu_next < 42.0 and ram_next < 45.0:
        return "scale_down"
    return "maintain"


def llm_only_action(cpu_obs: float, ram_obs: float, rng: random.Random) -> str:
    # Noisy heuristic proxy for LLM behavior.
    cpu_eff = cpu_obs + rng.gauss(0.0, 6.5)
    ram_eff = ram_obs + rng.gauss(0.0, 5.5)
    if cpu_eff > 58.0 or ram_eff > 62.0:
        return "scale_up"
    if cpu_eff < 40.0 and ram_eff < 43.0:
        return "scale_down"
    return "maintain"


def mcda_only_action(cpu_obs: float, ram_obs: float, replicas: int) -> str:
    risk_score = 0.55 * (cpu_obs / 100.0) + 0.45 * (ram_obs / 100.0)
    cost_pressure = replicas / MAX_REPLICAS
    up_score = 1.4 * risk_score - 0.25 * cost_pressure
    down_score = 0.9 * (1.0 - risk_score) - 0.15 * (1.0 - cost_pressure)
    if up_score > 0.70:
        return "scale_up"
    if down_score > 0.44:
        return "scale_down"
    return "maintain"


def apply_constraints(action: str, replicas: int) -> Tuple[str, bool]:
    if action == "scale_up" and replicas >= MAX_REPLICAS:
        return "maintain", True
    if action == "scale_down" and replicas <= MIN_REPLICAS:
        return "maintain", True
    return action, False


def apply_action(action: str, replicas: int) -> int:
    if action == "scale_up":
        return min(MAX_REPLICAS, replicas + 1)
    if action == "scale_down":
        return max(MIN_REPLICAS, replicas - 1)
    return replicas


def utility_step(action: str, oracle: str, cpu_next: float, ram_next: float, thrash: bool) -> float:
    score = 0.0
    if action == oracle:
        score += 1.0
    else:
        score -= 0.6

    sla_risk = cpu_next > 85.0 or ram_next > 90.0
    if sla_risk and action != "scale_up":
        score -= 1.2
    if not sla_risk and action == "scale_up":
        score -= 0.35

    if thrash:
        score -= 0.5
    return score


def f1_metrics(y_true: List[str], y_pred: List[str]) -> Tuple[float, Dict[str, float], float]:
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    acc = correct / len(y_true) if y_true else 0.0
    per_class = {}
    f1s = []
    for cls in ACTIONS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        per_class[cls] = f1
        f1s.append(f1)
    macro = sum(f1s) / len(f1s) if f1s else 0.0
    return acc, per_class, macro


def ci95(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return 1.96 * math.sqrt(var / len(values))


def maybe_corrupt(cpu: float, ram: float, rng: random.Random, condition: str, last: Tuple[float, float]) -> Tuple[float, float]:
    if condition == "noisy":
        return max(0.0, min(100.0, cpu + rng.gauss(0.0, 8.0))), max(0.0, min(100.0, ram + rng.gauss(0.0, 7.0)))
    if condition == "missing":
        if rng.random() < 0.15:
            return last
    return cpu, ram


def run_one(
    scenario: str,
    seed: int,
    condition: str,
    length: int,
    mcda_agreement_threshold: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    cfg = SCENARIOS[scenario]
    rng = random.Random(seed)

    cpu1, ram1 = _gen_stream(cfg, length, rng, phase=0.0)
    # evaluate on node-1 stream only for control decisions

    methods = ["llm_only", "mcda_only", "llm_mcda", "rule_only"]
    replicas = {m: MIN_REPLICAS for m in methods}
    prev_action = {m: "maintain" for m in methods}
    y_true = {m: [] for m in methods}
    y_pred = {m: [] for m in methods}
    util = {m: 0.0 for m in methods}
    violations = Counter()
    override_rows = []
    step_rows = []

    last_obs = (cpu1[0], ram1[0])
    for t in range(length - 1):
        cpu_raw, ram_raw = cpu1[t], ram1[t]
        cpu_obs, ram_obs = maybe_corrupt(cpu_raw, ram_raw, rng, condition, last_obs)
        last_obs = (cpu_obs, ram_obs)
        oracle = oracle_action(cpu1[t + 1], ram1[t + 1])

        for m in methods:
            if m == "llm_only":
                proposal = llm_only_action(cpu_obs, ram_obs, rng)
                action = proposal
                overridden = False
            elif m == "mcda_only":
                proposal = mcda_only_action(cpu_obs, ram_obs, replicas[m])
                action = proposal
                overridden = False
            elif m == "rule_only":
                if cpu_obs > 63.0 or ram_obs > 67.0:
                    proposal = "scale_up"
                elif cpu_obs < 41.0 and ram_obs < 44.0:
                    proposal = "scale_down"
                else:
                    proposal = "maintain"
                action = proposal
                overridden = False
            else:
                proposal = llm_only_action(cpu_obs, ram_obs, rng)
                mcda = mcda_only_action(cpu_obs, ram_obs, replicas[m])
                # Combine LLM proposal with MCDA + conservative guardrails.
                risk = 0.5 * (cpu_obs / 100.0) + 0.5 * (ram_obs / 100.0)
                disagreement = proposal != mcda
                should_override = disagreement and abs(risk - 0.5) > mcda_agreement_threshold
                action = proposal
                if replicas[m] >= MAX_REPLICAS - 1 and action == "scale_up":
                    action = "maintain"
                elif replicas[m] <= MIN_REPLICAS + 1 and action == "scale_down":
                    action = "maintain"
                if should_override:
                    if risk > 0.68:
                        action = "scale_up"
                    elif risk < 0.40:
                        action = "scale_down"
                    else:
                        action = "maintain"
                overridden = should_override
                if overridden:
                    override_rows.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "condition": condition,
                            "timestep": t,
                            "proposal": proposal,
                            "mcda_target": mcda,
                            "final_action": action,
                            "oracle_action": oracle,
                            "improved_vs_proposal": int(action == oracle and proposal != oracle),
                        }
                    )

            constrained_action, bounds_violation = apply_constraints(action, replicas[m])
            thrash = (
                (prev_action[m] == "scale_up" and constrained_action == "scale_down")
                or (prev_action[m] == "scale_down" and constrained_action == "scale_up")
            )
            if bounds_violation:
                violations[(scenario, seed, condition, m, "bounds")] += 1
            if thrash:
                violations[(scenario, seed, condition, m, "thrash")] += 1

            y_true[m].append(oracle)
            y_pred[m].append(constrained_action)
            util[m] += utility_step(constrained_action, oracle, cpu1[t + 1], ram1[t + 1], thrash)
            replicas[m] = apply_action(constrained_action, replicas[m])
            prev_action[m] = constrained_action

            step_rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "condition": condition,
                    "method": m,
                    "timestep": t,
                    "oracle_action": oracle,
                    "pred_action": constrained_action,
                    "proposal_action": proposal,
                    "replicas": replicas[m],
                    "cpu_obs": round(cpu_obs, 4),
                    "ram_obs": round(ram_obs, 4),
                    "cpu_next": round(cpu1[t + 1], 4),
                    "ram_next": round(ram1[t + 1], 4),
                    "overridden": int(overridden),
                }
            )

    metric_rows = []
    for m in methods:
        acc, per_class, macro = f1_metrics(y_true[m], y_pred[m])
        total_steps = max(len(y_true[m]), 1)
        b = violations[(scenario, seed, condition, m, "bounds")] / total_steps
        th = violations[(scenario, seed, condition, m, "thrash")] / total_steps
        metric_rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "condition": condition,
                "method": m,
                "accuracy": acc,
                "macro_f1": macro,
                "f1_up": per_class["scale_up"],
                "f1_maintain": per_class["maintain"],
                "f1_down": per_class["scale_down"],
                "bounds_violation_rate": b,
                "thrash_rate": th,
                "utility_score": util[m] / total_steps,
            }
        )
    return metric_rows, override_rows + step_rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def build_plots(out_dir: Path, agg_rows: List[Dict[str, object]], override_rows: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10

    # Figure 1: macro-F1 by method (baseline condition only)
    base = [r for r in agg_rows if r["condition"] == "baseline"]
    methods = sorted(set(r["method"] for r in base))
    macro = [next(r for r in base if r["method"] == m)["macro_f1_mean"] for m in methods]
    macro_ci = [next(r for r in base if r["method"] == m)["macro_f1_ci95"] for m in methods]

    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=DPI)
    ax.bar(
        methods,
        macro,
        yerr=macro_ci,
        capsize=4,
        edgecolor="black",
        linewidth=BAR_EDGEWIDTH,
    )
    ax.set_ylabel("Macro-F1", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
    ax.grid(axis="y", linestyle=GRID_STYLE, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    fig.tight_layout()
    fig.savefig(out_dir / "llm_mcda_decision_quality.pdf", format="pdf", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_dir / "llm_mcda_decision_quality.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: violation + utility
    util = [next(r for r in base if r["method"] == m)["utility_score_mean"] for m in methods]
    thr = [next(r for r in base if r["method"] == m)["thrash_rate_mean"] for m in methods]
    bnd = [next(r for r in base if r["method"] == m)["bounds_violation_rate_mean"] for m in methods]

    fig, ax1 = plt.subplots(figsize=(6.4, 3.6), dpi=DPI)
    ax1.plot(methods, util, marker="o", linewidth=LINEWIDTH, label="Utility")
    ax1.set_ylabel("Utility score", fontsize=FONT_SIZE_LABEL)
    ax1.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
    ax1.grid(axis="y", linestyle=GRID_STYLE, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    ax2 = ax1.twinx()
    ax2.plot(methods, thr, marker="s", linewidth=LINEWIDTH, label="Thrash rate")
    ax2.plot(methods, bnd, marker="^", linewidth=LINEWIDTH, label="Bounds violation rate")
    ax2.set_ylabel("Violation rate", fontsize=FONT_SIZE_LABEL)
    ax2.tick_params(axis="y", labelsize=FONT_SIZE_TICKS)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=FONT_SIZE_LEGEND, frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "llm_mcda_violation_utility.pdf", format="pdf", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_dir / "llm_mcda_violation_utility.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # override analysis summary figure not required; keep csv.


def main() -> None:
    parser = argparse.ArgumentParser(description="Decision-quality benchmark for LLM+MCDA defense")
    parser.add_argument("--scenarios", default="stable_api,bursty_web,growing_service")
    parser.add_argument("--seeds", default="101,102,103,104,105")
    parser.add_argument("--length", type=int, default=240)
    parser.add_argument("--mcda-agreement-threshold", type=float, default=0.15)
    parser.add_argument("--out-dir", default="thesis_reports")
    args = parser.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    conditions = ["baseline", "noisy", "missing"]

    all_metric_rows: List[Dict[str, object]] = []
    all_trace_rows: List[Dict[str, object]] = []
    override_only: List[Dict[str, object]] = []

    for scenario in scenarios:
        for seed in seeds:
            for cond in conditions:
                metric_rows, trace_rows = run_one(
                    scenario=scenario,
                    seed=seed,
                    condition=cond,
                    length=args.length,
                    mcda_agreement_threshold=args.mcda_agreement_threshold,
                )
                all_metric_rows.extend(metric_rows)
                all_trace_rows.extend(trace_rows)
                override_only.extend([r for r in trace_rows if "improved_vs_proposal" in r])

    out_dir = Path(args.out_dir)
    raw_path = out_dir / "llm_mcda_decision_quality_raw.csv"
    write_csv(
        raw_path,
        all_metric_rows,
        [
            "scenario",
            "seed",
            "condition",
            "method",
            "accuracy",
            "macro_f1",
            "f1_up",
            "f1_maintain",
            "f1_down",
            "bounds_violation_rate",
            "thrash_rate",
            "utility_score",
        ],
    )

    # Aggregate mean ± CI by method and condition.
    grouped: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in all_metric_rows:
        key = (str(r["method"]), str(r["condition"]))
        for k in [
            "accuracy",
            "macro_f1",
            "f1_up",
            "f1_maintain",
            "f1_down",
            "bounds_violation_rate",
            "thrash_rate",
            "utility_score",
        ]:
            grouped[key][k].append(float(r[k]))

    agg_rows = []
    for (method, condition), vals in sorted(grouped.items()):
        row = {"method": method, "condition": condition, "runs": len(vals["accuracy"])}
        for k, arr in vals.items():
            row[f"{k}_mean"] = sum(arr) / len(arr)
            row[f"{k}_ci95"] = ci95(arr)
        agg_rows.append(row)

    agg_path = out_dir / "llm_mcda_decision_quality_agg.csv"
    agg_fields = ["method", "condition", "runs"]
    metric_names = [
        "accuracy",
        "macro_f1",
        "f1_up",
        "f1_maintain",
        "f1_down",
        "bounds_violation_rate",
        "thrash_rate",
        "utility_score",
    ]
    for m in metric_names:
        agg_fields += [f"{m}_mean", f"{m}_ci95"]
    write_csv(agg_path, agg_rows, agg_fields)

    # Override analysis
    ov_group = defaultdict(lambda: {"count": 0, "improved": 0})
    for r in override_only:
        key = (r["scenario"], r["condition"])
        ov_group[key]["count"] += 1
        ov_group[key]["improved"] += int(r["improved_vs_proposal"])
    ov_rows = []
    for (scenario, cond), x in sorted(ov_group.items()):
        improved_rate = (x["improved"] / x["count"]) if x["count"] else 0.0
        ov_rows.append(
            {
                "scenario": scenario,
                "condition": cond,
                "overrides": x["count"],
                "improved_overrides": x["improved"],
                "improved_rate": round(improved_rate, 6),
            }
        )
    ov_path = out_dir / "llm_mcda_override_analysis.csv"
    write_csv(ov_path, ov_rows, ["scenario", "condition", "overrides", "improved_overrides", "improved_rate"])

    # Figures
    build_plots(out_dir, agg_rows, ov_rows)

    print(f"Saved raw decision quality: {raw_path}")
    print(f"Saved aggregated decision quality: {agg_path}")
    print(f"Saved override analysis: {ov_path}")
    print(
        "Saved figures: "
        f"{out_dir / 'llm_mcda_decision_quality.pdf'}, "
        f"{out_dir / 'llm_mcda_violation_utility.pdf'}"
    )


if __name__ == "__main__":
    main()
