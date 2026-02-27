#!/usr/bin/env python3
"""
Generate thesis-ready evaluation figures from autoscaling comparison JSON.

Supports both single-run and aggregated multi-run outputs.
Exports static vector PDFs suitable for LaTeX.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

# =========================
# GLOBAL STYLE (paper-ready)
# =========================
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"

DPI = 300
FONT_SIZE_LABEL = 10
FONT_SIZE_TICKS = 9
FONT_SIZE_LEGEND = 8.5
GRID_STYLE = ":"
GRID_ALPHA = 0.6
GRID_LINEWIDTH = 1.2
BAR_EDGEWIDTH = 1.2


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _method_order() -> List[Tuple[str, str]]:
    return [
        ("native_hpa", "Native HPA"),
        ("native_vpa", "Native VPA"),
        ("autosage", "AutoSage"),
    ]


def _method_colors() -> Dict[str, str]:
    return {
        "native_hpa": "#4C78A8",
        "native_vpa": "#59A14F",
        "autosage": "#E15759",
    }


def _value(data: Dict[str, Any], key: str, fallback: float = 0.0) -> float:
    val = data.get(key)
    if val is None:
        return fallback
    try:
        return float(val)
    except (TypeError, ValueError):
        return fallback


def _aggregate_metric(
    results: Dict[str, Any], method_key: str, metric_key: str
) -> Tuple[Optional[float], float]:
    aggregated = results.get("aggregated", {}).get(method_key, {}).get("metrics", {})
    metric = aggregated.get(metric_key, {})
    mean = metric.get("mean")
    margin = metric.get("margin_error")
    if mean is not None:
        return float(mean), float(margin or 0.0)

    # Legacy fallback: single-run format.
    row = results.get(method_key, {})
    if method_key == "native_hpa" and metric_key == "decision_reaction_latency_s":
        return _value(row, "first_scale_up_latency_s"), 0.0
    if method_key == "native_vpa" and metric_key == "decision_reaction_latency_s":
        return _value(row, "first_recommendation_latency_s"), 0.0
    if method_key == "autosage" and metric_key == "decision_reaction_latency_s":
        return _value(row, "recommendation_latency_s"), 0.0
    if metric_key == "p95_latency_s":
        return _value(row.get("latency_sla", {}), "p95_s"), 0.0
    if metric_key == "sla_violation_rate_pct":
        return _value(row.get("latency_sla", {}), "sla_violation_rate_pct"), 0.0
    if metric_key == "cost_proxy_avg_requested_vcpu":
        return _value(row.get("cost_proxy", {}), "avg_requested_vcpu"), 0.0
    return None, 0.0


def _plot_decision_timing(results: Dict[str, Any], out_dir: Path) -> None:
    labels: List[str] = []
    values: List[float] = []
    errors: List[float] = []
    colors: List[str] = []

    for method_key, label in _method_order():
        val, err = _aggregate_metric(results, method_key, "decision_reaction_latency_s")
        labels.append(label)
        values.append(val if val is not None else 0.0)
        errors.append(err)
        colors.append(_method_colors()[method_key])

    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=DPI)
    ax.bar(
        labels,
        values,
        color=colors,
        yerr=errors,
        capsize=4,
        ecolor="black",
        edgecolor="black",
        linewidth=BAR_EDGEWIDTH,
    )
    ax.set_ylabel("Decision/Reaction Time (s)", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
    ax.grid(axis="y", linestyle=GRID_STYLE, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / "evaluation_decision_latency.pdf", format="pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_latency_and_sla(results: Dict[str, Any], out_dir: Path) -> None:
    labels: List[str] = []
    p95_vals: List[float] = []
    p95_errs: List[float] = []
    sla_vals: List[float] = []
    sla_errs: List[float] = []
    colors: List[str] = []

    for method_key, label in _method_order():
        p95, p95_err = _aggregate_metric(results, method_key, "p95_latency_s")
        sla, sla_err = _aggregate_metric(results, method_key, "sla_violation_rate_pct")
        labels.append(label)
        p95_vals.append(p95 if p95 is not None else 0.0)
        p95_errs.append(p95_err)
        sla_vals.append(sla if sla is not None else 0.0)
        sla_errs.append(sla_err)
        colors.append(_method_colors()[method_key])

    x = list(range(len(labels)))
    w = 0.35

    fig, ax1 = plt.subplots(figsize=(6.8, 3.6), dpi=DPI)
    b1 = ax1.bar(
        [i - w / 2 for i in x],
        p95_vals,
        width=w,
        color=colors,
        alpha=0.95,
        yerr=p95_errs,
        capsize=4,
        ecolor="black",
        label="p95 latency (s)",
        edgecolor="black",
        linewidth=BAR_EDGEWIDTH,
    )
    ax1.set_ylabel("p95 Latency (s)", fontsize=FONT_SIZE_LABEL)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=FONT_SIZE_TICKS)
    ax1.tick_params(axis="y", labelsize=FONT_SIZE_TICKS)
    ax1.grid(axis="y", linestyle=GRID_STYLE, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    b2 = ax2.bar(
        [i + w / 2 for i in x],
        sla_vals,
        width=w,
        color="#B07AA1",
        alpha=0.60,
        yerr=sla_errs,
        capsize=4,
        ecolor="black",
        label="SLA violations (%)",
        edgecolor="black",
        linewidth=BAR_EDGEWIDTH,
    )
    ax2.set_ylabel("SLA Violations (%)", fontsize=FONT_SIZE_LABEL)
    ax2.tick_params(axis="y", labelsize=FONT_SIZE_TICKS)

    handles = [b1, b2]
    labels_legend = [h.get_label() for h in handles]
    ax1.legend(handles, labels_legend, loc="upper left", frameon=True, fontsize=FONT_SIZE_LEGEND)
    fig.tight_layout()
    fig.savefig(out_dir / "evaluation_service_levels.pdf", format="pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_cost_proxy(results: Dict[str, Any], out_dir: Path) -> None:
    labels: List[str] = []
    values: List[float] = []
    errors: List[float] = []
    colors: List[str] = []

    for method_key, label in _method_order():
        val, err = _aggregate_metric(results, method_key, "cost_proxy_avg_requested_vcpu")
        labels.append(label)
        values.append(val if val is not None else 0.0)
        errors.append(err)
        colors.append(_method_colors()[method_key])

    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=DPI)
    ax.bar(
        labels,
        values,
        color=colors,
        yerr=errors,
        capsize=4,
        ecolor="black",
        edgecolor="black",
        linewidth=BAR_EDGEWIDTH,
    )
    ax.set_ylabel("Average Requested vCPU", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
    ax.grid(axis="y", linestyle=GRID_STYLE, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / "evaluation_cost_proxy.pdf", format="pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_autosage_decomposition(results: Dict[str, Any], out_dir: Path) -> None:
    stage_keys = [
        ("metrics_collection_s", "Metrics"),
        ("forecast_s", "Forecast"),
        ("llm_inference_s", "LLM"),
        ("mcda_validation_s", "MCDA"),
        ("actuation_s", "Actuation"),
    ]
    labels: List[str] = []
    values: List[float] = []
    errors: List[float] = []

    for key, label in stage_keys:
        mean, err = _aggregate_metric(results, "autosage", key)
        labels.append(label)
        values.append(mean if mean is not None else 0.0)
        errors.append(err)

    # Plot in milliseconds and include a zoom panel for tiny stages.
    values_ms = [v * 1000.0 for v in values]
    errors_ms = [e * 1000.0 for e in errors]

    fig, (ax_main, ax_zoom) = plt.subplots(
        2,
        1,
        figsize=(6.8, 4.8),
        dpi=DPI,
        gridspec_kw={"height_ratios": [3.0, 1.35], "hspace": 0.22},
        constrained_layout=True,
    )

    ax_main.bar(
        labels,
        values_ms,
        color="#E15759",
        yerr=errors_ms,
        capsize=4,
        ecolor="black",
        alpha=0.9,
        edgecolor="black",
        linewidth=BAR_EDGEWIDTH,
    )
    ax_main.set_ylabel("Latency (ms)", fontsize=FONT_SIZE_LABEL)
    ax_main.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
    ax_main.grid(axis="y", linestyle=GRID_STYLE, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    ax_main.set_axisbelow(True)
    ax_main.set_xticklabels([])

    ax_zoom.bar(
        labels,
        values_ms,
        color="#E15759",
        yerr=errors_ms,
        capsize=4,
        ecolor="black",
        alpha=0.9,
        edgecolor="black",
        linewidth=BAR_EDGEWIDTH,
    )
    # Focus small-scale panel so forecast/MCDA are visible.
    ax_zoom.set_ylim(0, 20)
    ax_zoom.set_ylabel("Zoom (ms)", fontsize=FONT_SIZE_LABEL)
    ax_zoom.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
    ax_zoom.grid(axis="y", linestyle=GRID_STYLE, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    ax_zoom.set_axisbelow(True)
    fig.savefig(out_dir / "evaluation_autosage_decomposition.pdf", format="pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis evaluation plots from comparison JSON")
    parser.add_argument(
        "--input",
        default="thesis_reports/hpa_vpa_comparison_with_sla_cost_v2.json",
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "Agentic_AI_for_Automatic_Kubernetes_Operations_and_Autoscaling___CAIS2026/"
            "figures/evaluation"
        ),
        help="Directory for output PDF figures",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = _read_json(input_path)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": FONT_SIZE_LABEL,
            "xtick.labelsize": FONT_SIZE_TICKS,
            "ytick.labelsize": FONT_SIZE_TICKS,
            "legend.fontsize": FONT_SIZE_LEGEND,
        }
    )

    _plot_decision_timing(results, output_dir)
    _plot_latency_and_sla(results, output_dir)
    _plot_cost_proxy(results, output_dir)
    _plot_autosage_decomposition(results, output_dir)

    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
