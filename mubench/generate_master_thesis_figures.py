#!/usr/bin/env python3
"""
Reproducible figure generator for the Master Thesis (`Pedram_Nikjooy___Master_Thesis/images/figures`).

This script focuses on the muBench HPA/VPA/AutoSage comparison figures used in
`content/chapters/chapter5.tex`:
  - fig_comparison_p95.pdf
  - fig_comparison_cost.pdf
  - fig_comparison_sla.pdf
  - fig_shared_legend.pdf

Inputs are one or more `mubench/comparison_results*.json` files produced by
`mubench/run_comparison_eval.py`. Each input represents one load level (e.g., 48c, 96c).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


# =========================
# GLOBAL STYLE (thesis-ready)
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


METHOD_ORDER: List[Tuple[str, str]] = [
    ("HPA", "HPA"),
    ("VPA", "VPA"),
    ("AutoSage", "Ai4k8s"),
]

METHOD_COLORS: Dict[str, str] = {
    "HPA": "#4C78A8",
    "VPA": "#59A14F",
    "AutoSage": "#E15759",
}


@dataclass(frozen=True)
class TrialSummary:
    label: str  # e.g., "48c"
    n_runs: int
    sla_threshold_s: float
    metrics: Dict[str, Dict[str, Optional[float]]]
    # metrics[method] = {"p95_latency_s": mean, "p95_latency_ci": ci, ...}


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ci_from_half_ci(half_ci_95: Optional[float]) -> float:
    try:
        return float(half_ci_95 or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _mean_from_agg(agg: dict) -> Optional[float]:
    m = agg.get("mean")
    if m is None:
        return None
    try:
        return float(m)
    except (TypeError, ValueError):
        return None


def _extract_trial_summary(path: Path, label: Optional[str]) -> TrialSummary:
    doc = _read_json(path)
    cfg = doc.get("config", {}) or {}
    results = doc.get("results", {}) or {}

    wrk_c = cfg.get("wrk_connections")
    n_runs = int(cfg.get("n_runs", 1) or 1)
    sla_threshold_s = float(cfg.get("sla_threshold_s", 2.0) or 2.0)

    if not label:
        label = f"{wrk_c}c" if wrk_c is not None else path.stem

    metrics: Dict[str, Dict[str, Optional[float]]] = {}
    for method_key, _method_label in METHOD_ORDER:
        m = results.get(method_key, {}) or {}
        if m.get("na"):
            metrics[method_key] = {
                "p95_latency_s": None,
                "p95_latency_ci": 0.0,
                "cost_proxy": None,
                "cost_proxy_ci": 0.0,
                "sla_violation_rate": None,
                "sla_violation_rate_ci": 0.0,
            }
            continue

        agg = m.get("aggregate", {}) or {}
        p95 = agg.get("p95_latency_s", {}) or {}
        cost = agg.get("cost_proxy", {}) or {}
        sla = agg.get("sla_violation_rate", {}) or {}

        metrics[method_key] = {
            "p95_latency_s": _mean_from_agg(p95),
            "p95_latency_ci": _ci_from_half_ci(p95.get("half_ci_95")),
            "cost_proxy": _mean_from_agg(cost),
            "cost_proxy_ci": _ci_from_half_ci(cost.get("half_ci_95")),
            "sla_violation_rate": _mean_from_agg(sla),
            "sla_violation_rate_ci": _ci_from_half_ci(sla.get("half_ci_95")),
        }

    return TrialSummary(
        label=str(label),
        n_runs=n_runs,
        sla_threshold_s=sla_threshold_s,
        metrics=metrics,
    )


def _bar_group_positions(n_groups: int, n_bars: int, bar_width: float) -> List[List[float]]:
    x = list(range(n_groups))
    offsets = [(i - (n_bars - 1) / 2.0) * bar_width for i in range(n_bars)]
    return [[xi + offsets[bi] for xi in x] for bi in range(n_bars)]


def _plot_grouped_metric(
    summaries: Sequence[TrialSummary],
    metric_key: str,
    metric_ci_key: str,
    ylabel: str,
    out_path: Path,
    *,
    sla_line: Optional[float] = None,
    y_max: Optional[float] = None,
) -> None:
    group_labels = [s.label for s in summaries]
    n_groups = len(summaries)
    methods = [m for m, _ in METHOD_ORDER]

    bar_w = 0.22 if len(methods) >= 3 else 0.28
    positions = _bar_group_positions(n_groups, len(methods), bar_w)

    fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=DPI)

    for bi, (method_key, method_label) in enumerate(METHOD_ORDER):
        vals: List[float] = []
        errs: List[float] = []
        for s in summaries:
            v = s.metrics.get(method_key, {}).get(metric_key)
            e = s.metrics.get(method_key, {}).get(metric_ci_key)
            vals.append(float(v) if v is not None else float("nan"))
            errs.append(float(e or 0.0))

        ax.bar(
            positions[bi],
            vals,
            width=bar_w,
            label=method_label,
            color=METHOD_COLORS[method_key],
            yerr=errs,
            capsize=4,
            ecolor="black",
            edgecolor="black",
            linewidth=BAR_EDGEWIDTH,
        )

    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
    ax.set_xticks(list(range(n_groups)))
    ax.set_xticklabels(group_labels, fontsize=FONT_SIZE_TICKS)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICKS)
    ax.grid(axis="y", linestyle=GRID_STYLE, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    ax.set_axisbelow(True)

    if y_max is not None:
        ax.set_ylim(0, y_max)

    if sla_line is not None and math.isfinite(sla_line):
        ax.axhline(sla_line, color="red", linestyle="--", linewidth=1.6)

    # No title; captions are in LaTeX.
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_shared_legend(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 0.55), dpi=DPI)
    ax.axis("off")
    handles = []
    labels = []
    for method_key, method_label in METHOD_ORDER:
        h = ax.bar([0], [0], color=METHOD_COLORS[method_key], edgecolor="black", linewidth=BAR_EDGEWIDTH)
        handles.append(h[0])
        labels.append(method_label)
    ax.legend(
        handles,
        labels,
        loc="center",
        ncol=len(labels),
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
        handlelength=1.4,
        columnspacing=1.6,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Master Thesis muBench figures (fig_*.pdf).")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["mubench/comparison_results.json"],
        help="One or more muBench comparison JSON files (each one load level).",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=[],
        help="Optional labels aligned with --inputs (e.g., 48c 96c).",
    )
    parser.add_argument(
        "--out-dir",
        default="Pedram_Nikjooy___Master_Thesis/images/figures",
        help="Output directory for thesis figures.",
    )
    parser.add_argument(
        "--sla-line-s",
        type=float,
        default=2.0,
        help="Draw SLA threshold line (seconds) on p95 plot; set <=0 to disable.",
    )
    args = parser.parse_args()

    in_paths = [Path(p) for p in args.inputs]
    labels: List[Optional[str]] = [None] * len(in_paths)
    if args.labels:
        for i, lab in enumerate(args.labels[: len(in_paths)]):
            labels[i] = lab

    summaries = [_extract_trial_summary(p, labels[i]) for i, p in enumerate(in_paths)]
    out_dir = Path(args.out_dir)

    sla_line = args.sla_line_s if args.sla_line_s and args.sla_line_s > 0 else None

    _plot_grouped_metric(
        summaries,
        metric_key="p95_latency_s",
        metric_ci_key="p95_latency_ci",
        ylabel="p95 latency (s)",
        out_path=out_dir / "fig_comparison_p95.pdf",
        sla_line=sla_line,
    )
    _plot_grouped_metric(
        summaries,
        metric_key="cost_proxy",
        metric_ci_key="cost_proxy_ci",
        ylabel="Cost proxy (avg vCPU)",
        out_path=out_dir / "fig_comparison_cost.pdf",
    )
    _plot_grouped_metric(
        summaries,
        metric_key="sla_violation_rate",
        metric_ci_key="sla_violation_rate_ci",
        ylabel="SLA violation rate",
        out_path=out_dir / "fig_comparison_sla.pdf",
        y_max=1.0,
    )
    _plot_shared_legend(out_dir / "fig_shared_legend.pdf")

    print(f"Saved thesis figures to: {out_dir}")


if __name__ == "__main__":
    main()

