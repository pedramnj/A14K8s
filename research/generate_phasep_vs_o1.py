#!/usr/bin/env python3
"""Regenerate fig_phasep_vs_o1 (thesis Figure 5.12).

Two-panel grouped bar chart comparing Phase O.1 (v24, grey) against
Phase P (v28, orange) for all four methods on the same
session-compute + AWARE-shift workload:

  * left:  mean p95 latency, with the 0.5 s SLA threshold line;
  * right: mean SLA@0.5s violation rate.

v28 values are computed from mubench/comparison_results_v28.json.
Phase O.1 (v24) raw JSON lives only on the CrownLabs VM; its per-method
trial means (consistent with fig_phaseno_o1 and the tab:phase-p-closure
deltas) are inlined as constants.

Figure fix pass 2026-07-12: error bars and the two delta annotations
("AutoSage -0.86s (-42%)", "AutoSage -22.3pp") removed -- they covered
bars and were unreadable; the SLA-line label moved inside the axes
(it used to straddle the right border). Per-bar value labels added.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "Pedram_Nikjooy___Master_Thesis" / "images" / "figures"

DPI = 300
O1_COLOR = "#B5B5B5"
V28_COLOR = "#E8853D"
SLA_S = 0.5

METHODS = ["HPA", "VPA", "AutoSage", "AutoScaleAI"]

# Phase O.1 (v24) trial means; raw JSON only on the CrownLabs VM.
O1_P95 = {"HPA": 4.40, "VPA": 0.68, "AutoSage": 2.06, "AutoScaleAI": 2.65}
O1_SLA = {"HPA": 72.0, "VPA": 18.2, "AutoSage": 61.0, "AutoScaleAI": 86.5}


def _means(path: Path) -> tuple[dict, dict]:
    data = json.loads(path.read_text())
    p95, sla = {}, {}
    for method in METHODS:
        trials = data["results"][method]["trials"]
        pv = [t["p95_latency_s"] for t in trials if t.get("p95_latency_s") is not None]
        sv = [t["sla_violation_rate"] for t in trials if t.get("sla_violation_rate") is not None]
        p95[method] = sum(pv) / len(pv)
        sla[method] = 100.0 * sum(sv) / len(sv)
    return p95, sla


def _bar_labels(ax, bars, fmt: str) -> None:
    for bar in bars:
        ax.annotate(
            fmt.format(bar.get_height()),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 2), textcoords="offset points",
            ha="center", va="bottom", fontsize=7.5,
        )


def main() -> None:
    v28_p95, v28_sla = _means(ROOT / "mubench" / "comparison_results_v28.json")

    x = range(len(METHODS))
    width = 0.36
    xl = [i - width / 2 for i in x]
    xr = [i + width / 2 for i in x]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.1), dpi=DPI)

    # left: p95 latency
    b1 = ax1.bar(xl, [O1_P95[m] for m in METHODS], width, color=O1_COLOR,
                 edgecolor="black", linewidth=1.0,
                 label="v24 corrected (per-trial, no multiplier)")
    b2 = ax1.bar(xr, [v28_p95[m] for m in METHODS], width, color=V28_COLOR,
                 edgecolor="black", linewidth=1.0,
                 label="v28 (daemon + multiplier + Fix A)")
    _bar_labels(ax1, b1, "{:.2f}")
    _bar_labels(ax1, b2, "{:.2f}")
    ax1.axhline(SLA_S, color="red", linestyle="--", linewidth=1.0)
    ax1.text(1.5, SLA_S + 0.08, f"SLA {SLA_S}s", color="red",
             ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("p95 latency (s)", fontsize=10)
    ax1.set_title("p95 latency (lower is better)", fontsize=10.5)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(METHODS, fontsize=9, rotation=12)
    ax1.tick_params(axis="y", labelsize=9)
    ax1.grid(axis="y", linestyle=":", alpha=0.5, linewidth=1.0)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=8, loc="upper right")

    # right: SLA violation rate
    b3 = ax2.bar(xl, [O1_SLA[m] for m in METHODS], width, color=O1_COLOR,
                 edgecolor="black", linewidth=1.0, label="v24 corrected")
    b4 = ax2.bar(xr, [v28_sla[m] for m in METHODS], width, color=V28_COLOR,
                 edgecolor="black", linewidth=1.0, label="v28")
    _bar_labels(ax2, b3, "{:.1f}")
    _bar_labels(ax2, b4, "{:.1f}")
    ax2.set_ylabel("SLA violation rate (%)", fontsize=10)
    ax2.set_title("SLA @ 0.5s violation rate (lower is better)", fontsize=10.5)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(METHODS, fontsize=9, rotation=12)
    ax2.tick_params(axis="y", labelsize=9)
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", linestyle=":", alpha=0.5, linewidth=1.0)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Same workload, two architectures: v24 (corrected) vs v28\n"
        "WRK_SHIFT 50->100 @ T+60s, SLA 0.5s, VPA handicapped 30s, N=10",
        fontsize=10.5, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("pdf", "png"):
        fig.savefig(
            FIG_DIR / f"fig_phasep_vs_o1.{ext}",
            format=ext, dpi=DPI, bbox_inches="tight",
        )
    print("v28 p95:", {m: round(v28_p95[m], 3) for m in METHODS})
    print("v28 SLA%:", {m: round(v28_sla[m], 1) for m in METHODS})
    ratio_o1 = O1_P95["AutoSage"] / O1_P95["VPA"]
    ratio_v28 = v28_p95["AutoSage"] / v28_p95["VPA"]
    print(f"AutoSage-to-VPA ratio: O.1 {ratio_o1:.2f}x -> v28 {ratio_v28:.2f}x")
    print(f"written: {FIG_DIR / 'fig_phasep_vs_o1.[pdf|png]'}")


if __name__ == "__main__":
    main()
