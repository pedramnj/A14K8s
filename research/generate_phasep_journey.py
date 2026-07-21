#!/usr/bin/env python3
"""Regenerate fig_phasep_journey (thesis Figure 5.14).

Bar chart of mean p95 latency, native VPA vs AutoSage, across the three
Phase O.1 -> Phase P stages on the stateful session-compute + AWARE-shift
workload:

  * Phase O.1 (per-trial AutoSage, mult=1)      -- v24
  * Phase P   (continuous daemon, mult=1)       -- v25
  * Phase P+P.1 (daemon + request multiplier)   -- v26

v25/v26 means are computed from mubench/comparison_results_v{25,26}.json.
The v24 raw JSON lives only on the CrownLabs VM; its published means
(thesis section 5.7, Phase O.1 table) are inlined as constants.

No error bars by design: the CI whiskers overlapped the value labels
(figure fix pass, 2026-07-11).
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
VPA_COLOR = "#59A14F"
AUTOSAGE_COLOR = "#E8853D"

# Phase O.1 (v24) published means; raw JSON only on the CrownLabs VM.
V24_VPA_P95 = 0.68
V24_AUTOSAGE_P95 = 2.06


def _mean_p95(path: Path, method: str) -> float:
    data = json.loads(path.read_text())
    vals = [
        t.get("p95_latency_s")
        for t in data["results"][method]["trials"]
        if t.get("p95_latency_s") is not None
    ]
    return sum(vals) / len(vals)


def main() -> None:
    v25 = ROOT / "mubench" / "comparison_results_v25.json"
    v26 = ROOT / "mubench" / "comparison_results_v26.json"

    vpa = [V24_VPA_P95, _mean_p95(v25, "VPA"), _mean_p95(v26, "VPA")]
    autosage = [
        V24_AUTOSAGE_P95,
        _mean_p95(v25, "AutoSage"),
        _mean_p95(v26, "AutoSage"),
    ]

    groups = [
        "v24 corrected\n(per-trial,\nmult=1)",
        "v25\n(daemon,\nmult=1)",
        "v26\n(daemon + multiplier,\nmult=2.5)",
    ]

    x = range(len(groups))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=DPI)
    bars_vpa = ax.bar(
        [i - width / 2 for i in x], vpa, width,
        color=VPA_COLOR, edgecolor="black", linewidth=1.2, label="native VPA",
    )
    bars_as = ax.bar(
        [i + width / 2 for i in x], autosage, width,
        color=AUTOSAGE_COLOR, edgecolor="black", linewidth=1.2, label="AutoSage",
    )

    ax.set_yscale("log")
    ax.set_ylabel("p95 latency (s, log scale)", fontsize=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups, fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.6, linewidth=1.2)
    ax.set_axisbelow(True)

    for bar, val in zip(bars_vpa, vpa):
        ax.annotate(
            f"{val:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
        )
    for bar, val, ref in zip(bars_as, autosage, vpa):
        ax.annotate(
            f"{val:.2f}\n({val / ref:.1f}$\\times$ VPA)",
            xy=(bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylim(top=ax.get_ylim()[1] * 1.6)
    ax.set_title(
        "AutoSage p95 vs native VPA, stateful-compute + AWARE shift\n"
        "the daemon closed half the gap; the request multiplier closed most of the rest",
        fontsize=10.5,
    )
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            FIG_DIR / f"fig_phasep_journey.{ext}",
            format=ext, dpi=DPI, bbox_inches="tight",
        )
    print(f"VPA means:      {[round(v, 4) for v in vpa]}")
    print(f"AutoSage means: {[round(v, 4) for v in autosage]}")
    print(f"written: {FIG_DIR / 'fig_phasep_journey.[pdf|png]'}")


if __name__ == "__main__":
    main()
