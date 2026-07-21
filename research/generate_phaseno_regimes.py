#!/usr/bin/env python3
"""Regenerate fig_phaseno_regimes (thesis Figure 5.10).

Regime map across the Phase I / N / O sub-regimes on stateful workloads:
grouped log-scale bars of mean p95 latency, native VPA (green) vs
AutoSage (orange), for five evaluations (v9, v10, v23, Phase O
request-only, Phase O.1 fair).

The underlying JSONs (v9/v10/v23/v24) live only on the CrownLabs VM;
the per-method means below are the published values (chapter 5 text:
v23 VPA 0.006 s / AutoSage 1.09 s; Phase O.1 0.68 / 2.06; remaining
values as plotted in the original figure).

Figure fix pass 2026-07-12: error bars and per-bar value labels
removed (whiskers plunged across the whole log axis and covered the
labels); legend kept clear of all chart elements.
"""

from __future__ import annotations

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

GROUPS = [
    "v9\n(easy)",
    "v10\n(easy)",
    "v23\n(handicap)",
    "v24\n(shift,\nreq-only)",
    "v24 corrected\n(shift, fair)",
]
VPA_P95 = [0.006, 0.005, 0.006, 1.08, 0.68]
AUTOSAGE_P95 = [1.03, 1.46, 1.09, 3.99, 2.06]


def main() -> None:
    x = range(len(GROUPS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.8, 4.2), dpi=DPI)
    ax.bar(
        [i - width / 2 for i in x], VPA_P95, width,
        color=VPA_COLOR, edgecolor="black", linewidth=1.2, label="native VPA",
    )
    ax.bar(
        [i + width / 2 for i in x], AUTOSAGE_P95, width,
        color=AUTOSAGE_COLOR, edgecolor="black", linewidth=1.2, label="AutoSage",
    )

    ax.set_yscale("log")
    ax.set_ylabel("p95 latency (s, log scale)", fontsize=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels(GROUPS, fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.6, linewidth=1.2)
    ax.set_axisbelow(True)
    ax.set_title(
        "Native VPA wins stateful latency in every regime\n"
        "(LLM picks VPA 10/10; gap is structural, not observation budget)",
        fontsize=10.5,
    )
    ax.legend(fontsize=9, loc="upper left", framealpha=1.0,
              borderaxespad=1.0)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            FIG_DIR / f"fig_phaseno_regimes.{ext}",
            format=ext, dpi=DPI, bbox_inches="tight",
        )
    print(f"written: {FIG_DIR / 'fig_phaseno_regimes.[pdf|png]'}")


if __name__ == "__main__":
    main()
