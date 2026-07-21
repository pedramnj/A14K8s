#!/usr/bin/env python3
"""Regenerate fig_dsb_oscillating.pdf — the v42 multi-tier oscillating result.

Reads the committed N=10 (and N=3 pilot) DSB Hotel Reservation oscillating
results and produces a two-panel figure:

  (a) p95 latency by method at N=10 with 95% CIs — the three-way tie.
  (b) native VPA's per-trial p95 for the N=3 pilot vs N=10 — showing that the
      pilot happened to sample one ~43 ms tail trial that inflated its mean,
      an artefact the N=10 sample dissolves.

Figure-style rules (project memory): no text inside the plot areas — values
live in the caption; the legend sits in a reserved band below the axes.

Usage:  python3 research/generate_dsb_oscillating.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

try:
    from scipy import stats

    def t95(n: int) -> float:
        return float(stats.t.ppf(0.975, n - 1))
except Exception:  # scipy optional; fall back to a small-sample t table
    _T = {3: 4.303, 10: 2.262}

    def t95(n: int) -> float:
        return _T.get(n, 1.96)


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "Pedram_Nikjooy___Master_Thesis" / "images" / "figures"
N10 = ROOT / "mubench" / "comparison_results_v42_oscillating_N10.json"
N3 = ROOT / "mubench" / "comparison_results_v42_oscillating_N3.json"

DPI = 300
COLORS = {"HPA": "#7F7F7F", "VPA": "#2F5597", "AutoSage": "#E8853D"}
METHODS = ["HPA", "VPA", "AutoSage"]


def p95_ms(path: Path, method: str) -> np.ndarray:
    d = json.load(open(path))
    vals = d["results"][method]["aggregate"]["p95_latency_s"]["values"]
    return np.asarray(vals, dtype=float) * 1000.0


def ci95(a: np.ndarray) -> float:
    n = len(a)
    return t95(n) * a.std(ddof=1) / np.sqrt(n)


def main() -> None:
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.6, 3.1), dpi=DPI,
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.subplots_adjust(bottom=0.24, wspace=0.34, top=0.9)

    # ---- panel (a): p95 by method, N=10, with 95% CIs ----------------------
    means, errs, colors = [], [], []
    for m in METHODS:
        a = p95_ms(N10, m)
        means.append(a.mean())
        errs.append(ci95(a))
        colors.append(COLORS[m])
    x = np.arange(len(METHODS))
    axa.bar(x, means, width=0.62, color=colors, alpha=0.9,
            yerr=errs, capsize=5, error_kw={"elinewidth": 1.2, "ecolor": "#333"})
    axa.set_xticks(x)
    axa.set_xticklabels(METHODS)
    axa.set_ylabel("$p_{95}$ latency (ms)")
    axa.set_ylim(0, max(m + e for m, e in zip(means, errs)) * 1.25)
    axa.set_title("(a) Tail latency, $N{=}10$", fontsize=10)
    axa.grid(axis="y", linestyle=":", alpha=0.5)
    axa.set_axisbelow(True)

    # ---- panel (b): VPA per-trial p95, N=3 pilot vs N=10 -------------------
    rng = np.random.default_rng(7)
    for xi, path, n in [(0, N3, 3), (1, N10, 10)]:
        a = p95_ms(path, "VPA")
        jitter = (rng.random(len(a)) - 0.5) * 0.22
        axb.scatter(np.full(len(a), xi) + jitter, a, s=34,
                    color=COLORS["VPA"], alpha=0.75, edgecolor="white",
                    linewidth=0.6, zorder=3)
        axb.plot([xi - 0.2, xi + 0.2], [a.mean(), a.mean()],
                 color="#C0392B", linewidth=2.0, zorder=4)
    axb.set_xticks([0, 1])
    axb.set_xticklabels(["$N{=}3$ pilot", "$N{=}10$"])
    axb.set_xlim(-0.5, 1.5)
    axb.set_ylabel("VPA $p_{95}$ (ms)")
    axb.set_title("(b) Why the pilot misled", fontsize=10)
    axb.grid(axis="y", linestyle=":", alpha=0.5)
    axb.set_axisbelow(True)

    # ---- shared legend in a reserved band below the axes ------------------
    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=COLORS["VPA"], markeredgecolor="white",
               markersize=7, label="individual trial"),
        Line2D([0], [0], color="#C0392B", linewidth=2.0, label="mean"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"fig_dsb_oscillating.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
