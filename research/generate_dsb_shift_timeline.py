#!/usr/bin/env python3
"""Generate fig_dsb_shift_timeline (thesis chapter 4, section 4.7.6).

Illustrates the DSB oscillating-regime schedule (v42 configuration):
DSB_SHIFT_PHASES alternates the mongo-cold heavy-search.lua and the
memcached-warm light-search.lua every 60 s across a 600 s trial
(five heavy bursts), while AutoSage's advisor ticks every 30 s
(15 ticks), the harness polls the VPA recommender for up to 300 s,
and the latency probe fires at T+450 s.

Pure-configuration figure: the schedule is the v42 harness config,
not measured data.

No text is drawn inside the plot area (figure fix pass 2026-07-12):
every element is identified in the legend below the chart.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "Pedram_Nikjooy___Master_Thesis" / "images" / "figures"

DPI = 300
HEAVY_COLOR = "#E8853D"
LIGHT_COLOR = "#BDD7EE"
TICK_COLOR = "#2F5597"

TRIAL_S = 600
PHASE_S = 60          # heavy:60,light:60 x5
TICK_INTERVAL_S = 30
N_TICKS = 15
VPA_POLL_WINDOW_S = 300
PROBE_AT_S = 450


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 2.7), dpi=DPI)

    # alternating workload-script bands (heavy first)
    for start in range(0, TRIAL_S, 2 * PHASE_S):
        ax.axvspan(start, start + PHASE_S, ymin=0.0, ymax=0.62,
                   color=HEAVY_COLOR, alpha=0.85)
        ax.axvspan(start + PHASE_S, start + 2 * PHASE_S, ymin=0.0, ymax=0.62,
                   color=LIGHT_COLOR, alpha=0.9)

    # square wave outlining the script level
    xs, ys = [0], [1]
    for start in range(0, TRIAL_S, PHASE_S):
        level = 1 if (start // PHASE_S) % 2 == 0 else 0
        xs += [start, start + PHASE_S]
        ys += [level, level]
    ax.step(xs[1:], ys[1:], where="post", color="black", linewidth=1.3)

    # AutoSage advisor ticks (30 s cadence, 15 ticks)
    tick_times = [TICK_INTERVAL_S * (i + 1) for i in range(N_TICKS)]
    ax.plot(tick_times, [1.45] * len(tick_times), linestyle="none",
            marker="v", markersize=5, color=TICK_COLOR,
            markeredgecolor="black", markeredgewidth=0.5)

    # VPA recommendation poll window
    ax.annotate("", xy=(0, 1.85), xytext=(VPA_POLL_WINDOW_S, 1.85),
                arrowprops=dict(arrowstyle="<->", color="#59A14F", lw=1.6))

    # latency probe
    ax.axvline(PROBE_AT_S, color="darkred", linestyle="--", linewidth=1.2)

    ax.set_xlim(0, TRIAL_S)
    ax.set_ylim(-0.15, 2.1)
    ax.set_xticks(range(0, TRIAL_S + 1, 60))
    ax.set_yticks([])
    ax.set_xlabel("trial time (s)", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    for spine in ("left", "top", "right"):
        ax.spines[spine].set_visible(False)

    # all element descriptions live in the legend below the chart --
    # no text inside the plot area
    handles = [
        Patch(facecolor=HEAVY_COLOR, edgecolor="black", linewidth=0.5,
              label="heavy-search.lua (mongo-cold)"),
        Patch(facecolor=LIGHT_COLOR, edgecolor="black", linewidth=0.5,
              label="light-search.lua (memcached-warm)"),
        Line2D([], [], linestyle="none", marker="v", markersize=5,
               color=TICK_COLOR, markeredgecolor="black",
               markeredgewidth=0.5,
               label=f"AutoSage advisor tick (every {TICK_INTERVAL_S} s, "
                     f"{N_TICKS} per trial)"),
        Line2D([], [], color="#59A14F", linewidth=1.6,
               label=f"VPA recommendation poll window "
                     f"(first {VPA_POLL_WINDOW_S} s)"),
        Line2D([], [], color="darkred", linestyle="--", linewidth=1.2,
               label=f"latency probe (T+{PROBE_AT_S} s)"),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=True, framealpha=1.0,
              fancybox=True, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.30))

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_dsb_shift_timeline.{ext}",
                    format=ext, dpi=DPI, bbox_inches="tight")
    print(f"written: {FIG_DIR / 'fig_dsb_shift_timeline.[pdf|png]'}")


if __name__ == "__main__":
    main()
