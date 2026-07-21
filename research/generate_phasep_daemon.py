#!/usr/bin/env python3
"""Regenerate fig_phasep_daemon (thesis Figure 5.13).

Two-panel timeline of the Phase P continuous daemon during the v28
AutoSage evaluation block, rebuilt from the per-tick history in
mubench/comparison_results_v28.json (10 trials x 4 ticks = 40 Groq
decisions):

  * top: LLM-picked target_cpu and P.1-actuated request_cpu
    (= LLM x 2.5) per daemon tick, against wall time;
  * bottom: the ten probe-only trial windows T1..T10, showing that the
    daemon's control loop persists across trial boundaries.

Figure fix pass 2026-07-11: the top panel's x tick labels are hidden
(the two panels share one time axis; the between-panel text used to
cover the top panel's "10" tick label).
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
LLM_COLOR = "#E8853D"
ACTUATED_COLOR = "#A63D2F"
BAND_COLOR = "#F2DFD3"
MULTIPLIER = 2.5
CPU_LIMIT_M = 1000


def main() -> None:
    data = json.loads(
        (ROOT / "mubench" / "comparison_results_v28.json").read_text()
    )
    trials = data["results"]["AutoSage"]["trials"]

    t0 = trials[0]["tick_history"][0]["wall_time"]
    times, llm_cpu = [], []
    windows = []  # (start_min, end_min) per trial
    for trial in trials:
        hist = trial["tick_history"]
        start = (hist[0]["wall_time"] - t0) / 60.0
        end = (hist[-1]["wall_time"] - t0) / 60.0
        windows.append((start, end))
        for tick in hist:
            times.append((tick["wall_time"] - t0) / 60.0)
            llm_cpu.append(int(tick["target_cpu"].rstrip("m")))
    actuated = [v * MULTIPLIER for v in llm_cpu]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.8, 4.0), dpi=DPI, sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.28},
    )

    ax_top.plot(
        times, llm_cpu, marker="o", markersize=4.5, linewidth=1.4,
        color=LLM_COLOR, label="LLM picked target_cpu (m)",
    )
    ax_top.plot(
        times, actuated, marker="s", markersize=4.5, linewidth=1.4,
        linestyle="--", color=ACTUATED_COLOR,
        label="P.1 actuated request_cpu (m)  = LLM x 2.5",
    )
    ax_top.axhline(CPU_LIMIT_M, color="black", linestyle=":", linewidth=1.0)
    ax_top.annotate(
        f"MAX {CPU_LIMIT_M}m", xy=(times[-1], CPU_LIMIT_M),
        xytext=(0, 4), textcoords="offset points",
        ha="right", fontsize=8,
    )
    ax_top.set_ylabel("CPU (millicores)", fontsize=10)
    ax_top.set_ylim(0, 1120)
    ax_top.grid(axis="both", linestyle=":", alpha=0.5, linewidth=1.0)
    ax_top.set_axisbelow(True)
    ax_top.tick_params(axis="x", labelbottom=False)  # shared axis: labels below
    ax_top.tick_params(axis="y", labelsize=9)
    ax_top.legend(fontsize=8.5, loc="center right")
    ax_top.set_title(
        "v28 daemon-tick timeline: 40 Groq decisions across 25-min AutoSage block\n"
        "Continuous control loop persists across trial boundaries",
        fontsize=10.5,
    )

    for i, (start, end) in enumerate(windows):
        ax_bot.axvspan(start, end, color=BAND_COLOR)
        ax_bot.text(
            (start + end) / 2, 0.5, f"T{i + 1}",
            ha="center", va="center", fontsize=9,
        )
    ax_bot.set_yticks([])
    ax_bot.set_ylim(0, 1)
    ax_bot.set_title(
        "Trial windows (probe-only; daemon never stops)", fontsize=9.5
    )
    ax_bot.set_xlabel("wall time from daemon start (min)", fontsize=10)
    ax_bot.tick_params(axis="x", labelsize=9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            FIG_DIR / f"fig_phasep_daemon.{ext}",
            format=ext, dpi=DPI, bbox_inches="tight",
        )
    print(f"ticks plotted: {len(times)}, trials: {len(windows)}")
    print(f"written: {FIG_DIR / 'fig_phasep_daemon.[pdf|png]'}")


if __name__ == "__main__":
    main()
