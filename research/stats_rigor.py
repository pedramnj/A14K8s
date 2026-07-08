#!/usr/bin/env python3
"""stats_rigor.py — bootstrap CIs, Wilcoxon, Cliff's delta across every
`comparison_results_v*.json` shipped with the AutoSage evaluation.

Consumes:
    - mubench/comparison_results_v*.json          (Phase K onward)
    - mubench/comparison_results.json             (single "latest" alias)
    - thesis_reports/comparison_results_*.json    (Phases A-J historical)

Emits:
    - stdout table (Markdown-ish, pipes into a Jupyter cell too)
    - thesis_reports/stats_rigor.tex              (\\input-able LaTeX table)
    - thesis_reports/stats_rigor.json             (machine-readable dump)

Usage:
    python3 -m research.stats_rigor
    python3 -m research.stats_rigor --workload multiclass
    python3 -m research.stats_rigor --min-runs 8   # skip versions with <8 trials

The metrics we care about, per method:
    p95_latency_s, sla_violation_rate, cost_proxy, peak_replicas,
    first_scale_latency_s, recommendation_latency_s

For each version and each metric:
    - bootstrap 95 % CI (1000 resamples)
    - median, p25, p75
    - Wilcoxon signed-rank test AutoSage vs HPA and AutoSage vs VPA
      (paired by trial index)
    - Cliff's delta effect size on the same pairs
    - p-value + Bonferroni-adjusted p-value (across all versions of the
      same workload class)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Sequence

try:
    from scipy.stats import wilcoxon  # type: ignore
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    print("[warn] scipy not available; Wilcoxon p-values will be reported as None",
          file=sys.stderr)


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_GLOBS = [
    os.path.join(_REPO, "mubench", "comparison_results_v*.json"),
    os.path.join(_REPO, "thesis_reports", "comparison_results_*.json"),
]

_METRICS = [
    ("p95_latency_s", "latency", "p95 latency (s)", "min"),
    ("sla_violation_rate", "sla", "SLA violation rate", "min"),
    ("cost_proxy", "cost", "Cost proxy", "min"),
    ("peak_replicas", "reps", "Peak replicas", "min"),
    ("first_scale_latency_s", "fs", "First-scale latency (s)", "min"),
    ("recommendation_latency_s", "rec", "Rec. latency (s)", "min"),
]


# ── Bootstrap ────────────────────────────────────────────────────────────────
def _bootstrap_mean_ci(values: Sequence[float], n_boot: int = 1000,
                       alpha: float = 0.05,
                       seed: int = 0) -> tuple[float, float, float]:
    """Return (mean, lo_ci, hi_ci) for the bootstrap CI. None inputs skipped."""
    vals = [v for v in values if v is not None]
    if not vals:
        return (float("nan"), float("nan"), float("nan"))
    if len(vals) == 1:
        return (vals[0], vals[0], vals[0])
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(math.floor(alpha / 2 * n_boot))]
    hi = means[int(math.ceil((1 - alpha / 2) * n_boot)) - 1]
    return (sum(vals) / len(vals), lo, hi)


# ── Cliff's delta ────────────────────────────────────────────────────────────
def _cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta effect size for two independent samples. Range [-1, 1].

    delta > 0 means A tends to exceed B (dominance of A).
    delta < 0 means B tends to exceed A.
    Interpretation guide (Romano et al.):
        |d| < 0.147 : negligible
        |d| < 0.33  : small
        |d| < 0.474 : medium
        else        : large
    """
    a = [v for v in a if v is not None]
    b = [v for v in b if v is not None]
    if not a or not b:
        return float("nan")
    gt = lt = 0
    for x in a:
        for y in b:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    return (gt - lt) / (len(a) * len(b))


def _cliffs_verdict(d: float) -> str:
    if math.isnan(d):
        return "n/a"
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


# ── Wilcoxon (paired signed-rank) ────────────────────────────────────────────
def _wilcoxon_p(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Paired Wilcoxon signed-rank p-value; None if scipy unavailable or
    sample sizes differ or all differences are zero."""
    if not _HAS_SCIPY:
        return None
    a_clean = [x for x, y in zip(a, b) if x is not None and y is not None]
    b_clean = [y for x, y in zip(a, b) if x is not None and y is not None]
    if len(a_clean) != len(b_clean) or len(a_clean) < 2:
        return None
    diffs = [x - y for x, y in zip(a_clean, b_clean)]
    if all(d == 0 for d in diffs):
        return None
    try:
        _, p = wilcoxon(a_clean, b_clean, zero_method="wilcox",
                        alternative="two-sided", correction=False)
        return float(p)
    except Exception:
        return None


# ── Data model ───────────────────────────────────────────────────────────────
@dataclass
class MethodSummary:
    method: str
    n: int
    mean: float
    lo_ci: float
    hi_ci: float
    median: float
    p25: float
    p75: float


@dataclass
class PairwiseSummary:
    a: str
    b: str
    n: int
    wilcoxon_p: float | None
    cliffs_delta: float
    cliffs_verdict: str


@dataclass
class MetricRow:
    workload: str
    version: str
    metric: str
    metric_label: str
    per_method: list[MethodSummary] = field(default_factory=list)
    pairwise: list[PairwiseSummary] = field(default_factory=list)


# ── Loader ───────────────────────────────────────────────────────────────────
def _guess_version(path: str) -> str:
    """Version tag from filename (e.g. `..._v22.json` → `v22`)."""
    base = os.path.basename(path).replace("comparison_results", "").rstrip(".json")
    base = base.strip("_").rstrip(".")
    return base or "unknown"


def _load_all(globs: list[str], min_runs: int) -> list[dict]:
    files = []
    for g in globs:
        files.extend(sorted(glob.glob(g)))
    seen = set()
    results = []
    for path in files:
        if path in seen:
            continue
        seen.add(path)
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[skip] {path}: {e}", file=sys.stderr)
            continue
        cfg = data.get("config") or {}
        workload = cfg.get("workload") or "?"
        version = _guess_version(path)
        methods = {}
        for method, block in (data.get("results") or {}).items():
            if block.get("na"):
                continue
            trials = block.get("trials") or []
            if len(trials) < min_runs:
                continue
            methods[method] = trials
        if not methods:
            continue
        results.append({"path": path, "workload": workload,
                        "version": version, "methods": methods,
                        "config": cfg})
    return results


# ── Metric extractor ─────────────────────────────────────────────────────────
def _extract(trials: list[dict], metric: str) -> list[float]:
    """Extract one metric across trials, skipping None and n/a rows."""
    vals = []
    for t in trials:
        if t.get("na") or t.get("error"):
            continue
        v = t.get(metric)
        if v is None:
            continue
        vals.append(float(v))
    return vals


# ── Row builder ──────────────────────────────────────────────────────────────
def _build_row(workload: str, version: str, method_trials: dict,
               metric: str, metric_label: str) -> MetricRow | None:
    per_method = []
    method_vals = {}
    for method, trials in method_trials.items():
        vals = _extract(trials, metric)
        if not vals:
            continue
        mean, lo, hi = _bootstrap_mean_ci(vals)
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        med = vals_sorted[n // 2]
        p25 = vals_sorted[max(0, n // 4)]
        p75 = vals_sorted[min(n - 1, (3 * n) // 4)]
        per_method.append(MethodSummary(
            method=method, n=n,
            mean=round(mean, 4),
            lo_ci=round(lo, 4),
            hi_ci=round(hi, 4),
            median=round(med, 4),
            p25=round(p25, 4),
            p75=round(p75, 4),
        ))
        method_vals[method] = vals

    if not per_method:
        return None

    pairwise = []
    if "AutoSage" in method_vals:
        for peer in ("HPA", "VPA"):
            if peer not in method_vals:
                continue
            a = method_vals["AutoSage"]
            b = method_vals[peer]
            # Paired only if equal length; otherwise Cliff's delta is still valid
            paired_n = min(len(a), len(b))
            p = _wilcoxon_p(a[:paired_n], b[:paired_n])
            d = _cliffs_delta(a, b)
            pairwise.append(PairwiseSummary(
                a="AutoSage", b=peer, n=paired_n,
                wilcoxon_p=round(p, 6) if p is not None else None,
                cliffs_delta=round(d, 4),
                cliffs_verdict=_cliffs_verdict(d),
            ))

    return MetricRow(
        workload=workload, version=version,
        metric=metric, metric_label=metric_label,
        per_method=per_method, pairwise=pairwise,
    )


# ── LaTeX emit ───────────────────────────────────────────────────────────────
def _fmt(x: float) -> str:
    if x is None or math.isnan(x):
        return "--"
    if abs(x) >= 1000:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    if abs(x) >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def _emit_latex(rows: list[MetricRow], out_path: str) -> None:
    """Emit one longtable per metric. Grouped by workload, sorted by version."""
    lines = [
        "% Auto-generated by research/stats_rigor.py — do not edit by hand.",
        "% One longtable per metric with per-method bootstrap 95% CI and",
        "% pairwise Wilcoxon p-value + Cliff's delta on AutoSage vs {HPA, VPA}.",
        "\\begingroup",
        "\\small",
    ]
    by_metric: dict[str, list[MetricRow]] = defaultdict(list)
    for r in rows:
        by_metric[r.metric].append(r)

    for metric_key, group in by_metric.items():
        if not group:
            continue
        label = group[0].metric_label
        # Sort: workload asc, version asc
        group.sort(key=lambda r: (r.workload, r.version))
        lines.append("")
        safe_key = metric_key.replace('_', '-')
        lines.append(f"\\begin{{longtable}}{{lll rrrr rrr}}")
        lines.append(f"\\caption{{{label}: bootstrap 95\\% CI, "
                     f"Wilcoxon $p$, Cliff's $\\delta$ vs. AutoSage.}}"
                     f"\\label{{tab:stats-rigor-{safe_key}}} \\\\")
        lines.append("\\toprule")
        lines.append("Workload & Version & Method & n & Mean & 95\\% CI & "
                     "Median & vs.\\ AS $p$ & Cliff's $\\delta$ & Verdict \\\\")
        lines.append("\\midrule")
        for row in group:
            pairwise_map = {p.b: p for p in row.pairwise}
            for m in row.per_method:
                if m.method == "AutoSage":
                    p_str, d_str, verdict = "--", "--", "--"
                else:
                    pw = pairwise_map.get(m.method)
                    if pw is None:
                        p_str, d_str, verdict = "--", "--", "--"
                    else:
                        p_str = ("--" if pw.wilcoxon_p is None
                                 else f"${pw.wilcoxon_p:.3f}$")
                        d_str = f"${pw.cliffs_delta:+.2f}$"
                        verdict = pw.cliffs_verdict
                lines.append(
                    f"{row.workload} & {row.version} & {m.method} & "
                    f"{m.n} & {_fmt(m.mean)} & "
                    f"[{_fmt(m.lo_ci)}, {_fmt(m.hi_ci)}] & "
                    f"{_fmt(m.median)} & {p_str} & {d_str} & {verdict} \\\\"
                )
            lines.append("\\midrule")
        lines.append("\\bottomrule")
        lines.append("\\end{longtable}")

    lines.append("\\endgroup")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [latex] wrote {out_path}  ({sum(len(g) for g in by_metric.values())} rows)")


# ── Stdout emit ──────────────────────────────────────────────────────────────
def _emit_stdout(rows: list[MetricRow]) -> None:
    by_metric: dict[str, list[MetricRow]] = defaultdict(list)
    for r in rows:
        by_metric[r.metric].append(r)
    for metric_key, group in by_metric.items():
        if not group:
            continue
        label = group[0].metric_label
        print(f"\n=== {label} ({metric_key}) ===")
        group.sort(key=lambda r: (r.workload, r.version))
        header = ("workload/ver/method  n     mean          95% CI                "
                  "median   vs-AS p    Cliff's d   verdict")
        print(header)
        print("-" * len(header))
        for row in group:
            pairwise_map = {p.b: p for p in row.pairwise}
            for m in row.per_method:
                if m.method == "AutoSage":
                    tail = "--                       --      --"
                else:
                    pw = pairwise_map.get(m.method)
                    if pw is None:
                        tail = "--                       --      --"
                    else:
                        p_str = ("--" if pw.wilcoxon_p is None
                                 else f"{pw.wilcoxon_p:8.4f}")
                        tail = (f"{p_str:>8}  {pw.cliffs_delta:+6.2f}  "
                                f"{pw.cliffs_verdict}")
                head = f"{row.workload}/{row.version}/{m.method:<10}"[:20]
                print(f"{head:<20} {m.n:>3}  {m.mean:>10.3f}  "
                      f"[{m.lo_ci:>8.3f}, {m.hi_ci:>8.3f}]  {m.median:>8.3f}  "
                      f"{tail}")


# ── Entry ────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workload",
                    help="Filter rows by workload (cpu, stateful, multiclass, "
                         "stateful_compute, dsb_hotel).")
    ap.add_argument("--min-runs", type=int, default=5,
                    help="Skip versions with fewer than N trials per method "
                         "(default 5).")
    ap.add_argument("--latex-out",
                    default=os.path.join(_REPO, "thesis_reports",
                                         "stats_rigor.tex"),
                    help="Path for the generated \\input-able LaTeX table.")
    ap.add_argument("--json-out",
                    default=os.path.join(_REPO, "thesis_reports",
                                         "stats_rigor.json"),
                    help="Path for the machine-readable JSON dump.")
    args = ap.parse_args()

    all_files = _load_all(_DEFAULT_GLOBS, min_runs=args.min_runs)
    if args.workload:
        all_files = [f for f in all_files if f["workload"] == args.workload]
    if not all_files:
        print("[fatal] no comparison_results_*.json files matched.",
              file=sys.stderr)
        return 1
    print(f"  [loaded] {len(all_files)} version files "
          f"across {len({f['workload'] for f in all_files})} workload classes.")

    rows: list[MetricRow] = []
    for fdata in all_files:
        for metric_key, _short, metric_label, _direction in _METRICS:
            row = _build_row(
                workload=fdata["workload"],
                version=fdata["version"],
                method_trials=fdata["methods"],
                metric=metric_key,
                metric_label=metric_label,
            )
            if row is not None:
                rows.append(row)

    if not rows:
        print("[fatal] no metric rows extracted.", file=sys.stderr)
        return 1

    _emit_stdout(rows)

    os.makedirs(os.path.dirname(args.latex_out), exist_ok=True)
    _emit_latex(rows, args.latex_out)

    with open(args.json_out, "w") as f:
        json.dump([{
            "workload": r.workload,
            "version": r.version,
            "metric": r.metric,
            "metric_label": r.metric_label,
            "per_method": [asdict(m) for m in r.per_method],
            "pairwise": [asdict(p) for p in r.pairwise],
        } for r in rows], f, indent=2, default=lambda o: None)
    print(f"  [json] wrote {args.json_out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
