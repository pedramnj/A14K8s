import argparse
import csv
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _run(cmd: List[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _ci95(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return 1.96 * math.sqrt(var / len(xs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simulated forecast benchmark across scenarios and seeds")
    parser.add_argument("--python-bin", default="./.venv_forecast/bin/python")
    parser.add_argument("--scenarios", default="stable_api,bursty_web,growing_service")
    parser.add_argument("--seeds", default="101,102,103,104,105")
    parser.add_argument("--length", type=int, default=360)
    parser.add_argument("--lookback", type=int, default=3)
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", default="thesis_reports/forecast_simulated")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    py = args.python_bin
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    out_dir = (repo / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = []

    for scenario in scenarios:
        for seed in seeds:
            ds = out_dir / f"dataset_{scenario}_seed{seed}.csv"
            arima_out = out_dir / f"arima_{scenario}_seed{seed}.csv"
            lstm_out = out_dir / f"lstm_{scenario}_seed{seed}.csv"
            autosage_out = out_dir / f"autosage_{scenario}_seed{seed}.csv"

            _run(
                [
                    py,
                    "simulate_forecast_workloads.py",
                    "--scenario",
                    scenario,
                    "--length",
                    str(args.length),
                    "--seed",
                    str(seed),
                    "--output",
                    str(ds),
                ],
                repo,
            )

            _run(
                [
                    py,
                    "inference_eval_arima.py",
                    "--data-file",
                    str(ds),
                    "--nodes",
                    str(args.nodes),
                    "--lookback",
                    str(args.lookback),
                    "--output",
                    str(arima_out),
                ],
                repo,
            )

            _run(
                [
                    py,
                    "inference_eval_convlstm.py",
                    "--data-file",
                    str(ds),
                    "--nodes",
                    str(args.nodes),
                    "--lookback",
                    str(args.lookback),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--output",
                    str(lstm_out),
                ],
                repo,
            )

            _run(
                [
                    py,
                    "inference_eval_autosage.py",
                    "--data-file",
                    str(ds),
                    "--nodes",
                    str(args.nodes),
                    "--lookback",
                    str(args.lookback),
                    "--output",
                    str(autosage_out),
                ],
                repo,
            )

            for model_file in [arima_out, lstm_out, autosage_out]:
                rows = _read_csv(model_file)
                for r in rows:
                    raw_rows.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "model": r["model"],
                            "metric": r["metric"],
                            "mse": float(r["mse"]),
                            "mape_pct": float(r["mape_pct"]),
                        }
                    )

    raw_path = out_dir / "forecast_model_raw_results.csv"
    with raw_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "seed", "model", "metric", "mse", "mape_pct"])
        w.writeheader()
        w.writerows(raw_rows)

    grouped: Dict[Tuple[str, str, str], Dict[str, List[float]]] = defaultdict(lambda: {"mse": [], "mape": []})
    for r in raw_rows:
        key = (r["scenario"], r["model"], r["metric"])
        grouped[key]["mse"].append(r["mse"])
        grouped[key]["mape"].append(r["mape_pct"])

    agg_rows = []
    for (scenario, model, metric), vals in sorted(grouped.items()):
        agg_rows.append(
            {
                "scenario": scenario,
                "model": model,
                "metric": metric,
                "mse_mean": round(_mean(vals["mse"]), 6),
                "mse_ci95": round(_ci95(vals["mse"]), 6),
                "mape_mean": round(_mean(vals["mape"]), 6),
                "mape_ci95": round(_ci95(vals["mape"]), 6),
                "runs": len(vals["mse"]),
            }
        )

    agg_path = out_dir / "forecast_model_comparison_simulated.csv"
    with agg_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["scenario", "model", "metric", "mse_mean", "mse_ci95", "mape_mean", "mape_ci95", "runs"],
        )
        w.writeheader()
        w.writerows(agg_rows)

    print(f"Saved raw results to: {raw_path}")
    print(f"Saved aggregated results to: {agg_path}")


if __name__ == "__main__":
    main()
