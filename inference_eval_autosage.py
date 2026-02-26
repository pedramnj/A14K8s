import argparse
import csv
from datetime import datetime, timedelta

import numpy as np

from predictive_monitoring import ResourceMetrics, TimeSeriesForecaster


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def load_cpu_ram_series(data_file_path: str, nodes_number: int) -> dict:
    array = np.loadtxt(data_file_path, skiprows=1, delimiter=",")
    cpu_values = array[0::4][:, :nodes_number]
    ram_values = array[1::4][:, :nodes_number]
    return {"cpu": cpu_values, "ram": ram_values}


def evaluate_autosage_series(
    series: np.ndarray,
    split_idx: int,
    lookback: int,
    metric_name: str,
) -> tuple:
    """
    Walk-forward one-step evaluation using the project forecaster.
    We start at max(split_idx, 10) to avoid the synthetic insufficient-data fallback.
    """
    y_true = []
    y_pred = []

    start_idx = max(split_idx, 10, lookback)

    for t in range(start_idx, len(series)):
        forecaster = TimeSeriesForecaster(window_size=max(24, lookback * 2))
        base_ts = datetime.utcnow() - timedelta(minutes=t)

        for i in range(t):
            cpu_val = float(series[i]) if metric_name == "cpu" else 0.0
            ram_val = float(series[i]) if metric_name == "ram" else 0.0
            forecaster.add_data_point(
                ResourceMetrics(
                    timestamp=base_ts + timedelta(minutes=i),
                    cpu_usage=cpu_val,
                    memory_usage=ram_val,
                    network_io=0.0,
                    disk_io=0.0,
                    pod_count=1,
                    node_count=1,
                )
            )

        if metric_name == "cpu":
            pred = float(forecaster.forecast_cpu_usage(hours_ahead=1).predicted_values[0])
        else:
            pred = float(forecaster.forecast_memory_usage(hours_ahead=1).predicted_values[0])

        y_pred.append(pred)
        y_true.append(float(series[t]))

    return np.asarray(y_true, dtype=np.float32), np.asarray(y_pred, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoSage predictor evaluation for CPU/RAM (MSE, MAPE)")
    parser.add_argument("--data-file", required=True, help="Path to multivariate CSV dataset")
    parser.add_argument("--nodes", type=int, default=2, help="Number of nodes/columns to evaluate")
    parser.add_argument("--lookback", type=int, default=3, help="Lookback window")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--output", default="autosage_eval_mse_mape.csv", help="Output CSV path")
    args = parser.parse_args()

    metrics = load_cpu_ram_series(args.data_file, args.nodes)
    n_total = metrics["cpu"].shape[0]
    split_idx = int(n_total * args.train_split)

    rows = []
    for metric_name, metric_array in metrics.items():
        for node_idx in range(metric_array.shape[1]):
            series = metric_array[:, node_idx].astype(np.float32)
            y_true, y_pred = evaluate_autosage_series(
                series,
                split_idx=split_idx,
                lookback=args.lookback,
                metric_name=metric_name,
            )
            if len(y_true) == 0:
                continue
            mse = compute_mse(y_true, y_pred)
            mape = compute_mape(y_true, y_pred)
            rows.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "model": "AutoSagePredictor",
                    "metric": metric_name,
                    "node": node_idx,
                    "samples": len(y_true),
                    "lookback": args.lookback,
                    "mse": round(mse, 6),
                    "mape_pct": round(mape, 6),
                }
            )
            print(f"[AutoSagePredictor][{metric_name}][node={node_idx}] MSE={mse:.6f} MAPE={mape:.4f}%")

    if not rows:
        print("No evaluation rows produced. Check dataset size/config.")
        return

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "model", "metric", "node", "samples", "lookback", "mse", "mape_pct"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved AutoSage predictor MSE/MAPE results to: {args.output}")


if __name__ == "__main__":
    main()
