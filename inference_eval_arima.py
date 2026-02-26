import csv
import argparse
from datetime import datetime

import numpy as np
import pmdarima as pm

# =============================
# CONFIG
# =============================
NODES_NUMBER = 10
LOOKBACK = 3
TRAIN_SPLIT = 0.8
DATA_FILE_PATH = "data/data/10.csv"
RESULTS_CSV = "arima_eval_mse_mape_10.csv"
ARIMA_ORDER = (3, 0, 0)
ARIMA_WINDOW = 512


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


def one_step_walk_forward_arima(
    series: np.ndarray, split_idx: int, lookback: int, arima_order: tuple, arima_window: int
) -> tuple:
    y_true = []
    y_pred = []
    for t in range(split_idx, len(series)):
        history_end = t
        history_start = max(0, history_end - arima_window)
        history = series[history_start:history_end]
        if len(history) < max(lookback + 1, 5):
            continue

        model = pm.ARIMA(order=arima_order, suppress_warnings=True, with_intercept=True)
        model.fit(history, maxiter=25, disp=0)
        pred = float(model.predict(n_periods=1)[0])
        y_pred.append(pred)
        y_true.append(float(series[t]))
    return np.asarray(y_true, dtype=np.float32), np.asarray(y_pred, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="ARIMA forecast evaluation for CPU/RAM (MSE, MAPE)")
    parser.add_argument("--data-file", default=DATA_FILE_PATH, help="Path to multivariate CSV dataset")
    parser.add_argument("--nodes", type=int, default=NODES_NUMBER, help="Number of nodes/columns to evaluate")
    parser.add_argument("--lookback", type=int, default=LOOKBACK, help="Lookback window")
    parser.add_argument("--train-split", type=float, default=TRAIN_SPLIT, help="Train split ratio")
    parser.add_argument("--arima-order", default="3,0,0", help="ARIMA order as p,d,q")
    parser.add_argument("--arima-window", type=int, default=ARIMA_WINDOW, help="Max history window per step")
    parser.add_argument("--output", default=RESULTS_CSV, help="Output CSV path")
    args = parser.parse_args()

    arima_order = tuple(int(x.strip()) for x in args.arima_order.split(","))
    if len(arima_order) != 3:
        raise ValueError("--arima-order must have exactly 3 comma-separated integers (p,d,q)")

    metrics = load_cpu_ram_series(args.data_file, args.nodes)
    n_total = metrics["cpu"].shape[0]
    split_idx = int(n_total * args.train_split)

    rows = []
    for metric_name, metric_array in metrics.items():
        for node_idx in range(metric_array.shape[1]):
            series = metric_array[:, node_idx].astype(np.float32)
            y_true, y_pred = one_step_walk_forward_arima(
                series,
                split_idx,
                args.lookback,
                arima_order=arima_order,
                arima_window=args.arima_window,
            )
            if len(y_true) == 0:
                continue
            mse = compute_mse(y_true, y_pred)
            mape = compute_mape(y_true, y_pred)
            rows.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "model": "ARIMA",
                    "metric": metric_name,
                    "node": node_idx,
                    "samples": len(y_true),
                    "lookback": args.lookback,
                    "mse": round(mse, 6),
                    "mape_pct": round(mape, 6),
                }
            )
            print(f"[ARIMA][{metric_name}][node={node_idx}] MSE={mse:.6f} MAPE={mape:.4f}%")

    if not rows:
        print("No evaluation rows produced. Check dataset size/config.")
        return

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "model", "metric", "node", "samples", "lookback", "mse", "mape_pct"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved ARIMA MSE/MAPE results to: {args.output}")


if __name__ == "__main__":
    main()
