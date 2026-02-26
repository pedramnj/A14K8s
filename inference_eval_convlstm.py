import csv
import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# =============================
# CONFIG
# =============================
NODES_NUMBER = 50
LOOKBACK = 3  # keep aligned with autoscaling/bootstrap lookback
FORECAST_HORIZON = 1
TRAIN_SPLIT = 0.8
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DATA_FILE_PATH = "data/data/50.csv"
RESULTS_CSV = "lstm_eval_mse_mape_50.csv"


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def create_sequences(values: np.ndarray, lookback: int, horizon: int) -> tuple:
    X, y = [], []
    for i in range(len(values) - lookback - horizon + 1):
        X.append(values[i : i + lookback])
        y.append(values[i + lookback : i + lookback + horizon])
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if horizon == 1:
        y = y.reshape(-1, 1)
    return X, y


def build_lstm_model(input_shape: tuple, forecast_horizon: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(forecast_horizon),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=[tf.keras.metrics.MeanSquaredError()],
    )
    return model


def load_cpu_ram_series(data_file_path: str, nodes_number: int) -> dict:
    array = np.loadtxt(data_file_path, skiprows=1, delimiter=",")
    cpu_values = array[0::4][:, :nodes_number]
    ram_values = array[1::4][:, :nodes_number]
    return {"cpu": cpu_values, "ram": ram_values}


def evaluate_metric_node(
    series: np.ndarray,
    lookback: int,
    forecast_horizon: int,
    train_split: float,
    epochs: int,
    batch_size: int,
) -> tuple:
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = x_scaler.fit_transform(series.reshape(-1, 1))
    y_scaled = y_scaler.fit_transform(series.reshape(-1, 1))

    X, y = create_sequences(X_scaled, lookback, forecast_horizon)
    split_idx = int(len(X) * train_split)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if len(X_train) == 0 or len(X_test) == 0:
        return None, None, 0

    model = build_lstm_model((lookback, 1), forecast_horizon)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )

    y_pred = model.predict(X_test, verbose=0)

    y_test_inv = y_scaler.inverse_transform(y_test)
    y_pred_inv = y_scaler.inverse_transform(y_pred)

    mse = compute_mse(y_test_inv, y_pred_inv)
    mape = compute_mape(y_test_inv, y_pred_inv)
    return mse, mape, len(y_test_inv)


def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM forecast evaluation for CPU/RAM (MSE, MAPE)")
    parser.add_argument("--data-file", default=DATA_FILE_PATH, help="Path to multivariate CSV dataset")
    parser.add_argument("--nodes", type=int, default=NODES_NUMBER, help="Number of nodes/columns to evaluate")
    parser.add_argument("--lookback", type=int, default=LOOKBACK, help="Lookback window")
    parser.add_argument("--horizon", type=int, default=FORECAST_HORIZON, help="Forecast horizon")
    parser.add_argument("--train-split", type=float, default=TRAIN_SPLIT, help="Train split ratio")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--output", default=RESULTS_CSV, help="Output CSV path")
    args = parser.parse_args()

    tf.random.set_seed(42)
    np.random.seed(42)

    metrics = load_cpu_ram_series(args.data_file, args.nodes)
    rows = []

    for metric_name, metric_array in metrics.items():
        for node_idx in range(metric_array.shape[1]):
            series = metric_array[:, node_idx].astype(np.float32)
            mse, mape, samples = evaluate_metric_node(
                series,
                lookback=args.lookback,
                forecast_horizon=args.horizon,
                train_split=args.train_split,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            if mse is None:
                continue
            rows.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "model": "LSTM",
                    "metric": metric_name,
                    "node": node_idx,
                    "samples": samples,
                    "lookback": args.lookback,
                    "mse": round(mse, 6),
                    "mape_pct": round(mape, 6),
                }
            )
            print(f"[LSTM][{metric_name}][node={node_idx}] MSE={mse:.6f} MAPE={mape:.4f}%")

    if not rows:
        print("No evaluation rows produced. Check dataset size/config.")
        return

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "model", "metric", "node", "samples", "lookback", "mse", "mape_pct"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved LSTM MSE/MAPE results to: {args.output}")


if __name__ == "__main__":
    main()
