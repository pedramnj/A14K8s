import argparse
import csv
from collections import defaultdict


def _read_rows(path: str):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _mean(values):
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ARIMA/LSTM eval CSVs into one comparison table")
    parser.add_argument("--arima", required=True, help="Path to ARIMA eval CSV")
    parser.add_argument("--lstm", required=True, help="Path to LSTM eval CSV")
    parser.add_argument("--autosage", help="Optional path to AutoSage predictor eval CSV")
    parser.add_argument("--output", default="forecast_model_comparison.csv", help="Output merged CSV")
    args = parser.parse_args()

    groups = defaultdict(lambda: {"mse": [], "mape_pct": [], "samples": []})
    input_paths = [args.arima, args.lstm]
    if args.autosage:
        input_paths.append(args.autosage)

    for path in input_paths:
        rows = _read_rows(path)
        for r in rows:
            key = (r["model"], r["metric"])
            groups[key]["mse"].append(float(r["mse"]))
            groups[key]["mape_pct"].append(float(r["mape_pct"]))
            groups[key]["samples"].append(int(float(r["samples"])))

    merged_rows = []
    for (model, metric), vals in sorted(groups.items()):
        merged_rows.append(
            {
                "model": model,
                "metric": metric,
                "nodes_evaluated": len(vals["mse"]),
                "avg_samples_per_node": round(_mean(vals["samples"]) or 0.0, 2),
                "avg_mse": round(_mean(vals["mse"]) or 0.0, 6),
                "avg_mape_pct": round(_mean(vals["mape_pct"]) or 0.0, 6),
            }
        )

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "metric",
                "nodes_evaluated",
                "avg_samples_per_node",
                "avg_mse",
                "avg_mape_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"Saved merged comparison to: {args.output}")


if __name__ == "__main__":
    main()
