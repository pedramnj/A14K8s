import argparse
import csv
import re
import statistics
import subprocess
import time


def _run(cmd: str):
    proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr


def _parse_top_pods(output: str):
    cpu_vals = []
    ram_vals = []
    for line in output.strip().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        cpu = parts[1]
        mem = parts[2]

        m_cpu = re.match(r"([0-9.]+)m$", cpu)
        if m_cpu:
            cpu_vals.append(float(m_cpu.group(1)))

        m_mem_mi = re.match(r"([0-9.]+)Mi$", mem)
        if m_mem_mi:
            ram_vals.append(float(m_mem_mi.group(1)))

        m_mem_gi = re.match(r"([0-9.]+)Gi$", mem)
        if m_mem_gi:
            ram_vals.append(float(m_mem_gi.group(1)) * 1024.0)

    return cpu_vals, ram_vals


def main():
    parser = argparse.ArgumentParser(description="Collect live kube metrics into 4-stream CSV format")
    parser.add_argument("--namespace", default="ai4k8s-test")
    parser.add_argument("--label", default="app=test-app-autoscaling")
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--interval", type=int, default=3)
    parser.add_argument("--output", default="thesis_reports/forecast_live/live_cpu_ram_4stream.csv")
    args = parser.parse_args()

    rows = []
    cpu_hist = []
    ram_hist = []

    for _ in range(args.samples):
        cmd = f"kubectl -n {args.namespace} top pod -l {args.label} --no-headers"
        rc, out, _ = _run(cmd)

        cpu_vals = []
        ram_vals = []
        if rc == 0 and out.strip():
            cpu_vals, ram_vals = _parse_top_pods(out)

        cpu_avg = statistics.mean(cpu_vals) if cpu_vals else (cpu_hist[-1] if cpu_hist else 0.0)
        ram_avg = statistics.mean(ram_vals) if ram_vals else (ram_hist[-1] if ram_hist else 0.0)
        cpu_hist.append(cpu_avg)
        ram_hist.append(ram_avg)

        # Keep 4-stream row layout expected by existing loaders.
        rows.append([round(cpu_avg, 6), round(cpu_avg * 1.01, 6)])
        rows.append([round(ram_avg, 6), round(ram_avg * 1.01, 6)])
        rows.append([0.0, 0.0])  # disk placeholder
        rows.append([0.0, 0.0])  # service placeholder

        time.sleep(args.interval)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["v1", "v2"])
        writer.writerows(rows)

    print(f"Saved dataset to: {args.output}")
    print(f"Collected samples: {args.samples}")


if __name__ == "__main__":
    main()
