import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class ScenarioConfig:
    name: str
    base_cpu: float
    base_ram: float
    trend_cpu_per_step: float
    trend_ram_per_step: float
    seasonal_amp_cpu: float
    seasonal_amp_ram: float
    noise_cpu: float
    noise_ram: float
    burst_prob: float
    burst_cpu: Tuple[float, float]
    burst_ram: Tuple[float, float]


SCENARIOS = {
    "stable_api": ScenarioConfig(
        name="stable_api",
        base_cpu=30.0,
        base_ram=35.0,
        trend_cpu_per_step=0.005,
        trend_ram_per_step=0.003,
        seasonal_amp_cpu=4.0,
        seasonal_amp_ram=3.0,
        noise_cpu=1.0,
        noise_ram=0.8,
        burst_prob=0.02,
        burst_cpu=(8.0, 14.0),
        burst_ram=(4.0, 8.0),
    ),
    "bursty_web": ScenarioConfig(
        name="bursty_web",
        base_cpu=28.0,
        base_ram=30.0,
        trend_cpu_per_step=0.0,
        trend_ram_per_step=0.0,
        seasonal_amp_cpu=6.0,
        seasonal_amp_ram=4.0,
        noise_cpu=1.6,
        noise_ram=1.1,
        burst_prob=0.10,
        burst_cpu=(15.0, 35.0),
        burst_ram=(8.0, 16.0),
    ),
    "growing_service": ScenarioConfig(
        name="growing_service",
        base_cpu=24.0,
        base_ram=28.0,
        trend_cpu_per_step=0.06,
        trend_ram_per_step=0.045,
        seasonal_amp_cpu=5.0,
        seasonal_amp_ram=3.5,
        noise_cpu=1.2,
        noise_ram=1.0,
        burst_prob=0.05,
        burst_cpu=(10.0, 18.0),
        burst_ram=(6.0, 12.0),
    ),
}


def _clamp_pct(x: float) -> float:
    return max(0.0, min(100.0, x))


def _gen_stream(cfg: ScenarioConfig, length: int, rng: random.Random, phase: float) -> Tuple[List[float], List[float]]:
    cpu = []
    ram = []
    burst_cpu_remaining = 0
    burst_ram_remaining = 0
    burst_cpu_val = 0.0
    burst_ram_val = 0.0
    for t in range(length):
        # 24-step pseudo daily cycle
        seasonal = math.sin((2.0 * math.pi * (t + phase)) / 24.0)

        if burst_cpu_remaining <= 0 and rng.random() < cfg.burst_prob:
            burst_cpu_remaining = rng.randint(2, 6)
            burst_cpu_val = rng.uniform(*cfg.burst_cpu)
        if burst_ram_remaining <= 0 and rng.random() < cfg.burst_prob * 0.8:
            burst_ram_remaining = rng.randint(2, 6)
            burst_ram_val = rng.uniform(*cfg.burst_ram)

        c = (
            cfg.base_cpu
            + cfg.trend_cpu_per_step * t
            + cfg.seasonal_amp_cpu * seasonal
            + (burst_cpu_val if burst_cpu_remaining > 0 else 0.0)
            + rng.gauss(0.0, cfg.noise_cpu)
        )
        r = (
            cfg.base_ram
            + cfg.trend_ram_per_step * t
            + cfg.seasonal_amp_ram * seasonal
            + (burst_ram_val if burst_ram_remaining > 0 else 0.0)
            + rng.gauss(0.0, cfg.noise_ram)
        )
        cpu.append(_clamp_pct(c))
        ram.append(_clamp_pct(r))

        burst_cpu_remaining -= 1
        burst_ram_remaining -= 1
    return cpu, ram


def write_four_stream_csv(out_path: Path, cpu_nodes: List[List[float]], ram_nodes: List[List[float]]) -> None:
    length = len(cpu_nodes[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["v1", "v2"])
        for t in range(length):
            # cpu row
            w.writerow([round(cpu_nodes[0][t], 6), round(cpu_nodes[1][t], 6)])
            # ram row
            w.writerow([round(ram_nodes[0][t], 6), round(ram_nodes[1][t], 6)])
            # placeholders to keep legacy 4-stream layout
            w.writerow([0.0, 0.0])  # disk
            w.writerow([0.0, 0.0])  # service


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic CPU/RAM forecast benchmark dataset")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS.keys()), required=True)
    parser.add_argument("--length", type=int, default=480, help="Number of time steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    cfg = SCENARIOS[args.scenario]
    rng = random.Random(args.seed)

    cpu1, ram1 = _gen_stream(cfg, args.length, rng, phase=0.0)
    cpu2, ram2 = _gen_stream(cfg, args.length, rng, phase=3.0)
    write_four_stream_csv(Path(args.output), [cpu1, cpu2], [ram1, ram2])
    print(f"Saved synthetic dataset: {args.output}")


if __name__ == "__main__":
    main()
