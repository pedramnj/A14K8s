"""muBench-shaped Gymnasium environment for training PPO autoscaling policies.

Why this exists
---------------
The upstream AutoScaleAI repo ships a synthetic simulator (env/simulated_env.py)
that models a sinusoidal RPS workload with a 200 ms SLA against an RPS-capacity
queue. Our actual workload is muBench's CPU-bound microservice chain
(image_processing → matrix_compute → video_frames) under wrk burst traffic with
a 2 s SLA. The pretrained policy from upstream is therefore out-of-distribution
on our testbed.

This environment models a single muBench service (the entry-point `ingest`)
with the following faithful-but-simple dynamics:

  - A bounded number of replicas, each with a fixed CPU "service rate".
  - Incoming HTTP traffic represented as a target arrival rate (req/s) that
    follows a wrk-like burst pattern: idle → ramp → sustained → ramp-down.
  - Latency derived from an M/M/c-like queueing approximation: linear under
    moderate load, super-linear once cumulative utilisation exceeds 80%, and
    saturated at a hard ceiling.
  - Cost = number of currently-running replicas × per-pod request weight.
  - Reward shape matches the upstream simulator so the training objective is
    apples-to-apples: −latency_norm − 0.5·cost − 5·sla_penalty.

State (3-d, all in [0, 1] when normalised):
    s[0] = cpu_util      = avg per-pod CPU rate / pod CPU capacity
    s[1] = replicas_norm = current_replicas / max_replicas
    s[2] = rps_norm      = current_arrival_rate / RPS_PROXY_MAX

Actions (discrete, 3):
    0 = scale_down (clamped to MIN_REPLICAS)
    1 = noop
    2 = scale_up   (clamped to MAX_REPLICAS)

The intent is **not** a perfect digital twin of muBench — it is a workload
shape that's closer to wrk-on-image_processing than to upstream's sinusoid,
so the trained policy generalises rather than overfits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class MubenchConfig:
    # Replica bounds — match the eval harness
    min_replicas: int = 2
    max_replicas: int = 4

    # Per-pod service capacity. Tuned so 4 replicas comfortably serve the
    # burst peak (cap=100 vs peak=60 → ~60% utilisation at max replicas) and
    # 2 replicas saturate (cap=50 vs peak=60 → ρ=1.2). This makes scaling
    # decisions consequential without forcing always-max.
    pod_capacity_req_per_s: float = 25.0

    # SLA target (2 s) — matches the eval harness `SLA_THRESHOLD_S`.
    sla_seconds: float = 2.0

    # Scaling cost — same shape as upstream: 0.1 per replica per step
    cost_per_replica: float = 0.1

    # Reward weights — tuned so cost differential at idle (0.39 per step
    # between 2 and 4 replicas) accumulates over the long idle phase to
    # outweigh peak-burst SLA penalties. The previous (1.0 / 0.5 / 5.0)
    # made the always-max strategy globally optimal.
    w_latency: float = 1.0
    w_cost: float = 2.0
    w_sla: float = 2.0

    # Episode length — 200 control steps ≈ 200 × 5 s = ~16 min of simulated wall time
    episode_length: int = 200

    # Observation normalisation
    rps_proxy_max: float = 100.0   # upper bound for request_rate normalisation

    # Workload schedule (req/s at each step) — wrk-like burst
    # Burst peak well within max-replica capacity so 4 replicas can serve it
    # with mild SLA pressure; 2 replicas clearly saturate.
    burst_peak: float = 60.0
    burst_idle: float = 5.0


class MubenchScalingEnv(gym.Env):
    """muBench-shaped scaling environment for PPO training."""

    metadata = {"render_modes": []}

    def __init__(self, config: MubenchConfig | None = None, seed: int | None = None):
        super().__init__()
        self.cfg = config or MubenchConfig()
        self._rng = np.random.default_rng(seed)

        # 3-d Box state, all in [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )
        # 3 discrete actions
        self.action_space = spaces.Discrete(3)

        self.reset(seed=seed)

    # -- workload schedule -------------------------------------------------
    def _arrival_rate(self, t: int) -> float:
        cfg = self.cfg
        ramp_up_end       = 30
        sustained_end     = 130
        ramp_down_end     = 160
        if t < ramp_up_end:
            frac = t / ramp_up_end
            base = cfg.burst_idle + (cfg.burst_peak - cfg.burst_idle) * frac
        elif t < sustained_end:
            base = cfg.burst_peak
        elif t < ramp_down_end:
            frac = (ramp_down_end - t) / (ramp_down_end - sustained_end)
            base = cfg.burst_idle + (cfg.burst_peak - cfg.burst_idle) * frac
        else:
            base = cfg.burst_idle

        # Inject some episode-level variation so the policy doesn't memorise.
        # Multiplicative noise of ±20% on the peak.
        noise = 1.0 + 0.2 * self._rng.standard_normal() * (base / cfg.burst_peak)
        return max(0.5, base * (1.0 + 0.05 * self._rng.standard_normal())) \
               * (0.8 + 0.4 * self._episode_scale)

    # -- queueing latency model -------------------------------------------
    def _latency_seconds(self, replicas: int, arrival_rate: float) -> tuple[float, float]:
        """Return (latency_s, cpu_util) for the given replicas + arrival rate."""
        capacity = replicas * self.cfg.pod_capacity_req_per_s
        rho = arrival_rate / capacity if capacity > 0 else 10.0

        # Latency model — three regimes:
        # under moderate load: linear in rho around a 0.5 s base
        # near saturation:     super-linear (queueing blow-up)
        # over saturation:     hard ceiling (10 s)
        if rho < 0.6:
            latency = 0.4 + 0.5 * rho
        elif rho < 0.9:
            latency = 0.7 + 2.0 * (rho - 0.6)
        elif rho < 1.0:
            latency = 1.3 + 8.0 * (rho - 0.9)
        else:
            latency = min(10.0, 2.1 + 5.0 * (rho - 1.0))

        cpu_util = min(1.0, rho)
        return latency, cpu_util

    # -- gym API ----------------------------------------------------------
    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Per-episode multiplier on burst peak to expose different workload intensities.
        self._episode_scale = float(self._rng.uniform(0.7, 1.3))
        self.t = 0
        self.replicas = self.cfg.min_replicas
        self._last_cpu_util = 0.0
        self._last_arrival = self.cfg.burst_idle
        return self._obs(), {}

    def step(self, action: int):
        cfg = self.cfg

        # Apply action with hard min/max clamps.
        if action == 0:
            self.replicas = max(cfg.min_replicas, self.replicas - 1)
        elif action == 2:
            self.replicas = min(cfg.max_replicas, self.replicas + 1)
        # action == 1 is noop

        # Advance one control step.
        self.t += 1
        arrival = self._arrival_rate(self.t)
        latency, cpu_util = self._latency_seconds(self.replicas, arrival)

        # Reward — same shape as upstream simulator, retuned for our SLA.
        latency_norm = min(latency / 5.0, 2.0)   # cap the penalty
        cost         = cfg.cost_per_replica * self.replicas
        sla_penalty  = max(0.0, (latency - cfg.sla_seconds) / cfg.sla_seconds)
        reward = -(cfg.w_latency * latency_norm
                   + cfg.w_cost   * cost
                   + cfg.w_sla    * sla_penalty)

        self._last_cpu_util = cpu_util
        self._last_arrival = arrival

        terminated = False
        truncated  = self.t >= cfg.episode_length

        info = {
            "latency_s": latency,
            "cpu_util": cpu_util,
            "arrival_rps": arrival,
            "replicas": self.replicas,
            "cost": cost,
            "sla_violation": int(latency > cfg.sla_seconds),
        }
        return self._obs(), float(reward), terminated, truncated, info

    def _obs(self) -> np.ndarray:
        return np.array([
            self._last_cpu_util,
            self.replicas / max(1, self.cfg.max_replicas),
            min(1.0, self._last_arrival / self.cfg.rps_proxy_max),
        ], dtype=np.float32)

    def render(self):
        pass


# Sanity check when run directly: random policy should produce non-trivial rewards.
if __name__ == "__main__":
    env = MubenchScalingEnv(seed=0)
    obs, _ = env.reset()
    total = 0.0
    sla_viol = 0
    for step in range(env.cfg.episode_length):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total += reward
        sla_viol += info["sla_violation"]
        if step % 25 == 0:
            print(f"  t={step:3d} replicas={info['replicas']} arrival={info['arrival_rps']:5.1f} "
                  f"latency={info['latency_s']:.2f}s util={info['cpu_util']:.2f} reward={reward:+.3f}")
        if trunc:
            break
    print(f"\nEpisode total reward (random policy): {total:.3f}")
    print(f"SLA violations: {sla_viol}/{env.cfg.episode_length} steps")
