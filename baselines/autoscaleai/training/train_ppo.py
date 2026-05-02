"""Train a PPO policy on the muBench-shaped scaling environment.

Usage (on the CrownLabs VM):
    cd ~/ai4k8s
    source venv/bin/activate
    python -m baselines.autoscaleai.training.train_ppo \
        --steps 50000 \
        --out baselines/autoscaleai/artifacts/mubench_ppo_v1.zip \
        --tb-log /tmp/sb3-tb-mubench

Outputs
-------
- `mubench_ppo_v1.zip`            sb3-format archive (policy + value + arch)
- `mubench_ppo_v1.evaluation.txt` post-training rollout stats (mean reward,
                                  SLA violation rate, cost, etc.)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))   # repo root

from baselines.autoscaleai.training.mubench_env import (
    MubenchConfig,
    MubenchScalingEnv,
)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


def make_env(seed: int = 0):
    def _factory():
        env = MubenchScalingEnv(config=MubenchConfig(), seed=seed)
        return env
    return _factory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50_000,
                   help="total environment steps for training")
    p.add_argument("--n-envs", type=int, default=4,
                   help="parallel envs for vectorised rollouts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path,
                   default=HERE.parent / "artifacts" / "mubench_ppo_v1.zip")
    p.add_argument("--tb-log", type=Path, default=None,
                   help="optional tensorboard log dir (requires `pip install tensorboard`)")
    p.add_argument("--eval-episodes", type=int, default=20)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tb_log_arg = None
    if args.tb_log is not None:
        args.tb_log.mkdir(parents=True, exist_ok=True)
        tb_log_arg = str(args.tb_log)

    env = make_vec_env(make_env(args.seed), n_envs=args.n_envs, seed=args.seed)

    # MLP 128x128 to match the upstream PolicyNetwork architecture.
    policy_kwargs = dict(net_arch=[128, 128])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=args.seed,
        verbose=1,
        tensorboard_log=tb_log_arg,
    )

    print(f"Training PPO for {args.steps} env steps "
          f"({args.n_envs} parallel envs, "
          f"net_arch=128x128, lr=3e-4, gamma=0.99) …")
    model.learn(total_timesteps=args.steps, progress_bar=False)

    # Save the trained model.
    model.save(str(args.out))
    print(f"\nSaved model to {args.out}")

    # Evaluate on a fresh single env.
    print(f"\nEvaluating on {args.eval_episodes} fresh episodes …")
    eval_env = MubenchScalingEnv(config=MubenchConfig(), seed=args.seed + 1000)
    rewards = []
    sla_viols = []
    avg_replicas = []
    avg_latencies = []
    for ep in range(args.eval_episodes):
        obs, _ = eval_env.reset(seed=args.seed + 1000 + ep)
        ep_reward = 0.0
        ep_sla = 0
        replicas_hist = []
        latency_hist = []
        for _ in range(eval_env.cfg.episode_length):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = eval_env.step(int(action))
            ep_reward += r
            ep_sla += info["sla_violation"]
            replicas_hist.append(info["replicas"])
            latency_hist.append(info["latency_s"])
            if term or trunc:
                break
        rewards.append(ep_reward)
        sla_viols.append(ep_sla / eval_env.cfg.episode_length)
        avg_replicas.append(float(np.mean(replicas_hist)))
        avg_latencies.append(float(np.mean(latency_hist)))

    summary_lines = [
        f"Episodes evaluated:        {len(rewards)}",
        f"Mean episode reward:       {np.mean(rewards):+.3f} ± {1.96*np.std(rewards)/np.sqrt(len(rewards)):.3f}",
        f"Mean SLA violation rate:   {np.mean(sla_viols)*100:.2f}%",
        f"Mean avg replicas:         {np.mean(avg_replicas):.2f}",
        f"Mean avg latency (s):      {np.mean(avg_latencies):.3f}",
    ]
    summary = "\n".join(summary_lines)
    print("\n=== Evaluation summary ===")
    print(summary)

    eval_path = args.out.with_suffix(".evaluation.txt")
    with open(eval_path, "w") as f:
        f.write(f"Trained: {args.out.name}\n")
        f.write(f"Steps: {args.steps}\n")
        f.write(f"net_arch: 128x128\n\n")
        f.write(summary + "\n")
    print(f"\nSaved eval summary → {eval_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
