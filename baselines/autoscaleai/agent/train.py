"""Minimal training loop placeholder using a policy-gradient style update.
This file is a starting point — replace with DQN/PPO/A2C as desired.
"""
from __future__ import annotations
import torch
import torch.optim as optim
from agent.model import AutoScaleAgent
import numpy as np
import argparse
import torch.distributions

# Try to import canonical SimulatedEnvironment from env package (preferred).
try:
    from env.simulated_env import SimulatedEnvironment  # type: ignore
except Exception:
    # Fallback: define SimulatedEnvironment here (kept in sync with env/simulated_env.py)
    # This ensures backward compatibility for tests that import from agent.train.
    import numpy as _np
    import random as _random
    from typing import Tuple, Dict

    class SimulatedEnvironment:
        """Fallback simulated environment for local training & testing.

        State vector:
            [current_rps_normalized, current_replicas_normalized, avg_cpu_util_normalized]

        Actions:
            0 = scale down 1
            1 = no-op
            2 = scale up 1
        """

        def __init__(self, max_replicas: int = 10, base_capacity_per_pod: float = 100.0,
                     sla_latency: float = 200.0, seed: int | None = None):
            self.max_replicas = max_replicas
            self.base_capacity = base_capacity_per_pod
            self.sla = sla_latency  # ms
            self.seed = seed
            if seed is not None:
                _np.random.seed(seed)
                _random.seed(seed)
            self.reset()

        def reset(self) -> _np.ndarray:
            self.t = 0
            self.replicas = 1
            self.rps = 10.0
            self.history: list[Dict] = []
            self.last_cpu_util = 0.0
            return self._get_state()

        def step(self, action: int) -> Tuple[_np.ndarray, float, bool, Dict]:
            # Apply scaling action
            if action == 0:
                self.replicas = max(1, self.replicas - 1)
            elif action == 2:
                self.replicas = min(self.max_replicas, self.replicas + 1)

            # Simulate workload fluctuation: sinusoid + noise
            self.t += 1
            self.rps = 10 + 40 * abs(_np.sin(0.1 * self.t)) + _np.random.randn() * 2.0

            # Compute service capacity
            capacity = self.replicas * self.base_capacity
            load_ratio = self.rps / max(1e-6, capacity)

            # Latency model
            if load_ratio < 1.0:
                latency = 50 + 100 * (load_ratio)
            elif load_ratio < 2.0:
                latency = 150 + 200 * (load_ratio - 1.0)
            elif load_ratio < 3.0:
                latency = 350 + 400 * (load_ratio - 2.0)
            else:
                latency = 10000.0

            # CPU utilization is capped at 1
            cpu_util = min(1.0, load_ratio)
            self.last_cpu_util = cpu_util

            # Cost increases with number of replicas
            cost = 0.1 * self.replicas

            # Reward penalizes latency + cost + SLA violations
            sla_penalty = max(0.0, (latency - self.sla) / self.sla)
            reward = - (latency / 1000.0) - 0.5 * cost - 5.0 * sla_penalty

            done = self.t >= 200

            info = {
                "latency": latency,
                "rps": self.rps,
                "replicas": self.replicas,
                "cpu_util": cpu_util,
                "cost": cost
            }

            self.history.append(info)
            return self._get_state(), reward, done, info

        def _get_state(self) -> _np.ndarray:
            max_rps = 200.0
            return _np.array([
                min(1.0, self.rps / max_rps),
                self.replicas / self.max_replicas,
                float(self.last_cpu_util)
            ], dtype=_np.float32)

# --- Minimal training loop & CLI (keeps tests simple) ---

def simple_train_episode(env: "SimulatedEnvironment", policy_fn, gamma: float = 0.99):
    """Run a single episode given a policy function.

    policy_fn(state) -> action (int)
    Returns: list of rewards, and env.history
    """
    state = env.reset()
    rewards = []
    done = False
    while not done:
        action = policy_fn(state)
        _, reward, done, _ = env.step(action)
        rewards.append(reward)
    return rewards, env.history

# Example random policy used in quick smoke runs / tests
def random_policy(state):
    # state is a numpy array [rps_norm, replicas_norm, cpu_util]
    # choose uniformly among {0,1,2}
    import random
    return random.choice([0, 1, 2])

class Trainer:
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.agent = AutoScaleAgent(state_dim, action_dim)
        self.optimizer = optim.Adam(self.agent.policy.parameters(), lr=lr)

    def compute_loss(self, logits, actions, advantages):
        logp = torch.log_softmax(logits, dim=-1)
        selected = logp.gather(1, actions.unsqueeze(1)).squeeze(1)
        return -(selected * advantages).mean()


    def train_step(self, batch_states, batch_actions, batch_advantages):
        self.agent.policy.train()
        logits = self.agent.policy(torch.tensor(batch_states, dtype=torch.float32))
        loss = self.compute_loss(logits, torch.tensor(batch_actions, dtype=torch.long), torch.tensor(batch_advantages, dtype=torch.float32))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
 
if __name__ == '__main__':
    # Expose a CLI for manual quick runs (kept lightweight for CI)
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="Run a quick simulated training run (smoke test).")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to simulate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    env = SimulatedEnvironment(seed=args.seed)
    for ep in range(1, args.episodes + 1):
        rewards, history = simple_train_episode(env, random_policy)
        print(f"Episode {ep} | total_reward={sum(rewards):.3f} | steps={len(rewards)}")