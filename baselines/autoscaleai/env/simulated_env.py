import numpy as np
import random

class SimulatedEnvironment:
    """A minimal environment that simulates incoming request rate, pod capacity,
    latency and cost.

    State vector:
        [current_rps_normalized, current_replicas_normalized, avg_cpu_util_normalized]

    Actions:
        0 = scale down 1
        1 = no-op
        2 = scale up 1

    Reward:
        Negative latency penalty and resource cost penalty.
    """

    def __init__(self, max_replicas=10, base_capacity_per_pod=100.0, sla_latency=200.0, seed=None):
        self.max_replicas = max_replicas
        self.base_capacity = base_capacity_per_pod
        self.sla = sla_latency  # ms
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.reset()

    def reset(self):
        self.t = 0
        self.replicas = 1
        self.rps = 10.0
        self.history = []
        self.last_cpu_util = 0.0
        return self._get_state()

    def step(self, action):
        # Apply scaling action
        if action == 0:
            self.replicas = max(1, self.replicas - 1)
        elif action == 2:
            self.replicas = min(self.max_replicas, self.replicas + 1)

        # Simulate workload fluctuation: sinusoid + noise
        self.t += 1
        self.rps = 10 + 40 * abs(np.sin(0.1 * self.t)) + np.random.randn() * 2.0

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

    def _get_state(self):
        max_rps = 200.0
        return np.array([
            min(1.0, self.rps / max_rps),
            self.replicas / self.max_replicas,
            float(self.last_cpu_util)
        ], dtype=np.float32)
