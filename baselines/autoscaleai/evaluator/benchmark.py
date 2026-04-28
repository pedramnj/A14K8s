"""Small harness to run the SimulatedEnvironment and print a summary.
"""
import time
from agent.train import SimulatedEnvironment

def run_demo():
    env = SimulatedEnvironment()
    s = env.reset()
    for _ in range(50):
        # naive policy: scale up if rps growing quickly
        action = 1
        if env.rps > env.replicas * env.base_capacity * 0.8:
            action = 2
        elif env.rps < env.replicas * env.base_capacity * 0.4:
            action = 0
        s, r, done, info = env.step(action)
        print(f"t={env.t} rps={info['rps']:.1f} replicas={info['replicas']} latency={info['latency']:.1f}ms")
        if done:
            break


if __name__ == '__main__':
    run_demo()