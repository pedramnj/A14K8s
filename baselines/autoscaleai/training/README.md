# Training AutoScaleAI from scratch on muBench

The upstream AutoScaleAI ships with a synthetic simulator
(`env/simulated_env.py`) that models a sinusoidal RPS workload against a
RPS-bound queue with a 200 ms SLA. Our muBench testbed is a CPU-bound
microservice chain with a 2 s SLA under wrk-style burst traffic. The
out-of-distribution gap caused the bundled checkpoint to over-violate
the SLA in our 48 c / 96 c eval (49–53% violation rate).

This directory retrains the policy on a muBench-shaped simulator so the
RL baseline is a fair comparison rather than a strawman.

## What's in here

- `mubench_env.py` — Gymnasium environment that mirrors muBench:
  - 3-d state `[cpu_util, replicas_norm, request_rate_norm]`
  - 3 discrete actions (down / noop / up) with min/max clamping
  - CPU-bound service capacity, M/M/c-like queueing latency model
  - wrk-burst workload schedule (idle → ramp → sustained → ramp-down)
  - Reward `−latency_norm − 0.5·cost − 5·SLA_penalty` (same shape as upstream)
- `train_ppo.py` — sb3 PPO training script (MLP 128×128, 50 k env steps default)
- `__init__.py` — keeps the directory importable as a package
- `../artifacts/mubench_ppo_v1.zip` — produced by `train_ppo.py`; sb3 archive
  the runner loads at startup
- `../artifacts/mubench_ppo_v1.evaluation.txt` — post-training rollout stats
  (mean reward, SLA violation rate, mean replicas, mean latency)

## Run it

On the CrownLabs VM (the laptop's venv doesn't have torch/sb3 installed):

```bash
cd ~/ai4k8s
source venv/bin/activate
pip install stable-baselines3 gymnasium

# fast iteration (~5–10 min on 4 vCPU)
python -m baselines.autoscaleai.training.train_ppo --steps 50000

# longer training if needed
python -m baselines.autoscaleai.training.train_ppo --steps 200000 \
    --out baselines/autoscaleai/artifacts/mubench_ppo_v2.zip
```

The script saves the model and writes a one-page summary file next to it.
A successful policy should:

- Hold replicas at the minimum during the idle and ramp-down phases
- Scale up during the burst sustain phase
- Keep SLA violation rate under 25% on the evaluation rollouts
- Average reward materially higher than a random or always-noop policy

## How the runner uses the trained policy

`baselines/autoscaleai/runner.py` is the production inference daemon. After
Phase F it loads the sb3 archive directly:

```python
from stable_baselines3 import PPO
model = PPO.load(MODEL_PATH)
action, _ = model.predict(state, deterministic=True)
```

The state vector built in `runner.py` (`build_state(...)`) matches the
training env's observation order exactly, so the deployed policy makes
the same predictions in production that it did during eval rollouts.

## Honesty notes for the paper

The simulator is **not** a digital twin of muBench:

- We don't model network jitter or kubelet/HPA reconciliation latency
- The latency curve is a piecewise approximation of M/M/c, not a real
  benchmark of msvcbench's image_processing function
- Workload is single-service (ingest only); the upstream chain
  (ingest → process → analyze) is not modelled

What we do faithfully model:

- 3-d state shape and 3-action discrete control matches deployment
- min/max replica clamps match the eval harness (2/4)
- 2 s SLA target matches `mubench/run_comparison_eval.py:SLA_THRESHOLD_S`
- Cost weight of 0.1·replicas matches both upstream's reward and the
  eval harness's `cost_proxy = avg_replicas × CPU_REQUEST_M / 1000`

The trained policy is therefore a **muBench-shaped** RL baseline — closer
to a fair comparison than the bundled-checkpoint version, but still a
simulator-trained policy that we deploy without retraining on real
metrics. We document this clearly in the methodology section.
