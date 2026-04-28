AutoScaleAI — Intelligent RL-Based Autoscaling for Kubernetes
Reinforcement Learning–driven scaling for microservices

Project Overview

AutoScaleAI is an intelligent autoscaling framework for Kubernetes microservices using Reinforcement Learning (RL).
Unlike the Kubernetes Horizontal Pod Autoscaler (HPA), which relies on static threshold-based rules (e.g., CPU > 80%), AutoScaleAI uses an RL agent that:
 - Learns from real-time metrics
 - Predicts optimal scaling actions
 - Minimizes latency and costs
 - Reduces scaling oscillations
 - Adapts dynamically to workload patterns

AutoScaleAI integrates with Prometheus, Kubernetes API, and supports both simulated and real cluster modes.

Key Features
 - Intelligent Autoscaling
    Uses Actor–Critic RL model to decide scale-up, scale-down, or no-op.
 - Kubernetes-Native
    Works with any Kubernetes cluster with Prometheus installed.
 - Prometheus Metric Collection
    Collects CPU, memory, latency, RPS in real time.
 - Pluggable Policy Network (PyTorch)
    Switch between A2C, PPO, DQN, etc.
 - Simulation Engine
    Allows offline RL training before deploying to production.
 - Evaluation vs HPA
    Direct comparison across latency, cost, stability, and responsiveness.

🏗️ Architecture Diagram (Mermaid)
flowchart LR

  subgraph K8sCluster[ Kubernetes Cluster ]
    direction TB

    MS[Microservices]
    Prom[Prometheus]
    RL[RL Agent Pod (AutoScaleAI)]
    AutoCtrl[Autoscaling Controller]
    HPA[HPA (Baseline)]
    Infer[Inference Engine]
    K8sAPI[(Kubernetes API)]
  end

  MS -->|exposes metrics| Prom
  Prom --> K8sAPI
  RL --> K8sAPI
  AutoCtrl --> K8sAPI
  HPA --> K8sAPI
  Infer --> RL

  LoadGen[Load Generator] --> MS

  style K8sCluster stroke:#2b6cb0,stroke-width:2px
  style RL fill:#fefcbf
  style AutoCtrl fill:#c6f6d5

Repository Structure

autoscaleai/
│── agent/
│     ├── policy_network.py
│     ├── model.py
│     └── train.py
│── collector/
│     └── metrics.py
│── executor/
│     └── scaler.py
│── env/
│     └── simulated_env.py
│── evaluator/
│     └── benchmark.py
│── kubernetes/
│     ├── deployment.yaml
│     ├── autoscaleai-deployment.yaml
│── scripts/
│     ├── run_local.sh
│     └── run_local.bat
│── docs/
│── README.md
│── requirements.txt

Setup Instructions
1. Clone the Repository
git clone https://github.com/shaktijit-r/AutoScaleAI.git
cd AutoScaleAI

2. Create Virtual Environment
Windows:
python -m venv .venv
.\.venv\Scripts\activate

Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

How to Run the RL Agent (Simulation Mode)

Train Actor–Critic Agent Locally:
python train_visualize_pytorch.py --episodes 200


This produces:
 - autoscale_agent.pt (trained model)
 - pytorch_viz_episode_rewards.png
 - pytorch_viz_final_latency.png
 - pytorch_viz_training_summary.csv

Visualize RL Behavior:
python visualize_simulation.py

Run in Windows

Use the batch file:
scripts\run_local.bat

How to Run AutoScaleAI on Kubernetes
1. Deploy RL Agent:
kubectl apply -f kubernetes/autoscaleai-deployment.yaml

2. Set Up Prometheus Scraping:
Ensure Prometheus scrapes your microservices and exposes metrics at:
http://prometheus.monitoring.svc.cluster.local:9090

3. Run AutoScaleAI Controller
The controller queries Prometheus → feeds RL agent → sends scale decisions via K8s API.

Evaluation Method vs HPA:
AutoScaleAI is evaluated against Kubernetes HPA on:

1. Latency
 - p90, p95, p99
 - end-to-end response time

2. Stability / Oscillation
 - frequency of scale toggling
 - replica variance

3. Cost Efficiency
 - average pod count
 - pod-hours consumed

4. Responsiveness
 -  reaction time during load spikes
 - overshoot / undershoot behavior

5. SLA Violations
 - % of requests exceeding defined thresholds

Results Summary:
    Key Improvements of AutoScaleAI vs HPA
    Metric	                HPA	        AutoScaleAI     Improvement
    Average Latency	        Higher	    Lower	        ✔ Faster Response
    Scaling Oscillations	Frequent	Minimal	        ✔ More Stable
    Cost Efficiency	        Moderate	High	        ✔ Fewer Pod-Hours
    Response to Spikes	    Slow	    Rapid	        ✔ Faster Scaling
    SLA Violations	        Higher	    Lower	        ✔ More Reliable

AutoScaleAI learns predictive scaling behavior, reducing both cost and latency.

Contributors:
Shaktijit Rautaray
M25AI1042