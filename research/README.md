# research/

Offline analysis and evaluation scripts used during the AutoSage thesis
work (not part of the runtime system).

## Contents

| Script | Purpose |
|---|---|
| `autoscaling_native_comparison.py` | Head-to-head comparison harness for HPA, VPA, AutoSage, and the AutoScaleAI baseline. |
| `collect_live_forecast_dataset.py` | Snapshots live cluster metrics into a dataset for offline forecaster training. |
| `generate_thesis_evaluation_plots.py` | Generates matplotlib figures used in the thesis and self-reports. |
| `inference_eval_arima.py` | Forecast-accuracy evaluation of the ARIMA baseline. |
| `inference_eval_autosage.py` | Forecast-accuracy evaluation of AutoSage's bootstrap-ARIMA forecaster. |
| `inference_eval_convlstm.py` | Forecast-accuracy evaluation of a ConvLSTM baseline. |
| `merge_forecast_eval_results.py` | Merges per-model forecast results into a single comparison table. |
| `run_llm_mcda_decision_quality.py` | Sweeps LLM+MCDA decision-quality across ablation configurations. |
| `run_simulated_forecast_benchmark.py` | Runs the forecaster benchmark on synthetic workloads. |
| `simulate_forecast_workloads.py` | Generates the synthetic workload traces used by the benchmark. |

## Running

Each script is self-contained; run from the repo root, e.g.:

```bash
python -m research.generate_thesis_evaluation_plots
```

They depend on the same `requirements.txt` as the runtime system.
