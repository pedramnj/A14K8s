#!/usr/bin/env python3
"""
Compare AutoSage against native HPA/VPA on a live cluster.

This script is designed for thesis experiments and produces JSON results with:
- Native HPA reaction latency under high CPU load
- Native VPA recommendation latency (if VPA controller is installed)
- AutoSage predictive autoscaling latency and action
"""

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional

from autoscaling_engine import HorizontalPodAutoscaler
from autoscaling_integration import AutoscalingIntegration
from vpa_engine import VerticalPodAutoscaler


class AutoscalingComparisonRunner:
    def __init__(
        self,
        deployment: str,
        namespace: str,
        min_replicas: int,
        max_replicas: int,
        warmup_s: int = 15,
        cooldown_s: int = 15,
        seed_base: int = 42,
        mcda_profile: str = "balanced",
        mcda_agreement_threshold: float = 0.15,
        scenario: str = "short_burst",
        autosage_profile: str = "full",
    ):
        self.deployment = deployment
        self.namespace = namespace
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.warmup_s = warmup_s
        self.cooldown_s = cooldown_s
        self.seed_base = seed_base
        self.mcda_profile = mcda_profile
        self.mcda_agreement_threshold = mcda_agreement_threshold
        self.scenario = scenario
        self.autosage_profile = autosage_profile

        self.hpa = HorizontalPodAutoscaler()
        self.vpa = VerticalPodAutoscaler()
        self.integration: Optional[AutoscalingIntegration] = None
        self._python_kubeconfig_path: Optional[str] = None

    def _run(self, cmd: str, timeout: int = 30) -> Dict[str, Any]:
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "success": proc.returncode == 0,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
                "returncode": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": "timeout", "returncode": -1}
        except Exception as exc:
            return {"success": False, "stdout": "", "stderr": str(exc), "returncode": -1}

    def _wait_rollout(self, timeout_s: int = 180) -> bool:
        cmd = (
            f"kubectl -n {self.namespace} rollout status deployment/{self.deployment} "
            f"--timeout={timeout_s}s"
        )
        res = self._run(cmd, timeout=timeout_s + 10)
        return res["success"]

    def _set_cpu_load_percent(self, percent: int) -> Dict[str, Any]:
        cmd = (
            f"kubectl -n {self.namespace} set env deployment/{self.deployment} "
            f"CPU_LOAD_PERCENT={percent}"
        )
        return self._run(cmd, timeout=40)

    def _set_replicas(self, replicas: int) -> Dict[str, Any]:
        cmd = (
            f"kubectl -n {self.namespace} scale deployment/{self.deployment} "
            f"--replicas={replicas}"
        )
        return self._run(cmd, timeout=40)

    def _reset_resource_baseline(self) -> None:
        # Keep CPU/memory requests consistent across methods for fair cost comparison.
        patch = (
            "{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"app\","
            "\"resources\":{\"requests\":{\"cpu\":\"100m\",\"memory\":\"128Mi\"},"
            "\"limits\":{\"cpu\":\"500m\",\"memory\":\"256Mi\"}}}]}}}}"
        )
        cmd = (
            f"kubectl -n {self.namespace} patch deployment {self.deployment} "
            f"--type=merge -p '{patch}'"
        )
        self._run(cmd, timeout=40)
        self._wait_rollout()

    def _get_deployment_replicas(self) -> int:
        res = self.hpa.get_deployment_replicas(self.deployment, self.namespace)
        if not res.get("success"):
            return 0
        return int(res.get("replicas", 0))

    def _burst_http_load(self, burst_requests: int = 20) -> Dict[str, Any]:
        # Execute from an ephemeral pod so load is generated inside the cluster network.
        cmd = (
            f"kubectl -n {self.namespace} run loadgen-oneshot --rm -i --restart=Never "
            f"--image=curlimages/curl:8.9.1 --command -- sh -c "
            f"\"for i in $(seq 1 {burst_requests}); do "
            f"curl -s http://{self.deployment}/cpu-load >/dev/null & "
            f"done; wait\""
        )
        return self._run(cmd, timeout=90)

    def _apply_scenario_initial_load(self) -> None:
        # Keep scenario setup deterministic across methods for fair comparison.
        if self.scenario == "periodic_burst":
            self._set_cpu_load_percent(75)
            self._burst_http_load(20)
        elif self.scenario == "sustained_drift":
            self._set_cpu_load_percent(65)
            self._burst_http_load(10)
        else:
            # short_burst baseline
            self._set_cpu_load_percent(95)
            self._burst_http_load(30)

    def _apply_scenario_midtest_load(self, elapsed_s: float) -> None:
        if self.scenario == "periodic_burst":
            # Burst around 20s and 45s for a repeating stress pattern.
            if 18 <= elapsed_s <= 24 or 43 <= elapsed_s <= 49:
                self._burst_http_load(18)
        elif self.scenario == "sustained_drift":
            # Increase baseline pressure mid-test to emulate drift.
            if 20 <= elapsed_s <= 28:
                self._set_cpu_load_percent(80)
            if 40 <= elapsed_s <= 50:
                self._set_cpu_load_percent(90)
                self._burst_http_load(12)
        else:
            # short_burst baseline
            if 50 <= elapsed_s <= 65:
                self._burst_http_load(20)

    def _get_cpu_request_m(self) -> int:
        # Use the first container CPU request as a simple cost proxy baseline.
        cmd = (
            f"kubectl -n {self.namespace} get deployment {self.deployment} "
            f"-o jsonpath='{{.spec.template.spec.containers[0].resources.requests.cpu}}'"
        )
        res = self._run(cmd, timeout=20)
        if not res.get("success") or not res.get("stdout"):
            return 100
        cpu = res["stdout"].strip().strip("'").strip('"')
        try:
            if cpu.endswith("m"):
                return int(cpu[:-1])
            return int(float(cpu) * 1000)
        except Exception:
            return 100

    @staticmethod
    def _percentile(values: List[float], pct: float) -> Optional[float]:
        if not values:
            return None
        s = sorted(values)
        idx = int(round((pct / 100.0) * (len(s) - 1)))
        return s[idx]

    def _measure_service_latency(
        self,
        requests: int = 20,
        path: str = "/health",
        sla_latency_s: float = 0.5,
    ) -> Dict[str, Any]:
        pod_res = self._run(
            (
                f"kubectl -n {self.namespace} get pods -l app={self.deployment} "
                f"-o jsonpath='{{.items[0].metadata.name}}'"
            ),
            timeout=20,
        )
        pod_name = pod_res.get("stdout", "").strip().strip("'")
        if not pod_name:
            return {"success": False, "error": "no running pod found for latency probe", "requests": requests}

        probe_code = (
            "import time, urllib.request, urllib.error\n"
            "for _ in range(" + str(requests) + "):\n"
            "    t0=time.time()\n"
            "    code=0\n"
            "    try:\n"
            "        r=urllib.request.urlopen('http://127.0.0.1:8080" + path + "', timeout=5)\n"
            "        code=getattr(r, 'status', 200)\n"
            "        r.read(1)\n"
            "    except urllib.error.HTTPError as e:\n"
            "        code=e.code\n"
            "    except Exception:\n"
            "        code=0\n"
            "    dt=time.time()-t0\n"
            "    print(f'{dt:.4f} {code}')\n"
        )
        cmd = (
            f"kubectl -n {self.namespace} exec {pod_name} -- "
            f"python -c \"{probe_code}\""
        )
        res = self._run(cmd, timeout=120)
        if not res.get("success"):
            return {
                "success": False,
                "error": res.get("stderr", "latency probe failed"),
                "requests": requests,
            }

        latencies: List[float] = []
        status_codes: List[int] = []
        for line in res.get("stdout", "").splitlines():
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                latencies.append(float(parts[0]))
                status_codes.append(int(parts[1]))
            except Exception:
                continue

        if not latencies:
            return {"success": False, "error": "no latency samples collected", "requests": requests}

        err_count = sum(1 for c in status_codes if c >= 400 or c == 0)
        sla_violations = sum(
            1 for i, lat in enumerate(latencies) if lat > sla_latency_s or status_codes[i] >= 400
        )
        return {
            "success": True,
            "requests": len(latencies),
            "avg_s": round(sum(latencies) / len(latencies), 4),
            "p50_s": round(self._percentile(latencies, 50) or 0.0, 4),
            "p95_s": round(self._percentile(latencies, 95) or 0.0, 4),
            "max_s": round(max(latencies), 4),
            "error_rate_pct": round((err_count / len(latencies)) * 100.0, 2),
            "sla_latency_threshold_s": sla_latency_s,
            "sla_violations_count": sla_violations,
            "sla_violation_rate_pct": round((sla_violations / len(latencies)) * 100.0, 2),
        }

    def _cost_proxy(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        cpu_request_m = self._get_cpu_request_m()
        replicas = [int(s.get("deployment_replicas", 0)) for s in samples if "deployment_replicas" in s]
        if not replicas:
            replicas = [0]
        avg_replicas = sum(replicas) / len(replicas)
        # Approximate requested vCPU footprint over the measured window.
        avg_requested_vcpu = (avg_replicas * cpu_request_m) / 1000.0
        return {
            "cpu_request_per_pod_m": cpu_request_m,
            "avg_replicas": round(avg_replicas, 3),
            "avg_requested_vcpu": round(avg_requested_vcpu, 3),
        }

    def _sleep_if_needed(self, seconds: int) -> None:
        if seconds > 0:
            time.sleep(seconds)

    def _cluster_context(self) -> str:
        res = self._run("kubectl config current-context", timeout=10)
        if res.get("success") and res.get("stdout"):
            return res["stdout"].splitlines()[0]
        return "unknown"

    def _ensure_python_kubeconfig(self) -> Optional[str]:
        """
        Create a kubeconfig file from the active kubectl context so Python client
        and kubectl operate on the same cluster context.
        """
        if self._python_kubeconfig_path:
            return self._python_kubeconfig_path

        cfg = self._run("kubectl config view --raw -o json", timeout=20)
        if not cfg.get("success") or not cfg.get("stdout"):
            return None

        tmp = NamedTemporaryFile(prefix="autosage_kubeconfig_", suffix=".json", delete=False)
        tmp.write(cfg["stdout"].encode("utf-8"))
        tmp.flush()
        tmp.close()
        self._python_kubeconfig_path = tmp.name
        return self._python_kubeconfig_path

    def _verify_realtime_metrics_source(self) -> Dict[str, Any]:
        if not self.integration or not getattr(self.integration, "ai_integration", None):
            return {"ok": False, "error": "ai_integration not initialized"}

        ai = self.integration.ai_integration
        collector = getattr(ai, "metrics_collector", None)
        if collector is None:
            return {"ok": False, "error": "metrics_collector missing"}
        if getattr(collector, "k8s_client", None) is None:
            return {"ok": False, "error": "k8s_client unavailable (demo fallback path likely active)"}

        try:
            analysis = ai.get_current_analysis()
        except Exception as exc:
            return {"ok": False, "error": f"failed to get current analysis: {exc}"}

        if analysis.get("demo_mode"):
            return {"ok": False, "error": "demo_mode=true detected in AI monitoring analysis"}

        # Compare context from generated kubeconfig with current kubectl context.
        expected_context = self._cluster_context()
        if self._python_kubeconfig_path:
            ctx_cmd = f"kubectl --kubeconfig '{self._python_kubeconfig_path}' config current-context"
            ctx = self._run(ctx_cmd, timeout=10)
            if ctx.get("success") and ctx.get("stdout"):
                py_context = ctx["stdout"].splitlines()[0]
                if py_context != expected_context:
                    return {
                        "ok": False,
                        "error": (
                            f"context mismatch: kubectl={expected_context}, "
                            f"python_kubeconfig={py_context}"
                        ),
                    }

        return {"ok": True, "context": expected_context}

    def _extract_metric(self, method_result: Dict[str, Any], metric_path: str) -> Optional[float]:
        current: Any = method_result
        for part in metric_path.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current.get(part)
        if current is None:
            return None
        try:
            return float(current)
        except (TypeError, ValueError):
            return None

    def _ci_95(self, values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {
                "n": 0,
                "mean": None,
                "std": None,
                "margin_error": None,
                "ci_lower": None,
                "ci_upper": None,
            }
        n = len(values)
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if n > 1 else 0.0
        margin_error = 1.96 * (std_val / math.sqrt(n)) if n > 1 else 0.0
        return {
            "n": n,
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "margin_error": round(margin_error, 4),
            "ci_lower": round(mean_val - margin_error, 4),
            "ci_upper": round(mean_val + margin_error, 4),
        }

    def _aggregate_method_runs(self, runs: List[Dict[str, Any]], metric_map: Dict[str, str]) -> Dict[str, Any]:
        successful = [r for r in runs if r.get("success")]
        aggregated: Dict[str, Any] = {
            "total_runs": len(runs),
            "successful_runs": len(successful),
            "success_rate_pct": round((len(successful) / len(runs)) * 100.0, 2) if runs else 0.0,
            "metrics": {},
        }
        for out_name, metric_path in metric_map.items():
            values: List[float] = []
            for run in successful:
                value = self._extract_metric(run, metric_path)
                if value is not None:
                    values.append(value)
            ci = self._ci_95(values)
            ci["values"] = [round(v, 4) for v in values]
            aggregated["metrics"][out_name] = ci
        return aggregated

    def _pre_method_setup(self) -> Dict[str, Any]:
        self._reset_resource_baseline()
        self._set_cpu_load_percent(30)
        self._set_replicas(self.min_replicas)
        rollout_ok = self._wait_rollout()
        dep_replicas = self._get_deployment_replicas()
        _ = self.hpa.delete_hpa(f"{self.deployment}-hpa", self.namespace)
        _ = self.vpa.delete_vpa(f"{self.deployment}-vpa", self.namespace)
        self._sleep_if_needed(self.warmup_s)
        return {
            "rollout_ok": rollout_ok,
            "deployment_replicas": dep_replicas,
            "expected_min_replicas": self.min_replicas,
            "baseline_ok": rollout_ok and dep_replicas == self.min_replicas,
        }

    def _post_method_cleanup(self) -> Dict[str, Any]:
        _ = self.hpa.delete_hpa(f"{self.deployment}-hpa", self.namespace)
        _ = self.vpa.delete_vpa(f"{self.deployment}-vpa", self.namespace)
        self._set_cpu_load_percent(30)
        self._set_replicas(self.min_replicas)
        rollout_ok = self._wait_rollout()
        self._sleep_if_needed(self.cooldown_s)
        dep_replicas = self._get_deployment_replicas()
        return {
            "rollout_ok": rollout_ok,
            "deployment_replicas_after_cleanup": dep_replicas,
            "cleanup_ok": rollout_ok and dep_replicas == self.min_replicas,
        }

    def run_native_hpa(self, timeout_s: int = 240, sla_latency_s: float = 0.5) -> Dict[str, Any]:
        hpa_name = f"{self.deployment}-hpa"
        self._reset_resource_baseline()
        self._set_replicas(self.min_replicas)
        self._wait_rollout()
        baseline_replicas = self._get_deployment_replicas()
        _ = self.hpa.delete_hpa(hpa_name, self.namespace)

        create_start = time.time()
        create_res = self.hpa.create_hpa(
            self.deployment,
            self.namespace,
            min_replicas=self.min_replicas,
            max_replicas=self.max_replicas,
            cpu_target=50,
            memory_target=80,
        )
        create_latency = time.time() - create_start
        if not create_res.get("success"):
            return {
                "success": False,
                "error": create_res.get("error", "failed to create hpa"),
                "create_latency_s": round(create_latency, 3),
            }

        # Drive high CPU and monitor HPA status.
        self._apply_scenario_initial_load()
        self._wait_rollout()

        started = time.time()
        first_scale_up_latency = None
        peak_current = baseline_replicas
        peak_desired = baseline_replicas
        samples = []

        while time.time() - started < timeout_s:
            hpa_status = self.hpa.get_hpa(hpa_name, self.namespace)
            dep_replicas = self._get_deployment_replicas()
            if hpa_status.get("success"):
                status = hpa_status.get("status", {})
                desired = int(status.get("desired_replicas", dep_replicas))
                current = int(status.get("current_replicas", dep_replicas))
                cpu_usage = status.get("metrics", {}).get("cpu_usage", 0)
            else:
                desired = dep_replicas
                current = dep_replicas
                cpu_usage = 0

            peak_current = max(peak_current, current, dep_replicas)
            peak_desired = max(peak_desired, desired)
            samples.append(
                {
                    "t_s": round(time.time() - started, 1),
                    "current_replicas": current,
                    "desired_replicas": desired,
                    "deployment_replicas": dep_replicas,
                    "cpu_usage_percent": cpu_usage,
                }
            )

            if first_scale_up_latency is None and (desired > baseline_replicas or dep_replicas > baseline_replicas):
                first_scale_up_latency = time.time() - started

            self._apply_scenario_midtest_load(time.time() - started)

            time.sleep(10)

        # Cleanup HPA and normalize load.
        self.hpa.delete_hpa(hpa_name, self.namespace)
        self._set_cpu_load_percent(30)
        self._wait_rollout()
        latency_metrics = self._measure_service_latency(requests=20, sla_latency_s=sla_latency_s)
        cost_proxy = self._cost_proxy(samples)

        return {
            "success": True,
            "baseline_replicas": baseline_replicas,
            "create_latency_s": round(create_latency, 3),
            "first_scale_up_latency_s": round(first_scale_up_latency, 3) if first_scale_up_latency else None,
            "peak_current_replicas": peak_current,
            "peak_desired_replicas": peak_desired,
            "latency_sla": latency_metrics,
            "cost_proxy": cost_proxy,
            "samples": samples,
        }

    def run_native_vpa(self, timeout_s: int = 240, sla_latency_s: float = 0.5) -> Dict[str, Any]:
        vpa_name = f"{self.deployment}-vpa"
        self._reset_resource_baseline()
        self._set_replicas(self.min_replicas)
        self._wait_rollout()
        self.vpa.delete_vpa(vpa_name, self.namespace)

        create_start = time.time()
        create_res = self.vpa.create_vpa(
            self.deployment,
            self.namespace,
            min_cpu="100m",
            max_cpu="1500m",
            min_memory="128Mi",
            max_memory="1024Mi",
            update_mode="Auto",
        )
        create_latency = time.time() - create_start
        if not create_res.get("success"):
            return {
                "success": False,
                "error": create_res.get("error", "failed to create vpa"),
                "create_latency_s": round(create_latency, 3),
                "vpa_not_installed": bool(create_res.get("vpa_not_installed")),
            }

        self._apply_scenario_initial_load()
        self._wait_rollout()

        started = time.time()
        first_recommendation_latency = None
        recommendation_snapshot = None
        samples = []

        while time.time() - started < timeout_s:
            status_res = self.vpa.get_vpa(vpa_name, self.namespace)
            if status_res.get("success"):
                recs = status_res.get("status", {}).get("recommendations", [])
            else:
                recs = []

            has_recommendation = bool(recs)
            samples.append(
                {
                    "t_s": round(time.time() - started, 1),
                    "has_recommendation": has_recommendation,
                    "recommendations_count": len(recs),
                }
            )

            if has_recommendation and first_recommendation_latency is None:
                first_recommendation_latency = time.time() - started
                recommendation_snapshot = recs
                break

            time.sleep(10)

        self.vpa.delete_vpa(vpa_name, self.namespace)
        self._set_cpu_load_percent(30)
        self._wait_rollout()
        latency_metrics = self._measure_service_latency(requests=20, sla_latency_s=sla_latency_s)
        # Approximate cost proxy for VPA path using constant baseline replica behavior.
        baseline_samples = [{"deployment_replicas": self._get_deployment_replicas()}]
        cost_proxy = self._cost_proxy(baseline_samples)

        return {
            "success": True,
            "create_latency_s": round(create_latency, 3),
            "first_recommendation_latency_s": round(first_recommendation_latency, 3)
            if first_recommendation_latency
            else None,
            "recommendation_snapshot": recommendation_snapshot,
            "latency_sla": latency_metrics,
            "cost_proxy": cost_proxy,
            "samples": samples,
        }

    def run_autosage(self, timeout_s: int = 300, sla_latency_s: float = 0.5) -> Dict[str, Any]:
        self._reset_resource_baseline()
        self._set_replicas(self.min_replicas)
        self._wait_rollout()
        baseline_replicas = self._get_deployment_replicas()
        self._apply_scenario_initial_load()
        self._wait_rollout()

        kubeconfig_path = self._ensure_python_kubeconfig()
        if kubeconfig_path:
            os.environ["KUBECONFIG"] = kubeconfig_path
        os.environ["MCDA_PROFILE"] = self.mcda_profile
        os.environ["MCDA_AGREEMENT_THRESHOLD"] = str(self.mcda_agreement_threshold)
        profile_flags = {
            "full": {"AUTOSAGE_USE_LLM": "1", "AUTOSAGE_ENABLE_MCDA_VALIDATION": "1"},
            "llm_no_mcda": {"AUTOSAGE_USE_LLM": "1", "AUTOSAGE_ENABLE_MCDA_VALIDATION": "0"},
            "rule_only": {"AUTOSAGE_USE_LLM": "0", "AUTOSAGE_ENABLE_MCDA_VALIDATION": "0"},
        }
        active_flags = profile_flags.get(self.autosage_profile, profile_flags["full"])
        os.environ.update(active_flags)

        self.integration = AutoscalingIntegration(kubeconfig_path=kubeconfig_path)

        source_check = self._verify_realtime_metrics_source()
        if not source_check.get("ok"):
            return {
                "success": False,
                "aborted": True,
                "aborted_reason": source_check.get("error", "unknown monitoring source issue"),
                "aborted_demo_mode": True,
            }

        try:
            started = time.time()
            apply_res = self.integration.enable_predictive_autoscaling(
                self.deployment,
                self.namespace,
                min_replicas=self.min_replicas,
                max_replicas=self.max_replicas,
            )
            recommendation_latency = time.time() - started
            timing_breakdown = {}
            if isinstance(apply_res, dict):
                timing_breakdown = apply_res.get("timing_breakdown", {}) or {}

            if timing_breakdown.get("total_decision_s") is not None:
                try:
                    recommendation_latency = float(timing_breakdown.get("total_decision_s"))
                except Exception:
                    pass

            first_replica_change_latency = None
            peak_replicas = baseline_replicas
            samples = []

            poll_start = time.time()
            while time.time() - poll_start < timeout_s:
                replicas = self._get_deployment_replicas()
                peak_replicas = max(peak_replicas, replicas)
                samples.append(
                    {"t_s": round(time.time() - poll_start, 1), "deployment_replicas": replicas}
                )
                if first_replica_change_latency is None and replicas != baseline_replicas:
                    first_replica_change_latency = time.time() - poll_start
                self._apply_scenario_midtest_load(time.time() - poll_start)
                time.sleep(10)
            latency_metrics = self._measure_service_latency(requests=20, sla_latency_s=sla_latency_s)
            cost_proxy = self._cost_proxy(samples)

            return {
                "success": bool(apply_res.get("success")),
                "apply_result": apply_res,
                "recommendation_latency_s": round(recommendation_latency, 3),
                "baseline_replicas": baseline_replicas,
                "first_replica_change_latency_s": round(first_replica_change_latency, 3)
                if first_replica_change_latency
                else None,
                "peak_replicas": peak_replicas,
                "latency_sla": latency_metrics,
                "cost_proxy": cost_proxy,
                "samples": samples,
                "timing_breakdown": {
                    "metrics_collection_s": timing_breakdown.get("metrics_collection_s"),
                    "forecast_s": timing_breakdown.get("forecast_s"),
                    "llm_inference_s": timing_breakdown.get("llm_inference_s"),
                    "mcda_validation_s": timing_breakdown.get("mcda_validation_s"),
                    "actuation_s": timing_breakdown.get("actuation_s"),
                    "total_decision_s": timing_breakdown.get("total_decision_s"),
                },
            }
        finally:
            try:
                self.integration.disable_predictive_autoscaling(self.deployment, self.namespace)
            except Exception:
                pass
            try:
                self.integration.stop_predictive_scaling_loop()
            except Exception:
                pass
            self._set_cpu_load_percent(30)
            self._wait_rollout()

    def run_all(self, timeout_s: int, sla_latency_s: float, runs: int) -> Dict[str, Any]:
        # Basic pre-check.
        dep = self.hpa.get_deployment_replicas(self.deployment, self.namespace)
        if not dep.get("success"):
            return {
                "success": False,
                "error": f"Deployment {self.namespace}/{self.deployment} not found",
                "details": dep,
            }

        hpa_runs: List[Dict[str, Any]] = []
        vpa_runs: List[Dict[str, Any]] = []
        autosage_runs: List[Dict[str, Any]] = []

        for run_idx in range(runs):
            run_seed = self.seed_base + run_idx
            random.seed(run_seed)
            run_meta = {
                "run_id": run_idx + 1,
                "seed": run_seed,
                "started_at": datetime.utcnow().isoformat() + "Z",
                "cluster_context": self._cluster_context(),
            }

            pre_hpa = self._pre_method_setup()
            hpa_res = self.run_native_hpa(timeout_s=timeout_s, sla_latency_s=sla_latency_s)
            post_hpa = self._post_method_cleanup()
            hpa_runs.append(
                {
                    **hpa_res,
                    "run_metadata": run_meta,
                    "fairness_checks": {"pre": pre_hpa, "post": post_hpa},
                }
            )

            pre_vpa = self._pre_method_setup()
            vpa_res = self.run_native_vpa(timeout_s=timeout_s, sla_latency_s=sla_latency_s)
            post_vpa = self._post_method_cleanup()
            vpa_runs.append(
                {
                    **vpa_res,
                    "run_metadata": run_meta,
                    "fairness_checks": {"pre": pre_vpa, "post": post_vpa},
                }
            )

            pre_as = self._pre_method_setup()
            autosage_res = self.run_autosage(timeout_s=timeout_s, sla_latency_s=sla_latency_s)
            post_as = self._post_method_cleanup()
            autosage_runs.append(
                {
                    **autosage_res,
                    "run_metadata": run_meta,
                    "fairness_checks": {"pre": pre_as, "post": post_as},
                }
            )
            if autosage_res.get("aborted_demo_mode"):
                return {
                    "success": False,
                    "aborted": True,
                    "aborted_reason": autosage_res.get("aborted_reason", "AutoSage demo mode detected"),
                    "aborted_stage": "autosage",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "target": {
                        "deployment": self.deployment,
                        "namespace": self.namespace,
                        "min_replicas": self.min_replicas,
                        "max_replicas": self.max_replicas,
                    },
                    "experiment": {
                        "runs_requested": runs,
                        "runs_completed_before_abort": run_idx + 1,
                        "timeout_s_per_method": timeout_s,
                        "warmup_s": self.warmup_s,
                        "cooldown_s": self.cooldown_s,
                        "seed_base": self.seed_base,
                        "mcda_profile": self.mcda_profile,
                        "mcda_agreement_threshold": self.mcda_agreement_threshold,
                        "scenario": self.scenario,
                        "cluster_context": self._cluster_context(),
                    },
                    "native_hpa_runs": hpa_runs,
                    "native_vpa_runs": vpa_runs,
                    "autosage_runs": autosage_runs,
                }

        results = {
            "success": True,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "target": {
                "deployment": self.deployment,
                "namespace": self.namespace,
                "min_replicas": self.min_replicas,
                "max_replicas": self.max_replicas,
            },
            "experiment": {
                "runs": runs,
                "timeout_s_per_method": timeout_s,
                "warmup_s": self.warmup_s,
                "cooldown_s": self.cooldown_s,
                "seed_base": self.seed_base,
                "mcda_profile": self.mcda_profile,
                "mcda_agreement_threshold": self.mcda_agreement_threshold,
                "scenario": self.scenario,
                "autosage_profile": self.autosage_profile,
                "cluster_context": self._cluster_context(),
            },
            "sla_policy": {
                "latency_threshold_s": sla_latency_s,
                "violation_definition": "HTTP status >= 400 or latency above threshold",
            },
            "native_hpa_runs": hpa_runs,
            "native_vpa_runs": vpa_runs,
            "autosage_runs": autosage_runs,
            "native_hpa": hpa_runs[-1] if hpa_runs else {},
            "native_vpa": vpa_runs[-1] if vpa_runs else {},
            "autosage": autosage_runs[-1] if autosage_runs else {},
            "aggregated": {
                "native_hpa": self._aggregate_method_runs(
                    hpa_runs,
                    {
                        "provisioning_latency_s": "create_latency_s",
                        "decision_reaction_latency_s": "first_scale_up_latency_s",
                        "p95_latency_s": "latency_sla.p95_s",
                        "sla_violation_rate_pct": "latency_sla.sla_violation_rate_pct",
                        "cost_proxy_avg_requested_vcpu": "cost_proxy.avg_requested_vcpu",
                    },
                ),
                "native_vpa": self._aggregate_method_runs(
                    vpa_runs,
                    {
                        "provisioning_latency_s": "create_latency_s",
                        "decision_reaction_latency_s": "first_recommendation_latency_s",
                        "p95_latency_s": "latency_sla.p95_s",
                        "sla_violation_rate_pct": "latency_sla.sla_violation_rate_pct",
                        "cost_proxy_avg_requested_vcpu": "cost_proxy.avg_requested_vcpu",
                    },
                ),
                "autosage": self._aggregate_method_runs(
                    autosage_runs,
                    {
                        "decision_reaction_latency_s": "recommendation_latency_s",
                        "first_applied_change_latency_s": "first_replica_change_latency_s",
                        "metrics_collection_s": "timing_breakdown.metrics_collection_s",
                        "forecast_s": "timing_breakdown.forecast_s",
                        "llm_inference_s": "timing_breakdown.llm_inference_s",
                        "mcda_validation_s": "timing_breakdown.mcda_validation_s",
                        "actuation_s": "timing_breakdown.actuation_s",
                        "total_decision_s": "timing_breakdown.total_decision_s",
                        "p95_latency_s": "latency_sla.p95_s",
                        "sla_violation_rate_pct": "latency_sla.sla_violation_rate_pct",
                        "cost_proxy_avg_requested_vcpu": "cost_proxy.avg_requested_vcpu",
                    },
                ),
            },
        }
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoSage vs native HPA/VPA comparison runner")
    parser.add_argument("--deployment", default="test-app-autoscaling")
    parser.add_argument("--namespace", default="ai4k8s-test")
    parser.add_argument("--min-replicas", type=int, default=2)
    parser.add_argument("--max-replicas", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180, help="Per-test timeout in seconds")
    parser.add_argument("--runs", type=int, default=10, help="Number of repeated runs per method")
    parser.add_argument("--warmup", type=int, default=15, help="Warm-up time before each method run (seconds)")
    parser.add_argument("--cooldown", type=int, default=15, help="Cooldown time after each method run (seconds)")
    parser.add_argument("--seed-base", type=int, default=42, help="Base seed used to tag repeated runs")
    parser.add_argument(
        "--mcda-profile",
        default="balanced",
        choices=["balanced", "performance_first", "cost_optimized", "stability_first", "aggressive_scaling"],
        help="MCDA weight profile used by AutoSage",
    )
    parser.add_argument(
        "--mcda-agreement-threshold",
        type=float,
        default=0.15,
        help="MCDA override threshold (lower = more likely to override LLM choice)",
    )
    parser.add_argument(
        "--scenario",
        default="short_burst",
        choices=["short_burst", "periodic_burst", "sustained_drift"],
        help="Workload scenario profile applied during each run",
    )
    parser.add_argument(
        "--autosage-profile",
        default="full",
        choices=["full", "llm_no_mcda", "rule_only"],
        help="AutoSage ablation profile: full (LLM+MCDA), llm_no_mcda, or rule_only",
    )
    parser.add_argument(
        "--sla-latency-threshold",
        type=float,
        default=0.5,
        help="SLA latency threshold in seconds",
    )
    parser.add_argument(
        "--output",
        default=f"thesis_reports/hpa_vpa_comparison_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
    )
    args = parser.parse_args()

    runner = AutoscalingComparisonRunner(
        deployment=args.deployment,
        namespace=args.namespace,
        min_replicas=args.min_replicas,
        max_replicas=args.max_replicas,
        warmup_s=args.warmup,
        cooldown_s=args.cooldown,
        seed_base=args.seed_base,
        mcda_profile=args.mcda_profile,
        mcda_agreement_threshold=args.mcda_agreement_threshold,
        scenario=args.scenario,
        autosage_profile=args.autosage_profile,
    )
    results = runner.run_all(timeout_s=args.timeout, sla_latency_s=args.sla_latency_threshold, runs=args.runs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
