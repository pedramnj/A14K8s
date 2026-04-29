"""Prometheus metrics collector with optional simulation mode.

Usage:
    from collector.metrics import PrometheusCollector
    pc = PrometheusCollector(mode='sim', sim_env=env)
    pc.get_cpu_usage()

When mode='sim', the collector reads from the provided sim_env (SimulatedEnvironment) instead of querying Prometheus.
"""
import requests
from typing import Dict

class PrometheusCollector:
    def __init__(self, base_url='http://localhost:9090', mode='real', sim_env=None):
        self.base_url = base_url.rstrip('/')
        self.mode = mode
        self.sim_env = sim_env

    def query(self, promql: str) -> Dict:
        if self.mode == 'sim':
            # simulation returns a tiny structure
            last = self.sim_env.history[-1] if self.sim_env and self.sim_env.history else None
            return {
                'status': 'success',
                'data': last
            }
        url = f"{self.base_url}/api/v1/query"
        r = requests.get(url, params={'query': promql}, timeout=5)
        r.raise_for_status()
        data = r.json()
        return data

    def get_cpu_usage(self, namespace='default', deployment=None):
        """Return total CPU rate (in cores) for the matching pods, as a float.

        Vendored fix: upstream returned the raw Prometheus response dict in
        real mode, which broke any caller treating the value as a number.
        """
        if self.mode == 'sim':
            last = self.sim_env.history[-1] if self.sim_env and self.sim_env.history else {'cpu_util': 0.0}
            return last.get('cpu_util', 0.0)
        if deployment:
            prom = f"sum(rate(container_cpu_usage_seconds_total{{namespace=\"{namespace}\", pod=~\"{deployment}.*\"}}[1m]))"
        else:
            prom = f"sum(rate(container_cpu_usage_seconds_total{{namespace=\"{namespace}\"}}[1m]))"
        data = self.query(prom)
        try:
            result = data.get('data', {}).get('result', [])
            if not result:
                return 0.0
            return float(result[0]['value'][1])
        except (KeyError, IndexError, ValueError, TypeError):
            return 0.0