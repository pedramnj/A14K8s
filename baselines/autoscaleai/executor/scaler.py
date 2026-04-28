"""Kubernetes scaler using the official python client.
Requires: pip install kubernetes
"""
try:
 from kubernetes import client, config
 _HAS_K8S = True
except Exception:
 _HAS_K8S = False

class K8sScaler:
    def __init__(self, in_cluster=False, mode='real', sim_env=None):
        self.mode = mode
        self.sim_env = sim_env
        if self.mode == 'real':
            if not _HAS_K8S:
                raise RuntimeError('kubernetes client not available')
            if in_cluster:
                config.load_incluster_config()
            else:
                config.load_kube_config()
            self.apps = client.AppsV1Api()


    def scale_deployment(self, namespace, deployment_name, replicas):
        if self.mode == 'sim':
            if self.sim_env:
                self.sim_env.replicas = int(replicas)
            return {'status': 'simulated', 'replicas': int(replicas)}
        # Vendored fix: original upstream had `else: return ...` here, which
        # made real-mode scaling unreachable. The real-mode path now runs.
        body = {"spec": {"replicas": int(replicas)}}
        resp = self.apps.patch_namespaced_deployment_scale(
            name=deployment_name, namespace=namespace, body=body
        )
        return resp


if __name__ == '__main__':
    scaler = K8sScaler(in_cluster=False)
    print('Set scale to 2 (example)')
    scaler.scale_deployment('default', 'my-deploy', 2)