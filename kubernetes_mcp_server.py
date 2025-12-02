#!/usr/bin/env python3
"""
Model Context Protocol (MCP) bridge for Kubernetes operations.
Exposes kubectl-driven tools via the MCP interface so the web app can
query Kubernetes deterministically.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency for MCP runtime
    import mcp.server  # type: ignore
    import mcp.server.stdio  # type: ignore
    import mcp.types as types  # type: ignore

    MCP_AVAILABLE = True
except ImportError:  # pragma: no cover - fall back to local mode
    MCP_AVAILABLE = False
    mcp = None  # type: ignore
    types = None  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _run(command: List[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """Helper that wraps subprocess.run with consistent parameters."""
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


class KubernetesMCP:
    """Utility class that exposes kubectl-backed helpers."""

    def __init__(self) -> None:
        self.kubectl_available = self._check_binary(["kubectl", "version", "--client"])
        self.docker_available = self._check_binary(["docker", "--version"])

    @staticmethod
    def _check_binary(command: List[str]) -> bool:
        try:
            proc = _run(command, timeout=10)
            return proc.returncode == 0
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("%s not available: %s", command[0], exc)
            return False

    # ------------------------------------------------------------------
    # Cluster information helpers
    # ------------------------------------------------------------------
    async def get_cluster_info(self) -> Dict[str, Any]:
        if not self.kubectl_available:
            return {
                "error": "kubectl not available",
                "suggestion": "Install kubectl or adjust PATH for the MCP service",
            }

        try:
            version_proc = _run(["kubectl", "version", "--output=json"], timeout=30)
            nodes_proc = _run(["kubectl", "get", "nodes", "-o", "json"], timeout=30)

            version_data = json.loads(version_proc.stdout or "{}") if version_proc.returncode == 0 else {}
            nodes_data = json.loads(nodes_proc.stdout or "{}") if nodes_proc.returncode == 0 else {}

            nodes: List[Dict[str, Any]] = []
            for node in nodes_data.get("items", []):
                conditions = node.get("status", {}).get("conditions", [])
                status = conditions[-1].get("type", "Unknown") if conditions else "Unknown"
                role = node.get("metadata", {}).get("labels", {}).get("kubernetes.io/role", "worker")
                nodes.append(
                    {
                        "name": node.get("metadata", {}).get("name", "unknown"),
                        "status": status,
                        "role": role,
                    }
                )

            return {
                "cluster_version": version_data.get("serverVersion", {}).get("gitVersion", "unknown"),
                "node_count": len(nodes),
                "nodes": nodes,
                "collected_at": datetime.utcnow().isoformat(),
            }
        except Exception as exc:  # pragma: no cover
            return {"error": f"Failed to get cluster info: {exc}"}

    async def get_pods(self, namespace: str = "default") -> Dict[str, Any]:
        if not self.kubectl_available:
            return {"error": "kubectl not available"}

        try:
            if namespace in {None, "", "all", "all-namespaces", "--all-namespaces"}:
                command = ["kubectl", "get", "pods", "--all-namespaces", "-o", "json"]
                display_namespace = "all"
            else:
                command = ["kubectl", "get", "pods", "-n", namespace, "-o", "json"]
                display_namespace = namespace

            proc = _run(command, timeout=30)
            if proc.returncode != 0:
                return {"error": proc.stderr or "kubectl get pods failed"}

            data = json.loads(proc.stdout or "{}")
            pods: List[Dict[str, Any]] = []
            for pod in data.get("items", []):
                meta = pod.get("metadata", {})
                status = pod.get("status", {})
                containers = pod.get("spec", {}).get("containers", [])
                container_statuses = status.get("containerStatuses", [])
                ready_count = sum(1 for c in container_statuses if c.get("ready"))

                pods.append(
                    {
                        "name": meta.get("name", "unknown"),
                        "namespace": meta.get("namespace", display_namespace),
                        "status": status.get("phase", "Unknown"),
                        "ready": f"{ready_count}/{len(containers)}",
                        "restarts": sum(cs.get("restartCount", 0) for cs in container_statuses),
                        "age": meta.get("creationTimestamp"),
                    }
                )

            return {"namespace": display_namespace, "pod_count": len(pods), "pods": pods}
        except Exception as exc:  # pragma: no cover
            return {"error": f"Failed to get pods: {exc}"}

    async def get_services(self, namespace: str = "default") -> Dict[str, Any]:
        if not self.kubectl_available:
            return {"error": "kubectl not available"}

        try:
            proc = _run(["kubectl", "get", "services", "-n", namespace, "-o", "json"], timeout=30)
            if proc.returncode != 0:
                return {"error": proc.stderr or "kubectl get services failed"}

            services: List[Dict[str, Any]] = []
            data = json.loads(proc.stdout or "{}")
            for svc in data.get("items", []):
                spec = svc.get("spec", {})
                status = svc.get("status", {})
                lb_ingress = status.get("loadBalancer", {}).get("ingress") or []
                services.append(
                    {
                        "name": svc.get("metadata", {}).get("name", "unknown"),
                        "type": spec.get("type", "Unknown"),
                        "cluster_ip": spec.get("clusterIP"),
                        "external_ip": lb_ingress[0].get("ip") if lb_ingress else None,
                        "ports": [f"{p.get('port')}:{p.get('targetPort')}" for p in spec.get("ports", [])],
                    }
                )

            return {"namespace": namespace, "service_count": len(services), "services": services}
        except Exception as exc:  # pragma: no cover
            return {"error": f"Failed to get services: {exc}"}

    async def get_deployments(self, namespace: str = "default") -> Dict[str, Any]:
        if not self.kubectl_available:
            return {"error": "kubectl not available"}

        try:
            proc = _run(["kubectl", "get", "deployments", "-n", namespace, "-o", "json"], timeout=30)
            if proc.returncode != 0:
                return {"error": proc.stderr or "kubectl get deployments failed"}

            deployments: List[Dict[str, Any]] = []
            data = json.loads(proc.stdout or "{}")
            for deploy in data.get("items", []):
                spec = deploy.get("spec", {})
                status = deploy.get("status", {})
                deployments.append(
                    {
                        "name": deploy.get("metadata", {}).get("name", "unknown"),
                        "replicas": f"{status.get('readyReplicas', 0)}/{spec.get('replicas', 0)}",
                        "updated": status.get("updatedReplicas", 0),
                        "available": status.get("availableReplicas", 0),
                    }
                )

            return {"namespace": namespace, "deployment_count": len(deployments), "deployments": deployments}
        except Exception as exc:  # pragma: no cover
            return {"error": f"Failed to get deployments: {exc}"}

    async def list_resources(
        self,
        api_version: str,
        kind: str,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.kubectl_available:
            return {"error": "kubectl not available"}

        resource = kind.lower()
        if not resource.endswith("s"):
            resource = f"{resource}s"

        try:
            command = ["kubectl", "get", resource, "-o", "json"]
            if namespace and namespace not in {"all", "all-namespaces", "--all-namespaces"}:
                command = ["kubectl", "get", resource, "-n", namespace, "-o", "json"]

            proc = _run(command, timeout=30)
            if proc.returncode != 0:
                return {"error": proc.stderr or f"Failed to list {kind}"}

            data = json.loads(proc.stdout or "{}")
            return {
                "apiVersion": api_version,
                "kind": kind,
                "count": len(data.get("items", [])),
                "items": data.get("items", []),
            }
        except Exception as exc:  # pragma: no cover
            return {"error": f"Failed to list resources: {exc}"}

    async def list_namespaces(self) -> Dict[str, Any]:
        return await self.list_resources("v1", "Namespace")

    async def list_events(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        if not self.kubectl_available:
            return {"error": "kubectl not available"}

        try:
            if namespace and namespace not in {"all", "all-namespaces", "--all-namespaces"}:
                command = ["kubectl", "get", "events", "-n", namespace, "-o", "json"]
            else:
                command = ["kubectl", "get", "events", "--all-namespaces", "-o", "json"]

            proc = _run(command, timeout=30)
            if proc.returncode != 0:
                return {"error": proc.stderr or "Failed to list events"}

            data = json.loads(proc.stdout or "{}")
            return {"count": len(data.get("items", [])), "events": data.get("items", [])}
        except Exception as exc:  # pragma: no cover
            return {"error": f"Failed to list events: {exc}"}

    # ------------------------------------------------------------------
    # Pod-level helpers
    # ------------------------------------------------------------------
    async def get_pod_details(self, name: str, namespace: str = "default") -> Dict[str, Any]:
        if not self.kubectl_available:
            return {"error": "kubectl not available"}

        try:
            proc = _run(["kubectl", "get", "pod", name, "-n", namespace, "-o", "json"], timeout=30)
            if proc.returncode != 0:
                return {"error": proc.stderr or "pod not found"}

            return json.loads(proc.stdout or "{}")
        except Exception as exc:  # pragma: no cover
            return {"error": f"Failed to get pod details: {exc}"}

    async def delete_pod(self, name: str, namespace: str = "default") -> Dict[str, Any]:
        if not self.kubectl_available:
            return {"error": "kubectl not available"}

        try:
            proc = _run(["kubectl", "delete", "pod", name, "-n", namespace], timeout=30)
            if proc.returncode != 0:
                return {"error": proc.stderr or "Failed to delete pod"}

            return {
                "name": name,
                "namespace": namespace,
                "status": "deleted",
                "message": proc.stdout.strip(),
            }
        except subprocess.TimeoutExpired:
            return {"error": "Delete operation timed out"}
        except Exception as exc:  # pragma: no cover
            return {"error": f"Failed to delete pod: {exc}"}

    async def get_pod_logs(self, pod_name: str, namespace: str = "default", lines: int = 100) -> Dict[str, Any]:
        if not self.kubectl_available:
            return {"error": "kubectl not available"}

        try:
            proc = _run(["kubectl", "logs", pod_name, "-n", namespace, f"--tail={lines}"], timeout=30)
            return {
                "pod_name": pod_name,
                "namespace": namespace,
                "lines": lines,
                "logs": proc.stdout,
                "error": proc.stderr if proc.returncode != 0 else None,
            }
        except Exception as exc:  # pragma: no cover
            return {"error": f"Failed to get logs: {exc}"}

    async def get_pod_top(self, pod_name: Optional[str] = None, namespace: str = "default") -> Dict[str, Any]:
        if not self.kubectl_available:
            return {"error": "kubectl not available"}

        try:
            if pod_name:
                command = ["kubectl", "top", "pod", pod_name, "-n", namespace, "--no-headers"]
            elif namespace in {"all", "--all-namespaces"}:
                command = ["kubectl", "top", "pods", "--all-namespaces", "--no-headers"]
            else:
                command = ["kubectl", "top", "pods", "-n", namespace, "--no-headers"]

            proc = _run(command, timeout=30)
            if proc.returncode != 0:
                return {
                    "error": proc.stderr or "Failed to get pod metrics",
                    "suggestion": "Ensure metrics-server is installed and accessible",
                }

            metrics: List[Dict[str, Any]] = []
            for line in proc.stdout.strip().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                if namespace in {"all", "--all-namespaces"} and len(parts) >= 4:
                    pod_namespace, name, cpu, memory = parts[:4]
                elif len(parts) >= 3:
                    pod_namespace = namespace
                    name, cpu, memory = parts[:3]
                else:
                    continue
                metrics.append(
                    {
                        "name": name,
                        "namespace": pod_namespace,
                        "cpu": cpu,
                        "memory": memory,
                    }
                )

            return {
                "namespace": namespace,
                "pod_name": pod_name,
                "metrics": metrics,
                "total_pods": len(metrics),
            }
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out - metrics-server may be slow"}
        except Exception as exc:  # pragma: no cover
            return {"error": f"Failed to get pod metrics: {exc}"}

    async def exec_into_pod(
        self,
        pod_name: str,
        namespace: str,
        command: str,
        container: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.kubectl_available:
            return {"error": "kubectl not available"}

        try:
            full_cmd = ["kubectl", "exec", pod_name, "-n", namespace]
            if container:
                full_cmd.extend(["-c", container])
            full_cmd.extend(["--", "sh", "-c", command])

            proc = _run(full_cmd, timeout=60)
            return {
                "pod": pod_name,
                "namespace": namespace,
                "container": container,
                "command": command,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "return_code": proc.returncode,
                "success": proc.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out"}
        except Exception as exc:  # pragma: no cover
            return {"error": f"Failed to exec into pod: {exc}"}

    async def run_container_in_pod(
        self,
        image: str,
        name: Optional[str] = None,
        namespace: str = "default",
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not self.kubectl_available:
            return {"error": "kubectl not available"}

        try:
            import uuid

            pod_name = name or f"run-{image.split('/')[-1].split(':')[0]}-{uuid.uuid4().hex[:8]}"
            cmd = ["kubectl", "run", pod_name, f"--image={image}", "-n", namespace, "--restart=Never"]
            if command:
                cmd.extend(["--command", "--", command])
            if args:
                cmd.extend(args)

            proc = _run(cmd, timeout=60)
            if proc.returncode != 0:
                return {
                    "error": proc.stderr or "Failed to run container",
                    "command": " ".join(cmd),
                }

            return {
                "action": "created",
                "pod_name": pod_name,
                "namespace": namespace,
                "image": image,
                "command": command,
                "args": args,
                "status": "success",
                "output": proc.stdout,
            }
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out"}
        except Exception as exc:  # pragma: no cover
            return {"error": f"Failed to run container: {exc}"}

    async def execute_kubectl_command(self, command: str) -> Dict[str, Any]:
        if not self.kubectl_available:
            return {"error": "kubectl not available"}
        if not command or not command.strip():
            return {"error": "No command provided"}

        try:
            args = shlex.split(command)
            proc = _run(["kubectl", *args], timeout=60)
            return {
                "command": f"kubectl {' '.join(args)}",
                "return_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        except subprocess.TimeoutExpired:
            return {"error": "kubectl command timed out"}
        except Exception as exc:  # pragma: no cover
            return {"error": f"Failed to execute kubectl: {exc}"}

    async def get_docker_containers(self) -> Dict[str, Any]:
        if not self.docker_available:
            return {"error": "docker not available"}

        try:
            proc = _run(["docker", "ps", "-a", "--format", "{{json .}}"], timeout=30)
            if proc.returncode != 0:
                return {"error": proc.stderr or "docker ps failed"}

            containers: List[Dict[str, Any]] = []
            for line in proc.stdout.strip().splitlines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                containers.append(
                    {
                        "id": payload.get("ID"),
                        "name": payload.get("Names"),
                        "image": payload.get("Image"),
                        "status": payload.get("Status"),
                        "ports": payload.get("Ports"),
                    }
                )

            return {"container_count": len(containers), "containers": containers}
        except subprocess.TimeoutExpired:
            return {"error": "docker command timed out"}
        except Exception as exc:  # pragma: no cover
            return {"error": f"Failed to list containers: {exc}"}


k8s_mcp = KubernetesMCP()


def _wrap_response(result: Dict[str, Any]) -> List[types.TextContent]:
    text = json.dumps(result, indent=2, default=str)
    return [types.TextContent(type="text", text=text)]


if MCP_AVAILABLE:
    server = mcp.server.Server("kubernetes-monitor")  # type: ignore

    @server.list_tools()
    async def handle_list_tools() -> List[types.Tool]:  # type: ignore[return-value]
        return [
            types.Tool(
                name="get_cluster_info",
                description="Get cluster version and node status",
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            types.Tool(
                name="get_pods",
                description="List pods in a namespace",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Namespace (default: default, use 'all' for cluster)",
                            "default": "default",
                        }
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="get_services",
                description="List services with cluster/external IPs",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Namespace (default: default)",
                            "default": "default",
                        }
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="get_deployments",
                description="List deployments and replica status",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Namespace (default: default)",
                            "default": "default",
                        }
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="get_pod_logs",
                description="Fetch logs from a pod",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pod_name": {"type": "string", "description": "Pod name"},
                        "namespace": {
                            "type": "string",
                            "description": "Namespace (default: default)",
                            "default": "default",
                        },
                        "lines": {
                            "type": "integer",
                            "description": "Lines to tail",
                            "default": 100,
                        },
                    },
                    "required": ["pod_name"],
                },
            ),
            types.Tool(
                name="execute_kubectl",
                description="Run an arbitrary kubectl command",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command string without the kubectl prefix",
                        }
                    },
                    "required": ["command"],
                },
            ),
            types.Tool(
                name="get_docker_containers",
                description="List Docker containers on the host",
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            types.Tool(
                name="get_pod_top",
                description="Get CPU/memory usage for pods",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pod_name": {
                            "type": "string",
                            "description": "Optional pod name",
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Namespace (default: default, use 'all' for cluster)",
                            "default": "default",
                        },
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="exec_into_pod",
                description="Execute a shell command inside a pod",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pod_name": {"type": "string", "description": "Pod name"},
                        "namespace": {
                            "type": "string",
                            "description": "Namespace (default: default)",
                            "default": "default",
                        },
                        "command": {"type": "string", "description": "Shell command"},
                        "container": {
                            "type": "string",
                            "description": "Optional container name",
                        },
                    },
                    "required": ["pod_name", "command"],
                },
            ),
            types.Tool(
                name="run_container_in_pod",
                description="Run a one-off pod with the given image",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "image": {"type": "string", "description": "Container image"},
                        "name": {"type": "string", "description": "Optional pod name"},
                        "namespace": {
                            "type": "string",
                            "description": "Namespace (default: default)",
                            "default": "default",
                        },
                        "command": {"type": "string", "description": "Optional command"},
                        "args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional command args",
                        },
                    },
                    "required": ["image"],
                },
            ),
            types.Tool(
                name="delete_pod",
                description="Delete a pod",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Pod name"},
                        "namespace": {
                            "type": "string",
                            "description": "Namespace (default: default)",
                            "default": "default",
                        },
                    },
                    "required": ["name"],
                },
            ),
            types.Tool(
                name="list_namespaces",
                description="List namespaces in the cluster",
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            types.Tool(
                name="list_events",
                description="List Kubernetes events",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Optional namespace filter",
                        }
                    },
                    "required": [],
                },
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:  # type: ignore[return-value]
        try:
            if name == "get_cluster_info":
                result = await k8s_mcp.get_cluster_info()
            elif name == "get_pods":
                result = await k8s_mcp.get_pods(arguments.get("namespace", "default"))
            elif name == "get_services":
                result = await k8s_mcp.get_services(arguments.get("namespace", "default"))
            elif name == "get_deployments":
                result = await k8s_mcp.get_deployments(arguments.get("namespace", "default"))
            elif name == "get_pod_logs":
                result = await k8s_mcp.get_pod_logs(
                    arguments["pod_name"],
                    arguments.get("namespace", "default"),
                    arguments.get("lines", 100),
                )
            elif name == "execute_kubectl":
                result = await k8s_mcp.execute_kubectl_command(arguments.get("command", ""))
            elif name == "get_docker_containers":
                result = await k8s_mcp.get_docker_containers()
            elif name == "get_pod_top":
                result = await k8s_mcp.get_pod_top(
                    arguments.get("pod_name"),
                    arguments.get("namespace", "default"),
                )
            elif name == "exec_into_pod":
                result = await k8s_mcp.exec_into_pod(
                    arguments["pod_name"],
                    arguments.get("namespace", "default"),
                    arguments["command"],
                    arguments.get("container"),
                )
            elif name == "run_container_in_pod":
                result = await k8s_mcp.run_container_in_pod(
                    arguments["image"],
                    arguments.get("name"),
                    arguments.get("namespace", "default"),
                    arguments.get("command"),
                    arguments.get("args"),
                )
            elif name == "delete_pod":
                result = await k8s_mcp.delete_pod(arguments["name"], arguments.get("namespace", "default"))
            elif name == "list_namespaces":
                result = await k8s_mcp.list_namespaces()
            elif name == "list_events":
                result = await k8s_mcp.list_events(arguments.get("namespace"))
            else:
                result = {"error": f"Unknown tool: {name}"}

            return _wrap_response(result)
        except Exception as exc:  # pragma: no cover
            logger.exception("MCP tool %s failed", name)
            return _wrap_response({"error": str(exc)})

    @server.list_resources()
    async def handle_list_resources() -> List[types.Resource]:  # type: ignore[return-value]
        return [
            types.Resource(
                uri="kubernetes://cluster/info",
                name="Cluster Info",
                description="Cluster version and nodes",
            ),
            types.Resource(
                uri="kubernetes://pods/default",
                name="Pods (default namespace)",
                description="Pods running in the default namespace",
            ),
            types.Resource(
                uri="kubernetes://services/default",
                name="Services (default namespace)",
                description="Services in the default namespace",
            ),
            types.Resource(
                uri="kubernetes://deployments/default",
                name="Deployments (default namespace)",
                description="Deployments in the default namespace",
            ),
        ]

    @server.read_resource()
    async def handle_read_resource(uri: str) -> List[types.TextContent]:  # type: ignore[return-value]
        if uri == "kubernetes://cluster/info":
            result = await k8s_mcp.get_cluster_info()
        elif uri == "kubernetes://pods/default":
            result = await k8s_mcp.get_pods("default")
        elif uri == "kubernetes://services/default":
            result = await k8s_mcp.get_services("default")
        elif uri == "kubernetes://deployments/default":
            result = await k8s_mcp.get_deployments("default")
        else:
            result = {"error": f"Unknown resource: {uri}"}

        return _wrap_response(result)

else:
    server = None


async def main() -> None:
    if not MCP_AVAILABLE:
        raise RuntimeError("MCP stdio server is not available (install the 'mcp' package)")

    await server.run_stdio_server()  # type: ignore[func-returns-value]


if __name__ == "__main__":
    asyncio.run(main())
