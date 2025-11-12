#!/usr/bin/env python3
"""
Proper MCP Server for Kubernetes Monitoring
This server follows the Model Context Protocol specification
"""

import asyncio
import json
import logging
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence
from datetime import datetime

try:  # pragma: no cover - optional dependency
    import mcp.server  # type: ignore
    import mcp.server.stdio  # type: ignore
    import mcp.types as types  # type: ignore
    MCP_AVAILABLE = True
except ImportError:  # pragma: no cover
    MCP_AVAILABLE = False
    types = None  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KubernetesMCP:
    """MCP Server for Kubernetes operations and monitoring"""
    
    def __init__(self):
        self.kubectl_available = self._check_kubectl()
        self.docker_available = self._check_docker()
    
    def _check_kubectl(self):
        """Check if kubectl is available"""
        try:
            result = subprocess.run(['kubectl', 'version', '--client'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"kubectl not available: {e}")
            return False
    
    def _check_docker(self):
        """Check if docker is available"""
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"docker not available: {e}")
            return False

    async def get_cluster_info(self) -> Dict[str, Any]:
        """Get basic cluster information"""
        if not self.kubectl_available:
            return {"error": "kubectl not available", "suggestion": "Start Docker Desktop or install minikube to enable Kubernetes"}
        
        try:
            # Get cluster version
            version_result = subprocess.run(
                "kubectl version --output=json",
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Get nodes
            nodes_result = subprocess.run(
                "kubectl get nodes -o json",
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            version_data = json.loads(version_result.stdout) if version_result.returncode == 0 else {}
            nodes_data = json.loads(nodes_result.stdout) if nodes_result.returncode == 0 else {}
            
            return {
                "cluster_version": version_data.get("serverVersion", {}).get("gitVersion", "unknown"),
                "node_count": len(nodes_data.get("items", [])),
                "nodes": [
                    {
                        "name": node.get("metadata", {}).get("name", "unknown"),
                        "status": node.get("status", {}).get("conditions", [{}])[-1].get("type", "Unknown"),
                        "role": "master" if "master" in node.get("metadata", {}).get("labels", {}).get("kubernetes.io/role", "") else "worker"
                    }
                    for node in nodes_data.get("items", [])
                ]
            }
        except Exception as e:
            return {"error": f"Failed to get cluster info: {str(e)}"}

    async def get_pods(self, namespace: str = "default") -> Dict[str, Any]:
        """Get pods in a namespace"""
        if not self.kubectl_available:
            return {"error": "kubectl not available"}
        
        try:
            if namespace in (None, "", "all", "all-namespaces", "--all-namespaces"):
                cmd = "kubectl get pods --all-namespaces -o json"
            else:
                cmd = f"kubectl get pods -n {namespace} -o json"
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            display_namespace = namespace if namespace not in (None, "", "all", "all-namespaces", "--all-namespaces") else "all"
            
            if result.returncode != 0:
                return {"error": f"kubectl error: {result.stderr}"}
            
            data = json.loads(result.stdout)
            pods = data.get("items", [])
            
            return {
                "namespace": display_namespace,
                "pod_count": len(pods),
                "pods": [
                    {
                        "name": pod.get("metadata", {}).get("name", "unknown"),
                        "status": pod.get("status", {}).get("phase", "Unknown"),
                        "ready": f"{len([c for c in pod.get('status', {}).get('containerStatuses', []) if c.get('ready', False)])}/{len(pod.get('spec', {}).get('containers', []))}",
                        "restarts": sum(c.get('restartCount', 0) for c in pod.get('status', {}).get('containerStatuses', [])),
                        "age": pod.get("metadata", {}).get("creationTimestamp", "unknown"),
                        "namespace": pod.get("metadata", {}).get("namespace", display_namespace)
                    }
                    for pod in pods
                ]
            }
        except Exception as e:
            return {"error": f"Failed to get pods: {str(e)}"}

    async def get_pod_details(self, name: str, namespace: str = "default") -> Dict[str, Any]:
        """Get detailed information for a single pod"""
        if not self.kubectl_available:
            return {"error": "kubectl not available"}
        
        try:
            result = subprocess.run(
                f"kubectl get pod {name} -n {namespace} -o json",
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                return {"error": f"kubectl error: {result.stderr or 'pod not found'}"}
            
            data = json.loads(result.stdout)
            return {
                "name": data.get("metadata", {}).get("name", name),
                "namespace": data.get("metadata", {}).get("namespace", namespace),
                "labels": data.get("metadata", {}).get("labels", {}),
                "status": data.get("status", {}),
                "spec": data.get("spec", {}),
            }
        except Exception as e:
            return {"error": f"Failed to get pod details: {str(e)}"}

    async def delete_pod(self, name: str, namespace: str = "default") -> Dict[str, Any]:
        """Delete a pod"""
        if not self.kubectl_available:
            return {"error": "kubectl not available"}
        try:
            result = subprocess.run(
                f"kubectl delete pod {name} -n {namespace}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                return {"error": result.stderr or "Failed to delete pod"}
            return {
                "name": name,
                "namespace": namespace,
                "status": "deleted",
                "message": result.stdout.strip()
            }
        except subprocess.TimeoutExpired:
            return {"error": "Delete operation timed out"}
        except Exception as e:
            return {"error": f"Failed to delete pod: {str(e)}"}

    async def get_services(self, namespace: str = "default") -> Dict[str, Any]:
        """Get services in a namespace"""
        if not self.kubectl_available:
            return {"error": "kubectl not available"}
        
        try:
            result = subprocess.run(
                f"kubectl get services -n {namespace} -o json",
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return {"error": f"kubectl error: {result.stderr}"}
            
            data = json.loads(result.stdout)
            services = data.get("items", [])
            
            return {
                "namespace": namespace,
                "service_count": len(services),
                "services": [
                    {
                        "name": service.get("metadata", {}).get("name", "unknown"),
                        "type": service.get("spec", {}).get("type", "Unknown"),
                        "cluster_ip": service.get("spec", {}).get("clusterIP", "None"),
                        "external_ip": service.get("status", {}).get("loadBalancer", {}).get("ingress", [{}])[0].get("ip") if service.get("status", {}).get("loadBalancer", {}).get("ingress") else None,
                        "ports": [f"{p.get('port', '')}:{p.get('targetPort', '')}" for p in service.get("spec", {}).get("ports", [])]
                    }
                    for service in services
                ]
            }
        except Exception as e:
            return {"error": f"Failed to get services: {str(e)}"}

    async def get_deployments(self, namespace: str = "default") -> Dict[str, Any]:
        """Get deployments in a namespace"""
        if not self.kubectl_available:
            return {"error": "kubectl not available"}
        
        try:
            result = subprocess.run(
                f"kubectl get deployments -n {namespace} -o json",
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return {"error": f"kubectl error: {result.stderr}"}
            
            data = json.loads(result.stdout)
            deployments = data.get("items", [])
            
            return {
                "namespace": namespace,
                "deployment_count": len(deployments),
                "deployments": [
                    {
                        "name": deployment.get("metadata", {}).get("name", "unknown"),
                        "replicas": f"{deployment.get('status', {}).get('readyReplicas', 0)}/{deployment.get('spec', {}).get('replicas', 0)}",
                        "updated": deployment.get('status', {}).get('updatedReplicas', 0),
                        "available": deployment.get('status', {}).get('availableReplicas', 0),
                        "age": "unknown"  # Simplified for now
                    }
                    for deployment in deployments
                ]
            }
        except Exception as e:
            return {"error": f"Failed to get deployments: {str(e)}"}

    async def get_pod_logs(self, pod_name: str, namespace: str = "default", lines: int = 100) -> Dict[str, Any]:
        """Get logs from a pod"""
        if not self.kubectl_available:
            return {"error": "kubectl not available"}
        
        try:
            result = subprocess.run(
                f"kubectl logs {pod_name} -n {namespace} --tail={lines}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "pod_name": pod_name,
                "namespace": namespace,
                "lines": lines,
                "logs": result.stdout,
                "error": result.stderr if result.returncode != 0 else None
            }
        except Exception as e:
            return {"error": f"Failed to get logs: {str(e)}"}

    async def execute_kubectl_command(self, command: str) -> Dict[str, Any]:
        """Execute a kubectl command"""
        if not self.kubectl_available:
            return {"error": "kubectl not available"}
        
        try:
            result = subprocess.run(
                f"kubectl {command}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                "command": f"kubectl {command}",
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out"}
        except Exception as e:
            return {"error": f"Failed to execute command: {str(e)}"}

    async def get_docker_containers(self) -> Dict[str, Any]:
        """Get Docker containers information"""
        if not self.docker_available:
            return {"error": "docker not available"}
        
        try:
            result = subprocess.run(
                "docker ps -a --format json",
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return {"error": f"docker error: {result.stderr}"}
            
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    try:
                        container = json.loads(line)
                        containers.append({
                            "id": container.get("ID", "unknown"),
                            "name": container.get("Names", "unknown"),
                            "image": container.get("Image", "unknown"),
                            "status": container.get("Status", "unknown"),
                            "ports": container.get("Ports", "none")
                        })
                    except json.JSONDecodeError:
                        continue
            
            return {
                "container_count": len(containers),
                "containers": containers
            }
        except Exception as e:
            return {"error": f"Failed to get containers: {str(e)}"}

    async def get_pod_top(self, pod_name: str = None, namespace: str = "default") -> Dict[str, Any]:
        """Get pod resource usage (CPU and memory)"""
        if not self.kubectl_available:
            return {"error": "kubectl not available"}
        
        try:
            if pod_name:
                cmd = f"kubectl top pod {pod_name} -n {namespace} --no-headers"
            elif namespace == "all" or namespace == "--all-namespaces":
                # Support all-namespaces request
                cmd = "kubectl top pods --all-namespaces --no-headers"
            else:
                cmd = f"kubectl top pods -n {namespace} --no-headers"
            
            result = subprocess.run(
                cmd.split(),
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return {
                    "error": f"Failed to get pod metrics: {result.stderr}",
                    "suggestion": "Ensure metrics-server is installed and running"
                }
            
            # Parse the output
            lines = result.stdout.strip().split('\n')
            pod_metrics = []
            
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if namespace in ("all", "--all-namespaces") and len(parts) >= 4:
                        pod_namespace = parts[0]
                        pod_name = parts[1]
                        cpu = parts[2]
                        memory = parts[3]
                    elif len(parts) >= 3:
                        pod_namespace = namespace
                        pod_name = parts[0]
                        cpu = parts[1]
                        memory = parts[2]
                    else:
                        continue
                    pod_metrics.append({
                        "name": pod_name,
                        "cpu": cpu,
                        "memory": memory,
                        "namespace": pod_namespace
                    })
            
            return {
                "namespace": namespace,
                "pod_name": pod_name,
                "metrics": pod_metrics,
                "total_pods": len(pod_metrics)
            }
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out - metrics-server may be slow"}
        except Exception as e:
            return {"error": f"Failed to get pod metrics: {str(e)}"}

    async def exec_into_pod(self, pod_name: str, namespace: str, command: str, container: str = None) -> Dict[str, Any]:
        """Execute a command in a pod"""
        if not self.kubectl_available:
            return {"error": "kubectl not available"}
        
        try:
            if isinstance(command, list):
                command_str = " ".join(command)
            else:
                command_str = command
            # Build kubectl exec command
            cmd = ["kubectl", "exec", pod_name, "-n", namespace]
            
            if container:
                cmd.extend(["-c", container])
            
            cmd.extend(["--", "sh", "-c", command_str])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            return {
                "pod": pod_name,
                "namespace": namespace,
                "container": container,
                "command": command_str,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out"}
        except Exception as e:
            return {"error": f"Failed to exec into pod: {str(e)}"}

    async def run_container_in_pod(self, image: str, name: str = None, namespace: str = "default", 
                                 command: str = None, args: list = None) -> Dict[str, Any]:
        """Run a container image in a pod"""
        if not self.kubectl_available:
            return {"error": "kubectl not available"}
        
        try:
            import uuid
            
            # Generate pod name if not provided
            if not name:
                name = f"run-{image.split('/')[-1].split(':')[0]}-{str(uuid.uuid4())[:8]}"
            
            # Build kubectl run command
            cmd = ["kubectl", "run", name, f"--image={image}", f"-n", namespace, "--restart=Never"]
            
            if command:
                if isinstance(command, list):
                    cmd.extend(["--command", "--"] + command)
                else:
                    cmd.extend(["--command", "--", command])
            
            if args:
                cmd.extend(args)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                return {
                    "error": f"Failed to run container: {result.stderr}",
                    "command": " ".join(cmd)
                }
            
            return {
                "action": "created",
                "pod_name": name,
                "namespace": namespace,
                "image": image,
                "command": command,
                "args": args,
                "status": "success",
                "output": result.stdout
            }
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out"}
        except Exception as e:
            return {"error": f"Failed to run container: {str(e)}"}

    async def list_resources(self, api_version: str, kind: str, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Generic resource listing"""
        if not self.kubectl_available:
            return {"error": "kubectl not available"}
        resource = kind.lower()
        if not resource.endswith('s'):
            resource = resource + 's'
        try:
            if namespace and namespace not in ("all", "all-namespaces", "--all-namespaces"):
                cmd = f"kubectl get {resource} -n {namespace} -o json"
            else:
                cmd = f"kubectl get {resource} --all-namespaces -o json"
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                return {"error": result.stderr or f"Failed to list {kind}"}
            data = json.loads(result.stdout)
            return {
                "apiVersion": api_version,
                "kind": kind,
                "count": len(data.get("items", [])),
                "items": data.get("items", [])
            }
        except Exception as e:
            return {"error": f"Failed to list resources: {str(e)}"}

    async def list_namespaces(self) -> Dict[str, Any]:
        """List namespaces"""
        if not self.kubectl_available:
            return {"error": "kubectl not available"}
        try:
            result = subprocess.run(
                "kubectl get namespaces -o json",
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                return {"error": result.stderr or "Failed to list namespaces"}
            data = json.loads(result.stdout)
            return {
                "count": len(data.get("items", [])),
                "namespaces": [
                    {
                        "name": item.get("metadata", {}).get("name", "unknown"),
                        "status": item.get("status", {}).get("phase", "Unknown"),
                        "labels": item.get("metadata", {}).get("labels", {})
                    }
                    for item in data.get("items", [])
                ]
            }
        except Exception as e:
            return {"error": f"Failed to list namespaces: {str(e)}"}

    async def list_events(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """List cluster events"""
        if not self.kubectl_available:
            return {"error": "kubectl not available"}
        try:
            if namespace and namespace not in ("all", "all-namespaces", "--all-namespaces"):
                cmd = f"kubectl get events -n {namespace} -o json"
            else:
                cmd = "kubectl get events --all-namespaces -o json"
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                return {"error": result.stderr or "Failed to list events"}
            data = json.loads(result.stdout)
            return {
                "count": len(data.get("items", [])),
                "events": data.get("items", [])
            }
        except Exception as e:
            return {"error": f"Failed to list events: {str(e)}"}

# Initialize the Kubernetes MCP instance
k8s_mcp = KubernetesMCP()

if MCP_AVAILABLE:
    server = mcp.server.Server("kubernetes-monitor")  # type: ignore

    @server.list_tools()
    async def handle_list_tools() -> List[types.Tool]:
        """List available tools"""
        return [
            types.Tool(
                name="get_cluster_info",
                description="Get basic information about the Kubernetes cluster including version, nodes, and status",
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            types.Tool(
                name="get_pods",
                description="Get pods in a specific namespace with their status, readiness, and restart counts",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace (default: 'default')",
                            "default": "default",
                        }
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="get_services",
                description="Get services in a specific namespace with their types, IPs, and ports",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace (default: 'default')",
                            "default": "default",
                        }
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="get_deployments",
                description="Get deployments in a specific namespace with replica counts and status",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace (default: 'default')",
                            "default": "default",
                        }
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="get_pod_logs",
                description="Get logs from a specific pod",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pod_name": {"type": "string", "description": "Name of the pod"},
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace (default: 'default')",
                            "default": "default",
                        },
                        "lines": {
                            "type": "integer",
                            "description": "Number of log lines to retrieve (default: 100)",
                            "default": 100,
                        },
                    },
                    "required": ["pod_name"],
                },
            ),
            types.Tool(
                name="execute_kubectl",
                description="Execute a kubectl command and return the results",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "kubectl command to execute (without 'kubectl' prefix)",
                        }
                    },
                    "required": ["command"],
                },
            ),
            types.Tool(
                name="get_docker_containers",
                description="Get information about Docker containers running on the system",
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            types.Tool(
                name="get_pod_top",
                description="Get resource usage (CPU and memory) for pods",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pod_name": {
                            "type": "string",
                            "description": "Name of the pod (optional, if not provided returns all pods)",
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace (default: 'default')",
                            "default": "default",
                        },
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="exec_into_pod",
                description="Execute a command in a pod",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pod_name": {"type": "string", "description": "Name of the pod"},
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace (default: 'default')",
                            "default": "default",
                        },
                        "command": {
                            "type": "string",
                            "description": "Command to execute in the pod",
                        },
                        "container": {
                            "type": "string",
                            "description": "Container name (optional, for multi-container pods)",
                        },
                    },
                    "required": ["pod_name", "command"],
                },
            ),
            types.Tool(
                name="run_container_in_pod",
                description="Run a container image in a new pod",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "image": {"type": "string", "description": "Container image to run"},
                        "name": {
                            "type": "string",
                            "description": "Name for the pod (optional, will be generated if not provided)",
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace (default: 'default')",
                            "default": "default",
                        },
                        "command": {
                            "type": "string",
                            "description": "Command to run in the container (optional)",
                        },
                        "args": {
                            "type": "array",
                            "description": "Arguments for the command (optional)",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["image"],
                },
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Handle tool calls"""
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
                    arguments["pod_name"], arguments.get("namespace", "default"), arguments.get("lines", 100)
                )
            elif name == "execute_kubectl":
                result = await k8s_mcp.execute_kubectl_command(arguments["command"])
            elif name == "get_docker_containers":
                result = await k8s_mcp.get_docker_containers()
            elif name == "get_pod_top":
                result = await k8s_mcp.get_pod_top(arguments.get("pod_name"), arguments.get("namespace", "default"))
            elif name == "exec_into_pod":
                result = await k8s_mcp.exec_into_pod(
                    arguments["pod_name"], arguments.get("namespace", "default"), arguments["command"], arguments.get("container")
                )
            elif name == "run_container_in_pod":
                result = await k8s_mcp.run_container_in_pod(
                    arguments["image"], arguments.get("name"), arguments.get("namespace", "default"), arguments.get("command"), arguments.get("args")
                )
            else:
                result = {"error": f"Unknown tool: {name}"}

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [types.TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

    @server.list_resources()
    async def handle_list_resources() -> List[types.Resource]:
        """List available resources"""
        return [
            types.Resource(
                uri="kubernetes://cluster/info",
                name="Cluster Info",
                description="Basic information about the Kubernetes cluster",
            ),
            types.Resource(
                uri="kubernetes://pods/default",
                name="Pods (default namespace)",
                description="Current pods in the default namespace",
            ),
            types.Resource(
                uri="kubernetes://services/default",
                name="Services (default namespace)",
                description="Current services in the default namespace",
            ),
            types.Resource(
                uri="kubernetes://deployments/default",
                name="Deployments (default namespace)",
                description="Current deployments in the default namespace",
            ),
        ]

    @server.read_resource()
    async def handle_read_resource(uri: str) -> List[types.TextContent]:
        """Read a resource"""
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

        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    async def main():
        await server.run_stdio_server()
else:
    server = None

    async def main():
        raise RuntimeError("MCP stdio server is not available (requires Python >= 3.10)")

if __name__ == "__main__":
    asyncio.run(main())
