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

import mcp.server.stdio
import mcp.types as types

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
            result = subprocess.run(
                f"kubectl get pods -n {namespace} -o json",
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return {"error": f"kubectl error: {result.stderr}"}
            
            data = json.loads(result.stdout)
            pods = data.get("items", [])
            
            return {
                "namespace": namespace,
                "pod_count": len(pods),
                "pods": [
                    {
                        "name": pod.get("metadata", {}).get("name", "unknown"),
                        "status": pod.get("status", {}).get("phase", "Unknown"),
                        "ready": f"{len([c for c in pod.get('status', {}).get('containerStatuses', []) if c.get('ready', False)])}/{len(pod.get('spec', {}).get('containers', []))}",
                        "restarts": sum(c.get('restartCount', 0) for c in pod.get('status', {}).get('containerStatuses', [])),
                        "age": "unknown"  # Simplified for now
                    }
                    for pod in pods
                ]
            }
        except Exception as e:
            return {"error": f"Failed to get pods: {str(e)}"}

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
                    if len(parts) >= 3:
                        pod_metrics.append({
                            "name": parts[0],
                            "cpu": parts[1],
                            "memory": parts[2],
                            "namespace": namespace
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
            # Build kubectl exec command
            cmd = ["kubectl", "exec", pod_name, "-n", namespace]
            
            if container:
                cmd.extend(["-c", container])
            
            cmd.extend(["--", "sh", "-c", command])
            
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
                "command": command,
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

# Initialize the Kubernetes MCP instance
k8s_mcp = KubernetesMCP()

# Create the MCP server
server = mcp.server.Server("kubernetes-monitor")

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="get_cluster_info",
            description="Get basic information about the Kubernetes cluster including version, nodes, and status",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
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
                        "default": "default"
                    }
                },
                "required": []
            }
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
                        "default": "default"
                    }
                },
                "required": []
            }
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
                        "default": "default"
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="get_pod_logs",
            description="Get logs from a specific pod",
            inputSchema={
                "type": "object",
                "properties": {
                    "pod_name": {
                        "type": "string",
                        "description": "Name of the pod"
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Kubernetes namespace (default: 'default')",
                        "default": "default"
                    },
                    "lines": {
                        "type": "integer",
                        "description": "Number of log lines to retrieve (default: 100)",
                        "default": 100
                    }
                },
                "required": ["pod_name"]
            }
        ),
        types.Tool(
            name="execute_kubectl",
            description="Execute a kubectl command and return the results",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "kubectl command to execute (without 'kubectl' prefix)"
                    }
                },
                "required": ["command"]
            }
        ),
        types.Tool(
            name="get_docker_containers",
            description="Get information about Docker containers running on the system",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="get_pod_top",
            description="Get resource usage (CPU and memory) for pods",
            inputSchema={
                "type": "object",
                "properties": {
                    "pod_name": {
                        "type": "string",
                        "description": "Name of the pod (optional, if not provided returns all pods)"
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Kubernetes namespace (default: 'default')",
                        "default": "default"
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="exec_into_pod",
            description="Execute a command in a pod",
            inputSchema={
                "type": "object",
                "properties": {
                    "pod_name": {
                        "type": "string",
                        "description": "Name of the pod"
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Kubernetes namespace (default: 'default')",
                        "default": "default"
                    },
                    "command": {
                        "type": "string",
                        "description": "Command to execute in the pod"
                    },
                    "container": {
                        "type": "string",
                        "description": "Container name (optional, for multi-container pods)"
                    }
                },
                "required": ["pod_name", "command"]
            }
        ),
        types.Tool(
            name="run_container_in_pod",
            description="Run a container image in a new pod",
            inputSchema={
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": "Container image to run"
                    },
                    "name": {
                        "type": "string",
                        "description": "Name for the pod (optional, will be generated if not provided)"
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Kubernetes namespace (default: 'default')",
                        "default": "default"
                    },
                    "command": {
                        "type": "string",
                        "description": "Command to run in the container (optional)"
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Arguments for the command (optional)"
                    }
                },
                "required": ["image"]
            }
        )
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
                arguments["pod_name"],
                arguments.get("namespace", "default"),
                arguments.get("lines", 100)
            )
        elif name == "execute_kubectl":
            result = await k8s_mcp.execute_kubectl_command(arguments["command"])
        elif name == "get_docker_containers":
            result = await k8s_mcp.get_docker_containers()
        elif name == "get_pod_top":
            result = await k8s_mcp.get_pod_top(
                arguments.get("pod_name"),
                arguments.get("namespace", "default")
            )
        elif name == "exec_into_pod":
            result = await k8s_mcp.exec_into_pod(
                arguments["pod_name"],
                arguments.get("namespace", "default"),
                arguments["command"],
                arguments.get("container")
            )
        elif name == "run_container_in_pod":
            result = await k8s_mcp.run_container_in_pod(
                arguments["image"],
                arguments.get("name"),
                arguments.get("namespace", "default"),
                arguments.get("command"),
                arguments.get("args")
            )
        else:
            result = {"error": f"Unknown tool: {name}"}
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": str(e)}, indent=2)
        )]

@server.list_resources()
async def handle_list_resources() -> List[types.Resource]:
    """List available resources"""
    return [
        types.Resource(
            uri="kubernetes://cluster/info",
            name="Cluster Information",
            description="Basic information about the Kubernetes cluster",
            mimeType="application/json"
        ),
        types.Resource(
            uri="kubernetes://pods/default",
            name="Default Namespace Pods",
            description="Pods in the default namespace",
            mimeType="application/json"
        ),
        types.Resource(
            uri="kubernetes://services/default",
            name="Default Namespace Services",
            description="Services in the default namespace",
            mimeType="application/json"
        ),
        types.Resource(
            uri="kubernetes://deployments/default",
            name="Default Namespace Deployments",
            description="Deployments in the default namespace",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read a resource"""
    try:
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
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

async def main():
    """Main entry point"""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
