#!/usr/bin/env python3
import asyncio
import json
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Reuse the existing tool implementations
from kubernetes_mcp_server import KubernetesMCP


app = FastAPI(title="AI4K8s MCP HTTP Server", version="1.0")
k8s_mcp = KubernetesMCP()


def _list_tools_schema() -> Dict[str, Any]:
    return {
        "tools": [
            {
                "name": "pods_list",
                "description": "List pods in the specified namespace (or all namespaces).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace (use 'all' for all namespaces)",
                            "default": "default",
                        }
                    },
                    "required": [],
                },
            },
            {
                "name": "pods_get",
                "description": "Get details for a specific pod.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Pod name",
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace (default: 'default')",
                            "default": "default",
                        },
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "pods_delete",
                "description": "Delete a pod by name.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Pod name",
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace (default: 'default')",
                            "default": "default",
                        },
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "pods_run",
                "description": "Create a pod from the specified image.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Pod name (optional)"},
                        "image": {
                            "type": "string",
                            "description": "Container image to run",
                            "default": "nginx",
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Namespace for the pod (default: 'default')",
                            "default": "default",
                        },
                        "command": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Command to execute (optional)",
                        },
                        "args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Additional arguments (optional)",
                        },
                    },
                    "required": ["image"],
                },
            },
            {
                "name": "pods_log",
                "description": "Get logs from a specific pod",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the pod"},
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
                    "required": ["name"],
                },
            },
            {
                "name": "pods_exec",
                "description": "Execute a command in a pod",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the pod"},
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace (default: 'default')",
                            "default": "default",
                        },
                        "command": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Command to execute in the pod",
                        },
                        "container": {
                            "type": "string",
                            "description": "Container name (optional, for multi-container pods)",
                        },
                    },
                    "required": ["name", "command"],
                },
            },
            {
                "name": "pods_top",
                "description": "Get resource usage (CPU and memory) for pods",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pod_name": {
                            "type": "string",
                            "description": "Name of the pod (optional, if not provided returns all pods)",
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace (default: 'default', use 'all' for all namespaces)",
                            "default": "default",
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "resources_list",
                "description": "List Kubernetes resources for the specified apiVersion and kind.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "apiVersion": {
                            "type": "string",
                            "description": "API version (e.g., 'v1')",
                            "default": "v1",
                        },
                        "kind": {
                            "type": "string",
                            "description": "Kind of resource (e.g., 'Service', 'Deployment', 'Node')",
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Namespace (optional, defaults to all namespaces)",
                        },
                    },
                    "required": ["kind"],
                },
            },
            {
                "name": "namespaces_list",
                "description": "List namespaces in the cluster.",
                "inputSchema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "events_list",
                "description": "List recent Kubernetes events.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Namespace (optional, defaults to all namespaces)",
                        }
                    },
                    "required": [],
                },
            },
            {
                "name": "get_cluster_info",
                "description": "Get basic information about the Kubernetes cluster including version, nodes, and status",
                "inputSchema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "get_pods",
                "description": "Get pods in a specific namespace with their status, readiness, and restart counts",
                "inputSchema": {
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
            },
            {
                "name": "get_services",
                "description": "Get services in a specific namespace with their types, IPs, and ports",
                "inputSchema": {
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
            },
            {
                "name": "get_deployments",
                "description": "Get deployments in a specific namespace with replica counts and status",
                "inputSchema": {
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
            },
            {
                "name": "get_pod_logs",
                "description": "Get logs from a specific pod",
                "inputSchema": {
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
            },
            {
                "name": "execute_kubectl",
                "description": "Execute a kubectl command and return the results",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "kubectl command to execute (without 'kubectl' prefix)",
                        }
                    },
                    "required": ["command"],
                },
            },
            {
                "name": "get_docker_containers",
                "description": "Get information about Docker containers running on the system",
                "inputSchema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "get_pod_top",
                "description": "Get resource usage (CPU and memory) for pods",
                "inputSchema": {
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
            },
            {
                "name": "exec_into_pod",
                "description": "Execute a command in a pod",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pod_name": {"type": "string", "description": "Name of the pod"},
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace (default: 'default')",
                            "default": "default",
                        },
                        "command": {"type": "string", "description": "Command to execute in the pod"},
                        "container": {
                            "type": "string",
                            "description": "Container name (optional, for multi-container pods)",
                        },
                    },
                    "required": ["pod_name", "command"],
                },
            },
            {
                "name": "run_container_in_pod",
                "description": "Run a container image in a new pod",
                "inputSchema": {
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
                            "items": {"type": "string"},
                            "description": "Arguments for the command (optional)",
                        },
                    },
                    "required": ["image"],
                },
            },
        ]
    }


async def _dispatch_tool_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    namespace = arguments.get("namespace", "default")
    if name == "get_cluster_info":
        return await k8s_mcp.get_cluster_info()
    if name in ("get_pods", "pods_list"):
        return await k8s_mcp.get_pods(namespace)
    if name in ("get_services",):
        return await k8s_mcp.get_services(arguments.get("namespace", "default"))
    if name in ("get_deployments",):
        return await k8s_mcp.get_deployments(arguments.get("namespace", "default"))
    if name in ("get_pod_logs", "pods_log"):
        return await k8s_mcp.get_pod_logs(
            arguments.get("pod_name") or arguments.get("name"),
            namespace,
            arguments.get("lines", 100)
        )
    if name == "execute_kubectl":
        return await k8s_mcp.execute_kubectl_command(arguments["command"])
    if name == "get_docker_containers":
        return await k8s_mcp.get_docker_containers()
    if name in ("get_pod_top", "pods_top"):
        return await k8s_mcp.get_pod_top(arguments.get("pod_name"), namespace)
    if name in ("exec_into_pod", "pods_exec"):
        cmd = arguments.get("command", [])
        if isinstance(cmd, str):
            cmd_list = cmd.split()
        else:
            cmd_list = cmd
        return await k8s_mcp.exec_into_pod(
            arguments.get("pod_name") or arguments.get("name"),
            namespace,
            cmd_list,
            arguments.get("container")
        )
    if name in ("run_container_in_pod", "pods_run"):
        return await k8s_mcp.run_container_in_pod(
            arguments.get("image", "nginx"),
            arguments.get("name"),
            namespace,
            arguments.get("command"),
            arguments.get("args")
        )
    if name == "pods_get":
        return await k8s_mcp.get_pod_details(
            arguments.get("name"),
            namespace,
        )
    if name == "pods_delete":
        return await k8s_mcp.delete_pod(
            arguments.get("name"),
            namespace,
        )
    if name == "resources_list":
        return await k8s_mcp.list_resources(
            arguments.get("apiVersion", "v1"),
            arguments.get("kind", "Service"),
            arguments.get("namespace"),
        )
    if name == "namespaces_list":
        return await k8s_mcp.list_namespaces()
    if name == "events_list":
        return await k8s_mcp.list_events(arguments.get("namespace"))
    return {"error": f"Unknown tool: {name}"}


@app.post("/mcp")
async def mcp_router(request: Request):
    payload = await request.json()
    method = payload.get("method")
    req_id = payload.get("id")

    if method == "tools/list":
        return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": _list_tools_schema()})

    if method == "tools/call":
        params = payload.get("params", {})
        name = params.get("name")
        arguments = params.get("arguments", {})
        try:
            result = await _dispatch_tool_call(name, arguments)
            return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": result})
        except Exception as e:
            return JSONResponse({"jsonrpc": "2.0", "id": req_id, "error": str(e)}, status_code=500)

    return JSONResponse({"jsonrpc": "2.0", "id": req_id, "error": f"Unknown method: {method}"}, status_code=400)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mcp_http_server:app", host="127.0.0.1", port=5002, reload=False)
