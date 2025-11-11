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
    if name == "get_cluster_info":
        return await k8s_mcp.get_cluster_info()
    if name == "get_pods":
        return await k8s_mcp.get_pods(arguments.get("namespace", "default"))
    if name == "get_services":
        return await k8s_mcp.get_services(arguments.get("namespace", "default"))
    if name == "get_deployments":
        return await k8s_mcp.get_deployments(arguments.get("namespace", "default"))
    if name == "get_pod_logs":
        return await k8s_mcp.get_pod_logs(
            arguments["pod_name"], arguments.get("namespace", "default"), arguments.get("lines", 100)
        )
    if name == "execute_kubectl":
        return await k8s_mcp.execute_kubectl_command(arguments["command"])
    if name == "get_docker_containers":
        return await k8s_mcp.get_docker_containers()
    if name == "get_pod_top":
        return await k8s_mcp.get_pod_top(arguments.get("pod_name"), arguments.get("namespace", "default"))
    if name == "exec_into_pod":
        return await k8s_mcp.exec_into_pod(
            arguments["pod_name"], arguments.get("namespace", "default"), arguments["command"], arguments.get("container")
        )
    if name == "run_container_in_pod":
        return await k8s_mcp.run_container_in_pod(
            arguments["image"], arguments.get("name"), arguments.get("namespace", "default"), arguments.get("command"), arguments.get("args")
        )
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




