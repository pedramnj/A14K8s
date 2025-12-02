#!/usr/bin/env python3
"""
Enhanced AI Processor for AI4K8s with Create/Delete Support
"""

import json
import requests
from typing import Dict, List, Any, Optional
from mcp_client import call_mcp_tool

class EnhancedAIProcessor:
    def __init__(self, mcp_server_url="http://127.0.0.1:5002"):
        self.mcp_server_url = mcp_server_url
        self.available_tools = None
        self.use_ai = False
        self._load_tools()
    
    def _load_tools(self):
        try:
            response = requests.get(f"{self.mcp_server_url}/mcp", 
                                  json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
            if response.status_code == 200:
                result = response.json()
                self.available_tools = result.get("result", {}).get("tools", [])
        except Exception as e:
            print(f"Error loading tools: {e}")
            self.available_tools = []
    
    def _call_mcp_tool(self, tool_name: str, args: Dict) -> Dict:
        try:
            response = requests.post(f"{self.mcp_server_url}/mcp",
                                  json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                        "params": {"name": tool_name, "arguments": args}})
            if response.status_code == 200:
                result = response.json()
                # Check if the result has the expected structure
                if "result" in result:
                    return {"success": True, "result": result["result"]}
                else:
                    return {"success": False, "error": "No result in response"}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _ai_tool_selection(self, query: str) -> tuple:
        query_lower = query.lower()
        
        # Handle create commands
        if "create" in query_lower or "run" in query_lower:
            # Extract pod name and image
            pod_name = self._extract_pod_name(query)
            image = self._extract_image(query)
            return "run_container_in_pod", {
                "name": pod_name,
                "image": image,
                "namespace": "default"
            }
        
        # Handle delete commands
        elif "delete" in query_lower or "remove" in query_lower:
            pod_name = self._extract_pod_name(query)
            return "execute_kubectl", {
                "command": f"delete pod {pod_name}"
            }
        
        # Handle other queries
        elif "pod" in query_lower:
            return "get_pods", {}
        elif "service" in query_lower:
            return "get_services", {}
        elif "deployment" in query_lower:
            return "get_deployments", {}
        elif "node" in query_lower or "cluster" in query_lower:
            return "get_cluster_info", {}
        elif "cpu" in query_lower or "memory" in query_lower or "resource" in query_lower:
            return "get_pod_top", {}
        else:
            return "get_pods", {}
    
    def _extract_pod_name(self, query: str) -> str:
        """Extract pod name from query"""
        query_lower = query.lower()
        
        # Look for patterns like "name it X" or "pod X"
        if "name it" in query_lower:
            parts = query_lower.split("name it")
            if len(parts) > 1:
                name_part = parts[1].strip()
                # Extract first word as name
                name = name_part.split()[0]
                return name.replace("_", "-")
        
        # Look for "pod X" pattern
        if "pod" in query_lower:
            parts = query_lower.split("pod")
            if len(parts) > 1:
                name_part = parts[1].strip()
                # Extract first word as name
                name = name_part.split()[0]
                return name.replace("_", "-")
        
        # Default name
        return "new-pod"
    
    def _extract_image(self, query: str) -> str:
        """Extract container image from query"""
        query_lower = query.lower()
        
        # Common images
        if "nginx" in query_lower:
            return "nginx"
        elif "busybox" in query_lower:
            return "busybox"
        elif "alpine" in query_lower:
            return "alpine"
        elif "ubuntu" in query_lower:
            return "ubuntu"
        else:
            return "nginx"  # Default image
    
    def _smart_response_formatting(self, tool_name: str, real_data: dict, query: str) -> str:
        if tool_name == "get_pods":
            pods = real_data.get("pods", [])
            running_pods = [p for p in pods if p.get("status") == "Running"]
            pending_pods = [p for p in pods if p.get("status") == "Pending"]
            
            response = f"**Pod Status Summary 📊**\n\nI've analyzed the pod list for you. Here's a summary of the running pods:\n\n"
            response += f"**Total Pods: {len(pods)} 📈**\n"
            response += f"**Running Pods: {len(running_pods)} ✅**\n"
            response += f"**Pending Pods: {len(pending_pods)} ⏳**\n\n"
            
            if running_pods:
                response += "**Running Pods:**\n"
                for pod in running_pods[:10]:  # Show first 10
                    response += f"* `{pod['name']}` ✅ ({pod['status']})\n"
                if len(running_pods) > 10:
                    response += f"* ... and {len(running_pods) - 10} more running pods\n"
            
            if pending_pods:
                response += "\n**Pending Pods:**\n"
                for pod in pending_pods[:5]:  # Show first 5
                    response += f"* `{pod['name']}` ⏳ ({pod['status']})\n"
                if len(pending_pods) > 5:
                    response += f"* ... and {len(pending_pods) - 5} more pending pods\n"
            
            return response
            
        elif tool_name == "run_container_in_pod":
            response = "**Pod Creation 🚀**\n\nI've created a new pod for you. Here's what happened:\n\n"
            response += "✅ **Pod Created Successfully!**\n"
            response += f"* **Name:** `{real_data.get('name', 'new-pod')}`\n"
            response += f"* **Image:** `{real_data.get('image', 'nginx')}`\n"
            response += f"* **Namespace:** `{real_data.get('namespace', 'default')}`\n"
            response += f"* **Status:** {real_data.get('status', 'success')}\n\n"
            response += "**Next Steps:**\n"
            response += "* Use `kubectl get pods` to see the pod status\n"
            response += "* Use `kubectl describe pod <name>` for detailed information\n"
            response += "* Use `kubectl logs <name>` to see the logs\n"
            return response
            
        elif tool_name == "execute_kubectl":
            response = "**Kubectl Command Execution 🔧**\n\nI've executed the kubectl command for you. Here's what happened:\n\n"
            response += f"* **Command:** `{real_data.get('command', 'kubectl command')}`\n"
            response += f"* **Return Code:** {real_data.get('return_code', 0)}\n"
            response += f"* **Output:** {real_data.get('output', 'Command executed')}\n"
            return response
            
        elif tool_name == "get_cluster_info":
            nodes = real_data.get("nodes", [])
            response = f"**Cluster Information 📊**\n\nI've analyzed your cluster:\n\n"
            response += f"**Total Nodes: {len(nodes)}**\n\n"
            
            for node in nodes:
                response += f"* `{node.get('name', 'Unknown')}` - {node.get('status', 'Unknown')}\n"
            
            return response
            
        elif tool_name == "get_pod_top":
            metrics = real_data.get("metrics", [])
            response = f"**Resource Usage Summary 📊**\n\nI've analyzed the resource usage for you. Here's what I found:\n\n"
            
            if metrics:
                total_cpu = sum(int(m.get("cpu", "0").replace("m", "")) for m in metrics)
                total_memory = sum(int(m.get("memory", "0").replace("Mi", "")) for m in metrics)
                
                response += f"**Total CPU: {total_cpu}m cores**\n"
                response += f"**Total Memory: {total_memory}Mi**\n\n"
                
                # Show top resource users
                high_cpu = [m for m in metrics if int(m.get("cpu", "0").replace("m", "")) > 10]
                if high_cpu:
                    response += "**High CPU Usage:**\n"
                    for m in high_cpu[:5]:
                        response += f"* `{m['name']}` - {m['cpu']} CPU, {m['memory']} Memory\n"
            else:
                response += "No resource metrics available.\n"
            
            return response
            
        else:
            return f"**{tool_name.replace('_', ' ').title()} Summary 📊**\n\nI've analyzed the data for you. Here's what I found:\n\n{json.dumps(real_data, indent=2)}"
    
    def _process_with_groq_nl(self, query: str, system_prompt: str) -> dict:
        try:
            tool_name, tool_args = self._ai_tool_selection(query)
            result = self._call_mcp_tool(tool_name, tool_args)
            
            if result.get("success"):
                real_data = result["result"]
                explanation = self._smart_response_formatting(tool_name, real_data, query)
                return {
                    "command": f"AI: {tool_name}",
                    "explanation": explanation,
                    "ai_processed": True,
                    "tool_results": [{"tool_name": tool_name, "result": result}],
                    "mcp_result": result
                }
            else:
                return {
                    "command": f"AI: {tool_name}",
                    "explanation": f"❌ **Error executing {tool_name}:** {result.get('error', 'Unknown error')}",
                    "ai_processed": True,
                    "tool_results": [{"tool_name": tool_name, "result": result}],
                    "mcp_result": result
                }
        except Exception as e:
            return {
                "command": "AI: error",
                "explanation": f"❌ **Error in processing:** {str(e)}",
                "ai_processed": False,
                "tool_results": [],
                "mcp_result": None
            }
    
    def process_query(self, query: str) -> dict:
        return self._process_with_groq_nl(query, "")
