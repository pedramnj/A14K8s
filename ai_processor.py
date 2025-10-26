#!/usr/bin/env python3
"""
Enhanced AI Processor for AI4K8s with Post-Processing
Implements polished responses by passing MCP results through AI.
"""

import json
import re
import requests
import asyncio
from typing import Dict, List, Any, Optional
from mcp_client import call_mcp_tool

class EnhancedAIProcessor:
    """Enhanced AI processor with post-processing for polished responses"""
    
    def __init__(self, mcp_server_url="http://127.0.0.1:5002/mcp"):
        self.mcp_server_url = mcp_server_url
        self.available_tools = None
        self.use_ai = False
        self.anthropic = None
        self._load_tools()
        self._initialize_ai()
    
    def _initialize_ai(self):
        """Initialize AI client (Groq primary, Anthropic fallback)"""
        import os
        
        # Try to load .env file from client directory
        self._load_env_file()
        
        # Try Groq first (free)
        if os.getenv('GROQ_API_KEY'):
            try:
                import groq
                self.anthropic = groq.Groq(api_key=os.getenv('GROQ_API_KEY'))
                self.use_ai = True
                self.ai_provider = "groq"
                print("🤖 Groq AI processing enabled (free tier)")
                return
            except ImportError:
                print("⚠️  Groq package not installed, trying Anthropic...")
            except Exception as e:
                print(f"⚠️  Groq initialization failed: {e}, trying Anthropic...")
        
        # Fallback to Anthropic (paid)
        if os.getenv('ANTHROPIC_API_KEY'):
            try:
                from anthropic import Anthropic
                os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')
                self.anthropic = Anthropic(max_retries=0)
                self.use_ai = True
                self.ai_provider = "anthropic"
                print("🤖 Anthropic AI processing enabled (paid)")
                return
            except ImportError:
                print("⚠️  Anthropic package not installed")
            except Exception as e:
                print(f"⚠️  Anthropic initialization failed: {e}")
        
        # No AI available
        print("🔧 Regex-only processing (no AI keys)")
        self.ai_provider = None
    
    def _load_env_file(self):
        """Load environment variables from .env file"""
        import os
        
        env_paths = [
            'client/.env',
            '.env',
            os.path.expanduser('~/.anthropic_api_key')
        ]
        
        for env_path in env_paths:
            if os.path.exists(env_path):
                try:
                    with open(env_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                os.environ[key] = value
                    print(f"✅ Loaded environment from {env_path}")
                    return
                except Exception as e:
                    print(f"⚠️  Failed to load {env_path}: {e}")
        
        print("⚠️  No .env file found in expected locations")
    
    def _load_tools(self):
        """Load available MCP tools from the server"""
        try:
            # Use the proper MCP client to get tools
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            from mcp_client import get_mcp_client
            client = loop.run_until_complete(get_mcp_client())
            if client and client.available_tools:
                self.available_tools = list(client.available_tools.values())
                print(f"✅ Loaded {len(self.available_tools)} MCP tools")
            else:
                print("⚠️  No MCP tools available")
                self.available_tools = []
            loop.close()
        except Exception as e:
            print(f"⚠️  Error loading MCP tools: {e}")
            self.available_tools = []
    
    def _get_mcp_tools_for_ai(self) -> List[Dict]:
        """Convert MCP tools to Anthropic format"""
        if not self.available_tools:
            return []
        
        tools = []
        for tool in self.available_tools:
            if tool.get('name') in ['get_pods', 'get_pod_logs', 'get_pod_top', 'exec_into_pod', 'run_container_in_pod', 'execute_kubectl', 'get_cluster_info', 'get_services', 'get_deployments', 'get_docker_containers']:
                tools.append({
                    "name": tool['name'],
                    "description": tool.get('description', ''),
                    "input_schema": tool.get('inputSchema', {})
                })
        return tools
    
    def _process_with_groq_nl(self, query: str, system_prompt: str) -> dict:
        """Process query using AI for understanding + smart formatting (ONE AI call only)"""
        try:
            # Use AI to understand the query and determine MCP tool (ONE AI call)
            tool_name, tool_args = self._ai_tool_selection(query)
            
            print(f"🤖 AI Tool Selection: {tool_name}")
            print(f"🤖 AI Tool Args: {tool_args}")
            
            # Execute the tool call to get real data
            result = self._call_mcp_tool(tool_name, tool_args)
            
            if result['success']:
                # Use smart formatting (NO AI call - just intelligent formatting)
                real_data = result['result']
                explanation = self._smart_response_formatting(tool_name, real_data, query)
                
                return {
                    'command': f'AI: {tool_name}',
                    'explanation': explanation,
                    'ai_processed': True,
                    'tool_results': [{'tool_name': tool_name, 'result': result}],
                    'mcp_result': result
                }
            else:
                return {
                    'command': f'AI: {tool_name}',
                    'explanation': f"❌ **Error executing {tool_name}:** {result['error']}",
                    'ai_processed': True,
                    'tool_results': [{'tool_name': tool_name, 'result': result}],
                    'mcp_result': result
                }
                
        except Exception as e:
            return {
                'command': 'AI: error',
                'explanation': f"❌ **Error in processing:** {str(e)}",
                'ai_processed': False,
                'tool_results': [],
                'mcp_result': None
            }
    
    def _ai_tool_selection(self, query: str) -> tuple:
        """Use AI to understand natural language and select MCP tool (ONE AI call only)"""
        try:
            # Single AI call to understand the query
            ai_prompt = f"""You are a Kubernetes AI assistant. Analyze this user query and determine the appropriate MCP tool.

Available MCP Tools:
- get_pods: List pods in the cluster
- get_services: List services in the cluster  
- get_deployments: List deployments in the cluster
- get_cluster_info: Get cluster and node information
- get_pod_top: Get resource usage (CPU/memory) for pods
- run_container_in_pod: Create/run a new pod with specified image
- execute_kubectl: Execute any kubectl command

User Query: "{query}"

Respond with ONLY a JSON object:
{{
    "tool_name": "appropriate_mcp_tool_name",
    "tool_args": {{"param1": "value1", "param2": "value2"}}
}}

Examples:
- "how is my cluster doing?" → {{"tool_name": "get_cluster_info", "tool_args": {{}}}}
- "create a pod name it pedram with nginx" → {{"tool_name": "run_container_in_pod", "tool_args": {{"image": "nginx", "name": "pedram", "namespace": "default"}}}}
- "delete pedram pod" → {{"tool_name": "execute_kubectl", "tool_args": {{"command": "delete pod pedram"}}}}
- "show me all pods" → {{"tool_name": "get_pods", "tool_args": {{"namespace": "default"}}}}
- "how are cpu resources?" → {{"tool_name": "get_pod_top", "tool_args": {{"namespace": "default"}}}}
- "what services are running?" → {{"tool_name": "get_services", "tool_args": {{"namespace": "default"}}}}

JSON only:"""

            from groq import Groq
            groq_client = Groq(api_key=self.groq_api_key)
            
            ai_response = groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": ai_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            ai_result = ai_response.choices[0].message.content.strip()
            
            # Parse AI response
            import json
            try:
                tool_selection = json.loads(ai_result)
                tool_name = tool_selection.get('tool_name', 'get_pods')
                tool_args = tool_selection.get('tool_args', {})
                
                print(f"🤖 AI understood: {tool_name} with args: {tool_args}")
                return tool_name, tool_args
                
            except json.JSONDecodeError:
                print("⚠️ AI response malformed, using fallback")
                return self._fallback_tool_selection(query)
                
        except Exception as e:
            print(f"⚠️ AI tool selection failed: {e}, using fallback")
            return self._fallback_tool_selection(query)
    
    def _intelligent_tool_selection(self, query: str) -> tuple:
        """Intelligent tool selection without AI API calls"""
        query_lower = query.lower()
        
        # Create/run commands
        if any(word in query_lower for word in ['create', 'run', 'start', 'launch', 'deploy']):
            if any(word in query_lower for word in ['pod', 'container']):
                # Extract parameters intelligently
                image = self._extract_image(query_lower)
                name = self._extract_name(query_lower)
                return 'run_container_in_pod', {
                    'image': image,
                    'name': name,
                    'namespace': 'default'
                }
            else:
                return 'get_pods', {'namespace': 'default'}
        
        # Delete/remove commands
        elif any(word in query_lower for word in ['delete', 'remove', 'stop', 'kill', 'terminate']):
            if any(word in query_lower for word in ['pod', 'container']):
                pod_name = self._extract_pod_name(query_lower)
                if pod_name:
                    return 'execute_kubectl', {'command': f'delete pod {pod_name}'}
                else:
                    return 'get_pods', {'namespace': 'default'}
            else:
                return 'get_pods', {'namespace': 'default'}
        
        # Resource/monitoring queries
        elif any(word in query_lower for word in ['resource', 'cpu', 'memory', 'usage', 'performance', 'top', 'metrics']):
            return 'get_pod_top', {'namespace': 'default'}
        
        # Status/health queries
        elif any(word in query_lower for word in ['status', 'health', 'doing', 'how', 'what']):
            if any(word in query_lower for word in ['pod', 'pods', 'container']):
                return 'get_pods', {'namespace': 'default'}
            elif any(word in query_lower for word in ['service', 'services']):
                return 'get_services', {'namespace': 'default'}
            elif any(word in query_lower for word in ['deployment', 'deployments']):
                return 'get_deployments', {'namespace': 'default'}
            elif any(word in query_lower for word in ['node', 'nodes', 'cluster']):
                return 'get_cluster_info', {}
            else:
                return 'get_pods', {'namespace': 'default'}
        
        # Service queries
        elif any(word in query_lower for word in ['service', 'services']):
            return 'get_services', {'namespace': 'default'}
        
        # Deployment queries
        elif any(word in query_lower for word in ['deployment', 'deployments']):
            return 'get_deployments', {'namespace': 'default'}
        
        # Node/cluster queries
        elif any(word in query_lower for word in ['node', 'nodes', 'cluster']):
            return 'get_cluster_info', {}
        
        # Pod queries (default)
        elif any(word in query_lower for word in ['pod', 'pods', 'container', 'containers']):
            return 'get_pods', {'namespace': 'default'}
        
        # Default to pods
        else:
            return 'get_pods', {'namespace': 'default'}
    
    def _extract_image(self, query_lower: str) -> str:
        """Extract container image from query"""
        images = ['nginx', 'redis', 'mysql', 'postgres', 'mongo', 'elasticsearch', 'kibana', 'grafana', 'prometheus']
        for image in images:
            if image in query_lower:
                return image
        return 'nginx'  # default
    
    def _extract_name(self, query_lower: str) -> str:
        """Extract pod name from query"""
        # Look for "name it X" pattern
        if 'name it' in query_lower:
            parts = query_lower.split('name it')
            if len(parts) > 1:
                name = parts[1].strip().split()[0]
                return name
        
        # Look for "named X" pattern
        if 'named' in query_lower:
            parts = query_lower.split('named')
            if len(parts) > 1:
                name = parts[1].strip().split()[0]
                return name
        
        # Look for "pod X" pattern
        if 'pod' in query_lower:
            words = query_lower.split()
            for i, word in enumerate(words):
                if word == 'pod' and i + 1 < len(words):
                    return words[i + 1]
        
        return None
    
    def _extract_pod_name(self, query_lower: str) -> str:
        """Extract pod name for deletion"""
        # Look for "delete X pod" pattern
        if 'delete' in query_lower:
            words = query_lower.split()
            for i, word in enumerate(words):
                if word == 'delete' and i + 1 < len(words):
                    next_word = words[i + 1]
                    if next_word != 'pod' and next_word != 'the':
                        return next_word
        
        # Look for "delete pod X" pattern
        if 'pod' in query_lower:
            words = query_lower.split()
            for i, word in enumerate(words):
                if word == 'pod' and i + 1 < len(words):
                    return words[i + 1]
        
        # Look for "delete the X pod" pattern
        if 'the' in query_lower:
            words = query_lower.split()
            for i, word in enumerate(words):
                if word == 'the' and i + 1 < len(words):
                    return words[i + 1]
        
        return None
    
    def _smart_response_formatting(self, tool_name: str, real_data: dict, query: str) -> str:
        """Smart response formatting without AI API calls"""
        if tool_name == 'get_pods':
            return self._format_pods_response(real_data, query)
        elif tool_name == 'get_pod_top':
            return self._format_resources_response(real_data, query)
        elif tool_name == 'get_services':
            return self._format_services_response(real_data, query)
        elif tool_name == 'get_deployments':
            return self._format_deployments_response(real_data, query)
        elif tool_name == 'get_cluster_info':
            return self._format_cluster_response(real_data, query)
        elif tool_name == 'run_container_in_pod':
            return self._format_creation_response(real_data, query)
        elif tool_name == 'execute_kubectl':
            return self._format_kubectl_response(real_data, query)
        else:
            return f"**{tool_name.replace('_', ' ').title()} Result 📊**\n\n{json.dumps(real_data, indent=2)}"
    
    def _format_pods_response(self, real_data: dict, query: str) -> str:
        """Format pods response"""
        pods = real_data.get('pods', [])
        pod_count = real_data.get('pod_count', 0)
        running_pods = [p for p in pods if p.get('status') == 'Running']
        pending_pods = [p for p in pods if p.get('status') == 'Pending']
        failed_pods = [p for p in pods if p.get('status') == 'Failed']
        
        response = f"""**Pod Status Summary 📊**

I've analyzed your cluster pods. Here's what I found:

**Total Pods: {pod_count}** 📈
**Running Pods: {len(running_pods)}** ✅
**Pending Pods: {len(pending_pods)}** ⏳
**Failed Pods: {len(failed_pods)}** ❌

"""
        
        if running_pods:
            response += "**Running Pods:**\n"
            for pod in running_pods[:10]:
                response += f"* `{pod['name']}` ✅ (Running)\n"
            if len(running_pods) > 10:
                response += f"* ... and {len(running_pods) - 10} more running pods\n"
            response += "\n"
        
        if pending_pods:
            response += "**Pending Pods:**\n"
            for pod in pending_pods[:5]:
                response += f"* `{pod['name']}` ⏳ (Pending)\n"
            if len(pending_pods) > 5:
                response += f"* ... and {len(pending_pods) - 5} more pending pods\n"
            response += "\n"
        
        if failed_pods:
            response += "**Failed Pods:**\n"
            for pod in failed_pods[:5]:
                response += f"* `{pod['name']}` ❌ (Failed)\n"
            response += "\n"
        
        response += """**Next Steps:**
* Use `kubectl describe pod <pod-name>` for detailed information
* Use `kubectl logs <pod-name>` to see the logs
* Use `kubectl scale deployment <name> --replicas=<count>` to scale deployments"""
        
        return response
    
    def _format_resources_response(self, real_data: dict, query: str) -> str:
        """Format resources response"""
        if 'metrics' in real_data:
            metrics = real_data['metrics']
            total_pods = real_data.get('total_pods', len(metrics))
            
            # Calculate totals
            total_cpu = 0
            total_memory = 0
            high_cpu_pods = []
            high_memory_pods = []
            
            for pod in metrics:
                name = pod.get('name', 'Unknown')
                cpu_str = pod.get('cpu', '0m')
                memory_str = pod.get('memory', '0Mi')
                
                # Parse CPU
                cpu_value = 0
                if cpu_str.endswith('m'):
                    cpu_value = int(cpu_str[:-1])
                elif cpu_str.endswith('n'):
                    cpu_value = int(cpu_str[:-1]) / 1000000
                
                # Parse Memory
                memory_value = 0
                if memory_str.endswith('Mi'):
                    memory_value = int(memory_str[:-2])
                elif memory_str.endswith('Gi'):
                    memory_value = int(memory_str[:-2]) * 1024
                
                total_cpu += cpu_value
                total_memory += memory_value
                
                if cpu_value > 5:
                    high_cpu_pods.append((name, cpu_str))
                if memory_value > 50:
                    high_memory_pods.append((name, memory_str))
            
            response = f"""**Resource Usage Summary 📊**

I've analyzed your cluster resource usage:

**Total CPU Usage: {total_cpu:.0f}m** (millicores)
**Total Memory Usage: {total_memory:.0f}Mi** (MiB)

"""
            
            if high_cpu_pods:
                response += "**High CPU Usage Pods:**\n"
                for name, cpu in high_cpu_pods[:5]:
                    response += f"* `{name}` - CPU: {cpu} ⚠️\n"
                response += "\n"
            
            if high_memory_pods:
                response += "**High Memory Usage Pods:**\n"
                for name, memory in high_memory_pods[:5]:
                    response += f"* `{name}` - Memory: {memory} ⚠️\n"
                response += "\n"
            
            response += "**All Pods Resource Usage:**\n"
            for pod in metrics[:15]:
                name = pod.get('name', 'Unknown')
                cpu = pod.get('cpu', 'N/A')
                memory = pod.get('memory', 'N/A')
                cpu_status = "🔥" if cpu.endswith('m') and int(cpu[:-1]) > 5 else "✅"
                memory_status = "🔥" if memory.endswith('Mi') and int(memory[:-2]) > 50 else "✅"
                response += f"* `{name}` - CPU: {cpu} {cpu_status}, Memory: {memory} {memory_status}\n"
            
            if len(metrics) > 15:
                response += f"* ... and {len(metrics) - 15} more pods\n"
            
            response += "\n**Next Steps:**\n"
            response += "* Use `kubectl top nodes` to see node resource usage\n"
            response += "* Use `kubectl describe nodes` for detailed node information\n"
            response += "* Consider scaling down pods with low resource usage to save costs\n"
            
            return response
        else:
            return f"**Resource Data:** {real_data}"
    
    def _format_services_response(self, real_data: dict, query: str) -> str:
        """Format services response"""
        services = real_data.get('services', [])
        service_count = real_data.get('service_count', 0)
        
        response = f"""**Services Summary 📊**

I've analyzed your cluster services:

**Total Services: {service_count}**

"""
        
        for service in services[:10]:
            name = service.get('name', 'Unknown')
            service_type = service.get('type', 'Unknown')
            cluster_ip = service.get('cluster_ip', 'None')
            response += f"* `{name}` - Type: {service_type}, IP: {cluster_ip}\n"
        
        if len(services) > 10:
            response += f"* ... and {len(services) - 10} more services\n"
        
        response += "\n**Next Steps:**\n"
        response += "* Use `kubectl describe service <name>` for detailed information\n"
        response += "* Use `kubectl get endpoints` to see service endpoints\n"
        
        return response
    
    def _format_deployments_response(self, real_data: dict, query: str) -> str:
        """Format deployments response"""
        deployments = real_data.get('deployments', [])
        deployment_count = real_data.get('deployment_count', 0)
        
        response = f"""**Deployments Summary 📊**

I've analyzed your cluster deployments:

**Total Deployments: {deployment_count}**

"""
        
        for deployment in deployments[:10]:
            name = deployment.get('name', 'Unknown')
            replicas = deployment.get('replicas', 'Unknown')
            response += f"* `{name}` - Replicas: {replicas}\n"
        
        if len(deployments) > 10:
            response += f"* ... and {len(deployments) - 10} more deployments\n"
        
        response += "\n**Next Steps:**\n"
        response += "* Use `kubectl describe deployment <name>` for detailed information\n"
        response += "* Use `kubectl scale deployment <name> --replicas=<count>` to scale\n"
        
        return response
    
    def _format_cluster_response(self, real_data: dict, query: str) -> str:
        """Format cluster response"""
        if 'nodes' in real_data:
            nodes = real_data['nodes']
            node_count = len(nodes)
            
            response = f"""**Cluster Information 📊**

I've analyzed your cluster:

**Total Nodes: {node_count}**

"""
            
            for node in nodes[:5]:
                name = node.get('name', 'Unknown')
                status = node.get('status', 'Unknown')
                response += f"* `{name}` - {status}\n"
            
            if len(nodes) > 5:
                response += f"* ... and {len(nodes) - 5} more nodes\n"
            
            response += "\n**Next Steps:**\n"
            response += "* Use `kubectl describe nodes` for detailed node information\n"
            response += "* Use `kubectl get events` to see cluster events\n"
            
            return response
        else:
            return f"**Cluster Data:** {real_data}"
    
    def _format_creation_response(self, real_data: dict, query: str) -> str:
        """Format pod creation response"""
        if 'error' in real_data:
            return f"❌ **Error:** {real_data['error']}\n\n**Suggestion:** Check if the image exists and you have proper permissions."
        else:
            pod_name = real_data.get('pod_name', 'Unknown')
            image = real_data.get('image', 'Unknown')
            namespace = real_data.get('namespace', 'default')
            
            return f"""**Pod Creation 🚀**

✅ **Pod Created Successfully!**
* **Name:** `{pod_name}`
* **Image:** `{image}`
* **Namespace:** `{namespace}`
* **Status:** {real_data.get('status', 'Creating')}

**Next Steps:**
* Use `kubectl get pods` to see the pod status
* Use `kubectl describe pod {pod_name}` for detailed information
* Use `kubectl logs {pod_name}` to see the logs"""
    
    def _format_kubectl_response(self, real_data: dict, query: str) -> str:
        """Format kubectl command response"""
        command = real_data.get('command', 'unknown')
        
        if 'delete' in command.lower():
            if real_data.get('return_code') == 0:
                return f"""**Pod Deletion 🗑️**

✅ **Pod Deleted Successfully!**
* **Command:** `kubectl {command}`
* **Output:** {real_data.get('stdout', 'No output')}

**Next Steps:**
* Use `kubectl get pods` to verify the pod is gone
* Use `kubectl get events` to see deletion events"""
            else:
                return f"""**Pod Deletion 🗑️**

❌ **Deletion Failed:**
* **Command:** `kubectl {command}`
* **Error:** {real_data.get('stderr', 'Unknown error')}
* **Return Code:** {real_data.get('return_code', 'Unknown')}"""
        else:
            return f"""**Kubectl Command Execution 🔧**

* **Command:** `kubectl {command}`
* **Return Code:** {real_data.get('return_code', 'Unknown')}
* **Output:** {real_data.get('stdout', 'No output')}
{f"* **Error:** {real_data.get('stderr')}" if real_data.get('stderr') else ""}"""
    
    def _fallback_tool_selection(self, query: str) -> tuple:
        """Fallback pattern matching when AI fails"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['create', 'run', 'start', 'launch']):
            if any(word in query_lower for word in ['pod', 'container']):
                return 'run_container_in_pod', {'image': 'nginx', 'namespace': 'default'}
            else:
                return 'get_pods', {'namespace': 'default'}
        elif any(word in query_lower for word in ['delete', 'remove', 'stop', 'kill']):
            if any(word in query_lower for word in ['pod', 'container']):
                return 'execute_kubectl', {'command': 'get pods'}
            else:
                return 'get_pods', {'namespace': 'default'}
        elif any(word in query_lower for word in ['pod', 'pods', 'container', 'containers']):
            return 'get_pods', {'namespace': 'default'}
        elif any(word in query_lower for word in ['top', 'resource', 'cpu', 'memory', 'usage']):
            return 'get_pod_top', {'namespace': 'default'}
        elif any(word in query_lower for word in ['service', 'services']):
            return 'get_services', {'namespace': 'default'}
        elif any(word in query_lower for word in ['deployment', 'deployments']):
            return 'get_deployments', {'namespace': 'default'}
        elif any(word in query_lower for word in ['node', 'nodes', 'cluster']):
            return 'get_cluster_info', {}
        else:
            return 'get_pods', {'namespace': 'default'}
    
    def _basic_response_formatting(self, tool_name: str, real_data: dict, query: str) -> str:
        """Basic response formatting when AI polishing fails"""
        if tool_name == 'get_pods':
            pods = real_data.get('pods', [])
            pod_count = real_data.get('pod_count', 0)
            running_pods = [p for p in pods if p.get('status') == 'Running']
            
            return f"""**Pod Status Summary 📊**

I've analyzed the pod list for you. Here's a summary of the running pods:

**Total Pods: {pod_count}** 📈
**Running Pods: {len(running_pods)}** ✅

**Running Pods:**
{chr(10).join([f"* `{pod['name']}` ✅ (Running)" for pod in running_pods[:10]])}
{f"* ... and {len(running_pods) - 10} more running pods" if len(running_pods) > 10 else ""}

**Next Steps:**
* Use `kubectl describe pod <pod-name>` for detailed information
* Use `kubectl logs <pod-name>` to see the logs"""
        
        elif tool_name == 'run_container_in_pod':
            if 'error' in real_data:
                return f"❌ **Error:** {real_data['error']}"
            else:
                pod_name = real_data.get('pod_name', 'Unknown')
                return f"""**Pod Creation 🚀**

✅ **Pod Created Successfully!**
* **Name:** `{pod_name}`
* **Image:** `{real_data.get('image', 'Unknown')}`
* **Namespace:** `{real_data.get('namespace', 'default')}`

**Next Steps:**
* Use `kubectl get pods` to see the pod status
* Use `kubectl describe pod {pod_name}` for detailed information"""
        
        else:
            return f"**{tool_name.replace('_', ' ').title()} Result 📊**\n\n{json.dumps(real_data, indent=2)}"
    
    def _call_mcp_tool(self, tool_name: str, args: Dict) -> Dict:
        """Call MCP tool and return result"""
        try:
            # Use the proper MCP client
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(call_mcp_tool(tool_name, args))
            loop.close()
            
            if 'error' in result:
                return {
                    'success': False,
                    'error': result['error']
                }
            else:
                return {
                    'success': True,
                    'result': result.get('result', result)
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _post_process_with_ai(self, user_query: str, tool_name: str, raw_result: str) -> str:
        """Post-process MCP results with AI to create polished responses"""
        try:
            # Extract the actual content from MCP result
            if isinstance(raw_result, dict):
                content = raw_result.get('content', [{}])
                if isinstance(content, list) and len(content) > 0:
                    actual_content = content[0].get('text', '')
                else:
                    actual_content = str(raw_result)
            else:
                actual_content = str(raw_result)
            
            # Create a system prompt for post-processing
            system_prompt = (
                "You are a Kubernetes expert assistant. Your job is to take raw Kubernetes command output "
                "and transform it into a polished, user-friendly response that directly answers the user's question.\n\n"
                
                "**CRITICAL INSTRUCTIONS:**\n"
                "- Analyze the user's question to understand what they really want to know\n"
                "- Transform the raw output into a clean, conversational response\n"
                "- Use emojis and formatting to make it readable\n"
                "- Focus on the key information the user asked for\n"
                "- Be concise but informative\n"
                "- Group related information logically\n"
                "- Highlight important statuses and issues\n\n"
                
                "**RESPONSE STYLE:**\n"
                "- Start with a brief summary of what you found\n"
                "- Use bullet points or sections for clarity\n"
                "- Use appropriate emojis (✅ for running, ⚠️ for issues, ❌ for failures)\n"
                "- End with helpful next steps or suggestions\n"
                "- Keep it conversational and helpful\n\n"
                
                "**EXAMPLES:**\n"
                "- For pod lists: Group by namespace, show status with emojis\n"
                "- For pod details: Show key info like status, image, IP, labels clearly\n"
                "- For errors: Explain what went wrong and suggest fixes\n"
                "- For success: Confirm what was done and suggest next steps"
            )
            
            # Create the message for post-processing
            message = f"""User Question: "{user_query}"

Tool Used: {tool_name}

Raw Kubernetes Output:
{actual_content}

Please transform this raw output into a polished, user-friendly response that directly answers the user's question. Focus on what they asked for and make it easy to understand."""
            
            # Call AI for post-processing
            if self.ai_provider == "groq":
                response = self.anthropic.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": message}
                    ],
                    max_tokens=1000,
                    temperature=0.1
                )
            else:
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": message}]
                )
            
            # Extract the polished response
            if self.ai_provider == "groq":
                return response.choices[0].message.content
            else:
                if response.content and len(response.content) > 0:
                    return response.content[0].text
                else:
                    return actual_content
                
        except Exception as e:
            print(f"⚠️  Post-processing failed: {e}")
            # Fallback to basic formatting
            return self._basic_format_response(tool_name, raw_result)
    
    def _basic_format_response(self, tool_name: str, raw_result: str) -> str:
        """Basic fallback formatting when AI post-processing fails"""
        if isinstance(raw_result, dict):
            content = raw_result.get('content', [{}])
            if isinstance(content, list) and len(content) > 0:
                text_content = content[0].get('text', '')
            else:
                text_content = str(raw_result)
        else:
            text_content = str(raw_result)
        
        if tool_name == 'pods_list':
            return f"📋 **Pod List Results:**\n\n{text_content}"
        elif tool_name == 'pods_get':
            return f"📋 **Pod Details:**\n\n{text_content}"
        elif tool_name == 'pods_run':
            return f"✅ **Pod Creation:**\n\n{text_content}"
        else:
            return f"**{tool_name} Result:**\n\n{text_content}"
    
    def _process_with_ai(self, query: str) -> dict:
        """Process query using Anthropic AI with post-processing"""
        try:
            import json
            
            # Get available tools for AI
            available_tools = self._get_mcp_tools_for_ai()
            
            # Create system prompt for tool selection and execution
            system_prompt = (
                "You are an intelligent Kubernetes AI assistant. Your job is to:\n"
                "1. Understand the user's intent\n"
                "2. Execute the appropriate Kubernetes command\n"
                "3. The results will be automatically polished for the user\n\n"
                
                "**CRITICAL INSTRUCTIONS:**\n"
                "- ALWAYS use intelligent defaults (nginx for web pods)\n"
                "- NEVER ask for clarification - use smart defaults\n"
                "- Execute tools immediately when you understand the intent\n"
                "- Keep your explanations brief - the results will be polished\n\n"
                
                "**POD OPERATIONS:**\n"
                "- 'show me all pods' → Use pods_list\n"
                "- 'list pods in default namespace' → Use pods_list with namespace=default\n"
                "- 'get pod named X' → Use pods_get with name=X\n"
                "- 'create pod named X' → Use pods_run with name=X, image=nginx\n"
                "- 'delete pod X' → Use pods_delete with name=X\n"
                "- 'logs from pod X' → Use pods_log with name=X\n"
                "- 'execute command Y in pod X' → Use pods_exec\n\n"
                
                f"Available tools: {json.dumps([t['name'] for t in available_tools], indent=2)}"
            )
            
            messages = [{"role": "user", "content": query}]
            
            # Call AI with tools (Groq or Anthropic)
            if self.ai_provider == "groq":
                # Groq doesn't support function calling, use natural language approach
                return self._process_with_groq_nl(query, system_prompt)
            else:
                # Anthropic with function calling
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=800,
                    system=system_prompt,
                    messages=messages,
                    tools=available_tools
                )
            
            # Process AI response and handle tool calls
            final_text = []
            tool_results = []
            
            print(f"🔧 AI Response Content: {[c.type for c in response.content]}")
            
            for content in response.content:
                if content.type == 'text':
                    print(f"🔧 AI Text: {content.text}")
                    final_text.append(content.text)
                elif content.type == 'tool_use':
                    tool_name = content.name
                    tool_args = content.input
                    print(f"🔧 AI Tool Call: {tool_name} with args: {tool_args}")
                    
                    # Execute the tool call
                    result = self._call_mcp_tool(tool_name, tool_args)
                    tool_results.append({
                        'tool_name': tool_name,
                        'tool_args': tool_args,
                        'result': result
                    })
            
            # Post-process results with AI for polished responses
            if tool_results:
                # Get the AI's brief explanation
                ai_explanation = "\n\n".join([t.strip() for t in final_text if t and t.strip()])
                
                # Post-process each tool result
                polished_results = []
                for tool_result in tool_results:
                    if tool_result['result']['success']:
                        polished_result = self._post_process_with_ai(
                            query,
                            tool_result['tool_name'],
                            tool_result['result']['result']
                        )
                        polished_results.append(polished_result)
                    else:
                        polished_results.append(f"❌ **Error executing {tool_result['tool_name']}:** {tool_result['result']['error']}")
                
                # Combine AI explanation with polished results
                if ai_explanation:
                    response_text = f"{ai_explanation}\n\n{chr(10).join(polished_results)}"
                else:
                    response_text = "\n\n".join(polished_results)
                
                return {
                    'command': f'AI: {", ".join([tr["tool_name"] for tr in tool_results])}',
                    'explanation': response_text,
                    'ai_processed': True,
                    'tool_results': tool_results,
                    'mcp_result': tool_results[0]['result'] if tool_results else None
                }
            else:
                # No tools called, just return AI text
                response_text = "\n\n".join([t.strip() for t in final_text if t and t.strip()])
                return {
                    'command': 'AI: text_response',
                    'explanation': response_text,
                    'ai_processed': True,
                    'tool_results': [],
                    'mcp_result': None
                }
                
        except Exception as e:
            return {
                'command': None,
                'explanation': f"AI processing failed: {str(e)}",
                'ai_processed': True,
                'tool_results': [],
                'mcp_result': None
            }
    
    def process_query(self, query: str) -> dict:
        """Process natural language query using enhanced AI with post-processing"""
        
        # Try AI processing first if available
        if self.use_ai and self.anthropic:
            try:
                ai_result = self._process_with_ai(query)
                print(f"🔧 Enhanced AI Result: {ai_result}")
                if ai_result.get('ai_processed') and ai_result.get('command'):
                    print(f"✅ Using enhanced AI result: {ai_result.get('command')}")
                    return ai_result
                else:
                    print(f"⚠️  AI didn't process properly")
            except Exception as e:
                print(f"⚠️  AI processing failed: {e}")
        
        # Fallback to basic response
        return {
            'command': 'fallback',
            'explanation': 'AI processing is temporarily unavailable due to rate limits. Please wait a moment and try again.',
            'ai_processed': False,
            'tool_results': [],
            'mcp_result': None
        }
