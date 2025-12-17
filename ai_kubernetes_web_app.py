from simple_kubectl_executor import kubectl_executor
#!/usr/bin/env python3
"""
Simplified AI4K8s Web Application - Lightweight and Efficient
"""

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
import json
import requests
import uuid
import asyncio
import time
from typing import Any, List, Dict, Optional
from mcp_client import call_mcp_tool

# Import predictive monitoring components
try:
    from ai_monitoring_integration import AIMonitoringIntegration
    PREDICTIVE_MONITORING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Predictive monitoring not available: {e}")
    PREDICTIVE_MONITORING_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ai4k8s.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Session cookie configuration
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = False  # Allow JavaScript access
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Allow cross-site requests

db = SQLAlchemy(app)

# Context processor to make current year available in all templates
@app.context_processor
def inject_current_year():
    return {'current_year': datetime.now().year}

# AI-Powered MCP-based Natural Language Processor
class AIPoweredMCPKubernetesProcessor:
    def __init__(self):
        self.mcp_server_url = "http://172.18.0.1:5002/message"
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
                self.anthropic = Anthropic()
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
        
        # Try multiple locations for .env file
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
                self.available_tools = client.available_tools
                print(f"✅ Loaded {len(self.available_tools)} MCP tools")
            else:
                print("⚠️  No MCP tools available")
                self.available_tools = {}
            loop.close()
        except Exception as e:
            print(f"⚠️  Error loading MCP tools: {e}")
            self.available_tools = {}
    
    def _call_mcp_tool(self, tool_name: str, arguments: dict = None) -> dict:
        """Call an MCP tool with the given arguments"""
        try:
            # Use the proper MCP client
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(call_mcp_tool(tool_name, arguments or {}))
            loop.close()
            
            if 'error' in result:
                return {
                    'success': False,
                    'error': result['error'],
                    'tool': tool_name,
                    'arguments': arguments
                }
            else:
                return {
                    'success': True,
                    'result': result.get('result', result),
                    'tool': tool_name,
                    'arguments': arguments
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'tool': tool_name,
                'arguments': arguments
            }
    
    def _get_mcp_tools_for_ai(self):
        """Get MCP tools formatted for AI context"""
        if not self.available_tools:
            return []
        
        tools_for_ai = []
        for tool_name, tool_info in self.available_tools.items():
            # Format for Anthropic API v0.68.0
            tool_schema = tool_info.get('inputSchema', {})
            if not tool_schema:
                tool_schema = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            
            tools_for_ai.append({
                "name": tool_name,
                "description": tool_info.get('description', f'Execute {tool_name} operation'),
                "input_schema": tool_schema
            })
        return tools_for_ai
    
    def _process_with_groq_nl(self, query: str, system_prompt: str) -> dict:
        """Process query using Groq natural language approach"""
        try:
            # Create a system prompt for Groq to determine tool and args
            groq_system_prompt = (
                "You are a Kubernetes expert. Analyze the user's query and determine:\n"
                "1. What Kubernetes operation they want to perform\n"
                "2. What tool to use and with what parameters\n\n"
                
                "Available tools:\n"
                "- pods_list: List pods (params: namespace)\n"
                "- pods_get: Get pod details (params: name, namespace)\n"
                "- pods_run: Create pod (params: name, image, namespace)\n"
                "- pods_delete: Delete pod (params: name, namespace)\n"
                "- pods_log: Get pod logs (params: name, namespace)\n"
                "- pods_top: Get resource usage (params: pod_name, namespace)\n"
                "- pods_exec: Execute command (params: name, command, namespace)\n\n"
                
                "Respond with JSON format:\n"
                "{\n"
                '  "tool": "tool_name",\n'
                '  "args": {"param": "value"},\n'
                '  "explanation": "Brief explanation"\n'
                "}\n\n"
                
                "Use intelligent defaults: namespace=default, image=nginx for web pods."
            )
            
            response = self.anthropic.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": groq_system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            ai_response = response.choices[0].message.content
            
            # Try to extract JSON from response
            import json
            import re
            
            # Find JSON in the response
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                tool_name = parsed.get('tool')
                tool_args = parsed.get('args', {})
                explanation = parsed.get('explanation', '')
                
                if tool_name:
                    # Execute the tool
                    result = self._call_mcp_tool(tool_name, tool_args)
                    
                    if result['success']:
                        # Post-process the result
                        polished_result = self._post_process_with_ai(query, tool_name, result['result'])
                        
                        return {
                            'command': f'Groq: {tool_name}',
                            'explanation': f"{explanation}\n\n{polished_result}",
                            'ai_processed': True,
                            'tool_results': [{'tool_name': tool_name, 'result': result}],
                            'mcp_result': result
                        }
                    else:
                        return {
                            'command': f'Groq: {tool_name} (failed)',
                            'explanation': f"❌ **Error:** {result['error']}",
                            'ai_processed': True,
                            'tool_results': [],
                            'mcp_result': None
                        }
            
            # If no JSON found, return the AI response directly
            return {
                'command': 'Groq: text_response',
                'explanation': ai_response,
                'ai_processed': True,
                'tool_results': [],
                'mcp_result': None
            }
            
        except Exception as e:
            # Fallback to direct response
            return {
                'command': 'Groq: text_response',
                'explanation': f"Groq processing failed: {str(e)}",
                'ai_processed': True,
                'tool_results': [],
                'mcp_result': None
            }
    
    def _post_process_with_ai(self, user_query: str, tool_name: str, raw_result: str) -> str:
        """Post-process MCP results with AI"""
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
                "You are a Kubernetes expert. Transform the raw output into a polished, "
                "user-friendly response with emojis and clear formatting. "
                "Focus on what the user asked for and make it easy to understand."
            )
            
            message = f"""User Question: "{user_query}"
Tool Used: {tool_name}
Raw Output: {actual_content}

Transform this into a polished response."""
            
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
                return response.choices[0].message.content
            else:
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": message}]
                )
                return response.content[0].text if response.content else actual_content
                
        except Exception as e:
            print(f"⚠️  Post-processing failed: {e}")
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
        """Process query using Anthropic AI (like the client)"""
        try:
            import json
            
            # Get available tools for AI
            available_tools = self._get_mcp_tools_for_ai()
            
            # Create enhanced system prompt for intelligent Kubernetes operations
            system_prompt = (
                "You are an intelligent Kubernetes AI assistant with access to MCP tools for cluster management. "
                "You should be proactive and intelligent in your responses:\n\n"
                
                "**CRITICAL: ALWAYS USE INTELLIGENT DEFAULTS**\n"
                "- When user wants to create a pod with just a name, ALWAYS use 'nginx' as the default image\n"
                "- NEVER ask for clarification when you can use intelligent defaults\n"
                "- ALWAYS execute the tool immediately, don't ask questions\n\n"
                
                "**POD OPERATIONS:**\n"
                "- 'create a pod and name it BOBO' → Use pods_run with name='bobo', image='nginx'\n"
                "- 'create pod BOBO' → Use pods_run with name='bobo', image='nginx'\n"
                "- 'create a pod named BOBO' → Use pods_run with name='bobo', image='nginx'\n"
                "- 'delete the BOBO pod' → Use pods_delete with name='bobo'\n"
                "- 'show me all pods' → Use pods_list\n"
                "- 'get logs from BOBO' → Use pods_log with name='bobo'\n\n"
                
                "**MANDATORY BEHAVIOR:**\n"
                "- If user says 'create a pod and name it BOBO', IMMEDIATELY call pods_run with name='bobo' and image='nginx'\n"
                "- If user says 'i don't want to use any image', STILL use 'nginx' as default\n"
                "- NEVER ask 'what image do you want?' - always use 'nginx'\n"
                "- ALWAYS execute the tool, never ask for clarification\n\n"
                
                "**EXAMPLES OF CORRECT BEHAVIOR:**\n"
                "- User: 'create a pod and name it BOBO' → IMMEDIATELY call pods_run(name='bobo', image='nginx')\n"
                "- User: 'i don't want to use any image' → STILL call pods_run(name='bobo', image='nginx')\n"
                "- User: 'create pod test' → IMMEDIATELY call pods_run(name='test', image='nginx')\n\n"
                
                f"Available MCP Tools: {json.dumps(available_tools, indent=2)}"
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
                    max_tokens=1200,
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
            
            # Format the response
            if tool_results:
                # If tools were called, format the results
                response_text = "\n\n".join([t.strip() for t in final_text if t and t.strip()])
                
                # Add tool results
                for tool_result in tool_results:
                    if tool_result['result']['success']:
                        response_text += f"\n\n**{tool_result['tool_name']} Result:**\n{str(tool_result['result']['result'])}"
                    else:
                        response_text += f"\n\n**{tool_result['tool_name']} Error:** {tool_result['result']['error']}"
                
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
    
    def _process_with_regex(self, query: str) -> dict:
        """Process query using regex patterns (fallback)"""
        import re
        query_lower = query.lower().strip()
        
        # Enhanced patterns with better parameter extraction
        
        # 1. POD DETAILS/GET PATTERN
        pod_details_match = re.search(r'(?:show|get|describe|details).*pod.*(?:named|called)\s+[\'"]([^\'"]+)[\'"]', query_lower)
        if pod_details_match:
            pod_name = pod_details_match.group(1)
            result = self._call_mcp_tool('pods_get', {
                'name': pod_name,
                'namespace': 'default'
            })
            return {
                'command': f'Regex: pods_get ({pod_name})',
                'explanation': f"I'll get details for pod '{pod_name}' using MCP tools",
                'ai_processed': False,
                'mcp_result': result
            }
        
        # 2. POD LOGS PATTERN
        pod_logs_match = re.search(r'(?:show|get|view).*logs.*pod.*[\'"]([^\'"]+)[\'"]', query_lower)
        if pod_logs_match:
            pod_name = pod_logs_match.group(1)
            result = self._call_mcp_tool('pods_log', {
                'name': pod_name,
                'namespace': 'default'
            })
            return {
                'command': f'Regex: pods_log ({pod_name})',
                'explanation': f"I'll get logs from pod '{pod_name}' using MCP tools",
                'ai_processed': False,
                'mcp_result': result
            }
        
        # 3. POD EXEC PATTERN
        pod_exec_match = re.search(r'(?:execute|run|exec).*command.*[\'"]([^\'"]+)[\'"].*pod.*[\'"]([^\'"]+)[\'"]', query_lower)
        if pod_exec_match:
            command = pod_exec_match.group(1)
            pod_name = pod_exec_match.group(2)
            result = self._call_mcp_tool('pods_exec', {
                'name': pod_name,
                'command': [command],
                'namespace': 'default'
            })
            return {
                'command': f'Regex: pods_exec ({command} in {pod_name})',
                'explanation': f"I'll execute '{command}' in pod '{pod_name}' using MCP tools",
                'ai_processed': False,
                'mcp_result': result
            }
        
        # 4. POD TOP/RESOURCE USAGE PATTERN
        if re.search(r'(?:resource|usage|top|metrics).*pod', query_lower):
            result = self._call_mcp_tool('pods_top', {
                'all_namespaces': True
            })
            return {
                'command': 'Regex: pods_top',
                'explanation': f"I'll show pod resource usage using MCP tools",
                'ai_processed': False,
                'mcp_result': result
            }
        
        # 5. ENHANCED POD LISTING
        if 'pods' in query_lower and ('show' in query_lower or 'list' in query_lower or 'get' in query_lower or 'display' in query_lower or 'what' in query_lower or 'running' in query_lower):
            result = self._call_mcp_tool('pods_list', {})
            return {
                'command': 'Regex: pods_list',
                'explanation': f"I'll show you all pods using MCP tools",
                'ai_processed': False,
                'mcp_result': result
            }
        
        # 6. ENHANCED POD CREATION with image extraction
        elif 'create' in query_lower and 'pod' in query_lower:
            pod_name = self._extract_pod_name(query)
            # Extract image if specified, otherwise use nginx as default
            image_match = re.search(r'(?:with|using|from).*image.*[\'"]([^\'"]+)[\'"]', query_lower)
            image = image_match.group(1) if image_match else 'nginx'
            
            # Handle "i don't want to use any image" case - still use nginx
            if 'don\'t want' in query_lower and 'image' in query_lower:
                image = 'nginx'
                print(f"🔧 User doesn't want image, using default: {image}")
            
            if not pod_name:
                return {
                    'command': None,
                    'explanation': f"I understand you want to create a pod, but I couldn't extract the pod name from: '{query}'. Please try 'create pod <name>' or 'create a pod named <name>'.",
                    'ai_processed': False,
                    'mcp_result': None
                }
            
            print(f"🔧 Creating pod: {pod_name} with image: {image}")
            result = self._call_mcp_tool('pods_run', {
                'name': pod_name,
                'image': image
            })
            return {
                'command': f'Regex: pods_run ({pod_name} with {image})',
                'explanation': f"I'll create a pod named '{pod_name}' with {image} image using MCP tools",
                'ai_processed': False,
                'mcp_result': result
            }
        
        # 7. ENHANCED POD DELETION
        elif ('delete' in query_lower or 'stop' in query_lower or 'remove' in query_lower) and 'pod' in query_lower:
            pod_name = self._extract_pod_name_from_delete(query)
            if pod_name:
                result = self._call_mcp_tool('pods_delete', {
                    'name': pod_name,
                    'namespace': 'default'
                })
                return {
                    'command': f'Regex: pods_delete ({pod_name})',
                    'explanation': f"I'll delete the pod named '{pod_name}' using MCP tools",
                    'ai_processed': False,
                    'mcp_result': result
                }
            else:
                return {
                    'command': None,
                    'explanation': f"I understand you want to delete a pod, but I couldn't extract the pod name from: '{query}'. Please try 'delete pod <name>' or 'stop pod <name>'.",
                    'ai_processed': False,
                    'mcp_result': None
                }
        
        # 8. SERVICES
        elif 'services' in query_lower and ('show' in query_lower or 'list' in query_lower or 'get' in query_lower or 'display' in query_lower or 'what' in query_lower):
            result = self._call_mcp_tool('resources_list', {
                'apiVersion': 'v1',
                'kind': 'Service'
            })
            return {
                'command': 'Regex: resources_list (services)',
                'explanation': f"I'll show you all services using MCP tools",
                'ai_processed': False,
                'mcp_result': result
            }
        
        # 9. DEPLOYMENTS
        elif 'deployments' in query_lower and ('show' in query_lower or 'list' in query_lower or 'get' in query_lower or 'display' in query_lower or 'what' in query_lower):
            result = self._call_mcp_tool('resources_list', {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment'
            })
            return {
                'command': 'Regex: resources_list (deployments)',
                'explanation': f"I'll show you all deployments using MCP tools",
                'ai_processed': False,
                'mcp_result': result
            }
        
        # 10. NODES
        elif 'nodes' in query_lower and ('show' in query_lower or 'list' in query_lower or 'get' in query_lower or 'display' in query_lower or 'what' in query_lower):
            result = self._call_mcp_tool('resources_list', {
                'apiVersion': 'v1',
                'kind': 'Node'
            })
            return {
                'command': 'Regex: resources_list (nodes)',
                'explanation': f"I'll show you all nodes using MCP tools",
                'ai_processed': False,
                'mcp_result': result
            }
        
        # 11. CLUSTER HEALTH CHECK
        elif any(word in query_lower for word in ['health', 'wrong', 'error', 'problem', 'issue', 'diagnose', 'check']) and ('cluster' in query_lower or 'how' in query_lower):
            result = self._call_mcp_tool('events_list', {})
            return {
                'command': 'Regex: events_list',
                'explanation': f"I'll check the cluster health by looking at events using MCP tools",
                'ai_processed': False,
                'mcp_result': result
            }
        
        # 12. EVENTS
        elif 'events' in query_lower:
            result = self._call_mcp_tool('events_list', {})
            return {
                'command': 'Regex: events_list',
                'explanation': f"I'll check the events using MCP tools",
                'ai_processed': False,
                'mcp_result': result
            }
        
        # Default response
        return {
            'command': None,
            'explanation': f"I understand you want to: '{query}'. I can help with pods, services, deployments, nodes, events, and cluster health using intelligent MCP tools.",
            'ai_processed': False,
            'mcp_result': None
        }
    
    def process_query(self, query: str) -> dict:
        """Process natural language query using AI-first approach with regex fallback"""
        
        # Try AI processing first if available
        if self.use_ai and self.anthropic:
            try:
                ai_result = self._process_with_ai(query)
                print(f"🔧 AI Result: {ai_result}")
                # If AI successfully processed the query, return it
                if ai_result.get('ai_processed') and ai_result.get('command'):
                    print(f"✅ Using AI result: {ai_result.get('command')}")
                    return ai_result
                else:
                    print(f"⚠️  AI didn't process properly, falling back to regex")
            except Exception as e:
                print(f"⚠️  AI processing failed, falling back to regex: {e}")
        
        # Fall back to regex processing
        print(f"🔧 Using regex fallback for: {query}")
        return self._process_with_regex(query)
    
    def _extract_pod_name(self, query: str) -> str:
        """Extract pod name from create pod query"""
        import re
        words = query.split()
        pod_name = None
        
        print(f"🔧 Extracting pod name from: '{query}'")
        
        # Pattern 1: "name it BOBO" or "name it BOBO pod"
        name_it_match = re.search(r'name\s+it\s+([a-zA-Z0-9-_]+)', query, re.IGNORECASE)
        if name_it_match:
            pod_name = name_it_match.group(1).lower()
            print(f"✅ Pattern 1 matched: {pod_name}")
            return pod_name
        
        # Pattern 2: "named BOBO" or "called BOBO"
        named_match = re.search(r'(?:named|called)\s+([a-zA-Z0-9-_]+)', query, re.IGNORECASE)
        if named_match:
            pod_name = named_match.group(1).lower()
            print(f"✅ Pattern 2 matched: {pod_name}")
            return pod_name
        
        # Pattern 3: "create pod BOBO" or "create a pod BOBO"
        create_pod_match = re.search(r'create\s+(?:a\s+)?pod\s+([a-zA-Z0-9-_]+)', query, re.IGNORECASE)
        if create_pod_match:
            pod_name = create_pod_match.group(1).lower()
            print(f"✅ Pattern 3 matched: {pod_name}")
            return pod_name
        
        # Pattern 4: "pod named BOBO" or "pod called BOBO"
        pod_named_match = re.search(r'pod\s+(?:named|called)\s+([a-zA-Z0-9-_]+)', query, re.IGNORECASE)
        if pod_named_match:
            pod_name = pod_named_match.group(1).lower()
            print(f"✅ Pattern 4 matched: {pod_name}")
            return pod_name
        
        # Pattern 5: Look for capitalized words that might be pod names
        # This handles cases like "create a pod and name it BOBO"
        capitalized_words = re.findall(r'\b[A-Z][A-Z0-9-_]*\b', query)
        if capitalized_words:
            print(f"🔧 Found capitalized words: {capitalized_words}")
            # Return the first capitalized word that's not common words
            common_words = {'POD', 'CREATE', 'NAME', 'IT', 'AND', 'THE', 'A', 'AN'}
            for word in capitalized_words:
                if word not in common_words:
                    pod_name = word.lower()
                    print(f"✅ Pattern 5 matched: {pod_name}")
                    return pod_name
        
        print(f"❌ No pattern matched for: '{query}'")
        return pod_name
    
    def _extract_pod_name_from_delete(self, query: str) -> str:
        """Extract pod name from delete/stop pod query"""
        words = query.split()
        pod_name = None
        
        # Look for "the <name> pod" pattern
        for i, word in enumerate(words):
            if word.lower() == "the" and i + 1 < len(words):
                # Check if next word is the pod name
                potential_name = words[i + 1]
                # Skip common words that might follow "the"
                if potential_name.lower() not in ["pod", "pods", "container", "containers"]:
                    pod_name = potential_name.lower()
                    break
        
        # Look for "pod <name>" pattern
        if not pod_name:
            for i, word in enumerate(words):
                if word.lower() == "pod" and i + 1 < len(words):
                    potential_name = words[i + 1]
                    if potential_name.lower() not in ["the", "a", "an"]:
                        pod_name = potential_name.lower()
                        break
        
        return pod_name

# Initialize AI-powered MCP processor
try:
    from ai_processor import EnhancedAIProcessor
    processor = EnhancedAIProcessor()
    print("✅ Using enhanced AI processor with post-processing")
except ImportError:
    from ai_kubernetes_web_app import AIPoweredMCPKubernetesProcessor
    processor = AIPoweredMCPKubernetesProcessor()
    print("⚠️  Using original AI processor (enhanced processor not available)")

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    servers = db.relationship('Server', backref='owner', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class Server(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    server_type = db.Column(db.String(50), nullable=False)
    connection_string = db.Column(db.String(200), nullable=False)
    
    # Authentication fields
    username = db.Column(db.String(100), nullable=True)
    password = db.Column(db.String(255), nullable=True)  # Will be encrypted
    ssh_key = db.Column(db.Text, nullable=True)  # SSH private key content
    ssh_key_path = db.Column(db.String(255), nullable=True)  # Path to SSH key file
    ssh_port = db.Column(db.Integer, default=22)
    
    # Kubernetes-specific fields
    kubeconfig = db.Column(db.Text, nullable=True)
    namespace = db.Column(db.String(100), default='default')
    
    # Connection settings
    connection_timeout = db.Column(db.Integer, default=30)
    verify_ssl = db.Column(db.Boolean, default=True)
    
    # Status and metadata
    status = db.Column(db.String(20), default='inactive')
    last_connection_test = db.Column(db.DateTime, nullable=True)
    connection_error = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_accessed = db.Column(db.DateTime, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    chats = db.relationship('Chat', backref='server', lazy=True, cascade='all, delete-orphan')

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    server_id = db.Column(db.Integer, db.ForeignKey('server.id'), nullable=False)
    user_message = db.Column(db.Text, nullable=False)
    ai_response = db.Column(db.Text, nullable=False)
    mcp_tool_used = db.Column(db.String(100), nullable=True)
    processing_method = db.Column(db.String(50), nullable=True)  # 'AI' or 'Regex'
    mcp_success = db.Column(db.Boolean, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Chat {self.id}: {self.user_message[:50]}...>'

    def set_password(self, password):
        """Encrypt and store password"""
        if password:
            self.password = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if provided password matches stored encrypted password"""
        if not self.password:
            return False
        return check_password_hash(self.password, password)
    
    def get_connection_info(self):
        """Get connection information for display (without sensitive data)"""
        return {
            'id': self.id,
            'name': self.name,
            'server_type': self.server_type,
            'connection_string': self.connection_string,
            'username': self.username,
            'ssh_port': self.ssh_port,
            'namespace': self.namespace,
            'status': self.status,
            'has_password': bool(self.password),
            'has_ssh_key': bool(self.ssh_key),
            'has_kubeconfig': bool(self.kubeconfig),
            'last_connection_test': self.last_connection_test,
            'connection_error': self.connection_error,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed
        }
    
    def __repr__(self):
        return f'<Server {self.name}>'

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))

        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please log in to access the dashboard.', 'warning')
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    # Handle stale/missing user gracefully
    if not user:
        session.clear()
        flash('Your session has expired. Please log in again.', 'warning')
        return redirect(url_for('login'))
    servers = Server.query.filter_by(user_id=user.id).all()
    
    # Check cluster status for each server (quick check)
    for server in servers:
        if server.kubeconfig:
            try:
                import subprocess
                import os
                import tempfile
                
                # Create temp kubeconfig
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                    f.write(server.kubeconfig)
                    temp_kubeconfig = f.name
                
                env = os.environ.copy()
                env['KUBECONFIG'] = temp_kubeconfig
                
                # Quick cluster-info check
                result = subprocess.run(
                    ['kubectl', 'cluster-info'],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    env=env
                )
                
                # Check for connection errors
                if result.returncode != 0:
                    error_output = result.stderr.lower()
                    if any(phrase in error_output for phrase in ["no such host", "connection refused", "timeout", "unable to connect", "dial tcp", "name or service not known", "name resolution"]):
                        server.status = "error"
                        server.connection_error = result.stderr
                        print(f"❌ Server {server.id} ({server.name}): Cluster disconnected - {result.stderr[:100]}")
                    else:
                        server.status = "inactive"
                        server.connection_error = result.stderr
                else:
                    server.status = "active"
                    server.connection_error = None
                    print(f"✅ Server {server.id} ({server.name}): Cluster connected")
                
                # Clean up
                try:
                    os.unlink(temp_kubeconfig)
                except:
                    pass
                    
            except Exception as e:
                # If check fails, mark as error
                server.status = "error"
                server.connection_error = str(e)
                print(f"❌ Server {server.id} ({server.name}): Status check failed - {str(e)}")
    
    # Commit status changes
    try:
        db.session.commit()
        print(f"✅ Updated status for {len(servers)} servers")
    except Exception as e:
        print(f"❌ Failed to commit status changes: {e}")
        db.session.rollback()
    
    return render_template('dashboard.html', user=user, servers=servers)


@app.route('/benchmark')
def benchmark_page():
    """Render benchmark dashboard."""
    if 'user_id' not in session:
        flash('Please log in to access benchmarks.', 'warning')
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        flash('Your session has expired. Please log in again.', 'warning')
        return redirect(url_for('login'))

    servers = Server.query.filter_by(user_id=user.id).all()
    return render_template('benchmark.html', user=user, servers=servers)


@app.route('/api/benchmark/<int:server_id>', methods=['POST'])
def run_benchmark(server_id):
    """Execute a lightweight benchmark comparing RAG vs LLM latency."""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()

    try:
        iterations = request.json.get('iterations', 2) if request.is_json else 2
        iterations = max(1, min(int(iterations), 20))

        ai_monitoring = get_ai_monitoring(server_id)
        if not ai_monitoring:
            return jsonify({'error': 'AI monitoring not available for this server'}), 500

        warmup_iterations = 1 if iterations > 1 else 0
        for _ in range(warmup_iterations):
            try:
                ai_monitoring.get_current_analysis()
                ai_monitoring.get_llm_recommendations()
            except Exception:
                pass
            time.sleep(0.1)

        rag_times, rag_sizes, rag_items = [], [], []
        llm_times, llm_sizes, llm_items = [], [], []

        def safe_dump(payload: Any) -> bytes:
            try:
                return json.dumps(payload, default=lambda o: o.isoformat() if hasattr(o, "isoformat") else str(o)).encode('utf-8')
            except TypeError:
                return json.dumps(payload, default=str).encode('utf-8')

        for _ in range(iterations):
            start = time.time()
            analysis = ai_monitoring.get_current_analysis()
            rag_times.append(time.time() - start)
            rag_data = safe_dump(analysis)
            rag_sizes.append(len(rag_data))
            rag_items.append(len(analysis.get('rag_recommendations', [])))
            time.sleep(0.05)

            start = time.time()
            llm_recs = ai_monitoring.get_llm_recommendations()
            llm_times.append(time.time() - start)
            llm_data = safe_dump({'llm_recommendations': llm_recs})
            llm_sizes.append(len(llm_data))
            llm_items.append(len(llm_recs))
            time.sleep(0.05)

        def summarize(values: List[float]) -> Dict[str, float]:
            if not values:
                return {'avg': 0, 'min': 0, 'max': 0, 'p50': 0, 'p95': 0}
            sorted_vals = sorted(values)
            return {
                'avg': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'p50': sorted_vals[len(sorted_vals)//2],
                'p95': sorted_vals[int(len(sorted_vals)*0.95)] if len(sorted_vals) > 1 else sorted_vals[0]
            }

        return jsonify({
            'server_id': server_id,
            'iterations': iterations,
            'timestamp': datetime.utcnow().isoformat(),
            'rag': {
                'latency': summarize(rag_times),
                'payload_size': summarize(rag_sizes),
                'item_count': summarize(rag_items),
            },
            'llm': {
                'latency': summarize(llm_times),
                'payload_size': summarize(llm_sizes),
                'item_count': summarize(llm_items),
            }
        })
    except Exception as exc:
        app.logger.exception("Benchmark failed for server %s", server_id)
        return jsonify({'error': str(exc)}), 500

@app.route('/add_server', methods=['GET', 'POST'])
def add_server():
    if 'user_id' not in session:
        flash('Please log in to add a server.', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        name = request.form['name']
        server_type = request.form['server_type']
        connection_string = request.form['connection_string']
        
        # Authentication fields
        username = request.form.get('username')
        password = request.form.get('password')
        ssh_key = request.form.get('ssh_key')
        ssh_key_path = request.form.get('ssh_key_path')
        ssh_port = int(request.form.get('ssh_port', 22))
        
        # Kubernetes fields
        kubeconfig = request.form.get('kubeconfig')
        namespace = request.form.get('namespace', 'default')
        
        # Connection settings
        connection_timeout = int(request.form.get('connection_timeout', 30))
        verify_ssl = request.form.get('verify_ssl') == 'on'
        
        new_server = Server(
            name=name,
            server_type=server_type,
            connection_string=connection_string,
            username=username,
            ssh_key=ssh_key,
            ssh_key_path=ssh_key_path,
            ssh_port=ssh_port,
            kubeconfig=kubeconfig,
            namespace=namespace,
            connection_timeout=connection_timeout,
            verify_ssl=verify_ssl,
            user_id=session['user_id']
        )
        
        # Set password if provided
        if password:
            new_server.set_password(password)
        
        db.session.add(new_server)
        db.session.commit()
        flash(f'Server "{name}" added successfully!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('add_server.html')

@app.route('/server/<int:server_id>')
def server_detail(server_id):
    if 'user_id' not in session:
        flash('Please log in to view server details.', 'warning')
        return redirect(url_for('login'))
    
    server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
    
    # Get recent chat activity
    recent_chats = Chat.query.filter_by(
        server_id=server_id, 
        user_id=session['user_id']
    ).order_by(Chat.timestamp.desc()).limit(10).all()
    
    return render_template('server_detail.html', server=server, recent_chats=recent_chats)

@app.route('/chat/<int:server_id>')
def chat(server_id):
    if 'user_id' not in session:
        flash('Please log in to use the AI chat.', 'warning')
        return redirect(url_for('login'))
    
    server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
    return render_template('chat.html', server=server)

@app.route('/api/chat/<int:server_id>', methods=['POST'])
def api_chat(server_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first()
    if not server:
        return jsonify({'error': 'Server not found'}), 404
    
    data = request.get_json()
    message = data.get('message', '')
    
    # Check if it's a direct kubectl command
    if message.strip().startswith('kubectl'):
        try:
            import shlex

            tokens = shlex.split(message.strip())
            if len(tokens) < 2:
                return jsonify({'error': 'Invalid kubectl command'}), 400

            command = tokens[1]

            def extract_namespace(args, default=None):
                if '-A' in args or '--all-namespaces' in args:
                    return 'all'
                if '-n' in args:
                    idx = args.index('-n')
                    if idx + 1 < len(args):
                        return args[idx + 1]
                for token in args:
                    if token.startswith('--namespace='):
                        return token.split('=', 1)[1]
                return default

            def run_tool(tool_name, params):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(call_mcp_tool(tool_name, params))
                finally:
                    loop.close()

            result = None

            if command == 'get':
                resource = tokens[2] if len(tokens) > 2 else ''

                if resource in ('pods', 'pod'):
                    namespace = extract_namespace(tokens, default='all')
                    if resource == 'pod' and len(tokens) > 3:
                        pod_name = tokens[3]
                        result = run_tool('pods_get', {
                            'name': pod_name,
                            'namespace': namespace if namespace and namespace != 'all' else 'default'
                        })
                    else:
                        result = run_tool('pods_list', {'namespace': namespace or 'all'})
                elif resource in ('events', 'event'):
                    namespace = extract_namespace(tokens, default='all')
                    result = run_tool('events_list', {'namespace': namespace or 'all'})
                elif resource in ('namespaces', 'ns'):
                    result = run_tool('namespaces_list', {})
                else:
                    return jsonify({'error': f'Unsupported kubectl get resource: {resource}'}), 400

            elif command == 'top':
                resource = tokens[2] if len(tokens) > 2 else ''
                if resource.startswith('pod'):
                    namespace = extract_namespace(tokens, default='all')
                    pod_name = None
                    for token in tokens[2:]:
                        if not token.startswith('-') and token != resource:
                            pod_name = token
                            break
                    result = run_tool('pods_top', {
                        'namespace': namespace or 'all',
                        'pod_name': pod_name
                    })
                else:
                    return jsonify({'error': f'Unsupported kubectl top resource: {resource}'}), 400

            elif command == 'describe':
                if 'pod' in tokens:
                    namespace = extract_namespace(tokens, default='default')
                    pod_idx = tokens.index('pod')
                    if pod_idx + 1 >= len(tokens):
                        return jsonify({'error': 'Pod name required for kubectl describe pod'}), 400
                    pod_name = tokens[pod_idx + 1]
                    result = run_tool('pods_get', {
                        'name': pod_name,
                        'namespace': namespace
                    })
                else:
                    return jsonify({'error': 'Unsupported kubectl describe command'}), 400

            elif command == 'logs':
                namespace = extract_namespace(tokens, default='default')
                pod_name = None
                for token in tokens[2:]:
                    if not token.startswith('-'):
                        pod_name = token
                        break
                if not pod_name:
                    return jsonify({'error': 'Pod name required for kubectl logs'}), 400
                result = run_tool('pods_log', {
                    'name': pod_name,
                    'namespace': namespace
                })

            elif command == 'delete':
                if 'pod' in tokens:
                    namespace = extract_namespace(tokens, default='default')
                    pod_idx = tokens.index('pod')
                    if pod_idx + 1 >= len(tokens):
                        return jsonify({'error': 'Pod name required for kubectl delete pod'}), 400
                    pod_name = tokens[pod_idx + 1]
                    result = run_tool('pods_delete', {
                        'name': pod_name,
                        'namespace': namespace
                    })
                else:
                    return jsonify({'error': 'Unsupported kubectl delete command'}), 400
            else:
                return jsonify({'error': f'Unsupported kubectl command: {command}'}), 400

            if result and result.get('success'):
                payload = result.get('result', result)
                if isinstance(payload, dict):
                    content = json.dumps(payload, indent=2)
                else:
                    content = str(payload)

                chat = Chat(
                    user_id=session['user_id'],
                    server_id=server_id,
                    user_message=message,
                    ai_response=content,
                    mcp_tool_used=f'kubectl_{command}',
                    processing_method='MCP',
                    mcp_success=True
                )
                db.session.add(chat)
                db.session.commit()

                return jsonify({'response': content, 'chat_id': chat.id})

            error_message = result.get('error', 'Command failed') if result else 'Unknown error'
            return jsonify({'error': error_message}), 500
        except Exception as e:
            app.logger.exception("Kubectl command processing failed")
            return jsonify({'error': str(e)}), 500
    else:
        # Natural language query - use intelligent MCP processor
        try:
            processed = processor.process_query(message)
            
            if processed['command'] and processed.get('mcp_result'):
                # MCP tool was called directly
                mcp_result = processed['mcp_result']
                
                if mcp_result['success']:
                    # Format the MCP result nicely
                    if 'content' in mcp_result['result']:
                        # Handle MCP content format
                        content = mcp_result['result']['content']
                        if isinstance(content, list) and len(content) > 0:
                            result_text = content[0].get('text', str(content))
                        else:
                            result_text = str(content)
                    else:
                        result_text = str(mcp_result['result'])
                    
                    response_text = processed["explanation"]
                    
                    # Store chat in database
                    chat = Chat(
                        user_id=session['user_id'],
                        server_id=server_id,
                        user_message=message,
                        ai_response=response_text,
                        mcp_tool_used=processed['command'],
                        processing_method='AI' if processed.get('ai_processed') else 'Regex',
                        mcp_success=True
                    )
                    db.session.add(chat)
                    db.session.commit()
                    
                    return jsonify({
                        'response': response_text,
                        'status': 'success',
                        'ai_processed': processed.get('ai_processed', False),
                        'mcp_tool': processed['command'],
                        'mcp_success': True,
                        'processing_method': 'AI' if processed.get('ai_processed') else 'Regex',
                        'chat_id': chat.id
                    })
                else:
                    error_response = f"{processed['explanation']}\n\n**MCP Error:** {mcp_result.get('error', 'Unknown error')}"
                    
                    # Store chat in database
                    chat = Chat(
                        user_id=session['user_id'],
                        server_id=server_id,
                        user_message=message,
                        ai_response=error_response,
                        mcp_tool_used=processed['command'],
                        processing_method='AI' if processed.get('ai_processed') else 'Regex',
                        mcp_success=False
                    )
                    db.session.add(chat)
                    db.session.commit()
                    
                    return jsonify({
                        'response': error_response,
                        'status': 'error',
                        'ai_processed': processed.get('ai_processed', False),
                        'mcp_tool': processed['command'],
                        'mcp_success': False,
                        'processing_method': 'AI' if processed.get('ai_processed') else 'Regex',
                        'chat_id': chat.id
                    })
            else:
                # Store chat in database
                chat = Chat(
                    user_id=session['user_id'],
                    server_id=server_id,
                    user_message=message,
                    ai_response=processed['explanation'],
                    mcp_tool_used=None,
                    processing_method='AI' if processed.get('ai_processed') else 'Regex',
                    mcp_success=None
                )
                db.session.add(chat)
                db.session.commit()
                
                return jsonify({
                    'response': processed['explanation'],
                    'status': 'info',
                    'ai_processed': processed.get('ai_processed', False),
                    'mcp_tool': None,
                    'processing_method': 'AI' if processed.get('ai_processed') else 'Regex',
                    'chat_id': chat.id
                })
                
        except Exception as e:
            # Store error in database
            error_response = f"I apologize, but I encountered an error: {str(e)}"
            chat = Chat(
                user_id=session['user_id'],
                server_id=server_id,
                user_message=message,
                ai_response=error_response,
                mcp_tool_used=None,
                processing_method='Error',
                mcp_success=False
            )
            db.session.add(chat)
            db.session.commit()
            
            return jsonify({
                'response': error_response,
                'status': 'error',
                'ai_processed': True,
                'chat_id': chat.id
            }), 500

@app.route('/api/chat_history/<int:server_id>')
def get_chat_history(server_id):
    """Get chat history for a specific server"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first()
    if not server:
        return jsonify({'error': 'Server not found'}), 404
    
    # Get chat history, ordered by most recent first
    chats = Chat.query.filter_by(
        server_id=server_id, 
        user_id=session['user_id']
    ).order_by(Chat.timestamp.desc()).limit(50).all()
    
    chat_history = []
    for chat in chats:
        chat_history.append({
            'id': chat.id,
            'user_message': chat.user_message,
            'ai_response': chat.ai_response,
            'mcp_tool_used': chat.mcp_tool_used,
            'processing_method': chat.processing_method,
            'mcp_success': chat.mcp_success,
            'timestamp': chat.timestamp.isoformat()
        })
    
    return jsonify({
        'chats': chat_history,
        'server_id': server_id,
        'server_name': server.name
    })

@app.route('/api/server_status/<int:server_id>')
def server_status(server_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
    
    # Update last accessed time
    server.last_accessed = datetime.utcnow()
    
    # Check if cluster is actually connected by testing kubectl
    try:
        import subprocess
        import os
        
        # Set up kubeconfig if available
        env = os.environ.copy()
        if server.kubeconfig:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(server.kubeconfig)
                temp_kubeconfig = f.name
            env['KUBECONFIG'] = temp_kubeconfig
        
        # Test cluster connection
        result = subprocess.run(
            ['kubectl', 'cluster-info'],
            capture_output=True,
            text=True,
            timeout=10,
            env=env
        )
        
        if result.returncode == 0:
            server.status = "active"
            server.connection_error = None
        else:
            # Check for connection errors
            error_output = result.stderr.lower()
            if any(phrase in error_output for phrase in ["no such host", "connection refused", "timeout", "unable to connect", "dial tcp"]):
                server.status = "error"
                server.connection_error = result.stderr
            else:
                server.status = "inactive"
                server.connection_error = result.stderr
        
        # Clean up temp kubeconfig
        if server.kubeconfig and 'temp_kubeconfig' in locals():
            try:
                os.unlink(temp_kubeconfig)
            except:
                pass
                
    except Exception as e:
        # If we can't test, mark as error
        server.status = "error"
        server.connection_error = str(e)
    
    db.session.commit()
    
    return jsonify({
        'status': server.status,
        'last_accessed': server.last_accessed.isoformat() if server.last_accessed else None,
        'connection_error': server.connection_error
    })


@app.route("/api/delete_server/<int:server_id>", methods=["DELETE"])
def api_delete_server(server_id):
    """Delete a server"""
    try:
        # Get the server
        server = Server.query.get_or_404(server_id)
        
        # Check if user owns this server
        if server.user_id != session.get("user_id"):
            return jsonify({"success": False, "message": "Unauthorized"}), 403
        
        # Delete the server
        db.session.delete(server)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": f"Server {server.name} has been deleted successfully."
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            "success": False,
            "message": f"Failed to delete server: {str(e)}"
        }), 500
@app.route("/api/test_connection/<int:server_id>", methods=["POST"])
def test_connection(server_id):
    """Test connection to a server"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
    
    try:
        # Test different connection methods based on server type
        if server.server_type == 'local':
            # For local servers, test kubectl connectivity
            import subprocess
            timeout = getattr(server, 'connection_timeout', 30)
            result = subprocess.run(['kubectl', 'cluster-info'], 
                                  capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                server.status = 'active'
                server.connection_error = None
                message = 'Local cluster connection successful'
            else:
                server.status = 'error'
                server.connection_error = result.stderr
                message = f'Local cluster connection failed: {result.stderr}'
        
        elif server.server_type in ['remote', 'cloud']:
            # For remote servers, test SSH connectivity
            import paramiko
            import socket
            
            # Extract host and port from connection string
            if ':' in server.connection_string:
                host, port = server.connection_string.split(':')
                port = int(port)
            else:
                host = server.connection_string
                port = getattr(server, 'ssh_port', 22)
            
            # Test SSH connection
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            try:
                username = getattr(server, 'username', None)
                timeout = getattr(server, 'connection_timeout', 30)
                
                if getattr(server, 'ssh_key', None):
                    # Use SSH key content
                    import io
                    key_file = io.StringIO(server.ssh_key)
                    private_key = paramiko.RSAKey.from_private_key(key_file)
                    ssh.connect(host, port=port, username=username, pkey=private_key, 
                               timeout=timeout)
                elif getattr(server, 'ssh_key_path', None):
                    # Use SSH key file path
                    private_key = paramiko.RSAKey.from_private_key_file(server.ssh_key_path)
                    ssh.connect(host, port=port, username=username, pkey=private_key,
                               timeout=timeout)
                elif getattr(server, 'password', None):
                    # Use password authentication
                    ssh.connect(host, port=port, username=username, password=server.password,
                               timeout=timeout)
                else:
                    raise Exception('No authentication method provided')
                
                # Test kubectl on remote server
                stdin, stdout, stderr = ssh.exec_command('kubectl cluster-info')
                if stdout.channel.recv_exit_status() == 0:
                    server.status = 'active'
                    server.connection_error = None
                    message = 'Remote server connection successful'
                else:
                    server.status = 'error'
                    server.connection_error = stderr.read().decode()
                    message = f'Remote kubectl test failed: {stderr.read().decode()}'
                
                ssh.close()
                
            except Exception as e:
                server.status = 'error'
                server.connection_error = str(e)
                message = f'SSH connection failed: {str(e)}'
        
        else:
            server.status = 'error'
            server.connection_error = 'Unknown server type'
            message = 'Unknown server type'
        
        # Update server status
        server.last_connection_test = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'success': server.status == 'active',
            'message': message,
            'status': server.status,
            'error': server.connection_error,
            'last_test': server.last_connection_test.isoformat()
        })
        
    except Exception as e:
        server.status = 'error'
        server.connection_error = str(e)
        server.last_connection_test = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'success': False,
            'message': f'Connection test failed: {str(e)}',
            'status': server.status,
            'error': server.connection_error
        }), 500

# Predictive Monitoring Routes (always register endpoints; guard at runtime)
if True:
    # Store AI monitoring instances per server
    ai_monitoring_instances = {}

    def _patch_kubeconfig_for_container(original_path: str) -> str:
        """Optionally patch kubeconfig when running inside a Docker container.
        
        When AI4K8s runs in Docker, a kubeconfig that points to 127.0.0.1 will
        actually point *inside* the container. In that case we rewrite the
        server URL to reach the host (e.g. host.docker.internal).
        
        On bare-metal / HPC environments (like AMD HPC), we must NOT patch the
        kubeconfig; 127.0.0.1 should remain as-is so SSH tunnels work.
        """
        try:
            import os
            import re
            import tempfile
            import socket

            if not original_path or not os.path.exists(original_path):
                return original_path

            # Detect if we are actually running inside a Docker container.
            # If not, skip any patching and use the kubeconfig as-is.
            in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
            if not in_docker:
                # Running directly on host (e.g. AMD HPC) - do not rewrite 127.0.0.1
                return original_path

            with open(original_path, 'r') as f:
                content = f.read()
            
            # Try different host addresses in order of preference
            host_addresses = [
                'host.docker.internal',  # Docker Desktop
                '172.17.0.1',           # Docker default bridge
                '172.18.0.1',           # Alternative bridge
                socket.gethostbyname('host.docker.internal') if os.environ.get('DOCKER_DESKTOP', False) else None
            ]
            
            # Filter out None values
            host_addresses = [addr for addr in host_addresses if addr is not None]
            
            # Use the first available host address
            host_ip = host_addresses[0] if host_addresses else '172.17.0.1'
            
            # Replace 127.0.0.1 with the appropriate host IP
            patched = re.sub(r"server:\s*https://127\.0\.0\.1:(\d+)",
                             rf"server: https://{host_ip}:\1", content)
            
            if patched != content:
                print(f"🔧 Patching kubeconfig: 127.0.0.1 -> {host_ip}")
                tf = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
                tf.write(patched)
                tf.flush()
                tf.close()
                return tf.name
            return original_path
        except Exception as e:
            print(f"⚠️ Failed to patch kubeconfig: {e}")
            return original_path
    
    def get_ai_monitoring(server_id):
        """Get or create AI monitoring instance for a specific server"""
        if not PREDICTIVE_MONITORING_AVAILABLE:
            return None
        if server_id not in ai_monitoring_instances:
            # Get server details to determine kubeconfig path
            server = Server.query.get(server_id)
            if server:
                kubeconfig_path = None
                import os
                # If a kubeconfig is stored in DB, always prefer it
                if getattr(server, 'kubeconfig', None):
                    try:
                        import tempfile
                        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
                        temp_file.write(server.kubeconfig)
                        temp_file.close()
                        kubeconfig_path = temp_file.name
                    except Exception:
                        kubeconfig_path = None
                # Otherwise, try well-known paths (inside container)
                if kubeconfig_path is None:
                    candidate_paths = [
                        '/app/instance/kubeconfig_admin',
                        os.path.expanduser('~/.kube/config'),
                        '/etc/kubernetes/admin.conf',
                        '/var/lib/kubelet/kubeconfig'
                    ]
                    for path in candidate_paths:
                        if os.path.exists(path):
                            kubeconfig_path = path
                            break
                # Patch kubeconfig if pointing to 127.0.0.1
                kubeconfig_path = _patch_kubeconfig_for_container(kubeconfig_path)
                
                print(f"🔧 Creating AI monitoring instance for server {server_id} ({server.name})")
                print(f"🔧 Using kubeconfig: {kubeconfig_path}")
                ai_monitoring_instances[server_id] = AIMonitoringIntegration(kubeconfig_path)
            else:
                print(f"⚠️  Server {server_id} not found")
        return ai_monitoring_instances.get(server_id)
    
    @app.route('/monitoring/<int:server_id>')
    def monitoring_dashboard(server_id):
        """Predictive monitoring dashboard for specific server"""
        if 'user_id' not in session:
            return redirect(url_for('login'))
        
        # Verify server belongs to user
        server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
        user = User.query.get(session['user_id'])
        return render_template('monitoring.html', user=user, server=server)

    # Support legacy/query-param based route as well
    @app.route('/monitoring')
    def monitoring_dashboard_query():
        """Predictive monitoring dashboard using ?server_id= query param"""
        if 'user_id' not in session:
            return redirect(url_for('login'))

        server_id = request.args.get('server_id', type=int)
        if not server_id:
            flash('Server ID is required', 'warning')
            return redirect(url_for('dashboard'))

        server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
        user = User.query.get(session['user_id'])
        return render_template('monitoring.html', user=user, server=server)
    
    @app.route('/api/monitoring/insights/<int:server_id>')
    def get_monitoring_insights(server_id):
        """Get AI monitoring insights for specific server"""
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        
        # Verify server belongs to user
        server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
        
        try:
            print(f"🔍 Getting AI monitoring insights for server {server_id}")
            ai_monitoring = get_ai_monitoring(server_id)
            if not ai_monitoring:
                print(f"❌ AI monitoring not available for server {server_id}")
                return jsonify({'error': 'AI monitoring not available for this server'}), 500
            
            print(f"📊 Getting dashboard data for server {server_id}")
            insights = ai_monitoring.get_dashboard_data()
            print(f"✅ Dashboard data retrieved: {type(insights)}")
            return jsonify(insights)
        except Exception as e:
            print(f"❌ Error getting insights for server {server_id}: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/monitoring/alerts/<int:server_id>')
    def get_monitoring_alerts(server_id):
        """Get anomaly alerts for specific server"""
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        
        # Verify server belongs to user
        server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
        
        try:
            ai_monitoring = get_ai_monitoring(server_id)
            if not ai_monitoring:
                return jsonify({'error': 'AI monitoring not available for this server'}), 500
            
            alerts = ai_monitoring.get_anomaly_alerts()
            # Only add demo alert if we're truly in demo mode without real data
            current_analysis = ai_monitoring.get_current_analysis()
            current_metrics = current_analysis.get('current_metrics', {})
            
            # Check if we have realistic-looking data (indicating real monitoring)
            has_realistic_data = (
                current_metrics.get('pod_count', 0) > 0 and
                current_metrics.get('cpu_usage', 0) > 0 and
                current_metrics.get('memory_usage', 0) > 0
            )
            
            # Only show demo alert if we're in demo mode AND don't have realistic data
            if (not alerts) and current_analysis.get('demo_mode') and not has_realistic_data:
                from datetime import datetime
                alerts = [{
                    'type': 'info',
                    'severity': 'low',
                    'message': 'Demo mode active: no anomalies detected',
                    'timestamp': datetime.now().isoformat()
                }]
            return jsonify({'alerts': alerts, 'count': len(alerts)})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/monitoring/llm_recommendations/<int:server_id>')
    def get_monitoring_llm_recommendations(server_id):
        """LLM or fallback recommendations for specific server."""
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401

        Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
        try:
            ai_monitoring = get_ai_monitoring(server_id)
            if not ai_monitoring:
                return jsonify({'error': 'AI monitoring not available for this server'}), 500
            recommendations = ai_monitoring.get_llm_recommendations()
            return jsonify({
                'llm_recommendations': recommendations,
                'count': len(recommendations)
            })
        except Exception as exc:
            app.logger.exception("Failed to fetch LLM recommendations for server %s", server_id)
            return jsonify({'llm_recommendations': [], 'count': 0, 'error': str(exc)}), 200

    @app.route('/api/monitoring/recommendations/<int:server_id>')
    def get_monitoring_recommendations(server_id):
        """Get performance recommendations for specific server"""
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        
        # Verify server belongs to user
        server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
        
        try:
            ai_monitoring = get_ai_monitoring(server_id)
            if not ai_monitoring:
                return jsonify({'error': 'AI monitoring not available for this server'}), 500
            
            recommendations = ai_monitoring.get_performance_recommendations()
            return jsonify({'recommendations': recommendations, 'count': len(recommendations)})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/monitoring/forecast/<int:server_id>')
    def get_monitoring_forecast(server_id):
        """Get capacity forecasts for specific server"""
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        
        # Verify server belongs to user
        server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
        
        try:
            ai_monitoring = get_ai_monitoring(server_id)
            if not ai_monitoring:
                return jsonify({'error': 'AI monitoring not available for this server'}), 500
            
            forecast = ai_monitoring.get_forecast_summary()
            return jsonify(forecast)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/monitoring/health/<int:server_id>')
    def get_monitoring_health(server_id):
        """Get cluster health score for specific server"""
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        
        # Verify server belongs to user
        server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
        
        try:
            ai_monitoring = get_ai_monitoring(server_id)
            if not ai_monitoring:
                return jsonify({'error': 'AI monitoring not available for this server'}), 500
            
            health = ai_monitoring.get_health_score()
            return jsonify(health)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/monitoring/start/<int:server_id>', methods=['POST'])
    def start_monitoring(server_id):
        """Start continuous monitoring for specific server"""
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        
        # Verify server belongs to user
        server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
        
        try:
            ai_monitoring = get_ai_monitoring(server_id)
            if not ai_monitoring:
                return jsonify({'error': 'AI monitoring not available for this server'}), 500
            
            interval = request.json.get('interval', 300) if request.is_json else 300
            ai_monitoring.start_monitoring(interval)
            return jsonify({'success': True, 'message': f'Monitoring started for {server.name}'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/monitoring/stop/<int:server_id>', methods=['POST'])
    def stop_monitoring(server_id):
        """Stop continuous monitoring for specific server"""
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        
        # Verify server belongs to user
        server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
        
        try:
            ai_monitoring = get_ai_monitoring(server_id)
            if not ai_monitoring:
                return jsonify({'error': 'AI monitoring not available for this server'}), 500
            
            ai_monitoring.stop_monitoring()
            return jsonify({'success': True, 'message': f'Monitoring stopped for {server.name}'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # ==================== Autoscaling Routes ====================
    
    def get_autoscaling_instance(server_id):
        """Get or create autoscaling instance for a specific server"""
        if server_id not in getattr(app, 'autoscaling_instances', {}):
            # Get server details to determine kubeconfig path
            server = Server.query.get(server_id)
            if server:
                kubeconfig_path = None
                import os
                # If a kubeconfig is stored in DB, always prefer it
                if getattr(server, 'kubeconfig', None):
                    try:
                        import tempfile
                        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
                        temp_file.write(server.kubeconfig)
                        temp_file.close()
                        kubeconfig_path = temp_file.name
                    except Exception:
                        kubeconfig_path = None
                # Otherwise, try well-known paths
                if kubeconfig_path is None:
                    candidate_paths = [
                        '/app/instance/kubeconfig_admin',
                        os.path.expanduser('~/.kube/config'),
                        '/etc/kubernetes/admin.conf',
                        '/var/lib/kubelet/kubeconfig'
                    ]
                    for path in candidate_paths:
                        if os.path.exists(path):
                            kubeconfig_path = path
                            break
                # Patch kubeconfig if pointing to 127.0.0.1
                kubeconfig_path = _patch_kubeconfig_for_container(kubeconfig_path)
                
                print(f"🔧 Creating autoscaling instance for server {server_id} ({server.name})")
                print(f"🔧 Using kubeconfig: {kubeconfig_path}")
                if not hasattr(app, 'autoscaling_instances'):
                    app.autoscaling_instances = {}
                from autoscaling_integration import AutoscalingIntegration
                app.autoscaling_instances[server_id] = AutoscalingIntegration(kubeconfig_path)
            else:
                print(f"⚠️  Server {server_id} not found")
        return getattr(app, 'autoscaling_instances', {}).get(server_id)
    
    @app.route('/autoscaling/<int:server_id>')
    def autoscaling_page(server_id):
        """Autoscaling dashboard page"""
        if 'user_id' not in session:
            return redirect(url_for('login'))
        
        server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
        return render_template('autoscaling.html', server=server)
    
    @app.route('/api/autoscaling/status/<int:server_id>')
    def get_autoscaling_status(server_id):
        """Get autoscaling status"""
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        
        server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
        
        try:
            autoscaling = get_autoscaling_instance(server_id)
            if not autoscaling:
                return jsonify({'error': 'Failed to initialize autoscaling integration'}), 500
            status = autoscaling.get_autoscaling_status()
            return jsonify(status)
        except Exception as e:
            import traceback
            print(f"❌ Error in get_autoscaling_status: {e}")
            print(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/autoscaling/hpa/create/<int:server_id>', methods=['POST'])
    def create_hpa(server_id):
        """Create HPA for deployment"""
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        
        server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
        data = request.get_json()
        
        try:
            autoscaling = get_autoscaling_instance(server_id)
            if not autoscaling:
                return jsonify({'error': 'Failed to initialize autoscaling integration'}), 500
            
            result = autoscaling.create_hpa(
                deployment_name=data.get('deployment_name'),
                namespace=data.get('namespace', 'default'),
                min_replicas=data.get('min_replicas', 2),
                max_replicas=data.get('max_replicas', 10),
                cpu_target=data.get('cpu_target', 70),
                memory_target=data.get('memory_target', 80)
            )
            return jsonify(result)
        except Exception as e:
            import traceback
            print(f"❌ Error in create_hpa: {e}")
            print(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/autoscaling/predictive/enable/<int:server_id>', methods=['POST'])
    def enable_predictive_autoscaling(server_id):
        """Enable predictive autoscaling"""
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        
        server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
        data = request.get_json()
        
        try:
            autoscaling = get_autoscaling_instance(server_id)
            if not autoscaling:
                return jsonify({'error': 'Failed to initialize autoscaling integration'}), 500
            
            # Trim deployment name to remove any leading/trailing whitespace
            deployment_name = data.get('deployment_name', '').strip()
            if not deployment_name:
                return jsonify({'error': 'deployment_name is required'}), 400
            
            result = autoscaling.enable_predictive_autoscaling(
                deployment_name=deployment_name,
                namespace=data.get('namespace', 'default').strip(),
                min_replicas=data.get('min_replicas', 2),
                max_replicas=data.get('max_replicas', 10),
                state_management=data.get('state_management')  # Optional: 'stateless', 'stateful', or None
            )
            return jsonify(result)
        except Exception as e:
            import traceback
            print(f"❌ Error in enable_predictive_autoscaling: {e}")
            print(traceback.format_exc())
            return jsonify({'error': str(e)}), 500

    @app.route('/api/autoscaling/predictive/apply/<int:server_id>', methods=['POST'])
    def apply_predictive_target(server_id):
        """Force-apply a specific predictive target (HPA or VPA)."""
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401

        server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
        data = request.get_json()

        try:
            autoscaling = get_autoscaling_instance(server_id)
            if not autoscaling:
                return jsonify({'error': 'Failed to initialize autoscaling integration'}), 500

            scaling_type = data.get('scaling_type', 'hpa')  # Default to HPA for backward compatibility
            
            result = autoscaling.apply_predictive_target(
                deployment_name=data.get('deployment_name'),
                namespace=data.get('namespace', 'default'),
                target_replicas=int(data.get('target_replicas', 0)) if scaling_type in ['hpa', 'both'] else None,
                target_cpu=data.get('target_cpu') if scaling_type in ['vpa', 'both'] else None,
                target_memory=data.get('target_memory') if scaling_type in ['vpa', 'both'] else None,
                scaling_type=scaling_type
            )
            return jsonify(result)
        except Exception as e:
            import traceback
            print(f"❌ Error in apply_predictive_target: {e}")
            print(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/autoscaling/vpa/create/<int:server_id>', methods=['POST'])
    def create_vpa(server_id):
        """Create VPA for deployment"""
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401

        server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
        data = request.get_json()

        try:
            autoscaling = get_autoscaling_instance(server_id)
            if not autoscaling:
                return jsonify({'error': 'Failed to initialize autoscaling integration'}), 500

            result = autoscaling.create_vpa(
                deployment_name=data.get('deployment_name'),
                namespace=data.get('namespace', 'default'),
                min_cpu=data.get('min_cpu', '100m'),
                max_cpu=data.get('max_cpu', '1000m'),
                min_memory=data.get('min_memory', '128Mi'),
                max_memory=data.get('max_memory', '512Mi'),
                update_mode=data.get('update_mode', 'Auto')
            )
            return jsonify(result)
        except Exception as e:
            import traceback
            print(f"❌ Error in create_vpa: {e}")
            print(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/autoscaling/vpa/delete/<int:server_id>', methods=['POST'])
    def delete_vpa(server_id):
        """Delete VPA"""
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401

        server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
        data = request.get_json()

        try:
            autoscaling = get_autoscaling_instance(server_id)
            if not autoscaling:
                return jsonify({'error': 'Failed to initialize autoscaling integration'}), 500

            result = autoscaling.delete_vpa(
                vpa_name=data.get('vpa_name'),
                namespace=data.get('namespace', 'default')
            )
            return jsonify(result)
        except Exception as e:
            import traceback
            print(f"❌ Error in delete_vpa: {e}")
            print(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/autoscaling/recommendations/<int:server_id>')
    def get_scaling_recommendations(server_id):
        """Get scaling recommendations"""
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        
        server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
        deployment_name = request.args.get('deployment', '').strip()
        namespace = request.args.get('namespace', 'default').strip()
        
        if not deployment_name:
            return jsonify({'error': 'deployment parameter required'}), 400
        
        try:
            autoscaling = get_autoscaling_instance(server_id)
            if not autoscaling:
                return jsonify({'error': 'Failed to initialize autoscaling integration'}), 500
            
            print(f"🔍🔍🔍 WEB APP: Getting recommendations for {deployment_name} in {namespace}")
            result = autoscaling.get_scaling_recommendations(deployment_name, namespace)
            
            # Log the result to see what's being returned
            if result.get('predictive') and result['predictive'].get('recommendation'):
                rec = result['predictive']['recommendation']
                target_replicas = rec.get('target_replicas')
                print(f"🔍🔍🔍 WEB APP RESULT: target_replicas={target_replicas}, action={rec.get('action')}, scaling_type={rec.get('scaling_type')}")
                
                # CRITICAL: Final validation at web app layer - get min/max from deployment
                if target_replicas is not None:
                    try:
                        # Get deployment to read min/max from annotation
                        import subprocess
                        kubectl_cmd = ['kubectl', 'get', 'deployment', deployment_name, '-n', namespace, '-o', 'json']
                        if server.kubeconfig_path:
                            kubectl_cmd.extend(['--kubeconfig', server.kubeconfig_path])
                        
                        kubectl_result = subprocess.run(kubectl_cmd, capture_output=True, text=True, timeout=10)
                        if kubectl_result.returncode == 0:
                            deployment_json = json.loads(kubectl_result.stdout)
                            annotations = deployment_json.get('metadata', {}).get('annotations', {})
                            config_str = annotations.get('ai4k8s.io/predictive-autoscaling-config', '{}')
                            config = json.loads(config_str) if config_str else {}
                            min_replicas = config.get('min_replicas', 2)
                            max_replicas = config.get('max_replicas', 10)
                            
                            print(f"🔍🔍🔍 WEB APP VALIDATION: target_replicas={target_replicas}, min={min_replicas}, max={max_replicas}")
                            
                            if target_replicas > max_replicas:
                                print(f"🚨🚨🚨 WEB APP LAYER: target_replicas={target_replicas} > max={max_replicas}, FORCING to {max_replicas}")
                                rec['target_replicas'] = max_replicas
                                result['predictive']['recommendation'] = rec
                            elif target_replicas < min_replicas:
                                print(f"🚨🚨🚨 WEB APP LAYER: target_replicas={target_replicas} < min={min_replicas}, FORCING to {min_replicas}")
                                rec['target_replicas'] = min_replicas
                                result['predictive']['recommendation'] = rec
                            
                            print(f"🔍🔍🔍 WEB APP FINAL: target_replicas={rec.get('target_replicas')}")
                        else:
                            print(f"⚠️ Could not get deployment for validation: {kubectl_result.stderr}")
                    except Exception as e:
                        print(f"⚠️ Error in web app validation: {e}")
                        import traceback
                        traceback.print_exc()
                
                print(f"🔍🔍🔍 WEB APP FULL REC: {json.dumps(rec, indent=2)}")
            
            return jsonify(result)
        except Exception as e:
            import traceback
            print(f"❌ Error in get_scaling_recommendations: {e}")
            print(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/autoscaling/schedule/create/<int:server_id>', methods=['POST'])
    def create_schedule(server_id):
        """Create scheduled autoscaling"""
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        
        server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
        data = request.get_json()
        
        try:
            autoscaling = get_autoscaling_instance(server_id)
            if not autoscaling:
                return jsonify({'error': 'Failed to initialize autoscaling integration'}), 500
            
            result = autoscaling.create_schedule(
                deployment_name=data.get('deployment_name'),
                namespace=data.get('namespace', 'default'),
                schedule_rules=data.get('schedule_rules', [])
            )
            return jsonify(result)
        except Exception as e:
            import traceback
            print(f"❌ Error in create_schedule: {e}")
            print(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/autoscaling/hpa/delete/<int:server_id>', methods=['POST'])
    def delete_hpa(server_id):
        """Delete HPA"""
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        
        server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
        data = request.get_json()
        
        try:
            autoscaling = get_autoscaling_instance(server_id)
            if not autoscaling:
                return jsonify({'error': 'Failed to initialize autoscaling integration'}), 500
            
            result = autoscaling.delete_hpa(
                hpa_name=data.get('hpa_name'),
                namespace=data.get('namespace', 'default')
            )
            return jsonify(result)
        except Exception as e:
            import traceback
            print(f"❌ Error in delete_hpa: {e}")
            print(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/autoscaling/predictive/disable/<int:server_id>', methods=['POST'])
    def disable_predictive_autoscaling(server_id):
        """Disable predictive autoscaling"""
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        
        server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
        data = request.get_json()
        
        try:
            autoscaling = get_autoscaling_instance(server_id)
            if not autoscaling:
                return jsonify({'error': 'Failed to initialize autoscaling integration'}), 500
            
            result = autoscaling.disable_predictive_autoscaling(
                deployment_name=data.get('deployment_name'),
                namespace=data.get('namespace', 'default')
            )
            return jsonify(result)
        except Exception as e:
            import traceback
            print(f"❌ Error in disable_predictive_autoscaling: {e}")
            print(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/autoscaling/schedule/delete/<int:server_id>', methods=['POST'])
    def delete_schedule(server_id):
        """Delete scheduled autoscaling"""
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        
        server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
        data = request.get_json()
        
        try:
            autoscaling = get_autoscaling_instance(server_id)
            if not autoscaling:
                return jsonify({'error': 'Failed to initialize autoscaling integration'}), 500
            
            result = autoscaling.delete_schedule(
                deployment_name=data.get('deployment_name'),
                namespace=data.get('namespace', 'default')
            )
            return jsonify(result)
        except Exception as e:
            import traceback
            print(f"❌ Error in delete_schedule: {e}")
            print(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/autoscaling/patterns/<int:server_id>')
    def analyze_patterns(server_id):
        """Analyze historical patterns for schedule suggestions"""
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        
        server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
        deployment_name = request.args.get('deployment')
        namespace = request.args.get('namespace', 'default')
        
        if not deployment_name:
            return jsonify({'error': 'deployment parameter required'}), 400
        
        try:
            autoscaling = get_autoscaling_instance(server_id)
            if not autoscaling:
                return jsonify({'error': 'Failed to initialize autoscaling integration'}), 500
            
            result = autoscaling.analyze_patterns(deployment_name, namespace)
            return jsonify(result)
        except Exception as e:
            import traceback
            print(f"❌ Error in analyze_patterns: {e}")
            print(traceback.format_exc())
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Ensure a default admin user exists for first-run access
        try:
            default_admin_username = os.environ.get('DEFAULT_ADMIN_USERNAME', 'admin')
            default_admin_password = os.environ.get('DEFAULT_ADMIN_PASSWORD', 'admin123')
            default_admin_email = os.environ.get('DEFAULT_ADMIN_EMAIL', 'admin@local')

            admin_user = User.query.filter_by(username=default_admin_username).first()
            if not admin_user:
                admin_user = User(username=default_admin_username, email=default_admin_email)
                admin_user.set_password(default_admin_password)
                db.session.add(admin_user)
                db.session.commit()
                print('✅ Default admin user created')
            else:
                print('✅ Default admin user present')

            # Ensure the admin has at least one server
            admin_server = Server.query.filter_by(user_id=admin_user.id).first()
            if not admin_server:
                admin_server = Server(
                    name='Production K8s Cluster',
                    server_type='kubernetes',
                    connection_string='https://127.0.0.1:42019',
                    user_id=admin_user.id,
                    status='active'
                )
                db.session.add(admin_server)
                db.session.commit()
                print('✅ Default admin server created')
            else:
                print('✅ Default admin server present')
        except Exception as e:
            print(f'⚠️  Failed to ensure default admin: {e}')
    
    # Get configuration from environment variables
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5003))
    
    print("🚀 Starting AI-Powered AI4K8s Web Application...")
    print("✅ AI-Powered Natural Language Processing: Ready")
    print("✅ MCP Bridge Integration: Ready")
    print("✅ Database: Ready")
    print(f"🌐 Server: http://{host}:{port}")
    print(f"🔧 Debug Mode: {debug_mode}")
    
    app.run(debug=debug_mode, host=host, port=port)
