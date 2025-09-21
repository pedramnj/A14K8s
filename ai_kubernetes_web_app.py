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

db = SQLAlchemy(app)

# Context processor to make current year available in all templates
@app.context_processor
def inject_current_year():
    return {'current_year': datetime.now().year}

# AI-Powered MCP-based Natural Language Processor
class AIPoweredMCPKubernetesProcessor:
    def __init__(self):
        self.mcp_server_url = "http://localhost:5002/mcp"
        self.available_tools = None
        self.use_ai = False
        self.anthropic = None
        self._load_tools()
        self._initialize_ai()
    
    def _initialize_ai(self):
        """Initialize Anthropic AI if API key is available"""
        try:
            from anthropic import Anthropic
            import os
            
            # Try to load .env file from client directory
            self._load_env_file()
            
            if os.getenv('ANTHROPIC_API_KEY'):
                # Set the API key in environment and initialize client
                os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')
                self.anthropic = Anthropic()
                self.use_ai = True
                print("🤖 AI-powered processing enabled")
            else:
                print("🔧 Regex-only processing (no ANTHROPIC_API_KEY)")
        except ImportError:
            print("⚠️  Anthropic package not installed, using regex-only processing")
        except Exception as e:
            print(f"⚠️  AI initialization failed: {e}, using regex-only processing")
    
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
            response = requests.post(
                self.mcp_server_url,
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                timeout=5
            )
            if response.status_code == 200:
                result = response.json()
                self.available_tools = {tool['name']: tool for tool in result['result']['tools']}
                print(f"✅ Loaded {len(self.available_tools)} MCP tools")
            else:
                print(f"⚠️  Failed to load MCP tools: {response.status_code}")
                self.available_tools = {}
        except Exception as e:
            print(f"⚠️  Error loading MCP tools: {e}")
            self.available_tools = {}
    
    def _call_mcp_tool(self, tool_name: str, arguments: dict = None) -> dict:
        """Call an MCP tool with the given arguments"""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments or {}
                }
            }
            
            response = requests.post(
                self.mcp_server_url,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'result' in result:
                    return {
                        'success': True,
                        'result': result['result'],
                        'tool': tool_name,
                        'arguments': arguments
                    }
                else:
                    return {
                        'success': False,
                        'error': result.get('error', 'Unknown error'),
                        'tool': tool_name,
                        'arguments': arguments
                    }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}",
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
            
            # Call Anthropic AI with tools
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
    servers = Server.query.filter_by(user_id=user.id).all()
    
    return render_template('dashboard.html', user=user, servers=servers)

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
    return render_template('server_detail.html', server=server)

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
        # Direct kubectl command - use MCP bridge
        try:
            response = requests.post(
                'http://localhost:5001/api/chat',
                json={'message': message},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Store kubectl command in database
                chat = Chat(
                    user_id=session['user_id'],
                    server_id=server_id,
                    user_message=message,
                    ai_response=result.get('response', ''),
                    mcp_tool_used='kubectl_direct',
                    processing_method='Direct',
                    mcp_success=result.get('status') == 'success'
                )
                db.session.add(chat)
                db.session.commit()
                
                result['chat_id'] = chat.id
                return jsonify(result)
            else:
                return jsonify({'error': 'MCP bridge error'}), 500
                
        except Exception as e:
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
                    
                    response_text = f"{processed['explanation']}\n\n**MCP Tool Result:**\n{result_text}"
                    
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
    server.status = "active"
    db.session.commit()
    
    return jsonify({
        'status': server.status,
        'last_accessed': server.last_accessed.isoformat() if server.last_accessed else None
    })

@app.route('/api/test_connection/<int:server_id>', methods=['POST'])
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

# Predictive Monitoring Routes
if PREDICTIVE_MONITORING_AVAILABLE:
    # Store AI monitoring instances per server
    ai_monitoring_instances = {}
    
    def get_ai_monitoring(server_id):
        """Get or create AI monitoring instance for a specific server"""
        if server_id not in ai_monitoring_instances:
            # Get server details to determine kubeconfig path
            server = Server.query.get(server_id)
            if server:
                kubeconfig_path = None
                # For local servers, try to use default kubeconfig
                if server.server_type == 'local':
                    # Try to find kubeconfig in common locations
                    import os
                    kubeconfig_paths = [
                        os.path.expanduser('~/.kube/config'),
                        '/etc/kubernetes/admin.conf',
                        '/var/lib/kubelet/kubeconfig'
                    ]
                    for path in kubeconfig_paths:
                        if os.path.exists(path):
                            kubeconfig_path = path
                            break
                # For remote servers, use the stored kubeconfig
                elif server.server_type == 'remote' and server.kubeconfig:
                    # Save kubeconfig to a temporary file
                    import tempfile
                    import os
                    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
                    temp_file.write(server.kubeconfig)
                    temp_file.close()
                    kubeconfig_path = temp_file.name
                
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
            return jsonify({'alerts': alerts, 'count': len(alerts)})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
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

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
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
