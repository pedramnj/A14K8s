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

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ai4k8s.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

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
            
            # Create system prompt similar to the client
            system_prompt = (
                "You are an AI assistant operating an MCP client connected to a Kubernetes MCP server. "
                "When you need cluster data or to take action, choose the correct tool. "
                "Be concise and return clear, well-formatted answers.\n\n"
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
            
            for content in response.content:
                if content.type == 'text':
                    final_text.append(content.text)
                elif content.type == 'tool_use':
                    tool_name = content.name
                    tool_args = content.input
                    
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
            # Extract image if specified
            image_match = re.search(r'(?:with|using|from).*image.*[\'"]([^\'"]+)[\'"]', query_lower)
            image = image_match.group(1) if image_match else 'nginx'
            
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
                # If AI successfully processed the query, return it
                if ai_result.get('ai_processed') and ai_result.get('command'):
                    return ai_result
                # If AI couldn't process it, fall back to regex
            except Exception as e:
                print(f"⚠️  AI processing failed, falling back to regex: {e}")
        
        # Fall back to regex processing
        return self._process_with_regex(query)
    
    def _extract_pod_name(self, query: str) -> str:
        """Extract pod name from create pod query"""
        words = query.split()
        pod_name = "new-pod"
        
        # Look for "name it" pattern
        for i, word in enumerate(words):
            if word.lower() == "name" and i + 1 < len(words) and words[i + 1].lower() == "it":
                if i + 2 < len(words):
                    name_words = []
                    for j in range(i + 2, len(words)):
                        if words[j].lower() in ["with", "using", "from", "image"]:
                            break
                        name_words.append(words[j])
                    if name_words:
                        pod_name = "-".join(name_words).lower()
                break
            elif word.lower() in ["called", "named"] and i + 1 < len(words):
                name_words = []
                for j in range(i + 1, len(words)):
                    if words[j].lower() in ["with", "using", "from", "image"]:
                        break
                    name_words.append(words[j])
                if name_words:
                    pod_name = "-".join(name_words).lower()
                break
            elif word.lower() == "name" and i + 1 < len(words):
                # Direct "name toto" pattern
                name_words = []
                for j in range(i + 1, len(words)):
                    if words[j].lower() in ["with", "using", "from", "image", "pod"]:
                        break
                    name_words.append(words[j])
                if name_words:
                    pod_name = "-".join(name_words).lower()
                break
        
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
processor = AIPoweredMCPKubernetesProcessor()

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
    kubeconfig = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), default='inactive')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_accessed = db.Column(db.DateTime, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

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
        kubeconfig = request.form.get('kubeconfig')
        
        new_server = Server(
            name=name,
            server_type=server_type,
            connection_string=connection_string,
            kubeconfig=kubeconfig,
            user_id=session['user_id']
        )
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
                    
                    return jsonify({
                        'response': response_text,
                        'status': 'success',
                        'ai_processed': processed.get('ai_processed', False),
                        'mcp_tool': processed['command'],
                        'mcp_success': True,
                        'processing_method': 'AI' if processed.get('ai_processed') else 'Regex'
                    })
                else:
                    return jsonify({
                        'response': f"{processed['explanation']}\n\n**MCP Error:** {mcp_result.get('error', 'Unknown error')}",
                        'status': 'error',
                        'ai_processed': processed.get('ai_processed', False),
                        'mcp_tool': processed['command'],
                        'mcp_success': False,
                        'processing_method': 'AI' if processed.get('ai_processed') else 'Regex'
                    })
            else:
                return jsonify({
                    'response': processed['explanation'],
                    'status': 'info',
                    'ai_processed': processed.get('ai_processed', False),
                    'mcp_tool': None,
                    'processing_method': 'AI' if processed.get('ai_processed') else 'Regex'
                })
                
        except Exception as e:
            return jsonify({
                'response': f"I apologize, but I encountered an error: {str(e)}",
                'status': 'error',
                'ai_processed': True
            }), 500

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

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    print("🚀 Starting AI-Powered AI4K8s Web Application...")
    print("✅ AI-Powered Natural Language Processing: Ready")
    print("✅ MCP Bridge Integration: Ready")
    print("✅ Database: Ready")
    app.run(debug=True, host='0.0.0.0', port=5003)
