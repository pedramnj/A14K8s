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

# Intelligent MCP-based Natural Language Processor
class MCPKubernetesProcessor:
    def __init__(self):
        self.mcp_server_url = "http://localhost:5002/mcp"
        self.available_tools = None
        self._load_tools()
    
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
    
    def process_query(self, query: str) -> dict:
        """Process natural language query using intelligent MCP tools"""
        query_lower = query.lower().strip()
        
        # Map natural language to MCP tools
        if 'pods' in query_lower and ('show' in query_lower or 'list' in query_lower or 'get' in query_lower or 'display' in query_lower or 'what' in query_lower or 'running' in query_lower):
            # Use MCP pods_list tool
            result = self._call_mcp_tool('pods_list', {})
            return {
                'command': 'MCP: pods_list',
                'explanation': f"I'll show you all pods using MCP tools",
                'mcp_result': result
            }
        
        elif 'services' in query_lower and ('show' in query_lower or 'list' in query_lower or 'get' in query_lower or 'display' in query_lower or 'what' in query_lower):
            # Use MCP resources_list tool for services
            result = self._call_mcp_tool('resources_list', {
                'apiVersion': 'v1',
                'kind': 'Service'
            })
            return {
                'command': 'MCP: resources_list (services)',
                'explanation': f"I'll show you all services using MCP tools",
                'mcp_result': result
            }
        
        elif 'deployments' in query_lower and ('show' in query_lower or 'list' in query_lower or 'get' in query_lower or 'display' in query_lower or 'what' in query_lower):
            # Use MCP resources_list tool for deployments
            result = self._call_mcp_tool('resources_list', {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment'
            })
            return {
                'command': 'MCP: resources_list (deployments)',
                'explanation': f"I'll show you all deployments using MCP tools",
                'mcp_result': result
            }
        
        elif 'nodes' in query_lower and ('show' in query_lower or 'list' in query_lower or 'get' in query_lower or 'display' in query_lower or 'what' in query_lower):
            # Use MCP resources_list tool for nodes
            result = self._call_mcp_tool('resources_list', {
                'apiVersion': 'v1',
                'kind': 'Node'
            })
            return {
                'command': 'MCP: resources_list (nodes)',
                'explanation': f"I'll show you all nodes using MCP tools",
                'mcp_result': result
            }
        
        elif 'create' in query_lower and 'pod' in query_lower:
            # Use MCP pods_run tool
            pod_name = self._extract_pod_name(query)
            result = self._call_mcp_tool('pods_run', {
                'name': pod_name,
                'image': 'nginx'
            })
            return {
                'command': f'MCP: pods_run ({pod_name})',
                'explanation': f"I'll create a pod named '{pod_name}' using MCP tools",
                'mcp_result': result
            }
        
        elif ('delete' in query_lower or 'stop' in query_lower or 'remove' in query_lower) and 'pod' in query_lower:
            # Use MCP pods_delete tool
            pod_name = self._extract_pod_name_from_delete(query)
            if pod_name:
                result = self._call_mcp_tool('pods_delete', {
                    'name': pod_name,
                    'namespace': 'default'
                })
                return {
                    'command': f'MCP: pods_delete ({pod_name})',
                    'explanation': f"I'll delete the pod named '{pod_name}' using MCP tools",
                    'mcp_result': result
                }
            else:
                return {
                    'command': None,
                    'explanation': f"I understand you want to delete a pod, but I couldn't extract the pod name from: '{query}'. Please try 'delete pod <name>' or 'stop pod <name>'.",
                    'mcp_result': None
                }
        
        elif any(word in query_lower for word in ['health', 'wrong', 'error', 'problem', 'issue', 'diagnose', 'check']) and ('cluster' in query_lower or 'how' in query_lower):
            # Use MCP events_list tool for cluster health
            result = self._call_mcp_tool('events_list', {})
            return {
                'command': 'MCP: events_list',
                'explanation': f"I'll check the cluster health by looking at events using MCP tools",
                'mcp_result': result
            }
        
        elif 'events' in query_lower:
            # Use MCP events_list tool
            result = self._call_mcp_tool('events_list', {})
            return {
                'command': 'MCP: events_list',
                'explanation': f"I'll check the events using MCP tools",
                'mcp_result': result
            }
        
        # Default response
        return {
            'command': None,
            'explanation': f"I understand you want to: '{query}'. I can help with pods, services, deployments, nodes, events, and cluster health using intelligent MCP tools.",
            'mcp_result': None
        }
    
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

# Initialize intelligent MCP processor
processor = MCPKubernetesProcessor()

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
                        'ai_processed': True,
                        'mcp_tool': processed['command'],
                        'mcp_success': True
                    })
                else:
                    return jsonify({
                        'response': f"{processed['explanation']}\n\n**MCP Error:** {mcp_result.get('error', 'Unknown error')}",
                        'status': 'error',
                        'ai_processed': True,
                        'mcp_tool': processed['command'],
                        'mcp_success': False
                    })
            else:
                return jsonify({
                    'response': processed['explanation'],
                    'status': 'info',
                    'ai_processed': True,
                    'mcp_tool': None
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
    print("🚀 Starting Simplified AI4K8s Web Application...")
    print("✅ Natural Language Processing: Ready")
    print("✅ MCP Bridge Integration: Ready")
    print("✅ Database: Ready")
    app.run(debug=True, host='0.0.0.0', port=5003)
