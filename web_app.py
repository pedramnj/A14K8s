#!/usr/bin/env python3
"""
AI4K8s Web Application with User Authentication and Server Management
"""

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
import json
import requests
import uuid
from anthropic import Anthropic
import asyncio
import subprocess
import tempfile

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ai4k8s.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# AI Integration Class
class AIKubernetesAssistant:
    def __init__(self):
        self.anthropic = None
        self.available_commands = [
            "kubectl get pods", "kubectl get pod <name>", "kubectl get events", 
            "kubectl get nodes", "kubectl get services", "kubectl get deployments",
            "kubectl logs <pod_name>", "kubectl delete pod <name>", "kubectl top pods",
            "kubectl top pod <name>", "kubectl exec <pod_name> -- <command>",
            "kubectl run <name> --image=<image>", "kubectl create deployment <name> --image=<image>",
            "kubectl scale deployment <name> --replicas=<number>", "kubectl describe pod <name>"
        ]
        
        # Try to initialize Anthropic client
        try:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if api_key:
                self.anthropic = Anthropic(api_key=api_key)
                print("✅ AI Assistant initialized successfully")
            else:
                print("⚠️  ANTHROPIC_API_KEY not found, AI features disabled")
        except Exception as e:
            print(f"⚠️  Failed to initialize AI Assistant: {e}")
            self.anthropic = None
    
    def process_natural_language_query(self, query: str, server_info: dict) -> dict:
        """Process natural language query and return appropriate response"""
        try:
            # Check if AI is available
            if not self.anthropic:
                return self._fallback_response(query)
            
            # Create a prompt for the AI
            system_prompt = f"""You are a Kubernetes AI assistant. You can help users manage their Kubernetes cluster through natural language.

Available kubectl commands you can suggest or execute:
{', '.join(self.available_commands)}

Server Information:
- Name: {server_info.get('name', 'Unknown')}
- Type: {server_info.get('server_type', 'Unknown')}
- Connection: {server_info.get('connection_string', 'Unknown')}

When a user asks you to do something, you should:
1. Understand their intent
2. Suggest the appropriate kubectl command
3. If it's a simple query, provide the command directly
4. If it's complex, explain what you would do

Examples:
- "create a new pod name it funky dance" → "kubectl run funky-dance --image=nginx"
- "show me all pods" → "kubectl get pods"
- "what's wrong with my cluster" → "kubectl get events"
- "get logs from nginx pod" → "kubectl logs nginx"

Always be helpful and provide clear, actionable responses."""

            response = self.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": query}
                ]
            )
            
            ai_response = response.content[0].text
            
            # Check if the AI suggested a kubectl command
            if "kubectl" in ai_response.lower():
                # Extract the kubectl command from the response
                lines = ai_response.split('\n')
                kubectl_command = None
                for line in lines:
                    if line.strip().startswith('kubectl'):
                        kubectl_command = line.strip()
                        break
                
                if kubectl_command:
                    # Execute the command via MCP bridge
                    try:
                        mcp_response = requests.post(
                            'http://localhost:5001/api/chat',
                            json={'message': kubectl_command},
                            timeout=30
                        )
                        
                        if mcp_response.status_code == 200:
                            mcp_result = mcp_response.json()
                            return {
                                "ai_explanation": ai_response,
                                "command_executed": kubectl_command,
                                "command_result": mcp_result.get('response', 'No response'),
                                "status": "success"
                            }
                        else:
                            return {
                                "ai_explanation": ai_response,
                                "command_executed": kubectl_command,
                                "command_result": f"Error executing command: {mcp_response.status_code}",
                                "status": "error"
                            }
                    except Exception as e:
                        return {
                            "ai_explanation": ai_response,
                            "command_executed": kubectl_command,
                            "command_result": f"Error connecting to MCP bridge: {str(e)}",
                            "status": "error"
                        }
            
            # If no kubectl command was suggested, just return the AI response
            return {
                "ai_explanation": ai_response,
                "command_executed": None,
                "command_result": None,
                "status": "info"
            }
            
        except Exception as e:
            return {
                "ai_explanation": f"I apologize, but I encountered an error: {str(e)}",
                "command_executed": None,
                "command_result": None,
                "status": "error"
            }
    
    def _fallback_response(self, query: str) -> dict:
        """Fallback response when AI is not available"""
        query_lower = query.lower()
        
        # Simple pattern matching for common requests
        if "create" in query_lower and "pod" in query_lower:
            # Extract pod name from query - look for "name it" or "called" patterns
            words = query.split()
            pod_name = "new-pod"
            
            # Look for "name it" pattern
            for i, word in enumerate(words):
                if word.lower() == "name" and i + 1 < len(words) and words[i + 1].lower() == "it":
                    # Get the next word(s) after "it"
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
                    # Get the next word(s) after "called" or "named"
                    name_words = []
                    for j in range(i + 1, len(words)):
                        if words[j].lower() in ["with", "using", "from", "image"]:
                            break
                        name_words.append(words[j])
                    if name_words:
                        pod_name = "-".join(name_words).lower()
                    break
            
            kubectl_command = f"kubectl run {pod_name} --image=nginx"
            return {
                "ai_explanation": f"I understand you want to create a pod. I'll run: `{kubectl_command}`",
                "command_executed": kubectl_command,
                "command_result": None,
                "status": "success"
            }
        elif "show" in query_lower and "pods" in query_lower:
            kubectl_command = "kubectl get pods"
            return {
                "ai_explanation": f"I'll show you all the pods in your cluster: `{kubectl_command}`",
                "command_executed": kubectl_command,
                "command_result": None,
                "status": "success"
            }
        elif "events" in query_lower or "wrong" in query_lower:
            kubectl_command = "kubectl get events"
            return {
                "ai_explanation": f"I'll check the events to see what's happening: `{kubectl_command}`",
                "command_executed": kubectl_command,
                "command_result": None,
                "status": "success"
            }
        else:
            return {
                "ai_explanation": f"I understand you want to: '{query}'. However, AI features are currently disabled. Please use direct kubectl commands like 'kubectl get pods' or 'kubectl get events'.",
                "command_executed": None,
                "command_result": None,
                "status": "info"
            }

# Initialize AI Assistant
ai_assistant = AIKubernetesAssistant()

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    servers = db.relationship('Server', backref='owner', lazy=True)

class Server(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    server_type = db.Column(db.String(50), nullable=False)  # 'local', 'remote', 'cloud'
    connection_string = db.Column(db.String(500), nullable=False)
    kubeconfig = db.Column(db.Text)  # Base64 encoded kubeconfig
    status = db.Column(db.String(20), default='inactive')  # 'active', 'inactive', 'error'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_accessed = db.Column(db.DateTime)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return render_template('register.html')
        
        # Create new user
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        
        session['user_id'] = user.id
        session['username'] = user.username
        flash('Registration successful!')
        return redirect(url_for('dashboard'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    servers = Server.query.filter_by(user_id=user.id).all()
    
    return render_template('dashboard.html', user=user, servers=servers)

@app.route('/add_server', methods=['GET', 'POST'])
def add_server():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        name = request.form['name']
        server_type = request.form['server_type']
        connection_string = request.form['connection_string']
        kubeconfig = request.form.get('kubeconfig', '')
        
        # Validate server connection (basic validation)
        if server_type == 'local':
            connection_string = 'localhost:8080'  # Default local connection
        
        server = Server(
            name=name,
            server_type=server_type,
            connection_string=connection_string,
            kubeconfig=kubeconfig,
            user_id=session['user_id']
        )
        
        db.session.add(server)
        db.session.commit()
        
        flash(f'Server "{name}" added successfully!')
        return redirect(url_for('dashboard'))
    
    return render_template('add_server.html')

@app.route('/server/<int:server_id>')
def server_detail(server_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first()
    if not server:
        flash('Server not found')
        return redirect(url_for('dashboard'))
    
    return render_template('server_detail.html', server=server)

@app.route('/chat/<int:server_id>')
def chat(server_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first()
    if not server:
        flash('Server not found')
        return redirect(url_for('dashboard'))
    
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
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return jsonify(result)
            else:
                return jsonify({'error': 'MCP bridge error'}), 500
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        # Natural language query - use AI assistant
        try:
            server_info = {
                'name': server.name,
                'server_type': server.server_type,
                'connection_string': server.connection_string
            }
            
            ai_result = ai_assistant.process_natural_language_query(message, server_info)
            
            # Format the response for the frontend
            if ai_result['status'] == 'success':
                response_text = f"{ai_result['ai_explanation']}\n\n**Command Executed:** `{ai_result['command_executed']}`\n\n**Result:**\n{ai_result['command_result']}"
            elif ai_result['status'] == 'error':
                response_text = f"{ai_result['ai_explanation']}\n\n**Error:** {ai_result.get('command_result', 'Unknown error')}"
            else:
                response_text = ai_result['ai_explanation']
            
            return jsonify({
                'response': response_text,
                'status': 'success',
                'ai_processed': True,
                'command_executed': ai_result.get('command_executed'),
                'ai_explanation': ai_result['ai_explanation']
            })
            
        except Exception as e:
            return jsonify({
                'response': f"I apologize, but I encountered an error processing your request: {str(e)}",
                'status': 'error',
                'ai_processed': True
            }), 500

@app.route('/api/server_status/<int:server_id>')
def server_status(server_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first()
    if not server:
        return jsonify({'error': 'Server not found'}), 404
    
    # Update last accessed time
    server.last_accessed = datetime.utcnow()
    db.session.commit()
    
    # For now, return basic status
    # In the future, this will check actual server connectivity
    return jsonify({
        'status': server.status,
        'last_accessed': server.last_accessed.isoformat() if server.last_accessed else None
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5002)
