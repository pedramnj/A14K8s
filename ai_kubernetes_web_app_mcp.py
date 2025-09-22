#!/usr/bin/env python3
"""
AI4K8s Web Application with Proper MCP Integration
Uses stdio communication instead of HTTP
"""

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
import json
import asyncio
import threading
from typing import Dict, Any, Optional

# Import our proper MCP client
from mcp_client import get_mcp_client, call_mcp_tool

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ai4k8s.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Context processor to make current year available in all templates
@app.context_processor
def inject_current_year():
    return {'current_year': datetime.now().year}

# AI-Powered MCP-based Natural Language Processor with Proper Stdio Communication
class AIPoweredMCPKubernetesProcessor:
    def __init__(self):
        self.available_tools = {}
        self.use_ai = False
        self.anthropic = None
        self.mcp_client = None
        self._initialize_ai()
        self._load_tools_async()
    
    def _initialize_ai(self):
        """Initialize Anthropic AI if API key is available"""
        try:
            from anthropic import Anthropic
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if api_key and api_key != 'your-api-key-here':
                self.anthropic = Anthropic(api_key=api_key)
                self.use_ai = True
                print('🤖 Enhanced AI-powered processing enabled')
            else:
                print('⚠️  No Anthropic API key found')
        except ImportError:
            print('⚠️  Anthropic library not available')
        except Exception as e:
            print(f'⚠️  Error initializing AI: {e}')
    
    def _load_tools_async(self):
        """Load MCP tools using proper stdio communication"""
        try:
            # Run async function in thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.mcp_client = loop.run_until_complete(get_mcp_client())
            if self.mcp_client and self.mcp_client.available_tools:
                self.available_tools = self.mcp_client.available_tools
                print(f'✅ Loaded {len(self.available_tools)} MCP tools via stdio')
            else:
                print('⚠️  No MCP tools available')
        except Exception as e:
            print(f'⚠️  Error loading MCP tools: {e}')
    
    async def _call_tool_async(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP tool asynchronously"""
        try:
            if not self.mcp_client:
                return {'error': 'MCP client not initialized'}
            
            result = await self.mcp_client.call_tool(tool_name, arguments)
            return result
        except Exception as e:
            return {'error': str(e)}
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP tool synchronously"""
        try:
            if not self.mcp_client:
                return {'error': 'MCP client not initialized'}
            
            # Run async function in thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._call_tool_async(tool_name, arguments))
            return result
        except Exception as e:
            return {'error': str(e)}
    
    def process_query(self, query: str, server_id: int = 1) -> str:
        """Process natural language query using AI and MCP tools"""
        try:
            if not self.use_ai or not self.anthropic:
                return "AI processing not available. Please check your API key."
            
            # Get available tools
            tools = []
            if self.available_tools:
                tools = [{
                    'name': tool.name,
                    'description': tool.description,
                    'input_schema': tool.inputSchema
                } for tool in self.available_tools.values()]
            
            # Create messages for Claude
            messages = [{
                'role': 'user',
                'content': f"Kubernetes query: {query}"
            }]
            
            # Call Claude with tools
            response = self.anthropic.messages.create(
                model='claude-3-5-sonnet-20241022',
                max_tokens=1000,
                messages=messages,
                tools=tools
            )
            
            # Process response and handle tool calls
            final_response = []
            
            for content in response.content:
                if content.type == 'text':
                    final_response.append(content.text)
                elif content.type == 'tool_use':
                    tool_name = content.name
                    tool_args = content.input
                    
                    # Call the MCP tool
                    tool_result = self.call_tool(tool_name, tool_args)
                    
                    if tool_result.get('success'):
                        final_response.append(f"[Tool {tool_name} executed successfully]")
                        final_response.append(str(tool_result.get('result', '')))
                    else:
                        final_response.append(f"[Tool {tool_name} failed: {tool_result.get('error', 'Unknown error')}]")
            
            return '\n'.join(final_response)
            
        except Exception as e:
            return f"Error processing query: {str(e)}"

# Initialize the processor
processor = AIPoweredMCPKubernetesProcessor()

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    servers = db.relationship('Server', backref='user', lazy=True)

class Server(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    server_type = db.Column(db.String(50), nullable=False)
    connection_string = db.Column(db.String(200), nullable=False)
    kubeconfig = db.Column(db.Text)
    status = db.Column(db.String(20), default='inactive')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_accessed = db.Column(db.DateTime)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
            return render_template('register.html')
        
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    servers = Server.query.filter_by(user_id=user.id).all()
    return render_template('dashboard.html', servers=servers)

@app.route('/add_server', methods=['GET', 'POST'])
def add_server():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        name = request.form['name']
        server_type = request.form['server_type']
        connection_string = request.form['connection_string']
        kubeconfig = request.form.get('kubeconfig', '')
        
        server = Server(
            name=name,
            server_type=server_type,
            connection_string=connection_string,
            kubeconfig=kubeconfig,
            user_id=session['user_id']
        )
        db.session.add(server)
        db.session.commit()
        
        flash('Server added successfully!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('add_server.html')

@app.route('/server/<int:server_id>')
def server_detail(server_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first()
    if not server:
        flash('Server not found', 'error')
        return redirect(url_for('dashboard'))
    
    return render_template('server_detail.html', server=server)

@app.route('/chat/<int:server_id>')
def chat(server_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first()
    if not server:
        flash('Server not found', 'error')
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
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        response = processor.process_query(message, server_id)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/monitoring/<int:server_id>')
def monitoring(server_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first()
    if not server:
        flash('Server not found', 'error')
        return redirect(url_for('dashboard'))
    
    return render_template('monitoring.html', server=server)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    print('�� Server: http://0.0.0.0:5003')
    print('�� Debug Mode:', os.environ.get('FLASK_DEBUG', 'False'))
    app.run(host='0.0.0.0', port=5003, debug=False)
