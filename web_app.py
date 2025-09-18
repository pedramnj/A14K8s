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

# Simple Natural Language Processor
class SimpleKubernetesProcessor:
    def __init__(self):
        self.command_mapping = {
            # Pod operations
            'show pods': 'kubectl get pods',
            'list pods': 'kubectl get pods',
            'get pods': 'kubectl get pods',
            'show me all pods': 'kubectl get pods',
            'what pods are running': 'kubectl get pods',
            
            # Service operations
            'show services': 'kubectl get services',
            'list services': 'kubectl get services',
            'get services': 'kubectl get services',
            
            # Event operations
            'show events': 'kubectl get events',
            'get events': 'kubectl get events',
            'what is wrong': 'kubectl get events',
            'check errors': 'kubectl get events',
            'cluster problems': 'kubectl get events',
            
            # Cluster info
            'cluster info': 'kubectl cluster-info',
            'cluster status': 'kubectl cluster-info',
            'show cluster': 'kubectl cluster-info',
        }
    
    def process_query(self, query: str) -> dict:
        """Process natural language query and return kubectl command"""
        query_lower = query.lower().strip()
        
        # Direct mapping
        if query_lower in self.command_mapping:
            return {
                'command': self.command_mapping[query_lower],
                'explanation': f"I'll execute: {self.command_mapping[query_lower]}"
            }
        
        # Pattern matching for pod creation
        if 'create' in query_lower and 'pod' in query_lower:
            pod_name = self._extract_pod_name(query)
            command = f"kubectl run {pod_name} --image=nginx"
            return {
                'command': command,
                'explanation': f"I'll create a pod named '{pod_name}': {command}"
            }
        
        # Pattern matching for other operations
        if 'pods' in query_lower and ('show' in query_lower or 'list' in query_lower or 'get' in query_lower):
            return {
                'command': 'kubectl get pods',
                'explanation': f"I'll show you all pods: kubectl get pods"
            }
        
        if 'services' in query_lower and ('show' in query_lower or 'list' in query_lower or 'get' in query_lower):
            return {
                'command': 'kubectl get services',
                'explanation': f"I'll show you all services: kubectl get services"
            }
        
        if 'events' in query_lower or 'wrong' in query_lower or 'error' in query_lower:
            return {
                'command': 'kubectl get events',
                'explanation': f"I'll check the events: kubectl get events"
            }
        
        # Default response
        return {
            'command': None,
            'explanation': f"I understand you want to: '{query}'. Please try a more specific request like 'show pods', 'list services', or 'check events'."
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
        
        return pod_name

# Initialize processor
processor = SimpleKubernetesProcessor()

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
        # Natural language query - use simple processor
        try:
            processed = processor.process_query(message)
            
            if processed['command']:
                # Execute the command via MCP bridge
                try:
                    mcp_response = requests.post(
                        'http://localhost:5001/api/chat',
                        json={'message': processed['command']},
                        timeout=10
                    )
                    
                    if mcp_response.status_code == 200:
                        mcp_result = mcp_response.json()
                        response_text = f"{processed['explanation']}\n\n**Result:**\n{mcp_result.get('response', 'No response')}"
                        
                        return jsonify({
                            'response': response_text,
                            'status': 'success',
                            'ai_processed': True,
                            'command_executed': processed['command']
                        })
                    else:
                        return jsonify({
                            'response': f"{processed['explanation']}\n\n**Error:** Failed to execute command",
                            'status': 'error',
                            'ai_processed': True
                        })
                        
                except Exception as e:
                    return jsonify({
                        'response': f"{processed['explanation']}\n\n**Error:** {str(e)}",
                        'status': 'error',
                        'ai_processed': True
                    })
            else:
                return jsonify({
                    'response': processed['explanation'],
                    'status': 'info',
                    'ai_processed': True
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
