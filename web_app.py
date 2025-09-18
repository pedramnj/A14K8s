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

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ai4k8s.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

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
    
    # For now, we'll use the existing MCP bridge
    # In the future, this will be adapted to work with user-specific servers
    try:
        # Forward the request to the MCP bridge
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
