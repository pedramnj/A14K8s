# AI4K8s Web Application

A comprehensive web-based platform for managing Kubernetes clusters through AI-powered natural language interactions.

## 🚀 Features

### User Management
- **User Registration & Authentication** - Secure user accounts with password hashing
- **Session Management** - Persistent login sessions
- **User Dashboard** - Personalized interface for each user

### Server Management
- **Multi-Server Support** - Add and manage multiple Kubernetes clusters
- **Server Types** - Support for local, remote, and cloud-based clusters
- **Connection Management** - Store and manage server connection details
- **Status Monitoring** - Real-time server status checking

### AI Chat Interface
- **Natural Language Queries** - Interact with Kubernetes using plain English
- **Server-Specific Chat** - Each server has its own AI chat session
- **Quick Actions** - Pre-built commands for common operations
- **Real-time Responses** - Instant AI-powered responses

### Dashboard & Analytics
- **Server Overview** - Visual dashboard with server statistics
- **Activity Tracking** - Monitor server usage and access patterns
- **Status Indicators** - Real-time server health monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Browser   │    │   Flask Web     │    │   SQLite DB     │
│                 │◄──►│   Application   │◄──►│   (Users &      │
│   (Frontend)    │    │   (Backend)     │    │    Servers)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   MCP Bridge    │
                       │   (K8s API +    │
                       │    AI Client)   │
                       └─────────────────┘
```

## 📁 Project Structure

```
web_app/
├── web_app.py              # Main Flask application
├── templates/              # HTML templates
│   ├── base.html          # Base template with navigation
│   ├── index.html         # Landing page
│   ├── login.html         # User login page
│   ├── register.html      # User registration page
│   ├── dashboard.html     # User dashboard
│   ├── add_server.html    # Add server form
│   ├── server_detail.html # Server details page
│   └── chat.html          # AI chat interface
├── static/                # Static assets
│   ├── css/
│   │   └── style.css      # Custom styles
│   └── js/
│       └── app.js         # JavaScript functionality
├── requirements_web.txt   # Python dependencies
└── ai4k8s.db             # SQLite database (created on first run)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Flask and dependencies (see requirements_web.txt)
- Running MCP Bridge (for AI chat functionality)

### 1. Install Dependencies
```bash
pip install -r requirements_web.txt
```

### 2. Start the Web Application
```bash
python3 web_app.py
```

The application will be available at `http://localhost:5002`

### 3. Create Your Account
1. Visit `http://localhost:5002`
2. Click "Get Started" to register
3. Fill in your username, email, and password
4. You'll be redirected to your dashboard

### 4. Add Your First Server
1. Click "Add Server" on the dashboard
2. Choose server type (Local/Remote/Cloud)
3. Enter connection details
4. Optionally add kubeconfig for advanced users

### 5. Start AI Chatting
1. Click on your server in the dashboard
2. Click "AI Chat" to start interacting
3. Ask questions like:
   - "Show me all pods"
   - "What's the status of my cluster?"
   - "Get logs from the nginx pod"

## 🛠️ Database Schema

### Users Table
- `id` - Primary key
- `username` - Unique username
- `email` - Unique email address
- `password_hash` - Hashed password
- `created_at` - Account creation timestamp

### Servers Table
- `id` - Primary key
- `name` - Server display name
- `server_type` - Type (local/remote/cloud)
- `connection_string` - Server connection details
- `kubeconfig` - Optional kubeconfig content
- `status` - Server status (active/inactive/error)
- `created_at` - Server creation timestamp
- `last_accessed` - Last access timestamp
- `user_id` - Foreign key to users table

## 🔧 Configuration

### Environment Variables
- `SECRET_KEY` - Flask secret key for sessions (default: 'your-secret-key-change-this')
- `SQLALCHEMY_DATABASE_URI` - Database connection string (default: 'sqlite:///ai4k8s.db')

### Server Types

#### Local Development
- **minikube**: `minikube ip:8080`
- **Docker Desktop**: `localhost:8080`
- **kind**: `localhost:8080`

#### Remote Server
- **IP Address**: `192.168.1.100:8080`
- **Domain**: `k8s.company.com`
- **Custom Port**: Include port if not standard

#### Cloud Provider
- **EKS**: `your-cluster.region.eks.amazonaws.com`
- **GKE**: `your-cluster.zone.gcp.com`
- **AKS**: `your-cluster.region.azmk8s.io`

## 🔒 Security Features

- **Password Hashing** - Uses Werkzeug's secure password hashing
- **Session Management** - Secure session handling with Flask
- **User Isolation** - Each user can only access their own servers
- **Input Validation** - Form validation and sanitization
- **CSRF Protection** - Built-in Flask CSRF protection

## 🎨 UI/UX Features

- **Responsive Design** - Works on desktop, tablet, and mobile
- **Modern Interface** - Clean, professional design with Bootstrap 5
- **Interactive Elements** - Hover effects, animations, and transitions
- **Real-time Updates** - Live status updates and notifications
- **Accessibility** - ARIA labels and keyboard navigation support

## 🔌 API Endpoints

### Authentication
- `GET /` - Landing page
- `GET /login` - Login page
- `POST /login` - Process login
- `GET /register` - Registration page
- `POST /register` - Process registration
- `GET /logout` - Logout user

### Dashboard
- `GET /dashboard` - User dashboard
- `GET /add_server` - Add server form
- `POST /add_server` - Process server addition
- `GET /server/<id>` - Server details
- `GET /chat/<id>` - AI chat interface

### API
- `POST /api/chat/<server_id>` - Send message to AI
- `GET /api/server_status/<server_id>` - Get server status

## 🚀 Future Enhancements

- **Multi-cluster Management** - Advanced cluster orchestration
- **Team Collaboration** - Share servers and chat sessions
- **Advanced Monitoring** - Detailed metrics and alerting
- **Plugin System** - Extensible architecture for custom tools
- **Mobile App** - Native mobile application
- **API Documentation** - Swagger/OpenAPI documentation
- **Advanced Security** - OAuth, 2FA, and RBAC integration

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   - Change port in `web_app.py`: `app.run(port=5003)`
   - Or kill existing process: `lsof -ti:5002 | xargs kill`

2. **Database Errors**
   - Delete `ai4k8s.db` and restart the application
   - Check file permissions in the application directory

3. **MCP Bridge Connection**
   - Ensure MCP Bridge is running on port 5001
   - Check network connectivity between web app and bridge

4. **Static Files Not Loading**
   - Verify `static/` directory structure
   - Check file permissions and paths

### Debug Mode
Enable debug mode by setting `debug=True` in `web_app.py` for detailed error messages and auto-reload.

## 📄 License

This project is part of the AI4K8s thesis project and follows the same MIT License.
