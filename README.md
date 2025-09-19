# AI4K8s: AI-Powered Kubernetes Management Platform

**Master of Computer Engineering - Cloud Computing, Politecnico di Torino Thesis Project**

A comprehensive AI agent for Kubernetes cluster management using the Model Context Protocol (MCP), featuring intelligent natural language processing, real-time monitoring, and a professional web interface.

## 🎯 Project Overview

AI4K8s is an advanced AI-powered platform that enables natural language interaction with Kubernetes clusters through the Model Context Protocol (MCP). The system combines Claude AI with Kubernetes management capabilities, providing intelligent automation, monitoring, and user-friendly interfaces for cloud infrastructure management.

### Key Features

- **🤖 AI-Powered Natural Language Processing** - Interact with Kubernetes using natural language queries
- **🔗 Model Context Protocol (MCP) Integration** - Standardized AI-tool communication
- **🌐 Professional Web Interface** - Modern, responsive dashboard with user management
- **📊 Real-time Monitoring** - Integrated Prometheus and Grafana monitoring
- **🔒 Multi-User Support** - User authentication and server management
- **⚡ Intelligent Automation** - AI-driven cluster operations and recommendations
- **📈 Performance Analytics** - Comprehensive statistical analysis and benchmarking

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AI4K8s Platform                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   Web Interface │    │   AI Agent      │    │   MCP Server │ │
│  │   (Flask App)   │◄──►│   (Claude AI)   │◄──►│   (Official) │ │
│  │   Port: 5003    │    │   Port: 5002    │    │   Port: 5002 │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│           │                       │                       │     │
│           ▼                       ▼                       ▼     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   User Database │    │   MCP Bridge    │    │  Kubernetes  │ │
│  │   (SQLite)      │    │   (K8s Client)  │    │   Cluster    │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
ai4k8s/
├── 🚀 Core Application
│   ├── ai_kubernetes_web_app.py      # Main Flask web application
│   ├── kubernetes_mcp_server.py      # Custom MCP server implementation
│   └── requirements.txt              # Python dependencies
│
├── 🌐 Web Interface
│   ├── templates/                    # Jinja2 HTML templates
│   │   ├── base.html                # Base template with dark theme
│   │   ├── dashboard.html           # User dashboard
│   │   ├── chat.html                # AI chat interface
│   │   └── server_detail.html       # Server management
│   └── static/                      # Static assets
│       ├── css/style.css            # Dark theme styling
│       └── js/app.js                # Frontend JavaScript
│
├── 🤖 AI Integration
│   ├── client/                      # MCP client for AI integration
│   │   ├── ai_mcp_client.py        # AI-powered MCP client
│   │   └── pyproject.toml          # Client dependencies
│   └── setup_anthropic.sh          # API key setup script
│
├── ☸️ Kubernetes Deployments
│   ├── mcp-bridge-deployment.yaml   # MCP bridge deployment
│   ├── web-app-iframe-solution.yaml # Web app deployment
│   └── docker-compose.yml          # Docker Compose setup
│
├── 📊 Analytics & Benchmarking
│   └── netpress-integration/        # Statistical analysis
│       ├── statistical-analysis/    # Performance metrics
│       ├── benchmark_runner.py     # Benchmarking tools
│       └── test_results.json       # Test results
│
├── 📚 Documentation
│   ├── README.md                    # This file
│   ├── REPORT.md                    # Comprehensive project report
│   ├── WEB_APP_README.md           # Web application guide
│   ├── DOCKER_README.md            # Docker setup guide
│   └── OVERLEAF_REPORT.tex         # LaTeX thesis report
│
└── 🛠️ Utilities
    ├── run_chat.sh                  # Quick start script
    ├── migrate_database.py          # Database migration
    └── test_ai_processing.py        # AI processing tests
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** with pip
- **Docker Desktop** with Kubernetes enabled
- **Anthropic API Key** (for AI features)
- **kubectl** configured for your cluster

### 1. Clone and Setup

```bash
git clone https://github.com/pedramnj/A14K8s.git
cd ai4k8s

# Install Python dependencies
pip install -r requirements.txt

# Setup Anthropic API key
./setup_anthropic.sh
# or manually: export ANTHROPIC_API_KEY="your-api-key"
```

### 2. Start Kubernetes Services

```bash
# Ensure Docker Desktop Kubernetes is running
kubectl get nodes

# Deploy MCP bridge
kubectl apply -f mcp-bridge-deployment.yaml

# Port forward for MCP bridge
kubectl -n web port-forward service/mcp-bridge 5001:5001 &
```

### 3. Start Official MCP Server

```bash
# Install and run official Kubernetes MCP server
npx kubernetes-mcp-server@latest --port 5002 --log-level 3 &
```

### 4. Launch Web Application

```bash
# Start the Flask web application
python3 ai_kubernetes_web_app.py

# Access the application
open http://localhost:5003
```

### 5. Alternative: Terminal Chat Interface

```bash
# Run AI-powered terminal chat
./run_chat.sh
```

## 🛠️ Available Features

### 🤖 AI-Powered Operations

- **Natural Language Queries**: "Show me all running pods", "Create a pod named nginx"
- **Intelligent Pod Management**: Create, delete, scale pods with natural language
- **Cluster Health Analysis**: "How is the health of my cluster?"
- **Resource Monitoring**: Real-time pod, service, and deployment status

### 🌐 Web Interface Features

- **User Authentication**: Secure login and registration system
- **Server Management**: Add and manage multiple Kubernetes clusters
- **Real-time Chat**: Interactive AI chat interface
- **Connection Testing**: Test cluster connectivity and health
- **Dark Theme**: Modern, professional UI design

### 📊 Monitoring & Analytics

- **Prometheus Integration**: Metrics collection and querying
- **Grafana Dashboards**: Visualization and monitoring
- **Performance Benchmarking**: Comprehensive testing framework
- **Statistical Analysis**: AI agent performance evaluation

## 💬 Example AI Interactions

### Pod Management
```
User: "What pods are running in my cluster?"
AI: [Lists all pods with status, ready state, and restart counts]

User: "Create a pod named web-server with nginx image"
AI: [Creates pod and confirms deployment]

User: "Delete the web-server pod"
AI: [Removes pod and confirms deletion]
```

### Cluster Health
```
User: "How is the health of my cluster?"
AI: [Analyzes events, resource usage, and provides health assessment]

User: "Show me all services"
AI: [Lists services with types, IPs, and ports]

User: "What deployments are running?"
AI: [Shows deployments with replica counts and status]
```

## 🔧 Configuration

### Environment Variables

```bash
# Required for AI features
ANTHROPIC_API_KEY=your-anthropic-api-key

# Optional configuration
SECRET_KEY=your-secret-key-for-sessions
FLASK_ENV=development
```

### Database

The application uses SQLite for user and server management. The database is automatically created on first run.

```bash
# Manual database migration (if needed)
python3 migrate_database.py
```

## 📊 Performance & Analytics

### Statistical Analysis

The project includes comprehensive performance analysis through the NetPress integration:

- **Response Time Analysis**: AI query processing performance
- **Success Rate Metrics**: Operation success/failure rates
- **Confidence Intervals**: Statistical significance testing
- **Comparative Analysis**: Cross-method performance evaluation

### Benchmarking

```bash
# Run performance benchmarks
cd netpress-integration
./run_benchmark.sh

# Generate statistical analysis
cd statistical-analysis
./run_analysis.sh
```

## 🚀 Advanced Features

### Multi-Cluster Support
- Add multiple Kubernetes clusters
- Switch between different environments
- Centralized management interface

### Security Features
- User authentication and authorization
- Secure API key management
- RBAC integration with Kubernetes

### Monitoring Integration
- Real-time cluster metrics
- Prometheus metrics collection
- Grafana dashboard integration

## 🛠️ Development

### Running in Development Mode

```bash
# Enable debug mode
export FLASK_ENV=development
python3 ai_kubernetes_web_app.py
```

### Testing

```bash
# Test AI processing capabilities
python3 test_ai_processing.py

# Run comprehensive benchmarks
cd netpress-integration && ./run_benchmark.sh
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build individual containers
docker build -t ai4k8s-web .
docker run -p 5003:5003 ai4k8s-web
```

## 📈 Future Enhancements

### Planned Features
- **Multi-cluster Federation**: Cross-cluster workload management
- **Predictive Analytics**: ML-based resource prediction
- **Security Scanning**: AI-powered vulnerability detection
- **Cost Optimization**: Intelligent resource cost analysis
- **CI/CD Integration**: Automated deployment pipelines

### Research Areas
- **Autonomous Cloud Management**: Self-healing infrastructure
- **Performance Optimization**: AI-driven resource tuning
- **Security Intelligence**: Advanced threat detection
- **Cost Intelligence**: Predictive cost optimization

## 📚 Documentation

- **[Comprehensive Report](REPORT.md)** - Detailed project documentation
- **[Web App Guide](WEB_APP_README.md)** - Web interface documentation
- **[Docker Setup](DOCKER_README.md)** - Container deployment guide
- **[Statistical Analysis](netpress-integration/statistical-analysis/README.md)** - Performance metrics

## 👨‍💻 Author

**Pedram Nikjooy**  
Master of Computer Engineering - Cloud Computing  
Politecnico di Torino

- **Website**: [pedramnikjooy.me](https://pedramnikjooy.me)
- **Email**: pedramnikjooy@gmail.com
- **GitHub**: [@pedramnj](https://github.com/pedramnj)
- **LinkedIn**: [pedramnikjooy](https://linkedin.com/in/pedramnikjooy)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Anthropic** for Claude AI capabilities
- **Kubernetes Community** for the MCP server implementation
- **Politecnico di Torino** for academic support
- **Open Source Community** for the tools and libraries used

---

**© 2025 Pedram Nikjooy. All rights reserved.**