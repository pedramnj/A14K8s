# AI Agent for Kubernetes Management using MCP

A comprehensive implementation of an AI agent for Kubernetes cluster management using the Model Context Protocol (MCP). This project demonstrates how AI applications can interact with Kubernetes environments through standardized MCP tools and resources.

## 🎯 Project Overview

This thesis project showcases:
- **AI-powered Kubernetes management** through natural language queries
- **Model Context Protocol (MCP)** implementation for standardized AI-tool communication
- **Real-time cluster monitoring** and intelligent responses
- **Production-ready architecture** with proper error handling and security

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI Agent      │    │   MCP Server    │    │   Kubernetes    │
│   (Claude)      │◄──►│   (Python)      │◄──►│   Cluster       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Local Tools   │
                       │   (kubectl,     │
                       │    docker)      │
                       └─────────────────┘
```

## 📁 Project Structure

```
ai4k8s/
├── mcp_server.py              # Main MCP server implementation
├── client/                    # MCP client for AI integration
│   ├── client.py             # AI-powered MCP client
│   ├── .env                  # API configuration
│   └── pyproject.toml        # Dependencies
├── web-app-iframe-solution.yaml  # Web application deployment (Flask app inline)
├── mcp-bridge-deployment.yaml    # In-cluster bridge (Flask + K8s client + Claude)
├── run_chat.sh               # Helper to run MCP server + client
├── OVERLEAF_REPORT.tex       # Overleaf-ready thesis report
├── REPORT.md                 # Long-form project report
├── README.md                 # This file
└── LICENSE                   # MIT License
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker Desktop with Kubernetes enabled
- Anthropic API key

### 1. Start Your Kubernetes Cluster
```bash
# Ensure Docker Desktop is running with Kubernetes enabled
kubectl get nodes
```

### 2. Run the MCP Server + Client (Terminal Chat)
```bash
./run_chat.sh
# or
cd client && uv run client.py ../mcp_server.py
```

### 3. Run the Web Interface
```bash
# Apply deployments
kubectl apply -f web-app-iframe-solution.yaml
kubectl apply -f mcp-bridge-deployment.yaml

# Port-forward for local access
kubectl -n web port-forward --address 0.0.0.0 service/nginx-proxy 8080:80
# Open http://localhost:8080
```

## 🛠️ Available MCP Tools

### Core Monitoring Tools
- **`get_cluster_info`** - Get cluster version, nodes, and status
- **`get_pods`** - List pods with status, readiness, and restart counts
- **`get_services`** - List services with types, IPs, and ports
- **`get_deployments`** - List deployments with replica counts and status

### Advanced Tools
- **`get_pod_logs`** - Retrieve logs from specific pods
- **`execute_kubectl`** - Execute arbitrary kubectl commands
- **`get_docker_containers`** - Get Docker container information

## 💬 Example Queries

Once the AI client is running, you can ask natural language questions like:

```
"What pods are running in my cluster?"
"Show me the status of my nginx deployment"
"Get information about my cluster nodes"
"What services are available?"
"Show me logs from the grafana pod"
"Scale my nginx deployment to 5 replicas"
```

## 📊 Monitoring Integration

The system integrates with your existing monitoring stack:
- **Prometheus** - Metrics collection
- **Grafana** - Visualization dashboards
- **Kubernetes metrics** - Native cluster metrics

## 🚀 Future Enhancements

- **Multi-cluster support** for enterprise environments
- **Advanced monitoring** with predictive analytics
- **Security policies** and compliance checking
- **Workflow automation** for complex deployment scenarios
- **Web-based MCP client** for browser-based AI interactions

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.