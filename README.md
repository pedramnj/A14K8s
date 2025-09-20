# AI4K8s: AI-Powered Kubernetes Management Platform

**Master of Computer Engineering - Cloud Computing, Politecnico di Torino Thesis Project**

A comprehensive AI agent for Kubernetes cluster management using the Model Context Protocol (MCP), featuring intelligent natural language processing, real-time monitoring, and a professional web interface with **AI-powered predictive monitoring capabilities**.

## 🎯 Project Overview

AI4K8s is an advanced AI-powered platform that enables natural language interaction with Kubernetes clusters through the Model Context Protocol (MCP). The system combines Claude AI with Kubernetes management capabilities, providing intelligent automation, monitoring, and user-friendly interfaces for cloud infrastructure management.

### ✨ Key Features

- **🤖 AI-Powered Natural Language Processing** - Interact with Kubernetes using natural language queries
- **🔗 Model Context Protocol (MCP) Integration** - Standardized AI-tool communication
- **🌐 Professional Web Interface** - Modern, responsive dashboard with dark theme
- **🧠 AI-Powered Predictive Monitoring** - **NEW!** ML-based anomaly detection and forecasting
- **📊 Real-time Metrics Collection** - **NEW!** Kubernetes metrics server integration
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
│           │                       │                       │     │
│           ▼                       ▼                       ▼     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │  AI Monitoring  │    │  ML Models      │    │  Metrics     │ │
│  │  (Per Server)   │    │  (Anomaly/ML)   │    │  Collection  │ │
│  │  ✅ OPERATIONAL │    │  ✅ ACTIVE      │    │  ✅ REAL-TIME │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
ai4k8s/
├── 🚀 Core Application
│   ├── ai_kubernetes_web_app.py      # Main Flask web application
│   ├── kubernetes_mcp_server.py      # Custom MCP server implementation
│   ├── predictive_monitoring.py      # AI/ML monitoring system ✅
│   ├── k8s_metrics_collector.py      # Kubernetes metrics collection ✅
│   ├── ai_monitoring_integration.py  # AI monitoring integration layer ✅
│   └── requirements.txt              # Python dependencies (includes ML libs)
│
├── 🌐 Web Interface
│   ├── templates/                    # Jinja2 HTML templates
│   │   ├── base.html                # Base template with dark theme ✅
│   │   ├── dashboard.html           # User dashboard
│   │   ├── chat.html                # AI chat interface ✅
│   │   ├── server_detail.html       # Server management ✅
│   │   └── monitoring.html          # AI monitoring dashboard ✅
│   └── static/                      # Static assets
│       ├── css/style.css            # Dark theme styling ✅
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
    └── migrate_database.py          # Database migration
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** with pip
- **Docker Desktop** with Kubernetes enabled
- **Anthropic API Key** (for AI features)
- **kubectl** configured for your cluster
- **ML Dependencies**: numpy, pandas, scikit-learn (automatically installed)

### 1. Clone and Setup

```bash
git clone https://github.com/pedramnj/A14K8s.git
cd ai4k8s

# Install Python dependencies (includes ML libraries)
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

### 5. Access AI Monitoring

```bash
# Navigate to any server and click "AI Monitoring" button
# Or access directly: http://localhost:5003/monitoring/<server_id>
```

### 6. Alternative: Terminal Chat Interface

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
- **Smart Defaults**: AI uses intelligent defaults (e.g., nginx image for pod creation)

### 🌐 Web Interface Features

- **User Authentication**: Secure login and registration system
- **Server Management**: Add and manage multiple Kubernetes clusters
- **Real-time Chat**: Interactive AI chat interface with dark theme
- **Connection Testing**: Test cluster connectivity and health
- **AI Monitoring Dashboard**: Server-specific predictive monitoring ✅
- **Dark Theme**: Modern, professional UI design ✅
- **Dynamic Footer**: Auto-updating year and university information ✅

### 📊 Monitoring & Analytics

- **🧠 AI-Powered Predictive Monitoring**: ML-based anomaly detection and forecasting ✅
- **📈 Time Series Forecasting**: Predict resource usage patterns using polynomial fitting ✅
- **🔍 Anomaly Detection**: Isolation Forest and DBSCAN for unusual behavior detection ✅
- **⚡ Performance Optimization**: AI-driven tuning recommendations ✅
- **📊 Capacity Planning**: Predictive scaling recommendations ✅
- **📊 Real-time Metrics**: Kubernetes metrics server integration ✅
- **Performance Benchmarking**: Comprehensive testing framework**
- **Statistical Analysis**: AI agent performance evaluation

## 💬 Example AI Interactions

### Pod Management
```
User: "What pods are running in my cluster?"
AI: [Lists all pods with status, ready state, and restart counts]

User: "Create a pod named web-server"  # AI uses nginx as default
AI: [Creates pod with nginx image and confirms deployment]

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

## 🧠 AI-Powered Predictive Monitoring (Phase 1) ✅ COMPLETED

### Overview

The AI monitoring system provides intelligent insights into Kubernetes cluster behavior using machine learning models. Each server has its own dedicated monitoring instance that analyzes **real metrics** and provides predictive recommendations.

### ✅ Key Features - FULLY OPERATIONAL

#### 📈 Time Series Forecasting ✅
- **CPU Usage Prediction**: Forecasts future CPU utilization trends
- **Memory Usage Prediction**: Predicts memory consumption patterns  
- **Trend Analysis**: Identifies increasing, decreasing, or stable resource usage
- **Polynomial Fitting**: Uses advanced mathematical models for accurate predictions
- **Real-time Data**: Works with actual Kubernetes metrics (CPU: 5%, Memory: 64%)

#### 🔍 Anomaly Detection ✅
- **Isolation Forest**: Detects statistical anomalies in resource usage
- **DBSCAN Clustering**: Identifies unusual patterns in cluster behavior
- **Severity Classification**: Categorizes anomalies as low, medium, high, or critical
- **Real-time Alerts**: Immediate notification of unusual cluster behavior

#### ⚡ Performance Optimization ✅
- **Resource Recommendations**: AI-driven suggestions for CPU and memory optimization
- **Scaling Recommendations**: Intelligent auto-scaling suggestions
- **Performance Tuning**: Automated recommendations for resource limits and requests
- **Cost Optimization**: Suggestions for right-sizing resources

#### 📊 Capacity Planning ✅
- **Predictive Scaling**: Forecasts when to scale up or down resources
- **Resource Forecasting**: Predicts future resource needs
- **Capacity Alerts**: Early warning system for resource exhaustion
- **Growth Planning**: Long-term capacity planning recommendations

### ML Models Used ✅

1. **Isolation Forest**: Unsupervised anomaly detection algorithm
2. **DBSCAN**: Density-based clustering for pattern recognition
3. **Linear Regression**: Time series forecasting for resource usage
4. **Polynomial Fitting**: Advanced trend analysis and prediction

### Real Metrics Collection ✅

- **Kubernetes Metrics Server**: Installed and configured
- **Real-time Data**: CPU usage (5%), Memory usage (64%), Pod count (29)
- **Automatic Parsing**: Fixed percentage parsing (removed % symbols)
- **Error Handling**: Comprehensive error handling and debug logging
- **Per-Server Monitoring**: Individual monitoring instances per server

### Demo Mode ✅

When Kubernetes is not available, the system automatically switches to demo mode:
- **Synthetic Data Generation**: Realistic metrics for demonstration
- **Full AI Capabilities**: All ML models work with generated data
- **Interactive Dashboard**: Complete monitoring experience
- **Clear Indicators**: Visual indication when in demo mode

### Usage ✅

1. **Access Monitoring**: Click "AI Monitoring" button on any server detail page
2. **View Insights**: Real-time dashboard with health scores and predictions
3. **Monitor Alerts**: Anomaly detection and performance recommendations
4. **Start/Stop Monitoring**: Control continuous monitoring with buttons
5. **Server-Specific**: Each server has independent monitoring instance

### API Endpoints ✅

- `GET /monitoring/<server_id>` - Monitoring dashboard
- `GET /api/monitoring/insights/<server_id>` - Complete AI analysis
- `GET /api/monitoring/alerts/<server_id>` - Anomaly alerts
- `GET /api/monitoring/recommendations/<server_id>` - Performance recommendations
- `GET /api/monitoring/forecast/<server_id>` - Capacity forecasts
- `GET /api/monitoring/health/<server_id>` - Cluster health score
- `POST /api/monitoring/start/<server_id>` - Start continuous monitoring
- `POST /api/monitoring/stop/<server_id>` - Stop continuous monitoring

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
- Real-time cluster metrics ✅
- Kubernetes metrics server integration ✅
- AI-powered predictive monitoring ✅

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

### Phase 1 Completed ✅
- **✅ AI-Powered Predictive Monitoring**: ML-based anomaly detection and forecasting
- **✅ Time Series Forecasting**: Resource usage prediction with polynomial fitting
- **✅ Anomaly Detection**: Isolation Forest and DBSCAN for unusual behavior
- **✅ Performance Optimization**: AI-driven tuning recommendations
- **✅ Capacity Planning**: Predictive scaling recommendations
- **✅ Server-Specific Monitoring**: Individual monitoring instances per server
- **✅ Demo Mode**: Works without Kubernetes for demonstration
- **✅ Real Metrics Collection**: Kubernetes metrics server integration
- **✅ Intelligent Defaults**: AI uses smart defaults for pod creation
- **✅ Dark Theme**: Modern, professional UI design

### Planned Features (Future Phases)
- **Multi-cluster Federation**: Cross-cluster workload management
- **Security Scanning**: AI-powered vulnerability detection
- **Cost Optimization**: Intelligent resource cost analysis
- **CI/CD Integration**: Automated deployment pipelines
- **Advanced ML Models**: Deep learning for more sophisticated predictions

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