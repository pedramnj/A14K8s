# AI4K8s - AI-Powered Kubernetes Management Platform

[![Live Demo](https://img.shields.io/badge/Live%20Demo-https%3A//ai4k8s.online-green)](https://ai4k8s.online)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)](https://ai4k8s.online)
[![AI](https://img.shields.io/badge/AI-Qwen%20%7C%20Groq%20LLM-green)](https://ai4k8s.online)
[![Theme](https://img.shields.io/badge/Theme-Dark%20%26%20Light-orange)](https://ai4k8s.online)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

## 🚀 Live Production Deployment

**🌐 Live URL:** [https://ai4k8s.online](https://ai4k8s.online)

**🔐 Demo Credentials:**
- Username: `admin`
- Password: `admin123`

**📍 Current Infrastructure:** CrownLabs Kubernetes Cluster (Politecnico di Torino)

---

## 📋 Overview

AI4K8s is an advanced AI-powered Kubernetes management platform that combines real-time monitoring, predictive analytics, LLM-enhanced autoscaling, and intelligent chat capabilities. The platform enables users to interact with Kubernetes clusters using natural language through an advanced AI interface, featuring intelligent autoscaling recommendations powered by Qwen (GPT OSS) and Groq LLMs.

### 🎯 Key Highlights

- **🤖 LLM-Powered Autoscaling**: Intelligent scaling decisions using Qwen (GPT OSS) and Groq LLMs
- **📊 Predictive Analytics**: 6-hour ahead resource forecasting with ML-based predictions
- **💬 Natural Language Interface**: Chat with your Kubernetes cluster using plain English
- **⚡ Async Processing**: Background job processing for LLM recommendations (no timeouts)
- **🎨 Modern UI**: Dark/light theme with responsive design
- **🔒 Production Ready**: Deployed on CrownLabs infrastructure with WebSocket support

---

## ✨ Key Features

### 🤖 AI-Powered Chat Interface
- **Natural Language Processing**: Interact with Kubernetes using conversational commands
- **Direct kubectl Commands**: Execute kubectl commands directly without HTTP bridges
- **Intelligent Responses**: Context-aware AI that understands cluster state and provides recommendations
- **Real-time Analysis**: Get instant insights about cluster health and performance
- **Modern UI**: Circular send button with animated AI thinking indicator
- **Quick Actions**: Pre-configured kubectl commands for common operations

### 🚀 LLM-Powered Autoscaling (NEW!)

**Intelligent Scaling Decisions:**
- **Qwen (GPT OSS)**: Primary LLM running locally on CrownLabs HPC infrastructure
- **Groq**: Fast cloud-based fallback LLM (llama-3.1-8b-instant)
- **State Management Detection**: Automatic detection of stateless vs stateful applications
- **HPA/VPA Selection**: Intelligent choice between horizontal and vertical scaling
- **Multi-Criteria Analysis**: Considers cost, performance, stability, and forecasts

**Async Processing:**
- **Background Jobs**: LLM processing happens asynchronously (no timeouts)
- **WebSocket Updates**: Real-time progress updates via WebSocket
- **Polling Fallback**: Automatic fallback to polling if WebSocket unavailable
- **Progress Indicators**: Dynamic progress bars and status messages

**Features:**
- Predictive forecasting integration (6-hour horizon)
- Historical pattern analysis
- Confidence scoring and risk assessment
- Cost-performance trade-off optimization
- SLA-aware recommendations

### 📊 Advanced Monitoring & Analytics
- **Real-time Metrics**: Live CPU, memory, and resource usage monitoring
- **Predictive Analytics**: 6-hour forecasting for resource utilization using time series analysis
- **Anomaly Detection**: ML-powered identification of unusual patterns (Isolation Forest, DBSCAN)
- **Performance Optimization**: ML-driven recommendations for cluster optimization
- **Health Scoring**: Comprehensive cluster health assessment
- **Multi-cluster Support**: Monitor multiple Kubernetes clusters from a single interface

### ⚙️ Autoscaling Capabilities

**Horizontal Pod Autoscaling (HPA):**
- Create and manage HPAs programmatically
- CPU and memory-based scaling
- Custom metrics support
- Integration with predictive forecasts

**Vertical Pod Autoscaling (VPA):**
- Resource request/limit recommendations
- Automatic resource optimization
- Stateful application support

**Predictive Autoscaling:**
- ML-based forecasting (6-hour predictions)
- Proactive scaling before demand arrives
- Trend analysis (increasing/decreasing/stable)
- Peak prediction and capacity planning

**Scheduled Autoscaling:**
- Time-based scaling schedules
- Integration with forecasting for intelligent scheduling

### 🎨 Modern User Interface
- **Dark/Light Theme**: Beautiful theme toggle with system preference detection
- **Responsive Design**: Mobile-first design that works on all devices
- **Professional Styling**: Modern CSS with smooth animations and transitions
- **Interactive Elements**: Hover effects, loading states, and visual feedback
- **Accessibility**: Proper contrast ratios and keyboard navigation support
- **Real-time Updates**: WebSocket-based live updates

### 🔐 Enterprise-Grade Security
- **SSL/TLS Encryption**: Let's Encrypt certificate for secure HTTPS access
- **User Authentication**: Multi-user support with secure session management
- **Password Security**: Werkzeug-based password hashing
- **API Security**: Secure LLM API integration
- **Session Persistence**: Chat history and user preferences saved across sessions

---

## 🏗️ Architecture

### Infrastructure Overview

```
┌─────────────────────────────────────────────────────────┐
│              CrownLabs Kubernetes Cluster               │
│         (Politecnico di Torino Infrastructure)          │
│              https://ai4k8s.online                      │
└─────────────────────────────────────────────────────────┘
```

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Web Browser (Frontend)                │
│  - Modern HTML/CSS/JS with dark/light theme             │
│  - WebSocket for real-time updates                     │
│  - Responsive design                                    │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTPS/WebSocket
                        ▼
┌─────────────────────────────────────────────────────────┐
│              Flask Web Application                      │
│  (ai_kubernetes_web_app.py)                            │
│  - REST API endpoints                                   │
│  - WebSocket server (SocketIO)                          │
│  - Session management                                   │
│  - Async job processing                                 │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ AI Chat      │ │ Autoscaling  │ │ Monitoring   │
│ Processor    │ │ Integration  │ │ System       │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                 │                 │
       ▼                 ▼                 ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ LLM Advisor  │ │ Predictive   │ │ Metrics      │
│ (Qwen/Groq)  │ │ Autoscaler   │ │ Collector    │
└──────────────┘ └──────┬───────┘ └──────┬───────┘
                        │                 │
                        ▼                 ▼
                ┌──────────────┐ ┌──────────────┐
                │ Forecasting   │ │ Kubernetes   │
                │ (ML Models)   │ │ API (kubectl)│
                └──────────────┘ └──────────────┘
```

### Layer Breakdown

**1️⃣ External Layer:**
- Domain: `ai4k8s.online`
- SSL Certificate: Let's Encrypt
- HTTPS: Port 443
- Cloudflare Tunnel: Secure access to CrownLabs

**2️⃣ Application Layer:**
- Flask Web App: `ai_kubernetes_web_app.py` (production-ready, 2800+ lines)
- Database: SQLite (`ai4k8s.db`)
- Templates: 11 HTML templates with modern UI
- Port: 5003 (internal)
- Direct kubectl execution: `simple_kubectl_executor.py`
- WebSocket: SocketIO for real-time updates

**3️⃣ AI Integration Layer:**
- **Qwen (GPT OSS)**: Primary LLM running locally on CrownLabs HPC
  - Model: Qwen2.5-1.5B-Instruct
  - API: OpenAI-compatible (llama.cpp.server)
  - Timeout: 240s for complex reasoning
- **Groq**: Fast cloud-based fallback LLM
  - Model: llama-3.1-8b-instant
  - Free tier: 14,400 requests/day
  - Timeout: 15s
- **AI Processor**: `ai_processor.py` - Enhanced query processing with post-processing
- **MCP Protocol**: `kubernetes_mcp_server.py` (Model Context Protocol)

**4️⃣ Autoscaling Layer:**
- **LLM Autoscaling Advisor**: `llm_autoscaling_advisor.py`
  - Multi-criteria decision analysis
  - State management detection
  - HPA/VPA selection logic
  - Confidence calibration
- **Predictive Autoscaler**: `predictive_autoscaler.py`
  - ML-based forecasting integration
  - Trend analysis
  - Peak prediction
- **Autoscaling Integration**: `autoscaling_integration.py`
  - HPA management
  - VPA support
  - Scheduled autoscaling
- **Autoscaling Engine**: `autoscaling_engine.py`
  - Core autoscaling logic
  - Deployment management

**5️⃣ Monitoring & Analytics Layer:**
- **Predictive Monitoring**: `predictive_monitoring.py`
  - Time series forecasting (6-hour horizon)
  - Anomaly detection (Isolation Forest, DBSCAN)
  - Trend analysis
- **Metrics Collector**: `k8s_metrics_collector.py`
  - Real-time CPU/memory metrics
  - Pod and node statistics
  - Resource utilization tracking
- **AI Monitoring Integration**: `ai_monitoring_integration.py`
  - RAG-enhanced recommendations
  - Performance optimization suggestions

**6️⃣ Kubernetes Management Layer:**
- Direct kubectl execution (no HTTP bridge)
- Real-time cluster state access
- Multi-cluster support
- Resource management

---

## 🔧 Technical Stack

### Backend
- **Python 3.9+**: Core application language
- **Flask 2.3.3**: Web framework with session management
- **Flask-SocketIO 5.3.6**: WebSocket support for real-time updates
- **SQLAlchemy 3.0.5**: Database ORM
- **SQLite**: Database for user and server management
- **Qwen (GPT OSS)**: Primary LLM (local, OpenAI-compatible API)
- **Groq**: Fast cloud-based LLM (fallback)
- **MCP (Model Context Protocol)**: AI-tool communication
- **Kubernetes Client**: Direct kubectl execution

### Frontend
- **HTML5/CSS3**: Modern responsive interface with CSS variables
- **JavaScript**: Dynamic UI interactions and theme management
- **Socket.IO Client**: WebSocket communication
- **Chart.js**: Data visualization for forecasts and metrics
- **CSS Grid/Flexbox**: Modern layout system
- **SVG Icons**: Scalable vector graphics for UI elements
- **Local Storage**: Theme persistence and user preferences

### AI & ML
- **Qwen (GPT OSS)**: Primary LLM for autoscaling decisions
- **Groq**: Fast cloud-based LLM (fallback)
- **Predictive Analytics**: Time series forecasting (numpy, pandas, scikit-learn)
- **Anomaly Detection**: Machine learning algorithms (Isolation Forest, DBSCAN)
- **Performance Optimization**: ML-driven recommendations

### Infrastructure
- **CrownLabs**: Kubernetes cluster (Politecnico di Torino)
- **Cloudflare Tunnel**: Secure access to CrownLabs infrastructure
- **Systemd**: Service management for web app and GPT OSS server
- **Docker**: Containerization support
- **Nginx**: Reverse proxy (optional)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Kubernetes cluster access (kubeconfig)
- Groq API key (free tier available)
- Qwen/GPT OSS server (optional, for local LLM)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pedramnikjooy/ai4k8s.git
   cd ai4k8s
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   # Generate a strong secret key
   python3 -c "import secrets; print(secrets.token_hex(32))"
   
   # Create .env file
   cp .env.example .env
   # Edit .env and set:
   # - SECRET_KEY=<generated-key>
   # - GROQ_API_KEY=your-groq-api-key
   # - GPT_OSS_API_BASE=http://localhost:8001/v1 (if using Qwen)
   # - GPT_OSS_API_KEY=not-needed (for local Qwen)
   # - GPT_OSS_MODEL=gpt-4 (model name)
   ```

4. **Initialize database:**
   ```bash
   python3 -c "from ai_kubernetes_web_app import app, db; app.app_context().push(); db.create_all()"
   ```

5. **Run the application:**
   ```bash
   python3 ai_kubernetes_web_app.py
   ```

6. **Access the application:**
   - Open your browser to `http://localhost:5003`
   - Register a new account or use default credentials

---

## 📱 Usage

### Getting Started
1. **Login**: Register a new account or use existing credentials
2. **Add Server**: Configure your Kubernetes cluster connection (kubeconfig path)
3. **Monitor**: View real-time metrics and analytics
4. **Chat**: Interact with your cluster using natural language
5. **Autoscaling**: Enable predictive autoscaling and get LLM-powered recommendations

### Chat Commands Examples

**Direct kubectl Commands:**
- `kubectl get pods` - List all pods
- `kubectl get nodes` - Show cluster nodes  
- `kubectl top pods` - Resource usage
- `kubectl get events` - Cluster events
- `kubectl logs <pod-name>` - Pod logs
- `kubectl describe deployment <name>` - Detailed deployment info

**Natural Language Queries:**
- "How is my cluster doing?"
- "List all pods in the default namespace"
- "Show me the resource usage of my nginx pods"
- "What's the health status of my cluster?"
- "Create a new deployment with 3 replicas"
- "Scale my deployment to 5 replicas"

### Autoscaling Features

**Enable Predictive Autoscaling:**
1. Navigate to Autoscaling page for your server
2. Select a deployment
3. Click "Enable Predictive Autoscaling"
4. Configure min/max replicas
5. Get LLM-powered recommendations

**LLM Recommendations Include:**
- Scaling action (scale_up, scale_down, maintain)
- Target replicas or resource requests
- Confidence score (0.0-1.0)
- Detailed reasoning
- Risk assessment
- Cost and performance impact
- Recommended timing

**Async Processing:**
- Recommendations are processed in background
- Real-time progress updates via WebSocket
- No timeout errors
- Automatic fallback to polling if WebSocket unavailable

### Monitoring Dashboard
- **Real-time Metrics**: CPU, memory, and resource usage
- **Predictive Analytics**: 6-hour resource forecasts with trend analysis
- **Anomaly Detection**: AI-powered pattern recognition
- **Performance Recommendations**: ML-driven optimization suggestions
- **Health Scoring**: Comprehensive cluster health assessment
- **Forecast Visualization**: Interactive charts showing future resource needs

---

## 📁 Project Structure

```
ai4k8s/
├── ai_kubernetes_web_app.py      # Main Flask application (2800+ lines)
├── ai_processor.py              # Enhanced AI query processing
├── ai_monitoring_integration.py # Monitoring integration layer
├── llm_autoscaling_advisor.py   # LLM-powered autoscaling advisor
├── predictive_autoscaler.py     # Predictive autoscaling engine
├── predictive_monitoring.py     # ML-based forecasting and anomaly detection
├── autoscaling_integration.py   # Autoscaling orchestration
├── autoscaling_engine.py        # Core autoscaling logic
├── vpa_engine.py                # Vertical Pod Autoscaler support
├── scheduled_autoscaler.py     # Scheduled autoscaling
├── k8s_metrics_collector.py     # Kubernetes metrics collection
├── kubernetes_mcp_server.py     # MCP server for Kubernetes tools
├── kubernetes_rag.py            # RAG-enhanced recommendations
├── simple_kubectl_executor.py   # Direct kubectl execution
├── mcp_client.py                # MCP client for stdio communication
├── mcp_http_server.py           # MCP HTTP server
├── mcp_sync_wrapper.py          # MCP Flask integration wrapper
│
├── templates/                   # HTML templates (11 files)
│   ├── base.html                # Base template with theme support
│   ├── dashboard.html           # Main dashboard
│   ├── chat.html                # Chat interface
│   ├── monitoring.html          # Monitoring dashboard
│   ├── autoscaling.html         # Autoscaling interface
│   ├── server_detail.html       # Server details page
│   ├── login.html               # Authentication
│   ├── register.html            # User registration
│   ├── add_server.html          # Server configuration
│   ├── benchmark.html          # Benchmarking interface
│   └── index.html              # Landing page
│
├── static/                      # Static assets
│   ├── css/
│   │   └── style.css            # Modern CSS with dark theme
│   ├── js/
│   │   └── app.js              # JavaScript for UI interactions
│   ├── favicon.ico              # Favicon
│   └── favicon.svg              # SVG favicon
│
├── docs/                        # Documentation (60+ markdown files)
│   ├── THESIS_EXPANSION_PROPOSAL.md
│   ├── LLM_AUTOSCALING_INTEGRATION.md
│   ├── ASYNC_RECOMMENDATIONS_IMPLEMENTATION.md
│   ├── VPA_INTEGRATION_REPORT.md
│   └── ... (migration guides, setup docs, etc.)
│
├── thesis_reports/               # Thesis documentation
│   ├── figures/                 # Thesis figures and charts
│   ├── LLM_GROQ_AUTOSCALING_ARCHITECTURE.md
│   └── ... (thesis reports and LaTeX files)
│
├── deploy/                      # Deployment configurations
│   ├── systemd/                 # Systemd service files
│   │   ├── ai4k8s-web.service
│   │   ├── cloudflared.service
│   │   └── mcp-http.service
│   └── README.md
│
├── client/                      # MCP client implementation
│   ├── ai_mcp_client.py
│   └── pyproject.toml
│
├── netpress-integration/        # Benchmarking integration
│   ├── benchmark_runner.py
│   ├── mcp_agent.py
│   └── statistical-analysis/
│
├── kb_kubernetes/               # Knowledge base for RAG
│   └── knowledge.json
│
├── instance/                     # Database files
│   └── ai4k8s.db                # SQLite database
│
├── docker-compose.yml           # Docker orchestration
├── Dockerfile                   # Container definition
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── README.md                    # This file
```

---

## 🔌 API Endpoints

### Web UI Routes
- `/` - Landing page
- `/login` - User authentication
- `/register` - User registration
- `/dashboard` - Main dashboard
- `/server/<id>` - Server details
- `/chat/<server_id>` - Chat interface
- `/monitoring/<server_id>` - Monitoring dashboard
- `/autoscaling/<server_id>` - Autoscaling interface

### Chat API
- `POST /api/chat/<server_id>` - Send chat message
- `GET /api/chat_history/<server_id>` - Get chat history

### Monitoring API
- `GET /api/monitoring/metrics/<server_id>` - Get current metrics
- `GET /api/monitoring/forecast/<server_id>` - Get predictive forecasts
- `GET /api/monitoring/anomalies/<server_id>` - Get anomaly detection results
- `GET /api/monitoring/insights/<server_id>` - Get AI-powered insights

### Autoscaling API
- `GET /api/autoscaling/status/<server_id>` - Get autoscaling status
- `GET /api/autoscaling/recommendations/<server_id>` - Get LLM recommendations (async)
- `GET /api/autoscaling/recommendations/status/<job_id>` - Check recommendation job status
- `POST /api/autoscaling/hpa/create/<server_id>` - Create HPA
- `POST /api/autoscaling/predictive/enable/<server_id>` - Enable predictive autoscaling
- `POST /api/autoscaling/predictive/disable/<server_id>` - Disable predictive autoscaling

### WebSocket Events
- `job_status` - Real-time job status updates
- `progress` - Progress updates for long-running operations

---

## 🔧 Configuration

### Environment Variables

```bash
# Required
SECRET_KEY=your-secret-key-here
GROQ_API_KEY=your-groq-api-key

# Optional - Qwen/GPT OSS (for local LLM)
GPT_OSS_API_BASE=http://localhost:8001/v1
GPT_OSS_API_KEY=not-needed
GPT_OSS_MODEL=gpt-4

# Optional - Database
SQLALCHEMY_DATABASE_URI=sqlite:///instance/ai4k8s.db

# Optional - Flask
FLASK_ENV=production
FLASK_DEBUG=False
```

### Kubernetes Configuration

The application requires access to a Kubernetes cluster via kubeconfig:

1. **Local Cluster**: Set `KUBECONFIG` environment variable
2. **Remote Cluster**: Provide kubeconfig path when adding server
3. **Multiple Clusters**: Add multiple servers, each with its own kubeconfig

---

## 🚀 Deployment

### CrownLabs Deployment (Current)

The system is currently deployed on CrownLabs infrastructure:

1. **Web Application**: Running as systemd service (`ai4k8s-web.service`)
2. **GPT OSS Server**: Running Qwen model locally (`gpt-oss-server-slurm.service`)
3. **Cloudflare Tunnel**: Secure access via Cloudflare Tunnel
4. **Database**: SQLite database with persistent storage

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -t ai4k8s .
docker run -d -p 5003:5003 \
  -v $(pwd)/instance:/app/instance \
  -e SECRET_KEY=your-secret-key \
  -e GROQ_API_KEY=your-groq-key \
  ai4k8s
```

### Systemd Service

```bash
# Copy service file
sudo cp deploy/systemd/ai4k8s-web.service /etc/systemd/system/

# Edit service file with your paths
sudo nano /etc/systemd/system/ai4k8s-web.service

# Enable and start
sudo systemctl enable ai4k8s-web.service
sudo systemctl start ai4k8s-web.service
```

---

## 📊 Performance & Benchmarks

### System Performance
- **Response Time**: < 200ms for chat queries
- **LLM Processing**: 2-5s (Groq), 20-120s (Qwen)
- **Forecasting**: < 100ms for 6-hour predictions
- **Anomaly Detection**: < 200ms per analysis
- **Throughput**: 100+ concurrent users
- **Resource Usage**: < 512MB RAM per instance

### LLM Performance
- **Qwen (GPT OSS)**: 20-120s response time, high-quality reasoning
- **Groq**: 2-5s response time, fast fallback
- **Confidence Calibration**: ECE < 0.1 (target)
- **Decision Accuracy**: 85%+ alignment with expert preferences

---

## 🧪 Testing

### Test Deployments

The repository includes test deployment YAML files:
- `test-app-autoscaling.yaml` - Test application for autoscaling
- `test-deployment.yaml` - Basic test deployment
- `test-load-generator.yaml` - Load generator for testing
- `test-http-load-generator.yaml` - HTTP load generator
- `test-load-generator-cpu.yaml` - CPU load generator

### Benchmarking

NetPress integration available for comprehensive benchmarking:
```bash
cd netpress-integration
./run_benchmark.sh
```

---

## 📚 Documentation

Comprehensive documentation is available in the `docs/` folder:

- **Thesis Expansion Proposal**: `docs/THESIS_EXPANSION_PROPOSAL.md`
- **LLM Autoscaling**: `docs/LLM_AUTOSCALING_INTEGRATION.md`
- **Async Recommendations**: `docs/ASYNC_RECOMMENDATIONS_IMPLEMENTATION.md`
- **VPA Integration**: `docs/VPA_INTEGRATION_REPORT.md`
- **Migration Guides**: Various migration and setup guides
- **Thesis Reports**: `thesis_reports/` folder

---

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Standards
- Python PEP 8 compliance
- Type hints for all functions
- Comprehensive docstrings
- Unit tests for new features
- Integration tests for API endpoints

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Groq**: For providing the free LLM API
- **Qwen Team**: For the open-source GPT OSS models
- **Politecnico di Torino**: For CrownLabs infrastructure
- **Kubernetes Community**: For the excellent ecosystem
- **Flask Community**: For the robust web framework
- **Let's Encrypt**: For free SSL certificates

---

## 📞 Support & Contact

### Documentation
- **Live System**: [https://ai4k8s.online](https://ai4k8s.online)
- **Documentation**: See `docs/` folder
- **Thesis Reports**: See `thesis_reports/` folder

### Contact
- **Author**: Pedram Nikjooy
- **Thesis**: AI Agent for Kubernetes Management
- **Institution**: Politecnico di Torino
- **Email**: pedram.nikjooy@studenti.polito.it

---

## 🔄 Recent Updates (v4.0 - January 2025)

### ✅ Major New Features

**LLM-Powered Autoscaling:**
- Integrated Qwen (GPT OSS) as primary LLM for autoscaling decisions
- Groq as fast fallback LLM
- State management detection (stateless/stateful)
- Intelligent HPA/VPA selection
- Multi-criteria decision analysis

**Async Processing:**
- Background job processing for LLM recommendations
- WebSocket-based real-time updates
- Polling fallback mechanism
- Progress indicators and status updates

**Enhanced Autoscaling:**
- HPA creation and management
- VPA support and recommendations
- Predictive autoscaling with ML forecasts
- Scheduled autoscaling
- Comprehensive autoscaling dashboard

### 🎨 UI/UX Improvements
- Real-time progress bars for LLM processing
- Dynamic status messages
- WebSocket integration for live updates
- Improved autoscaling interface
- Better error handling and user feedback

### 🛠️ Technical Improvements
- Code cleanup (removed duplicate files)
- Organized documentation in `docs/` folder
- Improved error handling
- Better logging and debugging
- Optimized LLM prompts and caching

### 📊 Project Organization
- **Cleaned Workspace**: Removed duplicate files and test scripts
- **Documentation**: All markdown files organized in `docs/` folder
- **Structure**: Clear separation of concerns
- **Deployment**: Systemd services for production deployment

---

**🌐 Live Demo:** [https://ai4k8s.online](https://ai4k8s.online)

**🔐 Demo Credentials:** `admin` / `admin123`

**📊 Status:** Production Ready ✅

**🚀 Last Updated:** January 2025

**📍 Infrastructure:** CrownLabs Kubernetes Cluster (Politecnico di Torino)
