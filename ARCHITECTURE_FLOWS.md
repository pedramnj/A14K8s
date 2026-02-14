# AI4K8s - Architecture & Data Flows
**Detailed Technical Architecture and Data Flow Diagrams**

---

## 1. System Architecture Layers

### Complete System Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL ACCESS LAYER                            │
│                                                                          │
│  ┌────────────────┐    ┌──────────────┐    ┌─────────────────┐        │
│  │   End Users    │───▶│  Cloudflare  │───▶│ Let's Encrypt   │        │
│  │   (Browser)    │    │     CDN      │    │   SSL/TLS       │        │
│  └────────────────┘    └──────────────┘    └─────────────────┘        │
│                                │                                         │
│                                │ HTTPS/WSS (Port 443)                   │
└────────────────────────────────┼─────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────────┐
│                    CLOUDFLARE TUNNEL LAYER                               │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │         Cloudflared Service (systemd)                        │      │
│  │  - Encrypted tunnel to CrownLabs                             │      │
│  │  - Domain: ai4k8s.online                                     │      │
│  │  - Auto-reconnect on failure                                 │      │
│  └──────────────────────────────────────────────────────────────┘      │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────────┐
│                   CROWNLABS INFRASTRUCTURE                               │
│                  (Politecnico di Torino K8s Cluster)                     │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                  Kubernetes Pod: ai4k8s-web                  │      │
│  │                                                              │      │
│  │  ┌────────────────────────────────────────────────────┐    │      │
│  │  │     Flask Application (Port 5003)                  │    │      │
│  │  │  - REST API Server                                 │    │      │
│  │  │  - SocketIO WebSocket Server                       │    │      │
│  │  │  - Session Manager                                 │    │      │
│  │  │  - Async Job Queue                                 │    │      │
│  │  └────────────────────────────────────────────────────┘    │      │
│  │                          │                                   │      │
│  │                          ▼                                   │      │
│  │  ┌────────────────────────────────────────────────────┐    │      │
│  │  │          SQLite Database (Persistent Volume)       │    │      │
│  │  │  - Users, Servers, Chats                           │    │      │
│  │  └────────────────────────────────────────────────────┘    │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                │                                         │
│                                ▼                                         │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │              Kubernetes API Server                           │      │
│  │  - kubectl commands via Python subprocess                    │      │
│  │  - Kubernetes Python client API                              │      │
│  │  - Metrics Server API                                        │      │
│  └──────────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                      EXTERNAL LLM SERVICES                               │
│                                                                          │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐            │
│  │   GPT OSS   │      │    Groq     │      │  Anthropic  │            │
│  │   (Local)   │      │   (Cloud)   │      │   (Cloud)   │            │
│  │  Qwen Model │      │  Llama 3.1  │      │   Claude    │            │
│  │  Port 8001  │      │   API Free  │      │ (Optional)  │            │
│  └─────────────┘      └─────────────┘      └─────────────┘            │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Application Layer Architecture

### Flask Application Internal Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Flask App (ai_kubernetes_web_app.py)                  │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                    REQUEST ROUTER                              │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │     │
│  │  │   Auth       │  │   Server     │  │   Chat       │        │     │
│  │  │   Routes     │  │   Routes     │  │   Routes     │        │     │
│  │  └──────────────┘  └──────────────┘  └──────────────┘        │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │     │
│  │  │ Monitoring   │  │ Autoscaling  │  │  WebSocket   │        │     │
│  │  │   Routes     │  │   Routes     │  │   Handler    │        │     │
│  │  └──────────────┘  └──────────────┘  └──────────────┘        │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                  │                                       │
│                                  ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                 MIDDLEWARE LAYER                               │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │     │
│  │  │  Session     │  │     Auth     │  │    CORS      │        │     │
│  │  │  Manager     │  │  Decorator   │  │   Handler    │        │     │
│  │  └──────────────┘  └──────────────┘  └──────────────┘        │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                  │                                       │
│                                  ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │              BUSINESS LOGIC ORCHESTRATORS                      │     │
│  │                                                                │     │
│  │  ┌─────────────────────────────────────────────────┐          │     │
│  │  │  AIPoweredMCPKubernetesProcessor                │          │     │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │          │     │
│  │  │  │  Groq    │→ │Anthropic │→ │  Regex   │      │          │     │
│  │  │  │   LLM    │  │   LLM    │  │ Fallback │      │          │     │
│  │  │  └──────────┘  └──────────┘  └──────────┘      │          │     │
│  │  └─────────────────────────────────────────────────┘          │     │
│  │                                                                │     │
│  │  ┌─────────────────────────────────────────────────┐          │     │
│  │  │  AIMonitoringIntegration                        │          │     │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │          │     │
│  │  │  │ Metrics  │  │Predictive│  │ Anomaly  │      │          │     │
│  │  │  │Collector │  │  System  │  │ Detector │      │          │     │
│  │  │  └──────────┘  └──────────┘  └──────────┘      │          │     │
│  │  └─────────────────────────────────────────────────┘          │     │
│  │                                                                │     │
│  │  ┌─────────────────────────────────────────────────┐          │     │
│  │  │  AutoscalingIntegration                         │          │     │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │          │     │
│  │  │  │   HPA    │  │   VPA    │  │Predictive│      │          │     │
│  │  │  │  Manager │  │  Engine  │  │Autoscaler│      │          │     │
│  │  │  └──────────┘  └──────────┘  └──────────┘      │          │     │
│  │  └─────────────────────────────────────────────────┘          │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                  │                                       │
│                                  ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                   DATA ACCESS LAYER                            │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │     │
│  │  │    User      │  │    Server    │  │     Chat     │        │     │
│  │  │    Model     │  │    Model     │  │    Model     │        │     │
│  │  └──────────────┘  └──────────────┘  └──────────────┘        │     │
│  │                     SQLAlchemy ORM                             │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                  │                                       │
│                                  ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                SQLite Database (instance/ai4k8s.db)            │     │
│  └───────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow: Chat Query Processing

### Complete Chat Flow with LLM Integration

```
┌────────────────────────────────────────────────────────────────────────┐
│  User Input: "Show me pods with high CPU usage"                        │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Frontend (chat.html + app.js)                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  1. Capture user input from textarea                           │    │
│  │  2. Display user message in chat UI                            │    │
│  │  3. Show "AI is thinking..." animation                         │    │
│  │  4. POST /api/chat/<server_id>                                 │    │
│  │     Body: {message: "Show me pods with high CPU usage"}        │    │
│  └────────────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │ AJAX Request
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Flask Route: @app.route('/api/chat/<server_id>', methods=['POST'])    │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  1. Validate session (login_required decorator)                │    │
│  │  2. Get server from database (ownership check)                 │    │
│  │  3. Extract message from request JSON                          │    │
│  │  4. Create Chat record (user_message, timestamp)               │    │
│  │  5. Route to processor based on message type                   │    │
│  └────────────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  AIPoweredMCPKubernetesProcessor.process_query()                        │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Step 1: Check if direct kubectl command                       │    │
│  │  - Pattern match: "kubectl get pods"                           │    │
│  │  - If match: Execute directly via simple_kubectl_executor      │    │
│  │  - If not: Continue to LLM processing                          │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Step 2: LLM Provider Cascade                                  │    │
│  │                                                                 │    │
│  │  Try 1: Groq API                                               │    │
│  │  ┌───────────────────────────────────────────────────────┐    │    │
│  │  │ POST https://api.groq.com/openai/v1/chat/completions │    │    │
│  │  │ Model: llama-3.1-8b-instant                          │    │    │
│  │  │ Prompt: "Extract kubectl command to find high CPU pods" │  │    │
│  │  │ Timeout: 15s                                          │    │    │
│  │  │ Response: "kubectl top pods --all-namespaces --sort-by=cpu" │   │
│  │  └───────────────────────────────────────────────────────┘    │    │
│  │  ✅ Success → Use result                                       │    │
│  │  ❌ Failure → Try Anthropic                                    │    │
│  │                                                                 │    │
│  │  Try 2: Anthropic API (Fallback)                               │    │
│  │  ┌───────────────────────────────────────────────────────┐    │    │
│  │  │ POST https://api.anthropic.com/v1/messages           │    │    │
│  │  │ Model: claude-3-haiku-20240307                       │    │    │
│  │  │ Same prompt as Groq                                  │    │    │
│  │  └───────────────────────────────────────────────────────┘    │    │
│  │  ✅ Success → Use result                                       │    │
│  │  ❌ Failure → Try Regex                                        │    │
│  │                                                                 │    │
│  │  Try 3: Regex-based Extraction (Last Resort)                   │    │
│  │  ┌───────────────────────────────────────────────────────┐    │    │
│  │  │ Pattern: "pods? .* (high|cpu)"                       │    │    │
│  │  │ Fallback: "kubectl get pods --all-namespaces"        │    │    │
│  │  └───────────────────────────────────────────────────────┘    │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Step 3: MCP Tool Selection                                    │    │
│  │  - Analyze extracted command                                   │    │
│  │  - Determine MCP tool: "pods_top"                              │    │
│  │  - Build tool parameters: {namespace: "all", sort_by: "cpu"}   │    │
│  └────────────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MCP Client: call_mcp_tool()                                            │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  POST http://127.0.0.1:5002/mcp                                │    │
│  │  Body: {                                                        │    │
│  │    tool: "pods_top",                                            │    │
│  │    arguments: {namespace: "all", sort_by: "cpu"}                │    │
│  │  }                                                              │    │
│  └────────────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MCP Server: kubernetes_mcp_server.py                                   │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  1. Receive tool request                                       │    │
│  │  2. Validate tool name ("pods_top")                            │    │
│  │  3. Execute kubectl command via simple_kubectl_executor        │    │
│  │  4. Parse output into structured JSON                          │    │
│  └────────────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  kubectl Executor: simple_kubectl_executor.py                           │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Command: kubectl top pods --all-namespaces --sort-by=cpu      │    │
│  │  Execution: subprocess.run()                                    │    │
│  │  Output:                                                        │    │
│  │    NAMESPACE   NAME              CPU(cores)   MEMORY(bytes)    │    │
│  │    production  web-app-1         850m         2048Mi           │    │
│  │    production  web-app-2         720m         1900Mi           │    │
│  │    default     nginx-1           450m         512Mi            │    │
│  └────────────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MCP Server Response (JSON)                                             │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  {                                                              │    │
│  │    success: true,                                               │    │
│    content: [                                                      │    │
│  │      {namespace: "production", name: "web-app-1", cpu: 850},   │    │
│  │      {namespace: "production", name: "web-app-2", cpu: 720},   │    │
│  │      {namespace: "default", name: "nginx-1", cpu: 450}          │    │
│  │    ]                                                            │    │
│  │  }                                                              │    │
│  └────────────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  AI Processor: Post-Processing                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  1. Receive MCP result                                          │    │
│  │  2. Format for readability                                      │    │
│  │  3. Add AI insights (optional LLM call)                         │    │
│  │  4. Create human-friendly response:                             │    │
│  │                                                                  │    │
│  │     "I found 3 pods with notable CPU usage:                     │    │
│  │                                                                  │    │
│  │      ⚠️  production/web-app-1: 850m CPU (HIGH)                  │    │
│  │      ⚠️  production/web-app-2: 720m CPU (HIGH)                  │    │
│  │      ℹ️  default/nginx-1: 450m CPU (MODERATE)                   │    │
│  │                                                                  │    │
│  │      Recommendation: Consider scaling production/web-app-*"     │    │
│  └────────────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Flask Route: Update Database & Return                                  │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  1. Update Chat record:                                         │    │
│  │     - ai_response = formatted_response                          │    │
│  │     - mcp_tool_used = "pods_top"                                │    │
│  │     - processing_method = "groq"                                │    │
│  │     - mcp_success = True                                        │    │
│  │  2. Commit to database                                          │    │
│  │  3. Return JSON: {response: formatted_response}                 │    │
│  └────────────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Frontend: Display Response                                             │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  1. Receive AJAX response                                       │    │
│  │  2. Hide "AI thinking..." animation                             │    │
│  │  3. Append AI response to chat UI                               │    │
│  │  4. Scroll to bottom                                            │    │
│  │  5. Enable input field for next message                         │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow: Predictive Autoscaling

### Complete Autoscaling Workflow with LLM

```
┌────────────────────────────────────────────────────────────────────────┐
│  User Action: Enable Predictive Autoscaling for "nginx" deployment     │
│  Form Data: {deployment: "nginx", namespace: "default",                │
│              min_replicas: 2, max_replicas: 10}                        │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Frontend (autoscaling.html)                                            │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  POST /api/autoscaling/predictive/enable/<server_id>           │    │
│  │  Show progress bar (0%)                                         │    │
│  └────────────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Flask Route Handler                                                    │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  1. Generate job_id (UUID)                                      │    │
│  │  2. Store in recommendation_jobs: {status: "processing"}        │    │
│  │  3. Emit WebSocket: job_status(10%, "Initializing...")         │    │
│  │  4. Spawn background thread: process_recommendation_job()      │    │
│  │  5. Return immediately: {job_id: "...", status: "processing"}  │    │
│  └────────────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Background Thread (Async Job)                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 1: METRICS COLLECTION (10-30%)                          │    │
│  │  ────────────────────────────────────────                      │    │
│  │  Emit WebSocket: 10% "Collecting current metrics..."           │    │
│  │  ┌──────────────────────────────────────────────────────┐      │    │
│  │  │ KubernetesMetricsCollector.get_aggregated_metrics()  │      │    │
│  │  │  kubectl top pods -n default --no-headers            │      │    │
│  │  │  Output: nginx-1  150m  512Mi                        │      │    │
│  │  │          nginx-2  160m  520Mi                        │      │    │
│  │  │          nginx-3  145m  510Mi                        │      │    │
│  │  │                                                       │      │    │
│  │  │  Aggregation:                                         │      │    │
│  │  │    Average CPU: 151.67m → 15.17%                     │      │    │
│  │  │    Average Memory: 514Mi → 51.4%                     │      │    │
│  │  │    Pod Count: 3                                       │      │    │
│  │  └──────────────────────────────────────────────────────┘      │    │
│  │  Emit WebSocket: 30% "Metrics collected"                       │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 2: FORECASTING (30-60%)                                 │    │
│  │  ────────────────────────────────────────                      │    │
│  │  Emit WebSocket: 40% "Generating 6-hour forecast..."           │    │
│  │  ┌──────────────────────────────────────────────────────┐      │    │
│  │  │ TimeSeriesForecaster.forecast(metric="cpu", hours=6) │      │    │
│  │  │                                                       │      │    │
│  │  │  Historical Data (last 24h at 5-min intervals):      │      │    │
│  │  │    [12%, 13%, 14%, ..., 15.17%] (288 samples)        │      │    │
│  │  │                                                       │      │    │
│  │  │  Algorithm: Linear Regression + Seasonal              │      │    │
│  │  │    Trend: y = 0.05x + 10 (increasing)                │      │    │
│  │  │    Seasonal: +2% during peak hours                   │      │    │
│  │  │                                                       │      │    │
│  │  │  Predictions (next 6 hours):                         │      │    │
│  │  │    Hour 1: 16.2% (CI: 15.5-16.9%)                   │      │    │
│  │  │    Hour 2: 17.5% (CI: 16.7-18.3%)                   │      │    │
│  │  │    Hour 3: 18.8% (CI: 17.9-19.7%)                   │      │    │
│  │  │    Hour 4: 20.1% (CI: 19.1-21.1%)                   │      │    │
│  │  │    Hour 5: 21.4% (CI: 20.3-22.5%)                   │      │    │
│  │  │    Hour 6: 22.7% (CI: 21.5-23.9%)                   │      │    │
│  │  │                                                       │      │    │
│  │  │  Trend: "increasing"                                  │      │    │
│  │  └──────────────────────────────────────────────────────┘      │    │
│  │  Emit WebSocket: 60% "Forecast generated"                      │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 3: LLM ANALYSIS (60-90%)                                │    │
│  │  ────────────────────────────────────────                      │    │
│  │  Emit WebSocket: 70% "AI analyzing patterns and trends..."     │    │
│  │  ┌──────────────────────────────────────────────────────┐      │    │
│  │  │ LLMAutoscalingAdvisor.analyze_scaling_decision()     │      │    │
│  │  │                                                       │      │    │
│  │  │  Step 1: Cache Key Generation                        │      │    │
│  │  │    cpu_rounded = 15% (rounded to 25% bucket → 0%)   │      │    │
│  │  │    memory_rounded = 50% (rounded to 5% bucket)       │      │    │
│  │  │    trend = "increasing"                              │      │    │
│  │  │    cache_key = "nginx_default_0_50_increasing"       │      │    │
│  │  │                                                       │      │    │
│  │  │  Step 2: Cache Lookup                                │      │    │
│  │  │    Result: MISS (no cached decision)                 │      │    │
│  │  │                                                       │      │    │
│  │  │  Step 3: Rate Limit Check                            │      │    │
│  │  │    Last call for nginx: 2 minutes ago                │      │    │
│  │  │    Minimum interval: 30 seconds                      │      │    │
│  │  │    Status: OK, proceed with LLM call                 │      │    │
│  │  │                                                       │      │    │
│  │  │  Step 4: Context Assembly                            │      │    │
│  │  │  ┌─────────────────────────────────────────────┐    │      │    │
│  │  │  │ {                                            │    │      │    │
│  │  │  │   "deployment_name": "nginx",               │    │      │    │
│  │  │  │   "namespace": "default",                   │    │      │    │
│  │  │  │   "current_metrics": {                      │    │      │    │
│  │  │  │     "cpu_percent": 15.17,                   │    │      │    │
│  │  │  │     "memory_percent": 51.4,                 │    │      │    │
│  │  │  │     "pod_count": 3                          │    │      │    │
│  │  │  │   },                                         │    │      │    │
│  │  │  │   "forecast": {                             │    │      │    │
│  │  │  │     "cpu_trend": "increasing",              │    │      │    │
│  │  │  │     "predicted_cpu": [16.2, 17.5, ..., 22.7],│  │      │    │
│  │  │  │     "memory_trend": "stable",               │    │      │    │
│  │  │  │     "predicted_memory": [52, 52, ...]       │    │      │    │
│  │  │  │   },                                         │    │      │    │
│  │  │  │   "current_replicas": 3,                    │    │      │    │
│  │  │  │   "min_replicas": 2,                        │    │      │    │
│  │  │  │   "max_replicas": 10,                       │    │      │    │
│  │  │  │   "has_state_management": false             │    │      │    │
│  │  │  │ }                                            │    │      │    │
│  │  │  └─────────────────────────────────────────────┘    │      │    │
│  │  │                                                       │      │    │
│  │  │  Step 5: LLM Provider Selection                      │      │    │
│  │  │  ┌─────────────────────────────────────────────┐    │      │    │
│  │  │  │  Try GPT OSS (Qwen):                        │    │      │    │
│  │  │  │    URL: http://localhost:8001/v1/chat/completions│  │    │
│  │  │  │    Model: gpt-4                             │    │      │    │
│  │  │  │    Temperature: 0.1                         │    │      │    │
│  │  │  │    Timeout: 240s                            │    │      │    │
│  │  │  │                                              │    │      │    │
│  │  │  │    Prompt:                                   │    │      │    │
│  │  │  │    "You are an expert Kubernetes autoscaling │   │      │    │
│  │  │  │     advisor. Analyze the following metrics...│   │      │    │
│  │  │  │     Recommend: scale_up, scale_down, maintain"  │      │    │
│  │  │  │                                              │    │      │    │
│  │  │  │    ✅ Success (120s response time)          │    │      │    │
│  │  │  └─────────────────────────────────────────────┘    │      │    │
│  │  │                                                       │      │    │
│  │  │  Step 6: LLM Response                                │      │    │
│  │  │  ┌─────────────────────────────────────────────┐    │      │    │
│  │  │  │ {                                            │    │      │    │
│  │  │  │   "action": "scale_up",                     │    │      │    │
│  │  │  │   "target_replicas": 5,                     │    │      │    │
│  │  │  │   "confidence": 0.85,                       │    │      │    │
│  │  │  │   "reasoning": "CPU trend shows steady      │    │      │    │
│  │  │  │     increase. Forecast predicts 22.7% in 6h.│    │      │    │
│  │  │  │     Proactive scaling to 5 replicas will    │    │      │    │
│  │  │  │     distribute load and maintain headroom.",│    │      │    │
│  │  │  │   "risk_assessment": "low",                 │    │      │    │
│  │  │  │   "cost_impact": "moderate increase",       │    │      │    │
│  │  │  │   "recommended_timing": "immediate"         │    │      │    │
│  │  │  │ }                                            │    │      │    │
│  │  │  └─────────────────────────────────────────────┘    │      │    │
│  │  │                                                       │      │    │
│  │  │  Step 7: Cache Storage                               │      │    │
│  │  │    Store decision with 5-min TTL                     │      │    │
│  │  └──────────────────────────────────────────────────────┘      │    │
│  │  Emit WebSocket: 90% "Scaling decision made"                   │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 4: SCALING EXECUTION (90-100%)                          │    │
│  │  ────────────────────────────────────────                      │    │
│  │  Emit WebSocket: 95% "Applying scaling changes..."             │    │
│  │  ┌──────────────────────────────────────────────────────┐      │    │
│  │  │ kubectl scale deployment nginx --replicas=5          │      │    │
│  │  │ kubectl annotate deployment nginx                    │      │    │
│  │  │   ai4k8s.io/predictive-autoscaling-enabled=true      │      │    │
│  │  │   ai4k8s.io/predictive-autoscaling-config='{"min":2,"max":10}'│ │
│  │  └──────────────────────────────────────────────────────┘      │    │
│  │                                                                 │    │
│  │  Update Job Store:                                              │    │
│  │    recommendation_jobs[job_id] = {                              │    │
│  │      status: "completed",                                       │    │
│  │      result: {...LLM decision...},                              │    │
│  │      updated_at: now()                                          │    │
│  │    }                                                            │    │
│  │                                                                 │    │
│  │  Emit WebSocket: 100% "Autoscaling enabled successfully!"      │    │
│  └────────────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Frontend: Display Result                                               │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  1. WebSocket updates progress bar in real-time                │    │
│  │  2. Final result displayed:                                     │    │
│  │     "✅ Predictive autoscaling enabled for nginx               │    │
│  │      Scaled to 5 replicas (from 3)                             │    │
│  │      Confidence: 85%                                            │    │
│  │      Reasoning: [LLM reasoning...]"                             │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  BACKGROUND LOOP: Periodic Re-evaluation (Every 5 Minutes)              │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  PredictiveAutoscaler._scaling_loop()                           │    │
│  │  ┌──────────────────────────────────────────────────────┐      │    │
│  │  │  1. Find deployments with enabled annotation         │      │    │
│  │  │  2. For each deployment:                              │      │    │
│  │  │     - Collect current metrics                         │      │    │
│  │  │     - Generate new forecast                           │      │    │
│  │  │     - Call LLM (with caching!)                        │      │    │
│  │  │     - Apply scaling if recommended                    │      │    │
│  │  │  3. Sleep 5 minutes                                   │      │    │
│  │  │  4. Repeat indefinitely                               │      │    │
│  │  └──────────────────────────────────────────────────────┘      │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Monitoring System Architecture

### Monitoring Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MONITORING SYSTEM ARCHITECTURE                        │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │              DATA COLLECTION LAYER                             │     │
│  │                                                                │     │
│  │  ┌────────────────────────────────────────────────────┐       │     │
│  │  │  KubernetesMetricsCollector                        │       │     │
│  │  │  ┌──────────────────────────────────────────┐     │       │     │
│  │  │  │  kubectl top nodes                       │     │       │     │
│  │  │  │  kubectl top pods --all-namespaces       │     │       │     │
│  │  │  │  kubectl get nodes -o json               │     │       │     │
│  │  │  └──────────────────────────────────────────┘     │       │     │
│  │  │                                                    │       │     │
│  │  │  Output: ResourceMetrics {                        │       │     │
│  │  │    cpu: 65.5%,                                    │       │     │
│  │  │    memory: 48.2%,                                 │       │     │
│  │  │    network_io: 1.2 MB/s,                          │       │     │
│  │  │    disk_io: 0.8 MB/s,                             │       │     │
│  │  │    pod_count: 42,                                 │       │     │
│  │  │    node_count: 3,                                 │       │     │
│  │  │    timestamp: datetime.now()                      │       │     │
│  │  │  }                                                 │       │     │
│  │  └────────────────────────────────────────────────────┘       │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                 │                                        │
│                                 ▼                                        │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │              DATA STORAGE LAYER                                │     │
│  │                                                                │     │
│  │  ┌────────────────────────────────────────────��───────┐       │     │
│  │  │  History Buffer (deque)                            │       │     │
│  │  │  ┌──────────────────────────────────────────┐     │       │     │
│  │  │  │  Max Length: 288 samples                 │     │       │     │
│  │  │  │  Interval: 5 minutes                     │     │       │     │
│  │  │  │  Coverage: 24 hours                      │     │       │     │
│  │  │  │                                           │     │       │     │
│  │  │  │  [Sample 1, Sample 2, ..., Sample 288]   │     │       │     │
│  │  │  │   ↑                                ↑      │     │       │     │
│  │  │  │   24h ago                      now       │     │       │     │
│  │  │  └──────────────────────────────────────────┘     │       │     │
│  │  └────────────────────────────────────────────────────┘       │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                 │                                        │
│                                 ▼                                        │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │           PREDICTIVE ANALYTICS LAYER                           │     │
│  │                                                                │     │
│  │  ┌─────────────────────────────────────────────────────┐      │     │
│  │  │  1. TimeSeriesForecaster                            │      │     │
│  │  │  ┌───────────────────────────────────────────┐     │      │     │
│  │  │  │  Input: Last 24h data                     │     │      │     │
│  │  │  │  Algorithm: Linear Regression + Seasonal  │     │      │     │
│  │  │  │  Output: 6-hour predictions with CI       │     │      │     │
│  │  │  └───────────────────────────────────────────┘     │      │     │
│  │  └─────────────────────────────────────────────────────┘      │     │
│  │                                                                │     │
│  │  ┌─────────────────────────────────────────────────────┐      │     │
│  │  │  2. AnomalyDetector                                 │      │     │
│  │  │  ┌───────────────────────────────────────────┐     │      │     │
│  │  │  │  Algorithm: Isolation Forest (scikit)     │     │      │     │
│  │  │  │  Training: Requires ≥20 samples           │     │      │     │
│  │  │  │  Output: {is_anomaly, score, severity}    │     │      │     │
│  │  │  └───────────────────────────────────────────┘     │      │     │
│  │  └─────────────────────────────────────────────────────┘      │     │
│  │                                                                │     │
│  │  ┌─────────────────────────────────────────────────────┐      │     │
│  │  │  3. PerformanceOptimizer                            │      │     │
│  │  │  ┌───────────────────────────────────────────┐     │      │     │
│  │  │  │  Analyzes utilization patterns            │     │      │     │
│  │  │  │  Generates optimization recommendations   │     │      │     │
│  │  │  └───────────────────────────────────────────┘     │      │     │
│  │  └─────────────────────────────────────────────────────┘      │     │
│  │                                                                │     │
│  │  ┌─────────────────────────────────────────────────────┐      │     │
│  │  │  4. CapacityPlanner                                 │      │     │
│  │  │  ┌───────────────────────────────────────────┐     │      │     │
│  │  │  │  Uses forecasts for capacity planning     │     │      │     │
│  │  │  │  Recommends resource allocation           │     │      │     │
│  │  │  └───────────────────────────────────────────┘     │      │     │
│  │  └─────────────────────────────────────────────────────┘      │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                 │                                        │
│                                 ▼                                        │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │              OUTPUT LAYER                                      │     │
│  │                                                                │     │
│  │  Analysis JSON:                                                │     │
│  │  {                                                             │     │
│  │    current_metrics: {...},                                     │     │
│  │    forecasts: {                                                │     │
│  │      cpu: {predicted: [...], trend: "increasing"},            │     │
│  │      memory: {predicted: [...], trend: "stable"}              │     │
│  │    },                                                          │     │
│  │    anomaly_detection: {                                        │     │
│  │      is_anomaly: true,                                         │     │
│  │      score: 0.87,                                              │     │
│  │      severity: "high"                                          │     │
│  │    },                                                          │     │
│  │    performance_optimization: {                                 │     │
│  │      recommendations: ["Scale up CPU-bound pods"]             │     │
│  │    },                                                          │     │
│  │    capacity_planning: {                                        │     │
│  │      recommendations: ["Add node in 4 hours"]                 │     │
│  │    }                                                           │     │
│  │  }                                                             │     │
│  └───────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Component Interaction Map

### How All Components Work Together

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      COMPONENT INTERACTION MAP                           │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      USER INTERACTIONS                           │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │   │
│  │  │  Login   │  │   Add    │  │   Chat   │  │ Monitor  │        │   │
│  │  │  /Auth   │  │  Server  │  │  /Query  │  │ /Autoscale│       │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │   │
│  └───────┼─────────────┼─────────────┼─────────────┼──────────────────┘
│          │             │             │             │
│          ▼             ▼             ▼             ▼
│  ┌──────────────────────────────────────────────────────────────────┐
│  │                     FLASK ROUTES                                  │
│  │  /login           /add_server     /api/chat      /api/monitoring │
│  │  /register        /server/<id>    /api/autoscaling               │
│  └──────────┬──────────────┬──────────────┬──────────────┬──────────┘
│             │              │              │              │
│  ┌──────────▼────────┐     │     ┌────────▼──────┐      │
│  │                   │     │     │               │      │
│  │   User Model      │◀────┘     │  Server Model │◀─────┘
│  │   (Database)      │           │  (Database)   │
│  │                   │           │               │
│  └───────────────────┘           └───┬───────────┘
│                                      │
│                                      ▼
│  ┌─────────────────────────────────────────────────────────────────┐
│  │              BUSINESS LOGIC ORCHESTRATORS                        │
│  └─────────────────────────────────────────────────────────────────┘
│                   │                  │                  │
│      ┌────────────▼────┐  ┌──────────▼─────┐  ┌────────▼──────────┐
│      │                 │  │                │  │                   │
│      │  AI Processor   │  │   Monitoring   │  │   Autoscaling    │
│      │                 │  │   Integration  │  │   Integration    │
│      └────────┬────────┘  └────────┬───────┘  └────────┬─────────┘
│               │                    │                    │
│               ▼                    ▼                    ▼
│  ┌────────────────────────────────────────────────────────────────┐
│  │                  CORE SERVICES                                  │
│  ├─────────────────┬──────────────────┬───────────────────────────┤
│  │                 │                  │                           │
│  │  ┌──────────┐   │  ┌──────────┐   │  ┌──────────┐            │
│  │  │   MCP    │   │  │  Metrics │   │  │   LLM    │            │
│  │  │  Client  │   │  │Collector │   │  │ Advisor  │            │
│  │  └────┬─────┘   │  └────┬─────┘   │  └────┬─────┘            │
│  │       │         │       │         │       │                   │
│  │       ▼         │       ▼         │       ▼                   │
│  │  ┌──────────┐   │  ┌──────────┐   │  ┌──────────┐            │
│  │  │   MCP    │   │  │ kubectl  │   │  │Qwen/Groq │            │
│  │  │  Server  │   │  │   top    │   │  │   APIs   │            │
│  │  └────┬─────┘   │  └────┬─────┘   │  └──────────┘            │
│  │       │         │       │         │                           │
│  │       ▼         │       ▼         │                           │
│  │  ┌──────────┐   │  ┌──────────┐   │  ┌──────────┐            │
│  │  │ kubectl  │   │  │K8s API   │   │  │Predictive│            │
│  │  │ executor │   │  │ /Metrics │   │  │Autoscaler│            │
│  │  └──────────┘   │  └──────────┘   │  └──────────┘            │
│  └─────────────────┴──────────────────┴───────────────────────────┘
│                              │
│                              ▼
│  ┌─────────────────────────────────────────────────────────────────┐
│  │              KUBERNETES CLUSTER                                  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  │   Pods   │  │  Nodes   │  │   HPA    │  │   VPA    │        │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│  └─────────────────────────────────────────────────────────────────┘
└──────────────────────────────────────────────────────────────────────┘

LEGEND:
───▶  Synchronous request/response
····▶ Asynchronous/background processing
◀──▶  Bidirectional communication
```

---

**End of Architecture & Flows Document**

*This document provides detailed technical architecture diagrams and data flow visualizations for the AI4K8s platform.*
