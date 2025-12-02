# AI4K8s Autoscaling Strategy Report
## Final Thesis Implementation Path

**Author:** Pedram Nikjooy  
**Date:** December 2024  
**Project:** AI-Powered Kubernetes Management Platform  
**Status:** Live Production System on HPC Cluster

---

## Executive Summary

This report analyzes seven autoscaling approaches for the AI4K8s platform and recommends the optimal implementation strategy for the final thesis phase. Based on the current system's capabilities—predictive monitoring, ML-based anomaly detection, real-time metrics collection, and AI-powered recommendations—we recommend a **hybrid approach combining Predictive, Reactive, and Event-driven autoscaling** as the core implementation.

---

## Current System Capabilities

### ✅ Existing Infrastructure

1. **Predictive Monitoring System** (`predictive_monitoring.py`)
   - Time series forecasting (6-hour CPU/memory predictions)
   - ML-based anomaly detection (Isolation Forest, DBSCAN)
   - Capacity planning with scaling recommendations
   - Performance optimization analysis

2. **Real-time Metrics Collection** (`k8s_metrics_collector.py`)
   - Live CPU, memory, network, disk I/O metrics
   - Pod and node count tracking
   - Aggregated cluster-wide statistics

3. **AI-Powered Recommendations** (`ai_monitoring_integration.py`)
   - RAG-enhanced recommendations (Kubernetes best practices)
   - LLM-generated guidance (Groq integration)
   - Performance optimization suggestions
   - Capacity planning recommendations

4. **Kubernetes Integration** (`kubernetes_mcp_server.py`)
   - Direct kubectl command execution
   - MCP protocol for AI-tool communication
   - Real-time cluster state access

### ❌ Missing Components

- **No actual autoscaling execution** (only recommendations)
- **No HPA creation/management** (templates exist but not implemented)
- **No VPA (Vertical Pod Autoscaler)** implementation
- **No cluster/node autoscaling**
- **No scheduled scaling**
- **No event-driven scaling triggers**

---

## Autoscaling Approaches Analysis

### 1. Horizontal Pod Autoscaling (HPA) ⭐⭐⭐⭐⭐

**What it is:** Scales the number of pod replicas based on CPU, memory, or custom metrics.

**Current Status:** 
- ✅ HPA templates exist in RAG knowledge base
- ✅ Scaling recommendations generated
- ❌ No actual HPA creation/management

**Implementation Complexity:** Low-Medium  
**Thesis Value:** High (Industry Standard)

**Pros:**
- Industry-standard Kubernetes feature
- Well-documented and widely used
- Integrates with existing metrics server
- Supports CPU, memory, and custom metrics
- Native Kubernetes resource (no external dependencies)

**Cons:**
- Reactive (responds to current load, not future)
- Can cause "thrashing" if thresholds are too sensitive
- Requires metrics server to be running

**Recommendation:** ✅ **IMPLEMENT** - Essential baseline for any autoscaling system

**Implementation Plan:**
```python
# New module: autoscaling_engine.py
class HorizontalPodAutoscaler:
    def create_hpa(self, deployment_name, namespace, 
                   min_replicas, max_replicas, 
                   cpu_target, memory_target):
        """Create HPA resource via kubectl"""
        
    def update_hpa(self, hpa_name, namespace, new_targets):
        """Update existing HPA thresholds"""
        
    def delete_hpa(self, hpa_name, namespace):
        """Remove HPA resource"""
        
    def get_hpa_status(self, hpa_name, namespace):
        """Get current HPA metrics and replica count"""
```

**Integration Points:**
- Use existing `CapacityPlanner` recommendations to auto-create HPAs
- Integrate with monitoring dashboard to show HPA status
- Add "Enable Autoscaling" button in UI for deployments

---

### 2. Vertical Pod Autoscaling (VPA) ⭐⭐⭐

**What it is:** Automatically adjusts CPU and memory requests/limits for pods based on historical usage.

**Current Status:** 
- ❌ Not implemented
- ❌ No VPA knowledge in RAG system

**Implementation Complexity:** Medium-High  
**Thesis Value:** Medium (Less common, but valuable for cost optimization)

**Pros:**
- Optimizes resource allocation (reduces waste)
- Cost-effective (right-sizing containers)
- Works well with HPA (complementary)
- Reduces OOM kills and CPU throttling

**Cons:**
- Requires VPA admission controller (additional component)
- Can conflict with HPA if both modify same resources
- More complex to implement and debug
- Less commonly used in production

**Recommendation:** ⚠️ **CONSIDER** - Implement if time permits, but not critical for thesis

**Implementation Plan:**
- Install VPA components (recommender, updater, admission controller)
- Create VPA resources based on historical pod usage patterns
- Integrate with existing metrics history for recommendations

---

### 3. Scheduled Autoscaling (CronHPA/KEDA CronScaler) ⭐⭐⭐⭐

**What it is:** Scales resources based on time-based schedules (e.g., scale up during business hours, down at night).

**Current Status:**
- ❌ Not implemented
- ✅ Time series forecasting exists (can predict daily patterns)

**Implementation Complexity:** Low-Medium  
**Thesis Value:** High (Demonstrates intelligent scheduling)

**Pros:**
- Predictable cost optimization
- Proactive scaling (before load arrives)
- Easy to understand and demonstrate
- Can use existing forecasting data

**Cons:**
- Requires knowledge of workload patterns
- Less flexible than reactive scaling
- May over/under-provision if patterns change

**Recommendation:** ✅ **IMPLEMENT** - High value, leverages existing forecasting

**Implementation Plan:**
```python
# New module: scheduled_autoscaling.py
class ScheduledAutoscaler:
    def create_schedule(self, deployment_name, namespace, 
                       schedule_rules):
        """Create CronHPA or KEDA CronScaler"""
        # schedule_rules: [{"time": "0 9 * * 1-5", "replicas": 5}, ...]
        
    def analyze_historical_patterns(self, metrics_history):
        """Use existing TimeSeriesForecaster to detect patterns"""
        # Leverage existing forecasting to suggest schedules
```

**Integration Points:**
- Use `TimeSeriesForecaster` to detect daily/weekly patterns
- Auto-generate schedules from forecasting data
- Show schedule visualization in monitoring dashboard

---

### 4. Reactive Autoscaling ⭐⭐⭐⭐⭐

**What it is:** Scales immediately in response to current metrics (CPU > 80%, memory > 85%, etc.).

**Current Status:**
- ✅ Threshold-based recommendations exist
- ✅ Real-time metrics collection active
- ❌ No automatic execution

**Implementation Complexity:** Low  
**Thesis Value:** High (Core functionality)

**Pros:**
- Fast response to load spikes
- Simple to implement and understand
- Works with existing HPA
- Immediate value for users

**Cons:**
- Always "behind" the load (reactive, not proactive)
- Can cause thrashing with rapid changes
- May not prevent overload (only responds after it happens)

**Recommendation:** ✅ **IMPLEMENT** - Core baseline, works with HPA

**Implementation Plan:**
- Enhance existing `CapacityPlanner` to auto-execute scaling
- Add "Auto-apply recommendations" toggle in UI
- Create HPA resources automatically when thresholds are met

---

### 5. Predictive Autoscaling ⭐⭐⭐⭐⭐ **RECOMMENDED PRIMARY FOCUS**

**What it is:** Uses ML forecasting to predict future load and scale proactively before demand arrives.

**Current Status:**
- ✅ Time series forecasting implemented (`TimeSeriesForecaster`)
- ✅ 6-hour CPU/memory predictions available
- ✅ Trend analysis (increasing/decreasing/stable)
- ❌ No scaling execution based on predictions

**Implementation Complexity:** Medium  
**Thesis Value:** ⭐⭐⭐⭐⭐ **HIGHEST** - Most innovative, leverages AI/ML

**Pros:**
- **Most innovative** - Demonstrates AI/ML capabilities
- Proactive (scales before load arrives)
- Reduces latency spikes and user impact
- Leverages existing forecasting infrastructure
- **Unique differentiator** for thesis
- Aligns with AI4K8s brand (AI-powered)

**Cons:**
- Requires accurate forecasting (already implemented)
- May over-provision if predictions are wrong
- More complex than reactive scaling

**Recommendation:** ✅✅✅ **PRIMARY FOCUS** - This is the thesis differentiator!

**Implementation Plan:**
```python
# New module: predictive_autoscaler.py
class PredictiveAutoscaler:
    def __init__(self, monitoring_system):
        self.forecaster = monitoring_system.forecaster
        self.metrics_collector = monitoring_system.metrics_collector
        
    def predict_and_scale(self, deployment_name, namespace):
        """Predict future load and scale proactively"""
        # 1. Get 6-hour forecast
        cpu_forecast = self.forecaster.forecast_cpu_usage(hours_ahead=6)
        memory_forecast = self.forecaster.forecast_memory_usage(hours_ahead=6)
        
        # 2. Determine if scaling is needed
        max_predicted_cpu = max(cpu_forecast.predicted_values)
        max_predicted_memory = max(memory_forecast.predicted_values)
        
        # 3. Calculate required replicas
        if max_predicted_cpu > 75 or max_predicted_memory > 80:
            current_replicas = self.get_current_replicas(deployment_name, namespace)
            required_replicas = self.calculate_required_replicas(
                current_replicas, max_predicted_cpu, max_predicted_memory
            )
            
            # 4. Scale proactively (with safety bounds)
            if required_replicas > current_replicas:
                self.scale_up(deployment_name, namespace, required_replicas)
        
    def calculate_required_replicas(self, current, cpu_pred, mem_pred):
        """Calculate replicas needed based on predictions"""
        # Scale factor = predicted_usage / target_usage (70% CPU, 80% memory)
        cpu_scale = cpu_pred / 70.0
        mem_scale = mem_pred / 80.0
        scale_factor = max(cpu_scale, mem_scale)
        return int(current * scale_factor * 1.2)  # 20% buffer
```

**Integration Points:**
- Use existing `TimeSeriesForecaster.forecast_cpu_usage()` and `forecast_memory_usage()`
- Integrate with `CapacityPlanner` for recommendations
- Add "Predictive Autoscaling" toggle in monitoring dashboard
- Show prediction timeline visualization

**Thesis Value Proposition:**
- Demonstrates **real AI/ML application** (not just recommendations)
- Shows **proactive vs reactive** comparison
- Can measure **effectiveness** (latency reduction, cost savings)
- **Unique contribution** to Kubernetes autoscaling research

---

### 6. Event-driven Autoscaling ⭐⭐⭐⭐

**What it is:** Scales based on events (Kubernetes events, custom metrics, webhooks, message queue depth, etc.).

**Current Status:**
- ✅ Anomaly detection implemented (`AnomalyDetector`)
- ✅ Event monitoring exists (`events_list` MCP tool)
- ❌ No event-based scaling triggers

**Implementation Complexity:** Medium  
**Thesis Value:** High (Demonstrates intelligent event handling)

**Pros:**
- Responds to business events (not just metrics)
- Can integrate with anomaly detection
- Flexible (supports custom triggers)
- Works well with KEDA (Kubernetes Event-Driven Autoscaling)

**Cons:**
- Requires event source setup
- More complex than metric-based scaling
- May need external components (KEDA, message queues)

**Recommendation:** ✅ **IMPLEMENT** - High value, integrates with anomaly detection

**Implementation Plan:**
```python
# New module: event_driven_autoscaler.py
class EventDrivenAutoscaler:
    def __init__(self, anomaly_detector):
        self.anomaly_detector = anomaly_detector
        
    def on_anomaly_detected(self, anomaly_result):
        """Scale based on anomaly detection"""
        if anomaly_result.severity == "critical":
            # Scale up immediately
            self.emergency_scale_up(anomaly_result.affected_metrics)
        elif anomaly_result.severity == "high":
            # Scale up moderately
            self.scale_up(anomaly_result.affected_metrics)
            
    def on_kubernetes_event(self, event):
        """Scale based on Kubernetes events"""
        # e.g., "FailedScheduling" -> scale up nodes
        # e.g., "OOMKilled" -> scale up memory
        if event.reason == "FailedScheduling" and "Too many pods" in event.message:
            # Trigger cluster autoscaling or scale down other workloads
            pass
```

**Integration Points:**
- Use existing `AnomalyDetector.detect_anomaly()` results
- Monitor Kubernetes events via existing `events_list` tool
- Create KEDA ScaledObjects for custom metrics
- Add event timeline in monitoring dashboard

---

### 7. Serverless Autoscaling (Knative/KEDA) ⭐⭐

**What it is:** Scales to zero when no traffic, scales up instantly when requests arrive (serverless model).

**Current Status:**
- ❌ Not implemented
- ❌ No serverless infrastructure

**Implementation Complexity:** High  
**Thesis Value:** Low (Requires significant infrastructure changes)

**Pros:**
- Cost-effective (pay only for active usage)
- True serverless experience
- Scales to zero automatically

**Cons:**
- Requires Knative or similar platform
- Cold start latency issues
- Not suitable for all workloads
- Significant infrastructure changes needed
- Less relevant for current thesis focus

**Recommendation:** ❌ **SKIP** - Too complex, not aligned with current system

---

## Recommended Implementation Strategy

### Phase 1: Foundation (Weeks 1-2) ⭐⭐⭐⭐⭐

**Priority: CRITICAL**

1. **Horizontal Pod Autoscaling (HPA)**
   - Implement HPA creation/management
   - Integrate with existing recommendations
   - Add UI controls for enabling autoscaling
   - **Why:** Industry standard, essential baseline

2. **Reactive Autoscaling**
   - Auto-execute scaling based on current metrics
   - Integrate with `CapacityPlanner` recommendations
   - Add safety bounds and rate limiting
   - **Why:** Core functionality, works with HPA

**Deliverables:**
- `autoscaling_engine.py` module
- HPA management functions
- UI integration in monitoring dashboard
- Documentation and testing

---

### Phase 2: Innovation (Weeks 3-4) ⭐⭐⭐⭐⭐ **THESIS FOCUS**

**Priority: HIGHEST - This is your differentiator!**

1. **Predictive Autoscaling** ⭐⭐⭐⭐⭐
   - Implement proactive scaling based on ML forecasts
   - Use existing `TimeSeriesForecaster` predictions
   - Create prediction-based scaling algorithm
   - Add comparison metrics (predictive vs reactive)
   - **Why:** Most innovative, leverages AI/ML, unique thesis contribution

2. **Scheduled Autoscaling**
   - Implement time-based scaling
   - Use forecasting to suggest schedules
   - Create schedule visualization
   - **Why:** High value, leverages existing forecasting

**Deliverables:**
- `predictive_autoscaler.py` module
- `scheduled_autoscaler.py` module
- Prediction timeline visualization
- Performance comparison dashboard
- Research metrics (latency reduction, cost savings)

---

### Phase 3: Enhancement (Weeks 5-6) ⭐⭐⭐⭐

**Priority: MEDIUM**

1. **Event-driven Autoscaling**
   - Integrate with anomaly detection
   - Create event-based scaling triggers
   - Add Kubernetes event monitoring
   - **Why:** High value, integrates with existing anomaly detection

2. **Vertical Pod Autoscaling (VPA)** (Optional)
   - Implement if time permits
   - Focus on cost optimization use cases
   - **Why:** Less critical, but adds value

**Deliverables:**
- `event_driven_autoscaler.py` module
- Anomaly-triggered scaling
- Event monitoring dashboard
- VPA implementation (if time permits)

---

## Why This Strategy?

### 1. **Predictive Autoscaling is Your Thesis Differentiator** ⭐⭐⭐⭐⭐

**Current State:**
- Most Kubernetes autoscaling is **reactive** (HPA, VPA)
- Your system already has **predictive capabilities** (forecasting)
- **No one else is doing AI-powered predictive autoscaling** in your context

**Thesis Value:**
- Demonstrates **real AI/ML application** (not just monitoring)
- Shows **measurable improvements** (latency reduction, cost savings)
- **Unique contribution** to Kubernetes autoscaling research
- Aligns perfectly with "AI-Powered Kubernetes Management" brand

**Research Questions You Can Answer:**
- How much does predictive scaling reduce latency vs reactive?
- What's the cost difference between predictive and reactive?
- How accurate are ML predictions for autoscaling decisions?
- What's the optimal prediction horizon (1h, 3h, 6h)?

### 2. **Leverages Existing Infrastructure** ⭐⭐⭐⭐⭐

**You Already Have:**
- ✅ Time series forecasting (`TimeSeriesForecaster`)
- ✅ 6-hour CPU/memory predictions
- ✅ Trend analysis (increasing/decreasing/stable)
- ✅ Real-time metrics collection
- ✅ ML models (Isolation Forest, DBSCAN)

**You Just Need:**
- Execute scaling actions based on predictions
- Create HPA resources programmatically
- Add UI controls and visualizations

**Implementation Effort:** Medium (2-3 weeks)  
**Thesis Impact:** High (unique contribution)

### 3. **Comprehensive but Focused** ⭐⭐⭐⭐

**Core Implementation (Phases 1-2):**
- HPA (industry standard)
- Reactive (baseline)
- Predictive (innovation) ⭐
- Scheduled (intelligent scheduling)

**This gives you:**
- ✅ Industry-standard baseline (HPA)
- ✅ Innovative AI/ML contribution (Predictive)
- ✅ Practical scheduling (Scheduled)
- ✅ Complete autoscaling solution

**Total Implementation:** 4-6 weeks  
**Thesis Value:** Maximum

---

## Implementation Architecture

```
┌─────────────────────────────────────────────────────────┐
│              AI4K8s Autoscaling Engine                  │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Reactive   │   │  Predictive  │   │   Scheduled  │
│  Autoscaler  │   │  Autoscaler  │   │  Autoscaler   │
│              │   │              │   │              │
│ - HPA        │   │ - ML Forecast│   │ - CronHPA     │
│ - Thresholds │   │ - Proactive │   │ - Patterns    │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                ┌──────────────────────┐
                │  Kubernetes Cluster   │
                │  - HPA Resources      │
                │  - Deployments        │
                │  - Metrics Server     │
                └──────────────────────┘
                            │
                            ▼
                ┌──────────────────────┐
                │  Existing Components   │
                │  - TimeSeriesForecaster│
                │  - AnomalyDetector    │
                │  - MetricsCollector   │
                │  - CapacityPlanner    │
                └──────────────────────┘
```

---

## Success Metrics

### Technical Metrics

1. **Prediction Accuracy**
   - Forecast error rate (MAE, RMSE)
   - Prediction horizon effectiveness (1h vs 3h vs 6h)

2. **Scaling Performance**
   - Time to scale (predictive vs reactive)
   - Latency reduction during traffic spikes
   - Cost savings (over-provisioning reduction)

3. **System Reliability**
   - Scaling decision accuracy
   - False positive/negative rates
   - Thrashing prevention (stabilization)

### Business Metrics

1. **User Experience**
   - Request latency (p50, p95, p99)
   - Error rate reduction
   - Availability improvement

2. **Cost Optimization**
   - Resource utilization improvement
   - Over-provisioning reduction
   - Cost per request reduction

---

## Research Contributions

### 1. **AI-Powered Predictive Autoscaling**
- First implementation of ML-based predictive autoscaling in your context
- Comparison of predictive vs reactive approaches
- Optimal prediction horizon analysis

### 2. **Hybrid Autoscaling Strategy**
- Combination of reactive, predictive, and scheduled scaling
- Intelligent switching between strategies
- Cost-performance trade-off analysis

### 3. **Integration with Existing AI Infrastructure**
- Leveraging RAG for autoscaling best practices
- LLM-generated scaling recommendations
- Anomaly-driven emergency scaling

---

## Conclusion

### Recommended Focus: **Predictive Autoscaling** ⭐⭐⭐⭐⭐

**Why:**
1. **Most innovative** - Demonstrates real AI/ML application
2. **Leverages existing infrastructure** - You already have forecasting
3. **Unique thesis contribution** - No one else doing this in your context
4. **Measurable impact** - Can show latency reduction, cost savings
5. **Aligns with AI4K8s brand** - AI-powered Kubernetes management

### Implementation Priority:

1. **Phase 1 (Weeks 1-2):** HPA + Reactive ⭐⭐⭐⭐⭐
   - Essential baseline
   - Industry standard
   - Foundation for everything else

2. **Phase 2 (Weeks 3-4):** Predictive + Scheduled ⭐⭐⭐⭐⭐
   - **THESIS FOCUS** - Your innovation
   - Leverages existing ML infrastructure
   - Unique contribution

3. **Phase 3 (Weeks 5-6):** Event-driven + VPA (optional) ⭐⭐⭐⭐
   - Enhancement
   - Nice to have
   - If time permits

### Final Recommendation:

**Focus 70% of effort on Predictive Autoscaling** - This is your thesis differentiator. The combination of HPA (baseline) + Predictive (innovation) + Scheduled (intelligent) gives you a complete, innovative autoscaling solution that demonstrates real AI/ML value.

---

## Next Steps

1. **Review this report** with your advisor
2. **Create detailed implementation plan** for Phase 1 (HPA + Reactive)
3. **Design predictive autoscaling algorithm** (Phase 2)
4. **Set up testing environment** with load generators
5. **Define success metrics** and measurement approach
6. **Begin Phase 1 implementation**

---

## Autoscaling Dashboard UI Components & Calculations

This section explains the UI components in the autoscaling dashboard and how their values are calculated.

### Overview Cards

#### 1. **Total Replicas** 📊

**What it shows:** The sum of all current replica counts across all HPAs in the cluster.

**How it's calculated:**
```javascript
// From autoscaling.html JavaScript
const totalReplicas = data.hpas.reduce((sum, hpa) => sum + (hpa.current_replicas || 0), 0);
```

**Calculation Details:**
- Fetches all HPAs via `/api/autoscaling/status/<server_id>`
- Iterates through each HPA in the response
- Sums up the `current_replicas` field from each HPA
- Displays the total count

**Example:**
- HPA 1: `my-app-hpa` with 2 current replicas
- HPA 2: `backend-hpa` with 3 current replicas
- **Total Replicas: 5**

**Backend Implementation:**
```python
# From autoscaling_integration.py
def get_autoscaling_status(self, namespace: str = "default"):
    hpas = self.hpa_manager.list_hpas(namespace)
    # Returns list of HPAs with current_replicas field
    # Frontend sums these values
```

---

### 🤖 Predictive Autoscaling Section

#### 2. **Forecast & Scaling Chart** 📈

**What it shows:** A line chart displaying CPU and Memory usage predictions for the next 6 hours, used to make proactive scaling decisions.

**How it's calculated:**

1. **Data Source:**
   - Uses `TimeSeriesForecaster.forecast_cpu_usage(hours_ahead=6)`
   - Uses `TimeSeriesForecaster.forecast_memory_usage(hours_ahead=6)`
   - Leverages historical metrics from `PredictiveMonitoringSystem.metrics_history`

2. **Forecasting Algorithm:**
   ```python
   # From predictive_monitoring.py
   def forecast_cpu_usage(self, hours_ahead: int = 6):
       # 1. Extract historical CPU data (last 24 hours)
       cpu_data = [m.cpu_usage for m in self.history[-24:]]
       
       # 2. Calculate linear trend
       trend = np.polyfit(x, cpu_data, 1)[0]
       
       # 3. Calculate seasonal component (daily patterns)
       daily_pattern = np.mean([cpu_data[i::24] for i in range(24)], axis=1)
       
       # 4. Generate predictions for each hour
       for h in range(1, hours_ahead + 1):
           trend_value = current_cpu + trend * h
           seasonal_value = daily_pattern[hour_of_day]
           prediction = 0.7 * trend_value + 0.3 * seasonal_value
   ```

3. **Chart Display:**
   - **X-axis:** Time points: `['Now', '+1h', '+2h', '+3h', '+4h', '+5h', '+6h']`
   - **Y-axis:** Usage percentage (0-100%)
   - **Lines:**
     - Blue line: CPU Forecast
     - Green line: Memory Forecast
   - **Data Points:** Current value + 6 predicted values

4. **Scaling Decision:**
   - If predicted CPU > 75% OR predicted Memory > 80% → Scale Up
   - If predicted CPU < 25% AND predicted Memory < 25% → Scale Down
   - Otherwise → No Action

**Example Chart Data:**
```
Now:    CPU=45%, Memory=60%
+1h:    CPU=48%, Memory=62%  (predicted)
+2h:    CPU=52%, Memory=65%  (predicted)
+3h:    CPU=55%, Memory=68%  (predicted)
+4h:    CPU=58%, Memory=70%  (predicted)
+5h:    CPU=60%, Memory=72%  (predicted)
+6h:    CPU=62%, Memory=74%  (predicted)
```

**API Endpoint:** `/api/autoscaling/recommendations/<server_id>?deployment=<name>&namespace=<ns>`

---

### 📊 Horizontal Pod Autoscaler (HPA) Section

#### 3. **HPA Status Display: "Target: my-app | Replicas: 2/2 | Range: 2-10"**

**What each field means:**

- **Target: `my-app`**
  - The deployment name that the HPA is scaling
  - This is the `scaleTargetRef.name` from the HPA spec
  - The HPA monitors this deployment and adjusts its replica count

- **Replicas: `2/2`**
  - Format: `current_replicas / desired_replicas`
  - **Current Replicas:** Number of pods currently running for this deployment
  - **Desired Replicas:** Number of replicas the HPA wants (based on metrics)
  - If they match (2/2), the deployment is stable
  - If different (e.g., 2/4), the deployment is scaling up or down

- **Range: `2-10`**
  - Format: `min_replicas - max_replicas`
  - **Min Replicas:** Minimum number of pods the HPA will maintain (e.g., 2)
  - **Max Replicas:** Maximum number of pods the HPA can scale to (e.g., 10)
  - The HPA will never scale below min or above max

**How it's calculated:**
```python
# From autoscaling_engine.py
def list_hpas(self, namespace):
    # Gets HPA resources via kubectl
    result = self._execute_kubectl("get hpa --all-namespaces -o json")
    
    # Extracts for each HPA:
    hpa_info = {
        'name': item['metadata']['name'],                    # e.g., "my-app-hpa"
        'target': item['spec']['scaleTargetRef']['name'],    # e.g., "my-app"
        'min_replicas': item['spec']['minReplicas'],         # e.g., 2
        'max_replicas': item['spec']['maxReplicas'],         # e.g., 10
        'current_replicas': item['status']['currentReplicas'],  # e.g., 2
        'desired_replicas': item['status']['desiredReplicas']   # e.g., 2
    }
```

**Example Interpretation:**
- `Target: my-app` → HPA is managing the `my-app` deployment
- `Replicas: 2/2` → Currently 2 pods running, HPA wants 2 (stable)
- `Range: 2-10` → Will scale between 2 and 10 pods based on CPU/memory usage

**Scaling Behavior:**
- If CPU > 70% or Memory > 80% → HPA increases desired replicas (up to max)
- If CPU < 70% and Memory < 80% for 5 minutes → HPA decreases desired replicas (down to min)
- Kubernetes scheduler creates/deletes pods to match desired count

---

### ⏰ Scheduled Autoscaling Section

#### 4. **Time-based Scaling Calculation**

**What it does:** Analyzes historical usage patterns and creates cron-based schedules to scale deployments at specific times.

**How it's calculated:**

1. **Historical Pattern Analysis:**
   ```python
   # From scheduled_autoscaler.py
   def analyze_historical_patterns(self, deployment_name, namespace):
       # 1. Get last 7 days of metrics (168 hours)
       history = self.monitoring_system.metrics_history[-168:]
       
       # 2. Group metrics by hour of day
       hourly_usage = {}
       for metrics in history:
           hour = metrics.timestamp.hour
           avg_usage = (metrics.cpu_usage + metrics.memory_usage) / 2
           hourly_usage[hour].append(avg_usage)
       
       # 3. Calculate average usage per hour
       avg_hourly_usage = {
           hour: sum(usage) / len(usage)
           for hour, usage in hourly_usage.items()
       }
       
       # 4. Identify peak and off-peak hours
       sorted_hours = sorted(avg_hourly_usage.items(), key=lambda x: x[1], reverse=True)
       peak_hours = [h for h, u in sorted_hours[:8] if u > 50]      # Top 8 hours >50%
       off_peak_hours = [h for h, u in sorted_hours[-8:] if u < 30]  # Bottom 8 hours <30%
   ```

2. **Schedule Generation:**
   - **Peak Hours:** Scale up to handle high traffic (e.g., 9 AM - 5 PM weekdays)
   - **Off-Peak Hours:** Scale down to save costs (e.g., 10 PM - 6 AM)
   - **Business Hours:** Default schedule for 9 AM - 5 PM weekdays

3. **Cron Expression Format:**
   - Format: `minute hour day month weekday`
   - Example: `0 9 * * 1-5` = 9:00 AM, Monday-Friday
   - Example: `0 17 * * 1-5` = 5:00 PM, Monday-Friday (end of business)

4. **Schedule Application:**
   ```python
   def apply_schedule(self, deployment_name, namespace):
       current_time = datetime.now()
       current_hour = current_time.hour
       current_weekday = current_time.weekday()  # 0=Monday, 6=Sunday
       
       # Find matching schedule rule
       for rule in schedule['rules']:
           cron_parts = rule['time'].split()
           rule_hour = int(cron_parts[1])
           rule_days = cron_parts[4]  # e.g., "1-5" for weekdays
           
           if rule_hour == current_hour and day_matches(rule_days, current_weekday):
               # Scale deployment to rule['replicas']
               kubectl scale deployment {deployment_name} --replicas={replicas}
   ```

**Example Schedule Rules:**
```json
[
  {
    "time": "0 9 * * 1-5",    // 9 AM weekdays
    "replicas": 5,
    "reason": "Business hours start"
  },
  {
    "time": "0 17 * * 1-5",   // 5 PM weekdays
    "replicas": 2,
    "reason": "End of business hours"
  },
  {
    "time": "0 22 * * *",      // 10 PM daily
    "replicas": 1,
    "reason": "Off-peak hours"
  }
]
```

---

### 📈 Scaling History & Predictions Chart

#### 5. **Scaling History Chart**

**What it shows:** A line chart displaying the replica count over time, showing how the deployment has scaled historically.

**How it's calculated:**

1. **Current Implementation (Placeholder):**
   - Shows sample data: `[2, 2, 3, 3, 2, 2, 2]` for the last 7 time points
   - X-axis: `['1h ago', '2h ago', '3h ago', '4h ago', '5h ago', '6h ago', 'Now']`
   - Y-axis: Replica count

2. **Future Implementation (Recommended):**
   ```python
   # Proposed: Store scaling events in database or metrics
   def get_scaling_history(self, deployment_name, namespace):
       # Option 1: Query HPA status history
       # Option 2: Query deployment replica history
       # Option 3: Store scaling events in time-series database
       
       history = []
       for hour in range(6, 0, -1):  # Last 6 hours
           timestamp = datetime.now() - timedelta(hours=hour)
           # Get replica count at that time
           replicas = self.get_replicas_at_time(deployment_name, namespace, timestamp)
           history.append({
               'time': timestamp.isoformat(),
               'replicas': replicas
           })
       return history
   ```

3. **Data Sources (Potential):**
   - **HPA Status History:** Query HPA status over time to get `desiredReplicas`
   - **Deployment History:** Query deployment replica count from Kubernetes events
   - **Metrics Database:** Store scaling events in Prometheus or similar
   - **Application Logs:** Parse scaling events from application logs

4. **Chart Display:**
   - **Type:** Line chart (Chart.js)
   - **X-axis:** Time (last 6 hours or configurable)
   - **Y-axis:** Replica count
   - **Line Color:** Blue
   - **Shows:** How replicas changed over time (scaling up/down events)

**Example Chart Data:**
```
1h ago:  2 replicas
2h ago:  2 replicas
3h ago:  3 replicas  (scaled up due to load)
4h ago:  3 replicas
5h ago:  2 replicas  (scaled down after load decreased)
6h ago:  2 replicas
Now:     2 replicas
```

**Integration Points:**
- Can be enhanced to show actual historical data from:
  - HPA status API (`kubectl get hpa -o json --watch`)
  - Kubernetes events (`kubectl get events --field-selector involvedObject.kind=HorizontalPodAutoscaler`)
  - Prometheus metrics (if available)
  - Application database (storing scaling events)

---

## Summary of UI Calculations

| Component | Data Source | Calculation Method | Update Frequency |
|-----------|-------------|-------------------|------------------|
| **Total Replicas** | Sum of all HPA `current_replicas` | `sum(hpa.current_replicas for hpa in hpas)` | Every 30 seconds |
| **Forecast Chart** | `TimeSeriesForecaster` ML predictions | Linear trend + seasonal patterns | On-demand (when enabled) |
| **HPA Status** | Kubernetes HPA API | `kubectl get hpa -o json` | Every 30 seconds |
| **Scheduled Rules** | Historical pattern analysis | Average hourly usage over 7 days | On-demand (when analyzing) |
| **Scaling History** | Placeholder (future: HPA/Deployment history) | Sample data or time-series query | Every 30 seconds |

---

**Report Generated:** December 2024  
**Status:** Ready for Implementation  
**Next Review:** After Phase 1 completion

