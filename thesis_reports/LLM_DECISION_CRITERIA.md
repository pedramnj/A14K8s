# LLM Decision Criteria for HPA vs VPA Selection

## Overview

The LLM (Large Language Model) advisor uses a comprehensive prompt system to analyze deployment characteristics and make intelligent decisions between Horizontal Pod Autoscaling (HPA) and Vertical Pod Autoscaling (VPA).

## Decision Criteria

The LLM considers the following factors when choosing between HPA and VPA:

### 1. **Application Characteristics**

#### Choose HPA (Horizontal Scaling) When:
- **Stateless Applications**: Application can distribute load across multiple pods
- **High Availability Needs**: Need redundancy and fault tolerance
- **Traffic Spikes**: Need to handle sudden increases in traffic
- **Load Distribution**: Workload can be evenly distributed
- **Example**: Web servers, API gateways, stateless microservices

#### Choose VPA (Vertical Scaling) When:
- **Stateful Applications**: Application maintains state and cannot easily scale horizontally
- **Single-Pod Bottleneck**: Performance limited by individual pod resources, not pod count
- **Resource Constraints**: Current pods are under-resourced (CPU/Memory too low)
- **Simpler Architecture**: Fewer pods preferred for operational simplicity
- **Example**: Databases, stateful services, single-instance applications

### 2. **Current Metrics Analysis**

The LLM analyzes:
- **CPU Usage**: Current and predicted CPU utilization
- **Memory Usage**: Current and predicted memory utilization
- **Pod Count**: Current number of replicas
- **Resource Requests/Limits**: Current CPU and Memory configuration

### 3. **Forecast Data**

The LLM considers:
- **CPU Trends**: Increasing, decreasing, or stable
- **Memory Trends**: Increasing, decreasing, or stable
- **Peak Predictions**: Expected peak usage in next 6 hours
- **Prediction Values**: Hourly predictions for next 6 hours

### 4. **HPA/VPA Status**

The LLM checks:
- **HPA Active**: Whether HPA is already configured
- **VPA Active**: Whether VPA is already configured
- **Current Scaling Status**: Stable, scaling up, or scaling down
- **Target Thresholds**: Current HPA CPU/Memory targets

### 5. **Constraints**

The LLM respects:
- **Min/Max Replicas**: Deployment replica limits
- **Cluster Capacity**: Available cluster resources
- **Cost Considerations**: Minimize resource usage while maintaining performance
- **Stability**: Avoid rapid scaling changes

## Complete Prompts

### System Prompt

```
You are an expert Kubernetes autoscaling advisor with deep knowledge of:
- Resource optimization and cost management
- Performance requirements and SLA considerations
- Scaling best practices and anti-patterns
- Predictive analysis and trend interpretation
- Horizontal Pod Autoscaling (HPA) vs Vertical Pod Autoscaling (VPA)

Your role is to analyze deployment metrics, forecasts, and patterns to make intelligent scaling recommendations.

**IMPORTANT: You must decide between TWO scaling strategies:**

1. **HORIZONTAL SCALING (HPA)**: Scale by adjusting the NUMBER of replicas (pods)
   - Use when: Load can be distributed across multiple pods, need high availability, stateless applications
   - Example: 3 pods → 5 pods (same resources per pod)
   - Pros: Better fault tolerance, load distribution, can handle traffic spikes
   - Cons: More pods = more overhead, may hit node limits

2. **VERTICAL SCALING (VPA)**: Scale by adjusting RESOURCE requests/limits per pod
   - Use when: Application cannot scale horizontally, single-pod bottleneck, stateful applications
   - Example: CPU 100m → 200m, Memory 128Mi → 256Mi (same number of pods)
   - Pros: Better resource utilization, fewer pods, simpler architecture
   - Cons: Pod restart required, single point of failure, limited by node capacity

Consider these factors:
1. **Performance**: Ensure adequate resources to meet performance requirements
2. **Cost**: Minimize resource usage while maintaining performance
3. **Stability**: Avoid rapid scaling changes that could cause instability
4. **Predictions**: Use forecast data to proactively scale before demand arrives
5. **Constraints**: Respect min/max replica limits and current cluster capacity
6. **Scaling Type**: Choose HPA (horizontal) or VPA (vertical) based on application characteristics

Respond in JSON format with:
{
  "scaling_type": "hpa" | "vpa" | "both" | "maintain",
  "action": "scale_up" | "scale_down" | "maintain" | "at_max",
  "target_replicas": <number> (for HPA, null if VPA),
  "target_cpu": "<value>" (for VPA, e.g., "200m", null if HPA),
  "target_memory": "<value>" (for VPA, e.g., "256Mi", null if HPA),
  "confidence": <0.0-1.0>,
  "reasoning": "<detailed explanation including why HPA vs VPA was chosen>",
  "factors_considered": ["factor1", "factor2", ...],
  "risk_assessment": "low" | "medium" | "high",
  "cost_impact": "low" | "medium" | "high",
  "performance_impact": "positive" | "neutral" | "negative",
  "recommended_timing": "immediate" | "gradual" | "scheduled"
}
```

### User Prompt (Dynamic Context)

The user prompt includes the following dynamic information:

```
Analyze the following Kubernetes deployment autoscaling scenario and provide a recommendation:

**Deployment Information:**
- Name: {deployment_name}
- Namespace: {namespace}
- Current Replicas: {current_replicas}
- Min Replicas: {min_replicas}
- Max Replicas: {max_replicas}

**Current Resource Usage:**
- CPU: {cpu_usage_percent}%
- Memory: {memory_usage_percent}%
- Running Pods: {running_pods}/{pod_count}

**Forecast Data:**
- CPU Current: {cpu_current}%, Peak: {cpu_peak}%, Trend: {cpu_trend}
- Memory Current: {memory_current}%, Peak: {memory_peak}%, Trend: {memory_trend}
- CPU Predictions (next 6 hours): {cpu_predictions}
- Memory Predictions (next 6 hours): {memory_predictions}

**Current Resource Configuration:**
- CPU Request: {cpu_request}
- CPU Limit: {cpu_limit}
- Memory Request: {memory_request}
- Memory Limit: {memory_limit}

**HPA Status:**
- HPA Active: {Yes/No}
- Current Replicas: {current_replicas}
- Desired Replicas: {desired_replicas}
- Target CPU: {target_cpu}%
- Target Memory: {target_memory}%
- Scaling Status: {stable/scaling_up/scaling_down}

**VPA Status:**
- VPA Active: {Yes/No}
- Update Mode: {Auto/Initial/Off/Recreate}
- Recommendations: {vpa_recommendations}

**Historical Patterns:** (if available)
{historical_patterns}

**Analysis Request:**
Based on the above information, provide an intelligent scaling recommendation considering:
1. Current resource pressure (CPU/Memory usage)
2. Predicted future demand (forecast trends)
3. Cost optimization opportunities
4. Performance requirements
5. Stability and risk factors
6. **CRITICAL: Choose between HPA (horizontal - more replicas) or VPA (vertical - more resources per pod)**

**Scaling Strategy Decision Guidelines:**
- Choose HPA (horizontal) if: Application is stateless, can distribute load, needs high availability, traffic spikes are expected
- Choose VPA (vertical) if: Application is stateful, cannot scale horizontally, single-pod bottleneck, resource constraints per pod
- Choose "both" if: Need both more replicas AND more resources per pod (rare, but possible)

Provide your recommendation in the specified JSON format with scaling_type field.
```

## Example Decision Scenarios

### Scenario 1: Stateless Web Application (HPA Recommended)

**Context:**
- Application: Web server (stateless)
- Current: 3 replicas, CPU 150%, Memory 60%
- Forecast: CPU peak 200% in 2 hours
- Current Resources: CPU 100m, Memory 128Mi

**LLM Decision:**
```json
{
  "scaling_type": "hpa",
  "action": "scale_up",
  "target_replicas": 5,
  "target_cpu": null,
  "target_memory": null,
  "reasoning": "Stateless web application can distribute load across multiple pods. Current CPU usage is high (150%) and forecast predicts peak of 200%. Horizontal scaling to 5 replicas will distribute the load and improve performance without requiring pod restarts.",
  "factors_considered": ["stateless_application", "high_cpu_usage", "traffic_spike_predicted", "load_distribution_possible"]
}
```

### Scenario 2: Stateful Database (VPA Recommended)

**Context:**
- Application: Database (stateful)
- Current: 1 replica, CPU 180%, Memory 85%
- Forecast: CPU stable, Memory increasing
- Current Resources: CPU 100m, Memory 256Mi

**LLM Decision:**
```json
{
  "scaling_type": "vpa",
  "action": "scale_up",
  "target_replicas": null,
  "target_cpu": "200m",
  "target_memory": "512Mi",
  "reasoning": "Stateful database cannot scale horizontally. Current CPU usage is very high (180%) and memory is approaching limits (85%). Increasing CPU to 200m and Memory to 512Mi will improve performance. Single-pod bottleneck requires vertical scaling.",
  "factors_considered": ["stateful_application", "single_pod_bottleneck", "high_resource_pressure", "cannot_scale_horizontally"]
}
```

### Scenario 3: Mixed Workload (Both Recommended)

**Context:**
- Application: API gateway with caching (semi-stateful)
- Current: 2 replicas, CPU 160%, Memory 70%
- Forecast: CPU peak 220%, Memory stable
- Current Resources: CPU 100m, Memory 256Mi

**LLM Decision:**
```json
{
  "scaling_type": "both",
  "action": "scale_up",
  "target_replicas": 3,
  "target_cpu": "150m",
  "target_memory": "256Mi",
  "reasoning": "API gateway needs both more replicas for load distribution (2→3) and more CPU per pod (100m→150m) to handle the predicted CPU spike. This provides both horizontal redundancy and vertical resource increase.",
  "factors_considered": ["semi_stateful", "high_cpu_pressure", "need_redundancy", "resource_constraints"]
}
```

## Decision Flow

```
1. Analyze Application Type
   ├─ Stateless? → Consider HPA
   └─ Stateful? → Consider VPA

2. Check Resource Pressure
   ├─ CPU/Memory High? → Need scaling
   └─ CPU/Memory Low? → Consider scale down

3. Check Forecast Trends
   ├─ Increasing? → Scale up proactively
   ├─ Decreasing? → Scale down gradually
   └─ Stable? → Maintain or optimize

4. Check Current Configuration
   ├─ HPA Active? → May prefer HPA
   ├─ VPA Active? → May prefer VPA
   └─ Neither? → Choose based on app type

5. Consider Constraints
   ├─ At Max Replicas? → Consider VPA or "at_max"
   ├─ Resource Limits? → Consider HPA
   └─ Cost Concerns? → Optimize accordingly

6. Make Decision
   ├─ HPA: More replicas
   ├─ VPA: More resources per pod
   ├─ Both: Both strategies
   └─ Maintain: No changes needed
```

## Key Factors Weight

The LLM implicitly weights factors as follows:

1. **Application Type** (Highest Weight)
   - Stateless → Strong preference for HPA
   - Stateful → Strong preference for VPA

2. **Resource Pressure** (High Weight)
   - High CPU/Memory → Scale up (HPA or VPA)
   - Low CPU/Memory → Scale down

3. **Forecast Trends** (Medium Weight)
   - Increasing trend → Proactive scaling
   - Decreasing trend → Gradual scale down

4. **Current Configuration** (Medium Weight)
   - Existing HPA/VPA → Prefer same strategy
   - No existing config → Choose based on app type

5. **Constraints** (Low Weight)
   - Max replicas reached → Consider VPA
   - Resource limits → Consider HPA

## Response Format

The LLM returns a structured JSON response:

```json
{
  "scaling_type": "hpa" | "vpa" | "both" | "maintain",
  "action": "scale_up" | "scale_down" | "maintain" | "at_max",
  "target_replicas": <number> | null,
  "target_cpu": "<value>" | null,
  "target_memory": "<value>" | null,
  "confidence": 0.0-1.0,
  "reasoning": "<detailed explanation>",
  "factors_considered": ["factor1", "factor2", ...],
  "risk_assessment": "low" | "medium" | "high",
  "cost_impact": "low" | "medium" | "high",
  "performance_impact": "positive" | "neutral" | "negative",
  "recommended_timing": "immediate" | "gradual" | "scheduled"
}
```

## Caching and Rate Limiting

To prevent excessive API calls and ensure stability:

- **Cache TTL**: 5 minutes (300 seconds)
- **Rate Limit**: Minimum 5 minutes between LLM calls per deployment
- **Cache Key**: Based on aggressively rounded metrics (CPU rounded to 25%, Memory to 5%)

This ensures recommendations remain stable and don't change with minor metric fluctuations.

## Model Configuration

- **Primary Model**: `llama-3.1-8b-instant` (fast, reliable)
- **Fallback Model**: `llama-3.1-70b-versatile` (if primary unavailable)
- **Temperature**: 0.3 (lower for more consistent decisions)
- **Max Tokens**: 1000

## Summary

The LLM makes intelligent decisions based on:
1. **Application characteristics** (stateless vs stateful)
2. **Current resource pressure** (CPU/Memory usage)
3. **Forecast trends** (predicted future demand)
4. **Existing configuration** (HPA/VPA status)
5. **Constraints** (replica limits, cluster capacity)

The decision is explained in the `reasoning` field, which includes why HPA vs VPA was chosen, making the decision process transparent and auditable.

