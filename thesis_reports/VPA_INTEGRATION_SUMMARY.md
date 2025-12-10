# VPA Integration Summary

## Overview

The AI4K8s Predictive Autoscaling system has been expanded to support **Vertical Pod Autoscaling (VPA)** in addition to Horizontal Pod Autoscaling (HPA). The LLM-powered advisor now intelligently decides between horizontal scaling (more replicas) and vertical scaling (more resources per pod) based on application characteristics and workload patterns.

## Key Features

### 1. **LLM-Powered Scaling Type Decision**

The LLM advisor analyzes deployment characteristics and makes intelligent decisions:

- **HPA (Horizontal)**: Scale by adjusting replica count
  - Best for: Stateless applications, load distribution, high availability needs
  - Example: 3 pods → 5 pods (same resources per pod)

- **VPA (Vertical)**: Scale by adjusting resource requests/limits per pod
  - Best for: Stateful applications, single-pod bottlenecks, resource constraints
  - Example: CPU 100m → 200m, Memory 128Mi → 256Mi (same number of pods)

- **Both**: Apply both horizontal and vertical scaling simultaneously
  - Used when both more replicas AND more resources per pod are needed

### 2. **VPA Engine (`vpa_engine.py`)**

New module providing VPA management capabilities:

- `create_vpa()`: Create VPA resources with configurable CPU/Memory limits
- `get_vpa()`: Retrieve VPA status and recommendations
- `list_vpas()`: List all VPAs in cluster
- `delete_vpa()`: Remove VPA resources
- `patch_vpa_resources()`: Update VPA resource limits
- `get_deployment_resources()`: Get current resource requests/limits

### 3. **Enhanced LLM Advisor**

Updated `llm_autoscaling_advisor.py`:

- **System Prompt**: Now includes guidance on choosing HPA vs VPA
- **Context**: Includes VPA status and current resource configuration
- **Response Format**: Returns `scaling_type` field ('hpa', 'vpa', 'both', or 'maintain')
- **VPA Fields**: `target_cpu` and `target_memory` for vertical scaling recommendations

### 4. **Predictive Autoscaler Integration**

Updated `predictive_autoscaler.py`:

- Accepts optional `vpa_manager` parameter
- Passes VPA status and current resources to LLM advisor
- Handles VPA scaling actions (create/patch VPA resources)
- Supports combined HPA+VPA scaling

### 5. **API Endpoints**

New/Updated endpoints in `ai_kubernetes_web_app.py`:

- **`/api/autoscaling/predictive/apply/<server_id>`** (Updated):
  - Now accepts `scaling_type`, `target_cpu`, `target_memory` parameters
  - Supports HPA, VPA, or both scaling types

- **`/api/autoscaling/vpa/create/<server_id>`** (New):
  - Create VPA for a deployment

- **`/api/autoscaling/vpa/delete/<server_id>`** (New):
  - Delete VPA resource

### 6. **UI Enhancements**

Updated `templates/autoscaling.html`:

- **Scaling Type Badge**: Shows HPA, VPA, or HPA+VPA indicator
- **VPA Target Display**: Shows target CPU and Memory when VPA is recommended
- **Apply Button**: Updated to handle VPA scaling with appropriate confirmation messages
- **Recommendation Display**: Shows both replica count (HPA) and resource targets (VPA)

## Usage Flow

1. **Enable Predictive Autoscaling**: User enables predictive autoscaling for a deployment
2. **LLM Analysis**: System collects metrics, forecasts, HPA/VPA status, and current resources
3. **LLM Decision**: LLM analyzes and decides:
   - Scaling type (HPA, VPA, or both)
   - Target values (replicas, CPU, Memory)
   - Confidence score and reasoning
4. **Recommendation Display**: UI shows recommendation with scaling type badge
5. **Apply Recommendation**: User clicks "Apply Recommendation" button
6. **Scaling Execution**:
   - **HPA**: Scales deployment replicas (may patch HPA min/max)
   - **VPA**: Creates/updates VPA resource limits (VPA controller applies changes)
   - **Both**: Executes both HPA and VPA scaling

## Example LLM Recommendation

```json
{
  "scaling_type": "vpa",
  "action": "scale_up",
  "target_replicas": null,
  "target_cpu": "200m",
  "target_memory": "256Mi",
  "confidence": 0.85,
  "reasoning": "Application is stateful and cannot scale horizontally. Current CPU usage is high (180%) but memory is stable. Increasing CPU and Memory per pod will improve performance without requiring pod restarts.",
  "factors_considered": ["stateful_application", "cpu_pressure", "memory_stable"],
  "risk_assessment": "low",
  "cost_impact": "medium",
  "performance_impact": "positive"
}
```

## Technical Details

### VPA API Version
- Uses `autoscaling.k8s.io/v1` API
- Requires VPA controller to be installed in cluster

### Update Modes
- **Off**: VPA only provides recommendations, doesn't apply changes
- **Initial**: VPA sets resources only on pod creation
- **Auto**: VPA automatically updates resources (requires pod restart)
- **Recreate**: VPA recreates pods with new resources

### Resource Format
- CPU: Kubernetes format (e.g., "100m", "500m", "1", "2")
- Memory: Kubernetes format (e.g., "128Mi", "256Mi", "512Mi", "1Gi")

## Prerequisites

1. **VPA Controller**: Must be installed in Kubernetes cluster
   ```bash
   # Install VPA (example)
   git clone https://github.com/kubernetes/autoscaler.git
   cd autoscaler/vertical-pod-autoscaler/
   ./hack/vpa-up.sh
   ```

2. **Metrics Server**: Required for VPA recommendations

3. **Groq API Key**: Required for LLM-powered recommendations

## Benefits

1. **Intelligent Scaling**: LLM chooses optimal scaling strategy based on application characteristics
2. **Cost Optimization**: VPA can be more cost-effective for stateful applications
3. **Performance**: Right-sized resources per pod improve application performance
4. **Flexibility**: Supports both horizontal and vertical scaling strategies
5. **Automation**: Fully automated decision-making and execution

## Future Enhancements

- VPA recommendation history tracking
- Cost comparison between HPA and VPA strategies
- Automatic VPA controller installation check
- VPA recommendation visualization in UI
- Support for custom VPA update policies

## Files Modified/Created

### New Files
- `vpa_engine.py`: VPA management engine

### Modified Files
- `llm_autoscaling_advisor.py`: Added VPA support to LLM advisor
- `predictive_autoscaler.py`: Integrated VPA manager and scaling logic
- `autoscaling_integration.py`: Added VPA manager initialization and methods
- `ai_kubernetes_web_app.py`: Added VPA API endpoints
- `templates/autoscaling.html`: Updated UI for VPA recommendations

## Testing

To test VPA integration:

1. Ensure VPA controller is installed in cluster
2. Enable predictive autoscaling for a deployment
3. Monitor LLM recommendations for VPA suggestions
4. Apply VPA recommendations and verify resource changes
5. Check VPA status: `kubectl get vpa <deployment-name>-vpa -n <namespace>`

## Notes

- VPA requires pod restarts to apply resource changes (in Auto/Recreate modes)
- HPA and VPA can conflict if both are active - LLM considers this in recommendations
- VPA recommendations are based on historical usage patterns
- LLM considers application statefulness when choosing scaling type

