# VPA Testing Guide

This guide explains how to test Vertical Pod Autoscaler (VPA) functionality in AI4K8s.

## Prerequisites

1. **VPA Controller Installed**: VPA must be installed in your Kubernetes cluster
   ```bash
   # Check if VPA API is available
   kubectl get crd | grep verticalpodautoscaler
   
   # If not installed, install VPA (example for k3s/minikube)
   # Note: VPA installation varies by cluster type
   ```

2. **Test Deployment**: Create a test deployment
   ```bash
   kubectl create deployment test-vpa-app --image=nginx --replicas=1
   ```

## Testing Methods

### Method 1: Manual VPA Creation via UI

1. **Navigate to Autoscaling Page**: Go to `/autoscaling/<server_id>`

2. **Create VPA**:
   - Scroll to "Vertical Pod Autoscaler (VPA)" section
   - Fill in the form:
     - Deployment Name: `test-vpa-app`
     - Namespace: `default`
     - Min CPU: `100m`
     - Max CPU: `1000m`
     - Min Memory: `128Mi`
     - Max Memory: `512Mi`
     - Update Mode: `Auto` (or `Off` for recommendations only)
   - Click "Create VPA"

3. **Verify VPA Created**:
   ```bash
   kubectl get vpa test-vpa-app-vpa -n default
   kubectl describe vpa test-vpa-app-vpa -n default
   ```

### Method 2: Force VPA Recommendation via LLM

To test LLM VPA recommendations, you need to make the LLM think the app is stateful (state inside pod):

**Option A: Create a Stateful App**
```bash
# Create a deployment that appears stateful
kubectl create deployment stateful-test-app \
  --image=nginx \
  --replicas=1 \
  --dry-run=client -o yaml > stateful-app.yaml

# Edit the deployment to add annotations indicating stateful behavior
# Add annotation: ai4k8s.io/app-type: stateful
# Or use a name that suggests stateful behavior
```

**Option B: Modify LLM Context** (for testing only)
- The LLM checks if state is externalized
- If you want to force VPA, create a deployment that:
  - Doesn't use Redis/external DB
  - Has a name suggesting stateful behavior (e.g., `database-app`, `stateful-service`)
  - Or add annotations: `ai4k8s.io/state-type: internal`

### Method 3: Direct API Testing

```bash
# Create VPA via API
curl -X POST http://localhost:5003/api/autoscaling/vpa/create/<server_id> \
  -H "Content-Type: application/json" \
  -d '{
    "deployment_name": "test-vpa-app",
    "namespace": "default",
    "min_cpu": "100m",
    "max_cpu": "1000m",
    "min_memory": "128Mi",
    "max_memory": "512Mi",
    "update_mode": "Auto"
  }'
```

### Method 4: Test VPA Recommendations

1. **Create VPA with Update Mode "Off"** (recommendations only):
   - This allows you to see VPA recommendations without automatic updates
   - VPA will analyze pod resource usage and provide recommendations

2. **Check VPA Recommendations**:
   ```bash
   kubectl get vpa test-vpa-app-vpa -o yaml
   # Look for status.recommendation section
   ```

3. **Apply Recommendations Manually** (if Update Mode is "Off"):
   - Use the "Apply VPA" button in the UI recommendations
   - Or manually update deployment resource requests/limits

## Testing LLM VPA Recommendations

### Scenario 1: Stateful Application (State Inside Pod)

Create a deployment that the LLM will recognize as stateful:

```bash
# Create a deployment with a name suggesting stateful behavior
kubectl create deployment database-app --image=postgres:15 --replicas=1

# Enable predictive autoscaling
# The LLM should recommend VPA because:
# - Name suggests database (stateful)
# - No external state management visible
# - State likely stored inside pod
```

### Scenario 2: Application at Max Replicas

If your deployment is already at max_replicas and needs more resources:

1. Set max_replicas to 1 (or current replica count)
2. Enable predictive autoscaling
3. Generate high CPU/Memory load
4. LLM should recommend VPA instead of HPA (since HPA can't scale up)

### Scenario 3: Single-Pod Bottleneck

Create a deployment that can't scale horizontally:

```bash
# Create a single-replica deployment
kubectl create deployment single-pod-app --image=nginx --replicas=1

# Set max_replicas=1 in predictive autoscaling
# When resource pressure occurs, LLM should recommend VPA
```

## Verifying VPA Functionality

### Check VPA Status
```bash
# List all VPAs
kubectl get vpa --all-namespaces

# Get VPA details
kubectl get vpa <vpa-name> -n <namespace> -o yaml

# Check VPA recommendations
kubectl describe vpa <vpa-name> -n <namespace>
```

### Check Pod Resource Updates (if Update Mode is Auto)
```bash
# Get pod resource requests/limits
kubectl get pod <pod-name> -o jsonpath='{.spec.containers[0].resources}'

# Watch for pod restarts (VPA may restart pods to apply new resources)
kubectl get pods -w
```

### Monitor VPA Events
```bash
# Check VPA events
kubectl get events --field-selector involvedObject.kind=VerticalPodAutoscaler
```

## Troubleshooting

### Issue: VPA API Not Available
**Error**: `the server doesn't have a resource type "verticalpodautoscalers"`

**Solution**: 
- VPA controller is not installed
- Install VPA controller for your cluster type
- For k3s/minikube, VPA installation may require additional setup

### Issue: LLM Always Recommends HPA
**Possible Causes**:
1. Application appears stateless (uses external Redis/DB)
2. Application can scale horizontally
3. Max replicas allows horizontal scaling

**Solutions**:
- Create a deployment that appears stateful
- Set max_replicas to current replica count
- Use a deployment name suggesting stateful behavior
- Ensure deployment doesn't use external state management

### Issue: VPA Not Updating Pods
**Check**:
1. Update Mode: Should be "Auto" for automatic updates
2. VPA recommendations: Check if VPA is providing recommendations
3. Pod readiness: Pods must be ready for VPA to update
4. VPA controller: Ensure VPA controller is running

## Expected Behavior

### VPA with Update Mode "Auto"
- VPA analyzes pod resource usage
- Automatically updates pod resource requests/limits
- May restart pods to apply new resources
- Pods get new CPU/Memory requests based on actual usage

### VPA with Update Mode "Off"
- VPA analyzes pod resource usage
- Provides recommendations only
- Does not automatically update pods
- Recommendations can be applied manually via UI

### LLM VPA Recommendations
- LLM analyzes deployment characteristics
- Recommends VPA when:
  - State is stored inside pod (not externalized)
  - Application is at max replicas
  - Single-pod bottleneck
  - Cannot scale horizontally
- Provides target CPU/Memory values
- Can be applied via "Apply VPA" button

## Next Steps

1. Create a test deployment
2. Create VPA manually or enable predictive autoscaling
3. Monitor VPA recommendations
4. Apply recommendations and verify pod resource updates
5. Test LLM VPA recommendations with stateful-appearing deployments

