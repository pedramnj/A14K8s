# Test Deployment for Autoscaling

This directory contains test deployments for testing the autoscaling features.

## Test Applications

### 1. `my-app` - Basic Test Deployment

A simple nginx-based deployment with resource requests/limits configured for autoscaling testing.

**Deploy:**
```bash
kubectl apply -f test-deployment.yaml
```

**Verify:**
```bash
kubectl get deployment my-app
kubectl get pods -l app=my-app
```

### 2. `load-generator` - CPU Load Generator

A simple container that generates CPU load to test autoscaling behavior.

**Deploy:**
```bash
kubectl apply -f test-deployment-load-generator.yaml
```

**Note:** This is optional - you can also manually generate load on `my-app` pods.

## Testing Autoscaling

### Step 1: Deploy the Test App

```bash
kubectl apply -f test-deployment.yaml
```

Wait for pods to be ready:
```bash
kubectl wait --for=condition=ready pod -l app=my-app --timeout=60s
```

### Step 2: Create HPA (Reactive Autoscaling)

1. Go to the Autoscaling dashboard in the web UI
2. In the "Create New HPA" section:
   - Deployment Name: `my-app`
   - Namespace: `default`
   - Min Replicas: `2`
   - Max Replicas: `10`
   - CPU Target: `70`
   - Memory Target: `80`
3. Click "Create HPA"

**Or via kubectl:**
```bash
kubectl autoscale deployment my-app --cpu-percent=70 --min=2 --max=10
```

### Step 3: Enable Predictive Autoscaling

1. In the "Enable Predictive Autoscaling" section:
   - Deployment Name: `my-app`
   - Namespace: `default`
   - Min Replicas: `2`
   - Max Replicas: `10`
2. Click "Enable Predictive Autoscaling"

This will:
- Create an HPA if it doesn't exist
- Use ML forecasts to proactively scale before load arrives

### Step 4: Create Scheduled Autoscaling

1. In the "Create Schedule" section:
   - Deployment Name: `my-app`
   - Namespace: `default`
   - Cron Expression: `0 9 * * 1-5` (9 AM weekdays)
   - Replicas: `5`
2. Click "Create Schedule"

### Step 5: Generate Load (Optional)

To test reactive autoscaling, generate CPU load:

```bash
# Scale up load generator
kubectl scale deployment load-generator --replicas=3

# Or manually exec into a pod and generate load
kubectl exec -it deployment/my-app -- sh -c "while true; do timeout 5 dd if=/dev/zero of=/dev/null 2>/dev/null || true; sleep 1; done"
```

### Step 6: Monitor Scaling

Watch the HPA status:
```bash
kubectl get hpa my-app-hpa -w
```

Or use the web UI Autoscaling dashboard to see:
- Current replica counts
- Forecast charts
- Scaling recommendations
- Scaling history

## Cleanup

```bash
# Delete test deployments
kubectl delete -f test-deployment.yaml
kubectl delete -f test-deployment-load-generator.yaml

# Delete HPAs
kubectl delete hpa my-app-hpa
```

## Expected Behavior

### Reactive Autoscaling (HPA)
- When CPU > 70% or Memory > 80%, pods should scale up
- When CPU < 70% and Memory < 80% for 5 minutes, pods should scale down
- Scaling happens within 15-30 seconds

### Predictive Autoscaling
- Uses ML forecasts to predict future load
- Scales proactively before load arrives
- Shows forecast charts in the UI
- Provides recommendations based on predictions

### Scheduled Autoscaling
- Scales at specified times (cron schedule)
- Useful for known traffic patterns (business hours, etc.)

## Troubleshooting

### HPA not scaling
- Check metrics-server is installed: `kubectl get deployment metrics-server -n kube-system`
- Verify resource requests are set: `kubectl describe deployment my-app`
- Check HPA status: `kubectl describe hpa my-app-hpa`

### Predictive autoscaling not working
- Ensure monitoring system is collecting metrics
- Check that forecasting has enough historical data (at least 10 data points)
- Verify the deployment exists: `kubectl get deployment my-app`

### Charts not showing
- Check browser console for JavaScript errors
- Verify Chart.js is loading: Check network tab
- Ensure API endpoints are accessible: Check `/api/autoscaling/status/<server_id>`

