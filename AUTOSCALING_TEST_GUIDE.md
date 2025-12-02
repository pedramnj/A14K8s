# Autoscaling Testing Guide - CrownLabs k3s Cluster

## Test Application Deployed

**Deployment**: `test-app-autoscaling` in namespace `ai4k8s-test`
- **Current Replicas**: 2
- **Service**: `test-app-autoscaling` (ClusterIP)
- **Resource Requests**: CPU 100m, Memory 128Mi
- **Resource Limits**: CPU 500m, Memory 256Mi
- **Default CPU Load**: 30% (configurable via `CPU_LOAD_PERCENT` env var)

### Application Endpoints

- **Health Check**: `http://test-app-autoscaling.ai4k8s-test.svc.cluster.local/health`
- **Generate CPU Load**: `http://test-app-autoscaling.ai4k8s-test.svc.cluster.local/cpu-load`
- **Main Page**: `http://test-app-autoscaling.ai4k8s-test.svc.cluster.local/`

---

## 1. Horizontal Pod Autoscaler (HPA) Testing

### Step 1: Create HPA via AI4K8s Web UI

1. Navigate to **Clusters** → **CrownLabs k3s Cluster** → **Autoscaling**
2. In the **Horizontal Pod Autoscaler (HPA)** section:
   - **Target Deployment**: `test-app-autoscaling`
   - **Namespace**: `ai4k8s-test`
   - **Min Replicas**: 2
   - **Max Replicas**: 10
   - **Target CPU**: 50% (or 70% for more aggressive scaling)
   - Click **Create HPA**

### Step 2: Verify HPA Creation

```bash
export KUBECONFIG=~/crownlabs-k3s.yaml
kubectl get hpa -n ai4k8s-test test-app-autoscaling
kubectl describe hpa -n ai4k8s-test test-app-autoscaling
```

### Step 3: Generate Load to Trigger Scaling

**Option A: Increase CPU load via environment variable**

```bash
# Scale up CPU load to 80% to trigger HPA
kubectl set env deployment/test-app-autoscaling -n ai4k8s-test CPU_LOAD_PERCENT=80

# Wait and watch HPA
watch kubectl get hpa -n ai4k8s-test test-app-autoscaling
```

**Option B: Generate HTTP load**

```bash
# Port-forward to access the service
kubectl port-forward -n ai4k8s-test svc/test-app-autoscaling 8080:80 &

# Generate load using curl in a loop
for i in {1..100}; do
  curl -s http://localhost:8080/cpu-load > /dev/null &
done
```

**Option C: Use kubectl to scale CPU load**

```bash
# Patch deployment to increase CPU load
kubectl patch deployment test-app-autoscaling -n ai4k8s-test -p '{"spec":{"template":{"spec":{"containers":[{"name":"app","env":[{"name":"CPU_LOAD_PERCENT","value":"80"}]}]}}}}'
```

### Step 4: Monitor Scaling

```bash
# Watch HPA status
watch -n 2 'kubectl get hpa -n ai4k8s-test test-app-autoscaling && echo "" && kubectl get pods -n ai4k8s-test -l app=test-app-autoscaling'

# Check HPA events
kubectl describe hpa -n ai4k8s-test test-app-autoscaling | tail -20

# Monitor pod CPU usage
kubectl top pods -n ai4k8s-test -l app=test-app-autoscaling
```

### Expected Behavior

- **Initial**: 2 pods running at ~30% CPU each
- **After Load Increase**: CPU should spike to 80%+
- **HPA Response**: Should scale up to 3-4 pods within 1-2 minutes
- **After Load Decreases**: Should scale down after stabilization period (default 5 minutes)

---

## 2. Predictive Autoscaling Testing

### Step 1: Enable Predictive Autoscaling

1. Navigate to **Clusters** → **CrownLabs k3s Cluster** → **Autoscaling**
2. In the **🤖 Predictive Autoscaling** section:
   - Toggle **Enable Predictive Autoscaling** to ON
   - Select deployment: `test-app-autoscaling`
   - Namespace: `ai4k8s-test`
   - Click **Enable Predictive Autoscaling**

### Step 2: Monitor Predictive Forecasts

1. In the **Forecast & Scaling Chart** section, you should see:
   - **Current CPU Usage**: Real-time CPU percentage
   - **Predicted CPU Usage**: Forecast for next 1-6 hours
   - **Scaling Recommendations**: Based on predicted load

### Step 3: Generate Variable Load Pattern

To test predictive scaling, create a **time-based load pattern**:

```bash
# Morning: Low load (20%)
kubectl set env deployment/test-app-autoscaling -n ai4k8s-test CPU_LOAD_PERCENT=20

# Wait 5 minutes, then increase to medium load (50%)
sleep 300
kubectl set env deployment/test-app-autoscaling -n ai4k8s-test CPU_LOAD_PERCENT=50

# Wait 5 minutes, then increase to high load (80%)
sleep 300
kubectl set env deployment/test-app-autoscaling -n ai4k8s-test CPU_LOAD_PERCENT=80
```

### Step 4: Verify Predictive Scaling

- Check the **Forecast & Scaling Chart** in the web UI
- Predictive autoscaling should:
  - Analyze historical patterns
  - Predict future CPU usage
  - Proactively scale before load increases
  - Show scaling recommendations based on forecasts

### Expected Behavior

- **Pattern Recognition**: System learns from load patterns
- **Proactive Scaling**: Scales up before predicted load increase
- **Forecast Accuracy**: Predictions improve over time with more data

---

## 3. Scheduled Autoscaling Testing

### Step 1: Create Scheduled Scaling Rule

1. Navigate to **Clusters** → **CrownLabs k3s Cluster** → **Autoscaling**
2. In the **Time-based scaling based on historical patterns** section:
   - Click **Add Scheduled Rule**
   - **Rule Name**: `morning-scale-up`
   - **Schedule**: `0 9 * * *` (9 AM daily) or `*/5 * * * *` (every 5 minutes for testing)
   - **Target Replicas**: 5
   - **Deployment**: `test-app-autoscaling`
   - **Namespace**: `ai4k8s-test`
   - Click **Create Rule**

### Step 2: Create Multiple Scheduled Rules

Create additional rules for different times:

- **Morning Scale-Up** (9 AM): 5 replicas
- **Afternoon Scale-Down** (2 PM): 2 replicas
- **Evening Scale-Up** (6 PM): 4 replicas
- **Night Scale-Down** (11 PM): 1 replica

### Step 3: Test Scheduled Scaling

**For quick testing, use frequent schedules:**

```bash
# Create a rule that runs every 5 minutes (for testing)
# Via web UI or kubectl create CronJob
```

**Verify scheduled rules are active:**

- Check **Scheduled Rules** section in Autoscaling page
- Rules should show: Name, Schedule, Target Replicas, Status

### Step 4: Monitor Scheduled Scaling

```bash
# Watch deployment replicas
watch -n 10 'kubectl get deployment test-app-autoscaling -n ai4k8s-test -o wide'

# Check CronJobs (if scheduled scaling uses CronJobs)
kubectl get cronjobs -n ai4k8s-test

# View scheduled scaling history
# Check the "Scaling History & Predictions" chart in web UI
```

### Expected Behavior

- **At Scheduled Time**: Deployment should scale to target replicas
- **History Tracking**: Scaling events should appear in history chart
- **Multiple Rules**: Different rules can apply at different times

---

## 4. Combined Testing Scenario

### Scenario: Simulate Real-World Workload

1. **Initial State**: 2 replicas, 30% CPU load
2. **Morning Rush (9 AM)**: Scheduled scaling → 5 replicas
3. **Load Increase (10 AM)**: Increase CPU_LOAD_PERCENT to 70%
4. **HPA Response**: Should maintain 5 replicas or scale up if needed
5. **Predictive Forecast**: Should predict continued high load
6. **Afternoon (2 PM)**: Scheduled scaling → 2 replicas
7. **Load Decrease**: Reduce CPU_LOAD_PERCENT to 20%
8. **HPA Response**: Should scale down to 2 replicas

### Monitoring Commands

```bash
# Continuous monitoring script
while true; do
  clear
  echo "=== HPA Status ==="
  kubectl get hpa -n ai4k8s-test
  echo ""
  echo "=== Pod Status ==="
  kubectl get pods -n ai4k8s-test -l app=test-app-autoscaling
  echo ""
  echo "=== CPU Usage ==="
  kubectl top pods -n ai4k8s-test -l app=test-app-autoscaling
  echo ""
  echo "=== Deployment Replicas ==="
  kubectl get deployment test-app-autoscaling -n ai4k8s-test -o jsonpath='{.spec.replicas}'
  echo " desired, "
  kubectl get deployment test-app-autoscaling -n ai4k8s-test -o jsonpath='{.status.readyReplicas}'
  echo " ready"
  sleep 10
done
```

---

## 5. Cleanup

### Remove Test Resources

```bash
# Delete HPA
kubectl delete hpa -n ai4k8s-test test-app-autoscaling

# Delete deployment and service
kubectl delete deployment test-app-autoscaling -n ai4k8s-test
kubectl delete service test-app-autoscaling -n ai4k8s-test

# Or delete entire namespace
kubectl delete namespace ai4k8s-test
```

---

## Troubleshooting

### HPA Not Scaling

1. **Check metrics-server**: `kubectl get deployment metrics-server -n kube-system`
2. **Check pod CPU usage**: `kubectl top pods -n ai4k8s-test`
3. **Verify resource requests**: `kubectl describe deployment test-app-autoscaling -n ai4k8s-test`
4. **Check HPA events**: `kubectl describe hpa -n ai4k8s-test test-app-autoscaling`

### Predictive Autoscaling Not Working

1. **Ensure monitoring is active**: Check AI Monitoring dashboard
2. **Wait for data collection**: Predictive scaling needs historical data
3. **Check forecast chart**: Verify predictions are being generated

### Scheduled Scaling Not Triggering

1. **Verify CronJob**: `kubectl get cronjobs -n ai4k8s-test`
2. **Check CronJob logs**: `kubectl logs -n ai4k8s-test <cronjob-pod>`
3. **Verify timezone**: Ensure server timezone matches schedule

---

## Next Steps

1. ✅ Deploy test application
2. ✅ Create HPA via web UI
3. ✅ Test HPA scaling with load generation
4. ✅ Enable Predictive Autoscaling
5. ✅ Create Scheduled Scaling Rules
6. ✅ Monitor all three autoscaling types simultaneously
7. ✅ Document results and observations

