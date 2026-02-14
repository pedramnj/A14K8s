# AI4K8s — Recent Changes & Enhancements

**Pedram Nikjooy** | Politecnico di Torino | February 2026

---

## Overview

Phase 2 introduces two major analytical modules and a complete frontend overhaul:

| Module | Purpose | Key File |
|---|---|---|
| **MCDA Optimizer** | Formal multi-criteria optimization (TOPSIS) | `mcda_optimizer.py` |
| **UQ Quantifier** | Calibrated uncertainty for predictions & anomalies | `uncertainty_quantifier.py` |
| **Frontend Refresh** | SVG icons, grid layouts, live UQ dashboard | `templates/autoscaling.html`, `templates/monitoring.html` |
| **Backend APIs** | Trends, events, UQ pass-through | `ai_kubernetes_web_app.py`, `ai_monitoring_integration.py` |

---

## 1. MCDA Optimizer (TOPSIS)

### Why

Old autoscaling logic in `llm_autoscaling_advisor.py` was simple IF/ELSE:

```python
# Old heuristic approach — ignores trade-offs
if cpu > 75%:
    scale_up()
elif cpu < 25%:
    scale_down()
else:
    maintain()
```

This ignores cost, stability, forecast alignment, and response time trade-offs. An LLM might recommend scale-up when a cost-aware or stability-aware analysis would disagree.

### Where: `mcda_optimizer.py`

The full MCDA optimizer lives in a single file with these components:

| Component | Lines | Purpose |
|---|---|---|
| `WEIGHT_PROFILES` | 75–104 | 4 pre-defined priority profiles |
| `CRITERIA` | ~60–70 | 5 criteria definitions with direction |
| `generate_alternatives()` | 138–210 | Generates 5–8 candidate replica counts |
| `_evaluate_alternative()` | 212–284 | Scores each candidate on all 5 criteria |
| `topsis_rank()` | 286–349 | Full TOPSIS ranking algorithm |
| `validate_llm_decision()` | 440–518 | Cross-validates LLM vs MCDA |

### Flow

```
Current State (replicas, CPU, memory, forecasts)
        │
        ▼
┌─────────────────────────────────────┐
│  generate_alternatives()            │  → 5-8 candidate replica counts
│  candidates: current ± {-2,-1,1,2,3}, min, max, ideal(CPU/70%)
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  _evaluate_alternative()  ×N        │  → Score each on 5 criteria
│  cost, performance, stability,      │
│  forecast_alignment, response_time  │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  topsis_rank()                      │  → TOPSIS: normalize → weight →
│  ideal/anti-ideal → distances →     │     closeness coefficient
│  sorted ranking                     │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  validate_llm_decision()            │  → Compare LLM target vs MCDA top
│  agreement: full / partial / diverge│     Override if gap > 0.15
└─────────────────────────────────────┘
```

### Criteria & Weights

Defined in `mcda_optimizer.py` lines 75–104:

```python
WEIGHT_PROFILES = {
    'balanced': {
        'cost': 0.20,
        'performance': 0.30,
        'stability': 0.25,
        'forecast_alignment': 0.15,
        'response_time': 0.10
    },
    'performance_first': {
        'cost': 0.10, 'performance': 0.40, 'stability': 0.15,
        'forecast_alignment': 0.20, 'response_time': 0.15
    },
    'cost_optimized': {
        'cost': 0.40, 'performance': 0.20, 'stability': 0.20,
        'forecast_alignment': 0.10, 'response_time': 0.10
    },
    'stability_first': {
        'cost': 0.15, 'performance': 0.20, 'stability': 0.40,
        'forecast_alignment': 0.15, 'response_time': 0.10
    }
}
```

### How Each Criterion is Evaluated

From `_evaluate_alternative()` (lines 212–284):

```python
# Cost: normalized by max_replicas (more replicas = more cost)
cost = target / max(max_replicas, 1)

# Performance: how close CPU will be to the 60-80% sweet spot
estimated_cpu = cpu * (current_replicas / target)
perf = 1.0 - abs(estimated_cpu - 70.0) / 70.0  # peak at 70%

# Stability: penalize large changes + oscillation
change_magnitude = abs(target - current_replicas)
stability_risk = min(1.0, change_magnitude / replica_range)
if target < current_replicas and cpu_trend == 'increasing':
    stability_risk += 0.2  # penalty for scaling down while CPU rises

# Forecast alignment: handle predicted peak in 50-75% comfort zone
estimated_peak_cpu = peak_cpu * (current_replicas / target)
alignment = 0.7 * cpu_alignment + 0.3 * mem_alignment

# Response time: higher CPU → higher latency
response_time = estimated_avg_cpu / 100.0
```

### TOPSIS Algorithm

From `topsis_rank()` (lines 286–349):

```python
def topsis_rank(self, alternatives):
    # Step 1: Build decision matrix [alternatives × criteria]
    matrix = np.array([[getattr(a, attr) for attr in attr_names]
                        for a in alternatives], dtype=float)

    # Step 2: Vector normalization
    norms = np.sqrt((matrix ** 2).sum(axis=0))
    normalized = matrix / norms

    # Step 3: Apply weights
    weights = np.array([self.weights.get(k, 0.0) for k in weight_keys])
    weighted = normalized * weights

    # Step 4: Ideal and anti-ideal solutions
    for i, is_benefit in enumerate(is_benefit):
        if is_benefit:
            ideal[i] = weighted[:, i].max()       # best = max for benefit
            anti_ideal[i] = weighted[:, i].min()
        else:
            ideal[i] = weighted[:, i].min()       # best = min for cost
            anti_ideal[i] = weighted[:, i].max()

    # Step 5: Euclidean distances
    dist_to_ideal = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    dist_to_anti  = np.sqrt(((weighted - anti_ideal) ** 2).sum(axis=1))

    # Step 6: Closeness coefficient → score ∈ [0, 1]
    scores = dist_to_anti / (dist_to_ideal + dist_to_anti)
    return sorted(zip(alternatives, scores), key=lambda x: x[1], reverse=True)
```

### LLM Cross-Validation

From `validate_llm_decision()` (lines 440–518):

```python
def validate_llm_decision(self, llm_action, llm_target, current_replicas,
                           min_replicas, max_replicas, metrics, forecast):
    mcda_result = self.optimize(current_replicas, min_replicas, max_replicas,
                                 metrics, forecast)

    score_difference = mcda_result.mcda_score - llm_score

    if direction_agrees and abs(llm_target - mcda_result.target_replicas) <= 1:
        agreement = 'full'
        should_override = False
    elif direction_agrees:
        agreement = 'partial'
        should_override = score_difference > 0.15  # threshold
    else:
        agreement = 'disagree'
        should_override = score_difference > 0.15

    return {
        'agreement': agreement, 'should_override': should_override,
        'llm_score': llm_score, 'mcda_score': mcda_result.mcda_score,
        'score_difference': score_difference, 'mcda_target': mcda_result.target_replicas,
        'criteria_weights': mcda_result.criteria_weights, 'ranking': mcda_result.ranking
    }
```

### Integration Point

In `llm_autoscaling_advisor.py` (lines 1080–1101), after the LLM generates a recommendation:

```python
# Cross-validate with MCDA
validation = self.mcda_optimizer.validate_llm_decision(
    llm_action=llm_action, llm_target=llm_target,
    current_replicas=current_reps,
    min_replicas=min_replicas, max_replicas=max_replicas,
    metrics=mcda_metrics, forecast=mcda_forecast
)

# Attach to recommendation dict → flows to frontend
recommendation['mcda_validation'] = {
    'agreement': validation['agreement'],
    'llm_score': validation['llm_score'],
    'mcda_score': validation['mcda_score'],
    'mcda_target': validation['mcda_target'],
    'score_difference': validation['score_difference'],
    'dominance_margin': validation['dominance_margin'],
    'criteria_weights': validation['criteria_weights'],
    'should_override': validation['should_override']
}
```

---

## 2. Uncertainty Quantification (UQ)

### Why

Phase 1 provided only point predictions with:
- Constant-width confidence intervals (did not grow with horizon)
- Binary anomaly detection (anomaly or not, no probability)
- No information about whether predictions were reliable

### Where: `uncertainty_quantifier.py`

| Component | Lines | Purpose |
|---|---|---|
| `UncertainForecast` | 33–48 | Dataclass for forecast + uncertainty |
| `CalibratedAnomaly` | 51–62 | Dataclass for calibrated anomaly |
| `UncertaintyAwareForecaster` | 80–220 | Bootstrap forecasting engine |
| `CalibratedAnomalyDetector` | 300–455 | Platt scaling calibration |
| `UncertaintyAwareScaler` | 460–640 | Propagates uncertainty to decisions |

### Flow

```
Historical Metrics (CPU/Memory time series)
        │
        ▼
┌─────────────────────────────────────┐
│  UncertaintyAwareForecaster         │
│                                     │
│  1. Fit trend on original data      │
│  2. Bootstrap 50× resampled fits    │
│  3. Per-horizon uncertainty:        │
│     σ_total(h) = √(σ_alea²·√h      │
│                   + σ_epis²·h)      │
│  4. Prediction intervals [lo, hi]   │
│  5. Exceedance P(>50/70/80/90%)     │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  CalibratedAnomalyDetector          │
│                                     │
│  1. Isolation Forest → raw score    │
│  2. Platt: P = σ(A·score + B)      │
│  3. Severity distribution           │
│  4. Detection confidence            │
└───────────────┬─────────────────────┘
                │
                ▼
        UncertainForecast + CalibratedAnomaly
        (serialized to JSON in API response)
```

### Data Structures

From `uncertainty_quantifier.py` lines 33–62:

```python
@dataclass
class UncertainForecast:
    metric_name: str
    current_value: float
    point_forecasts: List[float]                        # one per hour
    prediction_intervals: List[Tuple[float, float]]     # [(lower, upper), ...]
    aleatoric_uncertainty: List[float]                   # data noise per hour
    epistemic_uncertainty: List[float]                   # model uncertainty per hour
    total_uncertainty: List[float]                       # combined per hour
    exceedance_probabilities: Dict[int, List[float]]    # {50: [...], 70: [...], ...}
    confidence_level: float                              # 0.95
    trend: str                                           # 'increasing'/'decreasing'/'stable'
    recommendation: str
    data_points_used: int
    model_quality: str                                   # 'good'/'limited'/'poor'

@dataclass
class CalibratedAnomaly:
    timestamp: datetime
    anomaly_probability: float          # P(anomaly) ∈ [0, 1]
    raw_score: float                    # Isolation Forest score
    is_anomaly: bool                    # probability > 0.5
    detection_confidence: float         # |P - 0.5| × 2
    severity_distribution: Dict[str, float]  # {low, medium, high, critical}
    affected_metrics: List[str]
    recommendation: str
```

### Bootstrap Prediction Intervals

From `forecast_with_uncertainty()` (lines 132–193):

```python
# --- Epistemic uncertainty via bootstrap ---
bootstrap_predictions = {h: [] for h in range(1, hours_ahead + 1)}

for _ in range(self.n_bootstrap):  # n_bootstrap = 50
    indices = np.random.choice(len(data), size=len(data), replace=True)
    boot_data = data[indices]
    boot_coeffs = np.polyfit(boot_x, boot_data, 1)
    for h in range(1, hours_ahead + 1):
        pred = np.polyval(boot_coeffs, len(data) + h)
        bootstrap_predictions[h].append(pred)

# --- Per-horizon uncertainty that GROWS ---
for h in range(1, hours_ahead + 1):
    point = np.polyval(coeffs, len(data) + h)

    # Epistemic: variance across bootstrap predictions
    epistemic_var = np.var(bootstrap_predictions[h])

    # Total grows with horizon: sqrt for aleatoric, linear for epistemic
    horizon_factor = np.sqrt(h)
    total_var = aleatoric_var * horizon_factor + epistemic_var * h
    total_std = np.sqrt(total_var)

    # Prediction interval
    lower = point - z_score * total_std    # z_score = 1.96 for 95%
    upper = point + z_score * total_std

    # Exceedance: P(metric > threshold)
    for threshold in [50, 70, 80, 90]:
        z = (threshold - point) / total_std
        prob = self._normal_sf(z)  # 1 - Φ(z), no scipy needed
        exceedance_probs[threshold].append(prob)
```

**Why it matters**: Hour 1 interval might be ±5%, while Hour 6 is ±8.5%. This tells operators that near-term forecasts are reliable but longer-horizon ones have significant uncertainty.

### Platt Scaling for Anomaly Calibration

From `compute_anomaly_probability()` (lines 342–369):

```python
def compute_anomaly_probability(self, raw_score):
    """Convert Isolation Forest score → calibrated P(anomaly)"""

    # Platt scaling: sigmoid(A * score + B)
    logit = self.platt_A * raw_score + self.platt_B  # A=-5.0, B=-0.5
    logit = max(-20.0, min(20.0, logit))             # clip for numerical stability
    anomaly_probability = 1.0 / (1.0 + np.exp(-logit))

    # Detection confidence: distance from decision boundary (0.5)
    detection_confidence = abs(anomaly_probability - 0.5) * 2.0

    # Severity distribution
    severity_distribution = self._compute_severity_distribution(anomaly_probability)

    return {
        'anomaly_probability': anomaly_probability,  # e.g., 0.0527
        'detection_confidence': detection_confidence, # e.g., 0.89
        'severity_distribution': severity_distribution
    }
```

**Why Platt scaling**: Isolation Forest outputs raw scores that are not probabilities. Platt scaling maps them to `[0, 1]` via a sigmoid, giving operators a meaningful "5.3% chance of anomaly" instead of a binary flag.

### Severity Distribution

From `_compute_severity_distribution()` (lines 423–455):

```python
def _compute_severity_distribution(self, anomaly_prob):
    p = anomaly_prob
    if p < 0.2:
        dist = {'low': 0.85, 'medium': 0.12, 'high': 0.02, 'critical': 0.01}
    elif p < 0.4:
        # Smooth interpolation
        dist = {'low': max(0, 0.5-p), 'medium': 0.4,
                'high': p*0.3, 'critical': 0.01}
    elif p < 0.6:
        dist = {'low': 0.05, 'medium': 0.35, 'high': 0.45, 'critical': 0.15}
    elif p < 0.8:
        dist = {'low': 0.02, 'medium': 0.13, 'high': 0.50, 'critical': 0.35}
    else:
        dist = {'low': 0.01, 'medium': 0.04, 'high': 0.25, 'critical': 0.70}
    return {k: round(v / sum(dist.values()), 4) for k, v in dist.items()}
```

### Where UQ Data Flows to the API

In `predictive_monitoring.py` (lines 800–850), `analyze()` builds the full response:

```python
"uncertainty_quantification": {
    "cpu": {
        "point_forecasts": cpu_forecast_uq.point_forecasts,
        "prediction_intervals": cpu_forecast_uq.prediction_intervals,
        "aleatoric_uncertainty": cpu_forecast_uq.aleatoric_uncertainty,
        "epistemic_uncertainty": cpu_forecast_uq.epistemic_uncertainty,
        "total_uncertainty": cpu_forecast_uq.total_uncertainty,
        "exceedance_probabilities": cpu_forecast_uq.exceedance_probabilities,
        "confidence_level": cpu_forecast_uq.confidence_level,
        "model_quality": cpu_forecast_uq.model_quality,
        "data_points_used": cpu_forecast_uq.data_points_used
    },
    "memory": { ... }  # same structure
},
"anomaly_detection": {
    "is_anomaly": anomaly_result.is_anomaly,
    "calibrated": {
        "anomaly_probability": anomaly_calibrated.anomaly_probability,
        "detection_confidence": anomaly_calibrated.detection_confidence,
        "severity_distribution": anomaly_calibrated.severity_distribution,
        "raw_score": anomaly_calibrated.raw_score
    }
}
```

---

## 3. Backend Changes

### Bug Fix: UQ Data Pass-Through

**Problem**: `get_dashboard_data()` in `ai_monitoring_integration.py` was building its own response dict and **dropping** the `uncertainty_quantification` and `anomaly_detection` keys from `PredictiveMonitoringSystem.analyze()`.

**Fix** (lines 934–961):

```python
def get_dashboard_data(self):
    analysis = self.get_current_analysis()  # calls monitoring_system.analyze()

    dashboard = {
        "timestamp": ..., "current_metrics": ..., "health_score": ...,
        "forecasts": ..., "alerts": ..., "recommendations": ...,
    }

    # NEW: Pass through UQ data that was previously dropped
    if "uncertainty_quantification" in analysis:
        dashboard["uncertainty_quantification"] = analysis["uncertainty_quantification"]
    if "anomaly_detection" in analysis:
        dashboard["anomaly_detection"] = analysis["anomaly_detection"]

    return dashboard
```

### New Route: `/api/monitoring/trends/<server_id>`

**Why**: The monitoring page needed trend history for charts, but no endpoint existed.

In `ai_kubernetes_web_app.py`:

```python
@app.route('/api/monitoring/trends/<int:server_id>')
def get_monitoring_trends(server_id):
    server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
    ai_monitoring = get_ai_monitoring(server_id)
    trends = ai_monitoring.get_trends_24h()  # returns {cpu: [...], memory: [...]}
    return jsonify(trends)
```

### New Route: `/api/monitoring/events/<server_id>`

**Why**: Kubernetes events were not being shown — no endpoint to fetch them.

```python
@app.route('/api/monitoring/events/<int:server_id>')
def get_monitoring_events(server_id):
    server = Server.query.filter_by(id=server_id, user_id=session['user_id']).first_or_404()
    # Try collector first, fallback to kubectl
    result = subprocess.run(
        ['kubectl', 'get', 'events', '--all-namespaces', '-o', 'json',
         '--sort-by=.lastTimestamp'],
        capture_output=True, text=True, timeout=10
    )
    # Parse and return last 20 events
    return jsonify({'events': events})
```

---

## 4. Frontend Changes

### SVG Icon System

**Why**: Emojis render inconsistently across browsers/OS and look unprofessional.

**Where**: `templates/autoscaling.html` (lines 700–715)

All 14 icons defined as Lucide-style inline SVGs in a single JS constant:

```javascript
const ICONS = {
    ai:       '<svg width="16" height="16" viewBox="0 0 24 24" ...><rect .../><path .../></svg>',
    chart:    '<svg ...><line x1="18" y1="20" x2="18" y2="10"/>...</svg>',
    hpa:      '<svg ...><path d="M18 8L22 12L18 16"/>...</svg>',
    vpa:      '<svg ...><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/>...</svg>',
    zap:      '<svg ...><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>...</svg>',
    check:    '<svg ...>...</svg>',
    warn:     '<svg ...>...</svg>',
    // ... 14 total
};
```

**Console logs**: All emojis replaced with text prefixes (`[INIT]`, `[SYNC]`, `[DEBUG]`, `[DATA]`, `[OK]`, `[ERR]`, `[BLOCKED]`, `[NET]`, `[WS]`, `[WARN]`).

### Recommendation Card — CSS Grid

**Why**: The old card had compressed elements stacked side-by-side without structure.

**Where**: `templates/autoscaling.html` (lines 2362–2405)

The entire card is one CSS Grid container:

```javascript
recommendationHTML += `
    <div class="alert" style="
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        grid-template-rows: auto auto auto auto;
    ">
        <!-- Row 1: Title — spans all 3 cols -->
        <div style="grid-column: 1 / -1;">
            ${ICONS.ai} LLM-Powered Recommendation: ${actionText}
        </div>

        <!-- Row 2: Confidence badge — spans all 3 cols -->
        <div style="grid-column: 1 / -1;">
            <span>${confidence}% Confidence</span>
        </div>

        <!-- Row 3: Reasoning — spans all 3 cols -->
        <div style="grid-column: 1 / -1;">
            <strong>Reasoning:</strong>
            <p>${rec.reasoning}</p>
        </div>

        <!-- Row 4: 3 equal columns -->
        <div style="grid-column: 1 / 2;">${statusColumnHTML}</div>
        <div style="grid-column: 2 / 3;">${scalingColumnHTML}</div>
        <div style="grid-column: 3 / 4;">${applyButtonHTML}</div>
    </div>`;
```

### Monitoring — UQ Dashboard

**Why**: UQ data was generated by the backend but never displayed.

**Where**: `templates/monitoring.html` — new `loadUncertaintyQuantification()` function

Key rendering logic for exceedance probabilities:

```javascript
thresholds.forEach(t => {
    const probs = exceedance[t] || [];
    const nextHourProb = probs[0] || 0;
    const pct = (nextHourProb * 100).toFixed(1);
    const barColor = nextHourProb > 0.5 ? '#ef4444'    // red
                   : nextHourProb > 0.2 ? '#f59e0b'    // amber
                   : '#10b981';                          // green

    exceedHTML += `
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span>P(>${t}%)</span>
            <div style="flex: 1; background: rgba(0,0,0,0.06); height: 18px;">
                <div style="width: ${barWidth}%; background: ${barColor};"></div>
            </div>
            <span>${pct}%</span>
        </div>`;
});
```

Uncertainty decomposition (aleatoric vs epistemic stacked bar):

```javascript
const aleRatio = (aleatoric[0] / total[0]) * 100;
const epiRatio = 100 - aleRatio;

decompHTML += `
    <div style="display: flex; height: 22px; border-radius: 0.25rem;">
        <div style="width: ${aleRatio}%; background: #f59e0b;">Aleatoric</div>
        <div style="width: ${epiRatio}%; background: #3b82f6;">Epistemic</div>
    </div>`;
```

---

## 5. Complete Data Flow

```
                    ┌─────────────────────────────────┐
                    │     Kubernetes Cluster           │
                    │  (CrownLabs, ai4k8s-test ns)    │
                    └───────────────┬─────────────────┘
                                    │ kubectl / metrics API
                                    ▼
┌──────────────────────────────────────────────────────────────┐
│  predictive_monitoring.py  —  PredictiveMonitoringSystem     │
│                                                              │
│  ┌──────────────────┐    ┌─────────────────────────────┐    │
│  │ TimeSeriesForecaster│   │ AnomalyDetector              │    │
│  │ + UncertaintyAware │   │ + CalibratedAnomalyDetector  │    │
│  │   Forecaster       │   │   (Platt scaling)            │    │
│  │                    │   │                              │    │
│  │ → point_forecasts  │   │ → anomaly_probability        │    │
│  │ → pred_intervals   │   │ → detection_confidence       │    │
│  │ → aleatoric/epist  │   │ → severity_distribution      │    │
│  │ → exceedance_probs │   │                              │    │
│  └────────┬───────────┘    └──────────┬───────────────────┘    │
│           │                           │                        │
│           └─────────┬─────────────────┘                        │
│                     ▼                                          │
│              analyze()  →  { uncertainty_quantification,       │
│                              anomaly_detection.calibrated }    │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  llm_autoscaling_advisor.py  —  LLMAutoscalingAdvisor        │
│                                                              │
│  1. Call Qwen (local) or Groq (cloud) for recommendation     │
│  2. Parse LLM response → {action, target_replicas, ...}      │
│  3. mcda_optimizer.validate_llm_decision()                   │
│     ├── TOPSIS ranking on 5-8 alternatives                   │
│     ├── Compare LLM target vs MCDA optimal                   │
│     └── Override if score_gap > 0.15                         │
│  4. Attach mcda_validation to recommendation dict            │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  ai_monitoring_integration.py  —  get_dashboard_data()       │
│                                                              │
│  Merges: forecasts + recommendations + UQ + anomaly_detection│
│  (previously dropped UQ — now fixed)                         │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  ai_kubernetes_web_app.py  —  Flask API Routes               │
│                                                              │
│  /api/monitoring/insights/<id>  → full dashboard + UQ        │
│  /api/monitoring/trends/<id>    → 24h CPU/memory history     │
│  /api/monitoring/events/<id>    → recent K8s events          │
│  /api/autoscaling/recommendations/<id>  → LLM + MCDA        │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Frontend (templates/)                                       │
│                                                              │
│  autoscaling.html                                            │
│  ├── Recommendation card (CSS Grid, SVG icons)               │
│  ├── MCDA validation card (agreement, scores, ranking)       │
│  └── Apply button (HPA/VPA/both)                             │
│                                                              │
│  monitoring.html                                             │
│  ├── UQ Dashboard (prediction intervals, exceedance bars)    │
│  ├── Uncertainty decomposition (aleatoric vs epistemic)      │
│  ├── Calibrated anomaly (P(anomaly), severity distribution)  │
│  ├── Resource trends (24h history)                           │
│  └── Kubernetes events (last 20)                             │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. Files Modified — Summary

| File | Changes | Reason |
|---|---|---|
| `mcda_optimizer.py` | New file (518 lines) | TOPSIS multi-criteria optimizer |
| `uncertainty_quantifier.py` | New file (640 lines) | Bootstrap UQ + Platt scaling |
| `predictive_monitoring.py` | Lines 800–850 | Serializes UQ data in `analyze()` |
| `llm_autoscaling_advisor.py` | Lines 1080–1101 | Integrates MCDA cross-validation |
| `ai_monitoring_integration.py` | Lines 934–961 | Passes UQ data through to API |
| `ai_kubernetes_web_app.py` | Lines 2092–2139 | New trends + events API routes |
| `templates/autoscaling.html` | ~200 lines changed | SVG icons, CSS Grid layout, MCDA card |
| `templates/monitoring.html` | ~250 lines added | UQ dashboard, exceedance bars, decomposition |

---

## 7. Key Numbers

| Metric | Value |
|---|---|
| MCDA criteria | 5 |
| Weight profiles | 4 |
| Alternatives per decision | 5–8 |
| Bootstrap resamples | 50 |
| Confidence level | 95% |
| Exceedance thresholds | 50%, 70%, 80%, 90% |
| Forecast horizon | 6 hours |
| Override threshold | score gap > 0.15 |
| Platt params | A = −5.0, B = −0.5 |
| SVG icons | 14 |
| Emojis removed | ~120+ |
| New API routes | 2 |

---

*AI4K8s — Intelligent Kubernetes Management Platform*
