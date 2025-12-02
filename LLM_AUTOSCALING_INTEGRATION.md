# LLM-Based Autoscaling Integration

## Overview

AI4K8s now uses **Groq LLM** to make intelligent autoscaling decisions, complementing the existing ML-based predictive monitoring system. The LLM analyzes multiple factors including current metrics, forecasts, HPA status, and historical patterns to provide intelligent scaling recommendations.

## Architecture

### Components

1. **LLMAutoscalingAdvisor** (`llm_autoscaling_advisor.py`)
   - Uses Groq LLM (llama-3.1-70b-versatile or llama-3.1-8b-instant)
   - Analyzes deployment metrics, forecasts, and patterns
   - Provides JSON-formatted recommendations with reasoning

2. **PredictiveAutoscaler** (Enhanced)
   - Integrates LLM advisor for intelligent recommendations
   - Falls back to rule-based decisions if LLM unavailable
   - Combines ML forecasts with LLM reasoning

3. **AutoscalingIntegration**
   - Orchestrates LLM, ML, and rule-based recommendations
   - Provides unified API for all recommendation sources

## How It Works

### Decision Flow

```
1. Get Current Metrics (CPU/Memory usage)
   ↓
2. Get Predictive Forecasts (ML-based, 6-hour horizon)
   ↓
3. Get HPA Status (if exists)
   ↓
4. LLM Analysis:
   - Analyzes all factors
   - Considers cost, performance, stability
   - Provides recommendation with reasoning
   ↓
5. Execute or Display Recommendation
```

### LLM Input Context

The LLM receives:
- **Deployment Info**: Name, namespace, current/min/max replicas
- **Current Metrics**: CPU%, Memory%, pod counts
- **Forecast Data**: Current values, peak predictions, trends, 6-hour predictions
- **HPA Status**: Current/desired replicas, targets, scaling status
- **Historical Patterns**: (Optional) Previous scaling decisions

### LLM Output

The LLM returns JSON with:
```json
{
  "action": "scale_up" | "scale_down" | "maintain" | "at_max",
  "target_replicas": <number>,
  "confidence": <0.0-1.0>,
  "reasoning": "<detailed explanation>",
  "factors_considered": ["factor1", "factor2", ...],
  "risk_assessment": "low" | "medium" | "high",
  "cost_impact": "low" | "medium" | "high",
  "performance_impact": "positive" | "neutral" | "negative",
  "recommended_timing": "immediate" | "gradual" | "scheduled"
}
```

## Setup

### 1. Set GROQ_API_KEY

```bash
export GROQ_API_KEY=your_groq_api_key_here
```

Or add to `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 2. Verify Installation

```python
from llm_autoscaling_advisor import LLMAutoscalingAdvisor
advisor = LLMAutoscalingAdvisor()
print(f"LLM Advisor available: {advisor.client is not None}")
```

## Usage

### Automatic Integration

LLM recommendations are automatically used when:
1. Predictive autoscaling is enabled for a deployment
2. `GROQ_API_KEY` is set
3. User requests scaling recommendations

### API Endpoints

- `GET /api/autoscaling/recommendations/<server_id>?deployment=<name>&namespace=<ns>`
  - Returns recommendations from all sources including LLM

### Frontend Display

The autoscaling dashboard shows:
- **LLM-Powered Recommendation** (if available) with:
  - Confidence score
  - Detailed reasoning
  - Risk assessment
  - Cost/Performance impact
  - Recommended timing
  - Factors considered

- **Rule-Based Recommendation** (fallback if LLM unavailable)

## Benefits

### LLM Advantages

1. **Contextual Understanding**: Considers multiple factors simultaneously
2. **Cost Optimization**: Balances performance vs. cost
3. **Risk Assessment**: Evaluates stability implications
4. **Explainable**: Provides detailed reasoning for decisions
5. **Adaptive**: Can learn from patterns and adjust recommendations

### Hybrid Approach

- **ML Forecasting**: Provides accurate predictions
- **LLM Reasoning**: Makes intelligent decisions based on predictions
- **Rule-Based Fallback**: Ensures system works even without LLM

## Example LLM Recommendation

```
🤖 LLM-Powered Recommendation: Scale Up
Confidence: 85%

Reasoning:
Current CPU usage is at 182% which is well above the 70% target threshold.
The forecast shows this trend continuing for the next 2-3 hours. Scaling up
to 15 replicas would bring CPU usage down to approximately 60-65%, providing
adequate headroom for traffic spikes while maintaining good performance.
The cost increase is moderate but justified given the performance requirements.

Target Replicas: 15
Risk: LOW
Cost Impact: MEDIUM
Performance: POSITIVE
Timing: IMMEDIATE

Factors Considered:
- Current resource pressure (CPU 182%)
- Predicted future demand (stable high)
- Cost optimization opportunities
- Performance requirements
- Stability considerations
```

## Configuration

### Model Selection

The system tries models in order:
1. `llama-3.1-70b-versatile` (preferred - better reasoning)
2. `llama-3.1-8b-instant` (fallback - faster)

### Temperature

Set to `0.3` for consistent, deterministic recommendations.

### Fallback Behavior

If LLM is unavailable:
- System falls back to rule-based recommendations
- ML forecasts still work
- No functionality is lost

## Testing

1. **Enable Predictive Autoscaling** for a deployment
2. **Set GROQ_API_KEY** environment variable
3. **View Recommendations** - should show LLM-powered recommendations
4. **Check Logs** - verify LLM calls are being made

## Troubleshooting

### LLM Not Working

1. Check `GROQ_API_KEY` is set: `echo $GROQ_API_KEY`
2. Check logs for LLM errors
3. Verify Groq package is installed: `pip show groq`
4. Test API key: Try calling Groq API directly

### Fallback to Rule-Based

If you see "LLM advisor available but using rule-based recommendation":
- LLM may have failed to parse response
- Check logs for JSON parsing errors
- System will still work with rule-based recommendations

## Future Enhancements

- **Historical Learning**: Train LLM on past scaling decisions
- **Multi-Factor Analysis**: Consider network, disk I/O, custom metrics
- **Cost Optimization**: Integrate cloud provider pricing
- **SLA-Aware**: Consider service level agreements
- **A/B Testing**: Compare LLM vs rule-based performance

