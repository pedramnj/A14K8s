# VPA Integration & LLM-Based Autoscaling: A First-Person Journey

**Presented by:** AI4K8s Development Team  
**Date:** December 2025

---

## Introduction: The Challenge I Faced

I was working on enhancing the AI4K8s platform to make it truly intelligent. The system could already do horizontal scaling (HPA) - adding more pods when needed. But I realized that wasn't enough. Some applications need vertical scaling (VPA) - giving each pod more CPU and memory resources instead of creating more pods.

The real challenge? **How do I teach an AI system to decide between HPA and VPA?** This isn't just about numbers - it's about understanding application architecture.

---

## What I Built: The Complete Solution

### 1. The VPA Engine - My Foundation

I created a complete VPA management system (`vpa_engine.py`) that can:
- Check if VPA is available in the cluster
- Patch deployment resources directly using `kubectl patch`
- Calculate optimal CPU and memory targets based on current usage
- Handle strategic merge patches to preserve all container fields

**Key Innovation:** Instead of creating VPA resources (which require the VPA controller), I made Predictive Autoscaling patch deployment resources directly. This gives me full control and avoids conflicts.

### 2. The LLM Decision Maker - The Brain

I enhanced the LLM Autoscaling Advisor (`llm_autoscaling_advisor.py`) to be truly intelligent. It now:

- **Analyzes application characteristics** - not just metrics, but state management patterns
- **Makes strategic decisions** - chooses between HPA (more replicas) or VPA (more resources per pod)
- **Provides detailed reasoning** - explains why it chose HPA or VPA
- **Respects constraints** - ensures recommendations stay within min/max replica limits

**The Magic:** I crafted detailed prompts that guide the LLM through a decision tree:
- Is state externalized (Redis, DB)? → HPA
- Is state inside the pod? → VPA  
- Uncertain? → VPA (safer choice)

### 3. State Management Detection - My Detective System

I built a comprehensive multi-source detection system that analyzes:

1. **Deployment Annotations** - Users can explicitly mark applications as stateless/stateful
2. **Environment Variables** - Looks for REDIS, DATABASE, POSTGRES, etc.
3. **Volume Mounts** - Detects persistent storage (PVCs, hostPath)
4. **Service Dependencies** - Checks for Redis/DB services in the namespace
5. **Labels** - Reads `ai4k8s.io/state-management` labels

**Confidence Levels:**
- **High confidence** - User annotation (most reliable)
- **Medium confidence** - Detected from environment variables or volumes
- **Low confidence** - No clear indicators found

### 4. Post-Processing Validation - My Safety Net

I implemented multiple layers of validation:

- **Replica Constraints:** Caps `target_replicas` to `max_replicas` if LLM exceeds limits
- **State Enforcement:** If user selects "stateless" but LLM recommends VPA, I correct it to HPA
- **Reasoning Validation:** Checks if LLM's reasoning matches its decision
- **Final Safety Check:** Validates one last time before returning to the frontend

**Example:** If LLM recommends 8 replicas but max is 5, I cap it to 5 and update the action to `at_max`.

### 5. The User Interface - Making It Accessible

I enhanced the web UI (`templates/autoscaling.html`) with:

- **State Management Dropdown** - Users can specify stateless/stateful/auto-detect
- **VPA Stat Card** - Shows count of active VPAs
- **VPA Recommendations Display** - Shows current vs target CPU/Memory resources
- **Apply VPA Button** - One-click application of VPA recommendations
- **Dark Theme Support** - Proper styling for both light and dark modes

**User Experience:** The dropdown has beautiful styling with icons, gradients, and hover effects. It adapts to the theme automatically.

---

## How It Works: The Complete Flow

### Step 1: User Enables Predictive Autoscaling
When a user enables predictive autoscaling, I:
- Store configuration in Kubernetes annotations
- Set the state management annotation if provided
- Track min/max replicas for validation

### Step 2: Gathering Context
Before making a decision, I collect:
- Current CPU/Memory usage metrics
- 6-hour forecast predictions
- HPA status (if exists)
- VPA status (if exists)
- Current resource requests/limits
- **State management information** (from my detection system)

### Step 3: LLM Analysis
I send all this context to the Groq LLM API with carefully crafted prompts:

**System Prompt:** Provides guidelines on when to use HPA vs VPA
**User Prompt:** Contains all the deployment-specific context

The LLM returns:
```json
{
  "scaling_type": "hpa" | "vpa" | "both",
  "target_replicas": 5,
  "target_cpu": "200m",
  "target_memory": "256Mi",
  "reasoning": "Detailed explanation...",
  "confidence": 0.8
}
```

### Step 4: Validation & Correction
I validate the LLM's recommendation:
- Check if `target_replicas` respects min/max limits
- Verify `scaling_type` matches the reasoning
- Correct HPA→VPA if state is detected inside pod
- Correct VPA→HPA if user explicitly selected stateless

### Step 5: Applying the Decision

**For HPA:**
```bash
kubectl scale deployment test-app --replicas=5
```

**For VPA:**
```bash
kubectl patch deployment test-app --type=merge -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "app",
          "resources": {
            "requests": {"cpu": "200m", "memory": "256Mi"},
            "limits": {"cpu": "400m", "memory": "384Mi"}
          }
        }]
      }
    }
  }
}'
```

---

## Key Technical Decisions I Made

### 1. Direct Resource Patching Instead of VPA Resources
**Why:** Avoids conflicts with VPA controller and gives Predictive Autoscaling full control.

**How:** I use `kubectl patch` with strategic merge to update only the resources field while preserving all other container properties (image, ports, probes, etc.).

### 2. Multi-Source State Detection
**Why:** Single-source detection is unreliable. Multiple sources increase confidence.

**Priority Order:**
1. User annotation (highest priority)
2. Environment variables
3. Volume mounts
4. Service dependencies
5. Labels

### 3. Aggressive Stateless Enforcement
**Why:** If a user explicitly selects "stateless", they know their application better than the LLM.

**Implementation:** I check multiple indicators:
- Direct detection from annotation
- State note contains "stateless" + "HPA"
- State note contains "STATELESS" (uppercase)

If ANY indicator suggests stateless, I force HPA even if LLM recommended VPA.

### 4. Cache Key Includes State Management
**Why:** When state management changes, recommendations should be regenerated.

**Implementation:** The cache key includes `state_management` (e.g., `stateless_annotation`), so changing from "no state" to "stateless" invalidates the cache and triggers a new LLM call.

### 5. Multiple Validation Layers
**Why:** Defense in depth - validate at multiple points to catch any issues.

**Layers:**
1. LLM advisor validation (after parsing response)
2. Predictive autoscaler validation (after receiving recommendation)
3. Return validation (before sending to frontend)

---

## Challenges I Overcame

### Challenge 1: LLM Sometimes Ignored State Management
**Problem:** Even with explicit "stateless" selection, LLM sometimes recommended VPA.

**Solution:** I implemented aggressive enforcement logic that checks multiple indicators and overrides LLM if needed. I also enhanced the prompts to be more explicit about state management rules.

### Challenge 2: Target Replicas Exceeding Max
**Problem:** LLM sometimes recommended more replicas than the max limit.

**Solution:** I added validation at three different points:
- In `_parse_llm_response` (LLM advisor)
- In `get_scaling_recommendation` (predictive autoscaler)
- Right before returning the response (final check)

### Challenge 3: Cached Recommendations Ignoring State Changes
**Problem:** Changing state management didn't invalidate cache, so old recommendations persisted.

**Solution:** I modified the cache key generation to include state management information, ensuring cache invalidation when state changes.

### Challenge 4: kubectl Patch Syntax Errors
**Problem:** Initial implementation had syntax errors with `kubectl patch`.

**Solution:** I switched from JSON patch (array format) to strategic merge patch (object format) and ensured all container fields are preserved.

### Challenge 5: Dark Theme UI Issues
**Problem:** State management dropdown had white text on white background in dark theme.

**Solution:** I replaced hardcoded colors with CSS variables (`var(--surface)`, `var(--text-primary)`) that adapt to the theme.

---

## What I Learned

### 1. LLM Prompt Engineering is Critical
The quality of LLM decisions directly correlates with prompt quality. I spent significant time refining prompts to be explicit about rules and constraints.

### 2. Multiple Validation Layers Are Essential
Even with good prompts, LLMs can make mistakes. Multiple validation layers catch edge cases.

### 3. User Input Should Override AI
When users explicitly specify state management, their knowledge should take precedence over AI inference.

### 4. Caching Must Consider All Context
Cache keys must include all relevant context (state management, min/max replicas) to prevent stale recommendations.

### 5. Direct Patching is More Reliable
Direct resource patching gives more control than creating VPA resources and avoids controller dependencies.

---

## Results & Impact

### Before VPA Integration
- ❌ Only horizontal scaling (HPA)
- ❌ No way to optimize resource allocation per pod
- ❌ LLM couldn't distinguish between stateless and stateful apps
- ❌ Manual resource adjustment required

### After VPA Integration
- ✅ Intelligent HPA/VPA decision making
- ✅ Automatic resource optimization
- ✅ State-aware scaling decisions
- ✅ One-click resource adjustment
- ✅ Comprehensive state detection
- ✅ Multi-layer validation for reliability

### Metrics
- **Files Modified:** 5 core files
- **Files Created:** 1 new file (`vpa_engine.py`)
- **Lines of Code:** ~2,000+ lines added/modified
- **API Endpoints:** 2 new endpoints
- **UI Components:** 3 new components (VPA stat card, state dropdown, VPA display)

---

## Future Enhancements I'm Planning

1. **Automatic State Detection from Pod Behavior** - Analyze pod patterns to detect stateful vs stateless automatically

2. **ML-Based Resource Calculation** - Use machine learning models to predict optimal resource requests/limits

3. **VPA Recommender Integration** - Integrate with VPA recommender API for better resource suggestions

4. **Resource Usage History** - Track resource usage over time to improve VPA target calculations

5. **Multi-Container Support** - Enhanced handling for deployments with multiple containers

6. **Custom Metrics Support** - Support custom metrics beyond CPU/Memory for VPA decisions

---

## Conclusion

This VPA integration represents a significant leap forward for AI4K8s. I've transformed it from a simple horizontal scaler into an intelligent, state-aware autoscaling system that understands application architecture and makes optimal scaling decisions.

The system now provides:
- **Intelligent scaling decisions** based on application characteristics
- **Comprehensive VPA support** with direct resource patching
- **Advanced state detection** from multiple sources
- **User-friendly interface** with beautiful UI components
- **Robust validation** ensuring reliability and correctness

Most importantly, I've created a system that learns from user input, respects constraints, and makes intelligent decisions that balance performance, cost, and reliability.

---

## Appendix: Prompts, Flow, and Code Locations

### LLM Prompt System

I designed a two-prompt system to guide the LLM through intelligent decision-making:

#### System Prompt
**Location:** `llm_autoscaling_advisor.py`, method `_create_system_prompt()` (lines 852-912)

**Purpose:** Provides the LLM with foundational knowledge and rules about Kubernetes autoscaling.

**Key Content:**
```
You are an expert Kubernetes autoscaling advisor with deep knowledge of:
- Resource optimization and cost management
- Performance requirements and SLA considerations
- Scaling best practices and anti-patterns
- Predictive analysis and trend interpretation
- Horizontal Pod Autoscaling (HPA) vs Vertical Pod Autoscaling (VPA)

**IMPORTANT: You must decide between TWO scaling strategies:**

1. **HORIZONTAL SCALING (HPA)**: Scale by adjusting the NUMBER of replicas (pods)
   - Use when: Load can be distributed across multiple pods, need high availability
   - **CRITICAL RULE**: Applications that externalize their state (use Redis, external 
     databases, external cache, shared storage) should be treated as STATELESS and prefer HPA
   - Example: 3 pods → 5 pods (same resources per pod)

2. **VERTICAL SCALING (VPA)**: Scale by adjusting RESOURCE requests/limits per pod
   - Use when: Application keeps critical state INSIDE the pod (not externalized), 
     cannot scale horizontally, single-pod bottleneck
   - **CRITICAL RULE**: Only prefer VPA if state is stored INSIDE the pod. If state is 
     externalized (Redis, DB, external cache), treat as stateless and prefer HPA instead
   - Example: CPU 100m → 200m, Memory 128Mi → 256Mi (same number of pods)

**IMPORTANT STATE MANAGEMENT RULES:**
- **NEVER assume "stateful = only VPA"**
- **ALWAYS check if state is externalized before ruling out HPA**
- If application uses Redis, external databases, external cache, or shared storage 
  → treat as STATELESS → prefer HPA
- If application keeps critical state inside the pod (local files, in-memory state 
  without externalization) → prefer VPA

**IMPORTANT CONSTRAINTS:**
- If you recommend HPA scaling, target_replicas MUST be >= min_replicas AND <= max_replicas
- If current replicas is already at max_replicas and you need more, recommend "at_max" 
  action or suggest VPA instead
- If current replicas is already at min_replicas and you need less, recommend "maintain" action

Respond in JSON format with:
{
  "scaling_type": "hpa" | "vpa" | "both" | "maintain",
  "action": "scale_up" | "scale_down" | "maintain" | "at_max",
  "target_replicas": <number> (for HPA, MUST be between min_replicas and max_replicas, null if VPA),
  "target_cpu": "<value>" (for VPA, e.g., "200m", null if HPA),
  "target_memory": "<value>" (for VPA, e.g., "256Mi", null if HPA),
  "confidence": <0.0-1.0>,
  "reasoning": "<detailed explanation including why HPA vs VPA was chosen and why target_replicas respects min/max limits>",
  "factors_considered": ["factor1", "factor2", ...],
  "risk_assessment": "low" | "medium" | "high",
  "cost_impact": "low" | "medium" | "high",
  "performance_impact": "positive" | "neutral" | "negative",
  "recommended_timing": "immediate" | "gradual" | "scheduled"
}
```

#### User Prompt
**Location:** `llm_autoscaling_advisor.py`, method `_create_user_prompt()` (lines 914-1010)

**Purpose:** Provides deployment-specific context for the LLM to analyze.

**Structure:**
```
Analyze the following Kubernetes deployment autoscaling scenario and provide a recommendation:

**Deployment Information:**
- Name: {deployment_name}
- Namespace: {namespace}
- Current Replicas: {current_replicas}
- Min Replicas: {min_replicas}
- Max Replicas: {max_replicas}

**Current Resource Usage:**
- CPU: {cpu_usage}%
- Memory: {memory_usage}%
- Running Pods: {running_pods}/{pod_count}

**Forecast Data:**
- CPU Current: {current}%, Peak: {peak}%, Trend: {trend}
- Memory Current: {current}%, Peak: {peak}%, Trend: {trend}
- CPU Predictions (next 6 hours): {predictions}
- Memory Predictions (next 6 hours): {predictions}

**Current Resource Configuration:**
- CPU Request: {cpu_request}
- CPU Limit: {cpu_limit}
- Memory Request: {memory_request}
- Memory Limit: {memory_limit}

**State Management Information:**
{state_management_note}
- This includes detection results from annotations, environment variables, 
  volume mounts, service dependencies, and labels
- Format: "✅✅✅ CRITICAL: State Management Detected (source, confidence): 
  Application is STATELESS/STATEFUL - [details]. **YOU MUST RECOMMEND HPA/VPA**"

**CRITICAL STATE MANAGEMENT RULES:**
- If the state management information above says "STATELESS" or "state is externalized" 
  → YOU MUST RECOMMEND HPA (horizontal scaling)
- If the state management information above says "STATEFUL" or "state is stored inside 
  the pod" → YOU MUST RECOMMEND VPA (vertical scaling)
- If the state management information says "No state management information detected" 
  → Default to VPA for safety (but prefer HPA if you have evidence of external state)
- DO NOT ignore the state management information provided above - it is critical for 
  choosing between HPA and VPA

**HPA Status:**
- HPA Active: {yes/no}
- Current Replicas: {replicas}
- Desired Replicas: {desired_replicas}
- Target CPU: {cpu}%
- Target Memory: {memory}%
- Scaling Status: {status}

**VPA Status:**
- VPA Active: {yes/no}
- Update Mode: {mode}
- Recommendations: {recommendations}

**IMPORTANT: State Detection Rules (CRITICAL - FOLLOW STRICTLY)**
- **DO NOT assume** the application uses Redis, external databases, or externalized 
  state unless explicitly mentioned in the deployment information above
- **DO NOT infer** external state from deployment names, metrics, or other indirect clues
- **DEFAULT BEHAVIOR**: If state management information is NOT provided in the context 
  above, you MUST assume the application stores state inside the pod and recommend VPA
- **ONLY recommend HPA** if you have EXPLICIT, CLEAR evidence that state is externalized 
  (Redis, DB, external cache explicitly mentioned in context)
- **If uncertain or no state information provided**, you MUST prefer VPA for safety 
  (stateful apps should not scale horizontally)
- **When in doubt, choose VPA** - it's safer for applications that may have internal state

**Analysis Request:**
Based on the above information, provide an intelligent scaling recommendation considering:
1. Current resource pressure (CPU/Memory usage)
2. Predicted future demand (forecast trends)
3. Cost optimization opportunities
4. Performance requirements
5. Stability and risk factors
6. **CRITICAL: Choose between HPA (horizontal - more replicas) or VPA (vertical - more resources per pod)**

**Scaling Strategy Decision Guidelines:**
- Choose HPA (horizontal) if: 
  * Application is stateless OR externalizes its state (Redis, external DB, external cache, shared storage)
  * Can distribute load across multiple pods
  * Needs high availability
  * Current replicas < max_replicas
  
- Choose VPA (vertical) if:
  * Application keeps state inside the pod
  * Cannot scale horizontally (stateful)
  * Single-pod bottleneck
  * Need to optimize resource allocation per pod

Provide your recommendation in the specified JSON format with scaling_type field.
```

---

### Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. USER ACTION: Enable Predictive Autoscaling                  │
│    Input: deployment_name, namespace, min_replicas,            │
│           max_replicas, state_management (optional)            │
│    Location: ai_kubernetes_web_app.py, route                   │
│              /api/autoscaling/predictive/enable/<server_id>    │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. CONTEXT GATHERING                                            │
│    Location: predictive_autoscaler.py,                          │
│              get_scaling_recommendation()                      │
│    • Get current metrics (CPU, Memory, Pods)                    │
│    • Get forecasts (6-hour predictions)                         │
│    • Get HPA status (if exists)                                 │
│    • Get VPA status (if exists)                                 │
│    • Get current resource requests/limits                       │
│    • Detect state management (multi-source)                     │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. STATE MANAGEMENT DETECTION                                   │
│    Location: llm_autoscaling_advisor.py,                       │
│              _detect_state_management() (lines 514-733)         │
│    Priority Order:                                              │
│    1. Deployment annotations (ai4k8s.io/state-management)      │
│    2. Environment variables (REDIS, DATABASE, etc.)            │
│    3. Volume mounts (PVC, hostPath, persistent storage)        │
│    4. Service dependencies (Redis, DB services in namespace)    │
│    5. Deployment labels (ai4k8s.io/state-management)            │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. CONTEXT PREPARATION                                          │
│    Location: llm_autoscaling_advisor.py,                       │
│              _prepare_context() (lines 735-850)                │
│    • Build context dictionary with all gathered information      │
│    • Format state management note                               │
│    • Include HPA/VPA status                                     │
│    • Include forecast data                                      │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. PROMPT CREATION                                              │
│    Location: llm_autoscaling_advisor.py                        │
│    • System Prompt: _create_system_prompt() (lines 852-912)    │
│    • User Prompt: _create_user_prompt() (lines 914-1010)       │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. LLM API CALL (Groq)                                         │
│    Location: llm_autoscaling_advisor.py,                       │
│              analyze_scaling_decision() (lines 196-229)         │
│    • Model: llama-3.1-8b-instant                                │
│    • Fallback: llama-3.1-70b-versatile                          │
│    • Temperature: 0.3 (for consistency)                          │
│    • Max Tokens: 1000                                           │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. RESPONSE PARSING                                             │
│    Location: llm_autoscaling_advisor.py,                       │
│              _parse_llm_response() (lines 279-520)             │
│    • Extract JSON from LLM response                             │
│    • Validate scaling_type consistency                          │
│    • Check min/max replica constraints                          │
│    • Apply state management corrections                         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. POST-PROCESSING VALIDATION                                   │
│    Location: llm_autoscaling_advisor.py,                       │
│              _parse_llm_response() (lines 418-520)               │
│    • Stateless enforcement: If user selected stateless but LLM  │
│      recommended VPA → correct to HPA                          │
│    • Replica validation: Cap target_replicas to min/max         │
│    • State correction: If state inside pod but HPA chosen → VPA  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 9. SAFETY VALIDATION                                            │
│    Location: predictive_autoscaler.py,                          │
│              get_scaling_recommendation() (lines 1276-1290)    │
│    • Validate target_replicas respects min/max                 │
│    • Cap if exceeds limits                                     │
│    • Update action and reasoning                               │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 10. FINAL RETURN VALIDATION                                      │
│     Location: predictive_autoscaler.py,                         │
│               get_scaling_recommendation() (lines 1353-1378)   │
│     • Final check before returning to frontend                  │
│     • Ensure target_replicas is within bounds                  │
│     • Log final values for debugging                           │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 11. RESPONSE TO FRONTEND                                        │
│     Location: autoscaling_integration.py,                      │
│               get_scaling_recommendations() (lines 532-600)    │
│     • Format recommendation for UI                             │
│     • Include forecast data                                    │
│     • Include current metrics                                  │
│     • Include VPA resource details                              │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 12. UI DISPLAY                                                  │
│     Location: templates/autoscaling.html                        │
│     • Display recommendation with reasoning                     │
│     • Show target_replicas (for HPA)                           │
│     • Show target_cpu/target_memory (for VPA)                  │
│     • Provide "Apply Recommendation" button                     │
└─────────────────────────────────────────────────────────────────┘
```

---

### Code File Locations

#### Core LLM Components
- **`llm_autoscaling_advisor.py`** - Main LLM advisor class
  - `_create_system_prompt()` - Lines 852-912
  - `_create_user_prompt()` - Lines 914-1010
  - `_prepare_context()` - Lines 735-850
  - `_detect_state_management()` - Lines 514-733
  - `_parse_llm_response()` - Lines 279-520
  - `analyze_scaling_decision()` - Lines 102-270
  - `get_intelligent_recommendation()` - Lines 1001-1029

#### Predictive Autoscaling
- **`predictive_autoscaler.py`** - Predictive autoscaling orchestrator
  - `get_scaling_recommendation()` - Lines 1114-1403
  - `enable_predictive_autoscaling()` - Lines 65-130
  - `disable_predictive_autoscaling()` - Lines 900-1050

#### VPA Engine
- **`vpa_engine.py`** - VPA resource management
  - `patch_deployment_resources()` - Lines 80-150
  - `check_vpa_available()` - Lines 30-70
  - `get_deployment_resources()` - Lines 150-200

#### Integration Layer
- **`autoscaling_integration.py`** - Main integration point
  - `get_scaling_recommendations()` - Lines 532-600
  - `apply_predictive_target()` - Lines 200-400

#### Web Application
- **`ai_kubernetes_web_app.py`** - Flask web app
  - `/api/autoscaling/recommendations/<server_id>` - Lines 2225-2249
  - `/api/autoscaling/predictive/enable/<server_id>` - Lines 2107-2137
  - `/api/autoscaling/predictive/apply/<server_id>` - Lines 2139-2200

#### Frontend
- **`templates/autoscaling.html`** - UI template
  - `displayRecommendations()` - Lines 1440-1520
  - `enablePredictiveAutoscaling()` - Lines 997-1050
  - `applyPredictiveRecommendation()` - Lines 1717-1800
  - State Management dropdown - Lines 850-900

---

### Key Function Call Flow

```
1. User clicks "Get Recommendations"
   ↓
2. Frontend: fetch('/api/autoscaling/recommendations/3?deployment=...')
   ↓
3. ai_kubernetes_web_app.py: get_scaling_recommendations()
   ↓
4. autoscaling_integration.py: get_scaling_recommendations()
   ↓
5. predictive_autoscaler.py: get_scaling_recommendation()
   ↓
6. llm_autoscaling_advisor.py: get_intelligent_recommendation()
   ↓
7. llm_autoscaling_advisor.py: analyze_scaling_decision()
   ├─→ _detect_state_management() [State Detection]
   ├─→ _prepare_context() [Context Building]
   ├─→ _create_system_prompt() [System Prompt]
   ├─→ _create_user_prompt() [User Prompt]
   ├─→ Groq API Call [LLM Analysis]
   └─→ _parse_llm_response() [Response Parsing & Validation]
   ↓
8. predictive_autoscaler.py: Safety Validation
   ↓
9. predictive_autoscaler.py: Final Return Validation
   ↓
10. autoscaling_integration.py: Format Response
   ↓
11. ai_kubernetes_web_app.py: Return JSON
   ↓
12. Frontend: Display Recommendation
```

---

**Thank you for your attention!**

*Questions? The complete technical documentation is available in `VPA_INTEGRATION_REPORT.md`*

