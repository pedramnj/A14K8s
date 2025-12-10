#!/usr/bin/env python3
"""
AI4K8s LLM-Based Autoscaling Advisor
====================================

Uses Groq LLM to make intelligent autoscaling decisions by analyzing:
- Current resource metrics
- Predictive forecasts
- Historical patterns
- Cost considerations
- Performance requirements

Author: Pedram Nikjooy
Thesis: AI Agent for Kubernetes Management
"""

import json
import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import groq
import hashlib

logger = logging.getLogger(__name__)

class LLMAutoscalingAdvisor:
    """LLM-powered autoscaling advisor using Groq"""
    
    def __init__(self, groq_api_key: Optional[str] = None, cache_ttl: int = 300):
        """Initialize LLM advisor with Groq
        
        Args:
            groq_api_key: Groq API key (or from env)
            cache_ttl: Cache TTL in seconds (default 5 minutes)
        """
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        self.client = None
        # Use supported models (70b was decommissioned, using 8b as primary)
        # Try 8b-instant first (fast and reliable), can add other models as fallback
        self.model = "llama-3.1-8b-instant"  # Primary model (fast, reliable)
        self.fallback_model = "llama-3.1-70b-versatile"  # Fallback (may be decommissioned)
        
        # Caching to prevent rapid successive LLM calls
        self.cache_ttl = cache_ttl  # 5 minutes default
        self.recommendation_cache: Dict[str, Dict[str, Any]] = {}
        # Track last LLM call time to enforce minimum interval
        self.last_llm_call_time: Dict[str, datetime] = {}
        self.min_llm_interval = 300  # Minimum 5 minutes between LLM calls for same deployment
        
        if self.groq_api_key:
            try:
                self.client = groq.Groq(api_key=self.groq_api_key)
                logger.info("✅ LLM Autoscaling Advisor initialized with Groq")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                self.client = None
        else:
            logger.warning("⚠️  No GROQ_API_KEY found, LLM advisor disabled")
    
    def _get_cache_key(self, deployment_name: str, namespace: str, 
                      current_metrics: Dict[str, Any], forecast: Dict[str, Any],
                      current_replicas: int) -> str:
        """Generate cache key from input parameters
        
        Uses very aggressive rounding to create stable cache keys that don't change
        with minor metric fluctuations. This prevents rapid recommendation changes.
        """
        # Round VERY aggressively to create stable cache keys:
        # - CPU/Memory: Round to nearest 25% (e.g., 173% -> 175%, 175% -> 175%, 180% -> 175%)
        #   This ensures 170-180% all use the same cache key
        # - Replicas: Use as-is (discrete value)
        # - Forecast peaks: Round to nearest 25%
        
        def round_to_25(value: float) -> int:
            """Round to nearest 25 (e.g., 173 -> 175, 175 -> 175, 180 -> 175)"""
            return int(round(value / 25.0) * 25)
        
        def round_to_5_percent(value: float) -> int:
            """Round to nearest 5% for memory (e.g., 11.7% -> 10%, 12.3% -> 10%)"""
            return int(round(value / 5.0) * 5)
        
        key_data = {
            'deployment': f"{namespace}/{deployment_name}",
            'replicas': current_replicas,
            # Round CPU to nearest 25% (170-180% all become 175%)
            'cpu': round_to_25(current_metrics.get('cpu_usage', 0)),
            # Round memory to nearest 5% (more stable for lower values)
            'memory': round_to_5_percent(current_metrics.get('memory_usage', 0)),
            # Round forecast peaks similarly
            'cpu_peak': round_to_25(forecast.get('cpu', {}).get('peak', 0)),
            'memory_peak': round_to_5_percent(forecast.get('memory', {}).get('peak', 0)),
            # Also include trend to differentiate between increasing/decreasing patterns
            'cpu_trend': forecast.get('cpu', {}).get('trend', 'unknown'),
            'memory_trend': forecast.get('memory', {}).get('trend', 'unknown')
        }
        key_str = json.dumps(key_data, sort_keys=True)
        cache_key = hashlib.md5(key_str.encode()).hexdigest()
        logger.debug(f"Cache key for {deployment_name}: cpu={key_data['cpu']}%, memory={key_data['memory']}%, key={cache_key[:8]}")
        return cache_key
    
    def analyze_scaling_decision(self, 
                                 deployment_name: str,
                                 namespace: str,
                                 current_metrics: Dict[str, Any],
                                 forecast: Dict[str, Any],
                                 hpa_status: Optional[Dict[str, Any]] = None,
                                 vpa_status: Optional[Dict[str, Any]] = None,
                                 current_resources: Optional[Dict[str, Any]] = None,
                                 historical_patterns: Optional[List[Dict[str, Any]]] = None,
                                 current_replicas: int = 1,
                                 min_replicas: int = 1,
                                 max_replicas: int = 10,
                                 use_cache: bool = True) -> Dict[str, Any]:
        """
        Use LLM to analyze and recommend scaling decisions
        
        Args:
            deployment_name: Name of the deployment
            namespace: Kubernetes namespace
            current_metrics: Current CPU/Memory usage metrics
            forecast: Predictive forecast data
            hpa_status: Current HPA status (optional)
            historical_patterns: Historical scaling patterns (optional)
            current_replicas: Current number of replicas
            min_replicas: Minimum allowed replicas
            max_replicas: Maximum allowed replicas
        
        Returns:
            Dict with LLM recommendation including action, reasoning, and confidence
        """
        if not self.client:
            return {
                'success': False,
                'error': 'LLM advisor not available (no API key)',
                'fallback': True
            }
        
        # Check cache and enforce minimum interval between LLM calls
        if use_cache:
            cache_key = self._get_cache_key(deployment_name, namespace, current_metrics, forecast, current_replicas)
            deployment_key = f"{namespace}/{deployment_name}"
            now = datetime.now()
            
            # Check if we have a cached recommendation
            if cache_key in self.recommendation_cache:
                cached = self.recommendation_cache[cache_key]
                cache_age = (now - cached['timestamp']).total_seconds()
                if cache_age < self.cache_ttl:
                    logger.debug(f"✅ Using cached LLM recommendation (age: {cache_age:.1f}s, key: {cache_key[:8]})")
                    return {
                        'success': True,
                        'recommendation': cached['recommendation'],
                        'llm_model': cached.get('llm_model', self.model),
                        'timestamp': cached['timestamp'].isoformat(),
                        'cached': True
                    }
                else:
                    # Cache expired, remove it
                    logger.debug(f"⏰ Cache expired (age: {cache_age:.1f}s), removing")
                    del self.recommendation_cache[cache_key]
            
            # Enforce minimum interval between LLM calls for same deployment
            if deployment_key in self.last_llm_call_time:
                time_since_last_call = (now - self.last_llm_call_time[deployment_key]).total_seconds()
                if time_since_last_call < self.min_llm_interval:
                    logger.info(f"⏸️ Skipping LLM call - only {time_since_last_call:.1f}s since last call (min: {self.min_llm_interval}s)")
                    # Return a fallback recommendation indicating we're rate-limiting
                    return {
                        'success': False,
                        'error': f'LLM call rate-limited (last call {time_since_last_call:.1f}s ago, min interval {self.min_llm_interval}s)',
                        'fallback': True,
                        'rate_limited': True
                    }
        
        try:
            # Prepare context for LLM
            context = self._prepare_context(
                deployment_name, namespace, current_metrics, forecast,
                hpa_status, vpa_status, current_resources, historical_patterns, 
                current_replicas, min_replicas, max_replicas
            )
            
            # Create system prompt
            system_prompt = self._create_system_prompt()
            
            # Create user prompt with context
            user_prompt = self._create_user_prompt(context)
            
            # Call Groq LLM
            # Try with preferred model first, fallback if needed
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,  # Lower temperature for more consistent decisions
                    max_tokens=1000
                )
            except Exception as e:
                # Fallback to 8b model if 70b not available
                logger.warning(f"Model {self.model} not available, trying fallback: {e}")
                try:
                    response = self.client.chat.completions.create(
                        model=self.fallback_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.3,
                        max_tokens=1000
                    )
                    self.model = self.fallback_model  # Use fallback for future calls
                except Exception as e2:
                    logger.error(f"Failed to call Groq API: {e2}")
                    raise
            
            # Parse LLM response
            llm_output = response.choices[0].message.content
            
            # Try to extract JSON from response (handle both JSON and text responses)
            # Pass min/max replicas for validation and context for state management check
            recommendation = self._parse_llm_response(llm_output, min_replicas, max_replicas, current_metrics, current_resources, context)
            
            now = datetime.now()
            result = {
                'success': True,
                'recommendation': recommendation,
                'llm_model': self.model,
                'timestamp': now.isoformat(),
                'cached': False
            }
            
            # Cache the result and track LLM call time
            if use_cache:
                cache_key = self._get_cache_key(deployment_name, namespace, current_metrics, forecast, current_replicas)
                deployment_key = f"{namespace}/{deployment_name}"
                
                self.recommendation_cache[cache_key] = {
                    'recommendation': recommendation,
                    'llm_model': self.model,
                    'timestamp': now
                }
                self.last_llm_call_time[deployment_key] = now
                
                logger.info(f"✅ LLM recommendation cached (key: {cache_key[:8]}, deployment: {deployment_key})")
                
                # Clean old cache entries (keep only last 100)
                if len(self.recommendation_cache) > 100:
                    # Remove oldest entries
                    sorted_cache = sorted(self.recommendation_cache.items(), 
                                        key=lambda x: x[1]['timestamp'])
                    for key, _ in sorted_cache[:-100]:
                        del self.recommendation_cache[key]
                
                # Clean old LLM call times (keep only last 50)
                if len(self.last_llm_call_time) > 50:
                    sorted_times = sorted(self.last_llm_call_time.items(), 
                                        key=lambda x: x[1])
                    for key, _ in sorted_times[:-50]:
                        del self.last_llm_call_time[key]
            
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM autoscaling advisor: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'raw_response': llm_output if 'llm_output' in locals() else None
            }
    
    def _parse_llm_response(self, llm_output: str, min_replicas: int = 1, max_replicas: int = 10,
                           current_metrics: Optional[Dict[str, Any]] = None,
                           current_resources: Optional[Dict[str, Any]] = None,
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Parse LLM response, extracting JSON if present and validate against constraints"""
        import re
        
        recommendation = None
        
        # Try direct JSON parse first
        try:
            recommendation = json.loads(llm_output)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        if not recommendation:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_output, re.DOTALL)
            if json_match:
                try:
                    recommendation = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
        
        # Try to find JSON object in text
        if not recommendation:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', llm_output, re.DOTALL)
            if json_match:
                try:
                    recommendation = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
        
        # Fallback: create recommendation from text analysis
        if not recommendation:
            logger.warning("Could not parse JSON from LLM response, creating fallback recommendation")
            recommendation = self._create_fallback_recommendation(llm_output)
        
        # Validate scaling_type consistency with reasoning
        # If reasoning mentions state inside pod but scaling_type is HPA, correct it to VPA
        if recommendation:
            reasoning_lower = recommendation.get('reasoning', '').lower()
            scaling_type = recommendation.get('scaling_type', 'hpa').lower()
            
            # Check if reasoning indicates stateful (state inside pod) but scaling_type is HPA
            state_inside_pod_indicators = [
                'state inside the pod',
                'state stored inside',
                'local files',
                'in-memory without externalization',
                'state is not externalized',
                'critical state inside'
            ]
            
            state_externalized_indicators = [
                'redis',
                'external database',
                'external db',
                'external cache',
                'shared storage',
                'state externalized',
                'externalized state'
            ]
            
            has_state_inside = any(indicator in reasoning_lower for indicator in state_inside_pod_indicators)
            has_state_externalized = any(indicator in reasoning_lower for indicator in state_externalized_indicators)
            
            # Check if no explicit state information was provided
            no_state_info_provided = False
            if context:
                state_note = context.get('state_management_note', '')
                no_state_info_provided = not state_note or 'no explicit state' in state_note.lower() or 'not provided' in state_note.lower()
            
            # CRITICAL: If no state info provided and LLM chose HPA, correct to VPA for safety
            if (has_state_inside and not has_state_externalized and scaling_type == 'hpa') or \
               (no_state_info_provided and scaling_type == 'hpa' and not has_state_externalized):
                logger.warning(f"⚠️ LLM recommended HPA but state is uncertain or inside pod. Correcting to VPA.")
                recommendation['scaling_type'] = 'vpa'
                # Clear target_replicas for VPA
                recommendation['target_replicas'] = None
                
                # Calculate VPA targets based on current usage and resources
                cpu_usage = current_metrics.get('cpu_usage', 0) if current_metrics else 0
                memory_usage = current_metrics.get('memory_usage', 0) if current_metrics else 0
                
                # Get current resource requests to calculate new targets
                cpu_request_str = current_resources.get('cpu_request', '100m') if current_resources else '100m'
                memory_request_str = current_resources.get('memory_request', '128Mi') if current_resources else '128Mi'
                
                # Parse current CPU request (handle formats like "100m", "0.5", "1")
                try:
                    if cpu_request_str.endswith('m'):
                        cpu_request_m = int(cpu_request_str[:-1])
                    elif cpu_request_str.endswith('n'):
                        cpu_request_m = int(cpu_request_str[:-1]) / 1000000
                    else:
                        cpu_request_m = float(cpu_request_str) * 1000
                except:
                    cpu_request_m = 100  # Default 100m
                
                # Calculate target CPU based on usage (add 20% headroom)
                target_cpu_m = int(cpu_request_m * (1 + cpu_usage / 100) * 1.2) if cpu_usage > 0 else cpu_request_m
                target_cpu_m = max(100, min(target_cpu_m, 4000))  # Clamp between 100m and 4000m
                
                # Parse current memory request (handle formats like "128Mi", "256Mi", "1Gi")
                try:
                    if memory_request_str.endswith('Mi'):
                        memory_request_mi = int(memory_request_str[:-2])
                    elif memory_request_str.endswith('Gi'):
                        memory_request_mi = int(memory_request_str[:-2]) * 1024
                    elif memory_request_str.endswith('Ki'):
                        memory_request_mi = int(memory_request_str[:-2]) / 1024
                    else:
                        memory_request_mi = int(memory_request_str) if memory_request_str.isdigit() else 128
                except:
                    memory_request_mi = 128  # Default 128Mi
                
                # Calculate target memory based on usage (add 30% headroom)
                target_memory_mi = int(memory_request_mi * (1 + memory_usage / 100) * 1.3) if memory_usage > 0 else memory_request_mi
                target_memory_mi = max(128, min(target_memory_mi, 4096))  # Clamp between 128Mi and 4Gi
                
                # Set VPA targets
                if not recommendation.get('target_cpu'):
                    recommendation['target_cpu'] = f"{target_cpu_m}m"
                if not recommendation.get('target_memory'):
                    recommendation['target_memory'] = f"{target_memory_mi}Mi"
                
                recommendation['reasoning'] = (recommendation.get('reasoning', '') + 
                    f' [Corrected: Changed from HPA to VPA because state is stored inside the pod. Calculated VPA targets: CPU={recommendation["target_cpu"]}, Memory={recommendation["target_memory"]} based on current usage]')
        
        # Validate and enforce min/max replica constraints (only for HPA)
        if recommendation and recommendation.get('scaling_type', 'hpa').lower() in ['hpa', 'both']:
            if 'target_replicas' in recommendation and recommendation['target_replicas'] is not None:
                target = recommendation['target_replicas']
                if target > max_replicas:
                    logger.warning(f"⚠️ LLM recommended {target} replicas but max is {max_replicas}. Capping to {max_replicas}.")
                    recommendation['target_replicas'] = max_replicas
                    recommendation['action'] = 'at_max' if target > max_replicas else recommendation.get('action', 'maintain')
                    recommendation['reasoning'] = (recommendation.get('reasoning', '') + 
                        f" [Note: Original recommendation was {target} replicas, but capped to max_replicas={max_replicas}]")
                elif target < min_replicas:
                    logger.warning(f"⚠️ LLM recommended {target} replicas but min is {min_replicas}. Setting to {min_replicas}.")
                    recommendation['target_replicas'] = min_replicas
                    recommendation['action'] = 'maintain' if target < min_replicas else recommendation.get('action', 'maintain')
                    recommendation['reasoning'] = (recommendation.get('reasoning', '') + 
                        f" [Note: Original recommendation was {target} replicas, but set to min_replicas={min_replicas}]")
        
        return recommendation
    
    def _create_fallback_recommendation(self, llm_output: str) -> Dict[str, Any]:
        """Create a fallback recommendation from text response"""
        import re
        
        # Try to extract action and replicas from text
        action_match = re.search(r'(scale_up|scale_down|maintain|at_max)', llm_output.lower())
        action = action_match.group(1) if action_match else 'maintain'
        
        replica_match = re.search(r'(\d+)\s*replicas?', llm_output.lower())
        target_replicas = int(replica_match.group(1)) if replica_match else 0
        
        return {
            'action': action,
            'target_replicas': target_replicas,
            'confidence': 0.5,
            'reasoning': llm_output[:500],  # First 500 chars
            'factors_considered': [],
            'risk_assessment': 'medium',
            'cost_impact': 'medium',
            'performance_impact': 'neutral',
            'recommended_timing': 'immediate'
        }
    
    def _prepare_context(self, deployment_name: str, namespace: str,
                        current_metrics: Dict[str, Any], forecast: Dict[str, Any],
                        hpa_status: Optional[Dict[str, Any]],
                        vpa_status: Optional[Dict[str, Any]],
                        current_resources: Optional[Dict[str, Any]],
                        historical_patterns: Optional[List[Dict[str, Any]]],
                        current_replicas: int, min_replicas: int, max_replicas: int) -> Dict[str, Any]:
        """Prepare context dictionary for LLM"""
        context = {
            'deployment': {
                'name': deployment_name,
                'namespace': namespace,
                'current_replicas': current_replicas,
                'min_replicas': min_replicas,
                'max_replicas': max_replicas
            },
            'current_metrics': {
                'cpu_usage_percent': current_metrics.get('cpu_usage', 0),
                'memory_usage_percent': current_metrics.get('memory_usage', 0),
                'pod_count': current_metrics.get('pod_count', 0),
                'running_pods': current_metrics.get('running_pod_count', 0)
            },
            'forecast': {
                'cpu': {
                    'current': forecast.get('cpu', {}).get('current', 0),
                    'peak': forecast.get('cpu', {}).get('peak', 0),
                    'trend': forecast.get('cpu', {}).get('trend', 'unknown'),
                    'predictions': forecast.get('cpu', {}).get('predictions', [])
                },
                'memory': {
                    'current': forecast.get('memory', {}).get('current', 0),
                    'peak': forecast.get('memory', {}).get('peak', 0),
                    'trend': forecast.get('memory', {}).get('trend', 'unknown'),
                    'predictions': forecast.get('memory', {}).get('predictions', [])
                }
            }
        }
        
        # Add current resource requests/limits if available
        if current_resources:
            context['current_resources'] = current_resources
        else:
            context['current_resources'] = {
                'cpu_request': 'N/A',
                'memory_request': 'N/A',
                'cpu_limit': 'N/A',
                'memory_limit': 'N/A'
            }
        
        # Add explicit note about state management detection
        # We don't have automatic detection, so we need to be conservative
        context['state_management_note'] = (
            "IMPORTANT: No explicit state management information is available. "
            "DO NOT assume the application uses Redis, external databases, or externalized state. "
            "When uncertain, prefer VPA (vertical scaling) as it's safer for applications that may store state inside pods."
        )
        
        if hpa_status:
            context['hpa'] = {
                'exists': True,
                'current_replicas': hpa_status.get('current_replicas', current_replicas),
                'desired_replicas': hpa_status.get('desired_replicas', current_replicas),
                'target_cpu': hpa_status.get('target_cpu', 70),
                'target_memory': hpa_status.get('target_memory', 80),
                'scaling_status': hpa_status.get('scaling_status', 'stable')
            }
        else:
            context['hpa'] = {'exists': False}
        
        if vpa_status:
            context['vpa'] = {
                'exists': True,
                'update_mode': vpa_status.get('update_mode', 'Auto'),
                'recommendations': vpa_status.get('recommendations', [])
            }
        else:
            context['vpa'] = {'exists': False}
        
        if historical_patterns:
            context['historical_patterns'] = historical_patterns[-10:]  # Last 10 patterns
        
        return context
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for LLM"""
        return """You are an expert Kubernetes autoscaling advisor with deep knowledge of:
- Resource optimization and cost management
- Performance requirements and SLA considerations
- Scaling best practices and anti-patterns
- Predictive analysis and trend interpretation
- Horizontal Pod Autoscaling (HPA) vs Vertical Pod Autoscaling (VPA)

Your role is to analyze deployment metrics, forecasts, and patterns to make intelligent scaling recommendations.

**IMPORTANT: You must decide between TWO scaling strategies:**

1. **HORIZONTAL SCALING (HPA)**: Scale by adjusting the NUMBER of replicas (pods)
   - Use when: Load can be distributed across multiple pods, need high availability
   - **CRITICAL RULE**: Applications that externalize their state (use Redis, external databases, external cache, shared storage) should be treated as STATELESS and prefer HPA
   - Example: 3 pods → 5 pods (same resources per pod)
   - Pros: Better fault tolerance, load distribution, can handle traffic spikes
   - Cons: More pods = more overhead, may hit node limits

2. **VERTICAL SCALING (VPA)**: Scale by adjusting RESOURCE requests/limits per pod
   - Use when: Application keeps critical state INSIDE the pod (not externalized), cannot scale horizontally, single-pod bottleneck
   - **CRITICAL RULE**: Only prefer VPA if state is stored INSIDE the pod. If state is externalized (Redis, DB, external cache), treat as stateless and prefer HPA instead
   - Example: CPU 100m → 200m, Memory 128Mi → 256Mi (same number of pods)
   - Pros: Better resource utilization, fewer pods, simpler architecture
   - Cons: Pod restart required, single point of failure, limited by node capacity

**IMPORTANT STATE MANAGEMENT RULES:**
- **NEVER assume "stateful = only VPA"**
- **ALWAYS check if state is externalized before ruling out HPA**
- If application uses Redis, external databases, external cache, or shared storage → treat as STATELESS → prefer HPA
- If application keeps critical state inside the pod (local files, in-memory state without externalization) → prefer VPA

Consider these factors:
1. **Performance**: Ensure adequate resources to meet performance requirements
2. **Cost**: Minimize resource usage while maintaining performance
3. **Stability**: Avoid rapid scaling changes that could cause instability
4. **Predictions**: Use forecast data to proactively scale before demand arrives
5. **Constraints**: **CRITICAL - You MUST respect min/max replica limits. target_replicas MUST be between min_replicas and max_replicas (inclusive). NEVER recommend more than max_replicas or less than min_replicas.**
6. **Scaling Type**: Choose HPA (horizontal) or VPA (vertical) based on application characteristics

**IMPORTANT CONSTRAINTS:**
- If you recommend HPA scaling, target_replicas MUST be >= min_replicas AND <= max_replicas
- If current replicas is already at max_replicas and you need more, recommend "at_max" action or suggest VPA instead
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
}"""
    
    def _create_user_prompt(self, context: Dict[str, Any]) -> str:
        """Create user prompt with context"""
        prompt = f"""Analyze the following Kubernetes deployment autoscaling scenario and provide a recommendation:

**Deployment Information:**
- Name: {context['deployment']['name']}
- Namespace: {context['deployment']['namespace']}
- Current Replicas: {context['deployment']['current_replicas']}
- Min Replicas: {context['deployment']['min_replicas']}
- Max Replicas: {context['deployment']['max_replicas']}

**Current Resource Usage:**
- CPU: {context['current_metrics']['cpu_usage_percent']:.1f}%
- Memory: {context['current_metrics']['memory_usage_percent']:.1f}%
- Running Pods: {context['current_metrics']['running_pods']}/{context['current_metrics']['pod_count']}

**Forecast Data:**
- CPU Current: {context['forecast']['cpu']['current']:.1f}%, Peak: {context['forecast']['cpu']['peak']:.1f}%, Trend: {context['forecast']['cpu']['trend']}
- Memory Current: {context['forecast']['memory']['current']:.1f}%, Peak: {context['forecast']['memory']['peak']:.1f}%, Trend: {context['forecast']['memory']['trend']}
- CPU Predictions (next 6 hours): {context['forecast']['cpu']['predictions']}
- Memory Predictions (next 6 hours): {context['forecast']['memory']['predictions']}

**Current Resource Configuration:**
- CPU Request: {context['current_resources'].get('cpu_request', 'N/A')}
- CPU Limit: {context['current_resources'].get('cpu_limit', 'N/A')}
- Memory Request: {context['current_resources'].get('memory_request', 'N/A')}
- Memory Limit: {context['current_resources'].get('memory_limit', 'N/A')}

**State Management Information:**
{context.get('state_management_note', 'No explicit state management information available. DO NOT assume external state (Redis, DB) unless explicitly mentioned.')}

**HPA Status:**
"""
        
        if context['hpa']['exists']:
            prompt += f"""- HPA Active: Yes
- Current Replicas: {context['hpa']['current_replicas']}
- Desired Replicas: {context['hpa']['desired_replicas']}
- Target CPU: {context['hpa']['target_cpu']}%
- Target Memory: {context['hpa']['target_memory']}%
- Scaling Status: {context['hpa']['scaling_status']}
"""
        else:
            prompt += "- HPA Active: No (reactive horizontal scaling not configured)\n"
        
        prompt += "\n**VPA Status:**\n"
        if context['vpa']['exists']:
            prompt += f"""- VPA Active: Yes
- Update Mode: {context['vpa']['update_mode']}
- Recommendations: {json.dumps(context['vpa'].get('recommendations', []), indent=2)}
"""
        else:
            prompt += "- VPA Active: No (vertical scaling not configured)\n"
        
        if context.get('historical_patterns'):
            prompt += f"\n**Historical Patterns:**\n{json.dumps(context['historical_patterns'], indent=2)}\n"
        
        prompt += """
**IMPORTANT: State Detection Rules (CRITICAL - FOLLOW STRICTLY)**
- **DO NOT assume** the application uses Redis, external databases, or externalized state unless explicitly mentioned in the deployment information above
- **DO NOT infer** external state from deployment names, metrics, or other indirect clues
- **DEFAULT BEHAVIOR**: If state management information is NOT provided in the context above, you MUST assume the application stores state inside the pod and recommend VPA
- **ONLY recommend HPA** if you have EXPLICIT, CLEAR evidence that state is externalized (Redis, DB, external cache explicitly mentioned in context)
- **If uncertain or no state information provided**, you MUST prefer VPA for safety (stateful apps should not scale horizontally)
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
  * Traffic spikes are expected
  * **Remember: Applications with externalized state should be treated as stateless and prefer HPA**

- Choose VPA (vertical) if: 
  * Application keeps critical state INSIDE the pod (not externalized)
  * Cannot scale horizontally due to state constraints
  * Single-pod bottleneck
  * Resource constraints per pod
  * **CRITICAL: Only choose VPA if state is stored inside the pod. If state is externalized, prefer HPA instead**

- Choose "both" if: Need both more replicas AND more resources per pod (rare, but possible)

**State Management Decision Tree:**
1. Does the application store critical state?
   - **If NOT mentioned in context** → Assume MAYBE (prefer VPA for safety)
   - NO (explicitly stateless) → Prefer HPA
   - YES → Continue to step 2
2. Where is the state stored?
   - **ONLY if EXPLICITLY mentioned**: Externalized (Redis, external DB, external cache, shared storage) → Treat as STATELESS → Prefer HPA
   - **Default assumption**: Inside the pod (local files, in-memory without externalization) → **MUST USE VPA** (NOT HPA)
3. **CRITICAL RULES:**
   - **DO NOT assume** Redis/external state exists unless explicitly mentioned in deployment information
   - **DO NOT infer** external state from deployment names or metrics
   - **When uncertain**, prefer VPA (safer for stateful applications)
   - **NEVER assume "stateful = only VPA"** - but also NEVER assume external state without evidence

**CRITICAL RULE FOR VPA:**
- If you determine state is stored INSIDE the pod (not externalized), you MUST set:
  - scaling_type: "vpa" (NOT "hpa")
  - target_replicas: null (do NOT set a replica count)
  - target_cpu: "<value>" (e.g., "200m", "500m", "1000m" based on current CPU usage)
  - target_memory: "<value>" (e.g., "256Mi", "512Mi", "1Gi" based on current memory usage)
  - action: "scale_up" or "scale_down" (based on resource pressure, NOT replica count)

**CRITICAL RULE FOR HPA:**
- Only use HPA if:
  - Application is stateless, OR
  - State is externalized (Redis, DB, external cache, shared storage)
- For HPA, set:
  - scaling_type: "hpa"
  - target_replicas: <number between min_replicas and max_replicas>
  - target_cpu: null
  - target_memory: null

**DO NOT scale down stateful applications (state inside pod) by reducing replicas - use VPA instead to adjust resources per pod.**

Provide your recommendation in the specified JSON format with scaling_type field."""
        
        return prompt
    
    def get_intelligent_recommendation(self, deployment_name: str, namespace: str,
                                      current_metrics: Dict[str, Any],
                                      forecast: Dict[str, Any],
                                      hpa_status: Optional[Dict[str, Any]] = None,
                                      vpa_status: Optional[Dict[str, Any]] = None,
                                      current_resources: Optional[Dict[str, Any]] = None,
                                      current_replicas: int = 1,
                                      min_replicas: int = 1,
                                      max_replicas: int = 10) -> Dict[str, Any]:
        """
        Get intelligent LLM-based scaling recommendation
        
        This is a convenience method that wraps analyze_scaling_decision
        """
        return self.analyze_scaling_decision(
            deployment_name=deployment_name,
            namespace=namespace,
            current_metrics=current_metrics,
            forecast=forecast,
            hpa_status=hpa_status,
            vpa_status=vpa_status,
            current_resources=current_resources,
            historical_patterns=None,  # Can be added later
            current_replicas=current_replicas,
            min_replicas=min_replicas,
            max_replicas=max_replicas
        )
    
    def explain_scaling_decision(self, recommendation: Dict[str, Any]) -> str:
        """Generate human-readable explanation of scaling decision"""
        if not recommendation.get('success'):
            return f"❌ Error: {recommendation.get('error', 'Unknown error')}"
        
        rec = recommendation.get('recommendation', {})
        action = rec.get('action', 'unknown')
        target = rec.get('target_replicas', 0)
        reasoning = rec.get('reasoning', 'No reasoning provided')
        confidence = rec.get('confidence', 0)
        risk = rec.get('risk_assessment', 'unknown')
        
        action_emoji = {
            'scale_up': '⬆️',
            'scale_down': '⬇️',
            'maintain': '➡️',
            'at_max': '⚠️'
        }.get(action, '❓')
        
        return f"""{action_emoji} **LLM Recommendation: {action.replace('_', ' ').title()}**

**Target Replicas:** {target}
**Confidence:** {confidence:.0%}
**Risk Level:** {risk.upper()}

**Reasoning:**
{reasoning}

**Factors Considered:** {', '.join(rec.get('factors_considered', []))}
**Cost Impact:** {rec.get('cost_impact', 'unknown').upper()}
**Performance Impact:** {rec.get('performance_impact', 'unknown').upper()}
**Recommended Timing:** {rec.get('recommended_timing', 'unknown').title()}"""

