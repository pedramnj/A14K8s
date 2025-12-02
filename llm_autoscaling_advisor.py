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
        # Use a model that supports longer context and better reasoning
        # Try 70b first, fallback to 8b if not available
        self.model = "llama-3.1-70b-versatile"  # More capable model for complex reasoning
        self.fallback_model = "llama-3.1-8b-instant"  # Fallback model
        
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
                hpa_status, historical_patterns, current_replicas,
                min_replicas, max_replicas
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
            recommendation = self._parse_llm_response(llm_output)
            
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
    
    def _parse_llm_response(self, llm_output: str) -> Dict[str, Any]:
        """Parse LLM response, extracting JSON if present"""
        import re
        
        # Try direct JSON parse first
        try:
            return json.loads(llm_output)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object in text
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', llm_output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Fallback: create recommendation from text analysis
        logger.warning("Could not parse JSON from LLM response, creating fallback recommendation")
        return self._create_fallback_recommendation(llm_output)
    
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

Your role is to analyze deployment metrics, forecasts, and patterns to make intelligent scaling recommendations.

Consider these factors:
1. **Performance**: Ensure adequate resources to meet performance requirements
2. **Cost**: Minimize resource usage while maintaining performance
3. **Stability**: Avoid rapid scaling changes that could cause instability
4. **Predictions**: Use forecast data to proactively scale before demand arrives
5. **Constraints**: Respect min/max replica limits and current cluster capacity

Respond in JSON format with:
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
            prompt += "- HPA Active: No (reactive scaling not configured)\n"
        
        if context.get('historical_patterns'):
            prompt += f"\n**Historical Patterns:**\n{json.dumps(context['historical_patterns'], indent=2)}\n"
        
        prompt += """
**Analysis Request:**
Based on the above information, provide an intelligent scaling recommendation considering:
1. Current resource pressure (CPU/Memory usage)
2. Predicted future demand (forecast trends)
3. Cost optimization opportunities
4. Performance requirements
5. Stability and risk factors

Provide your recommendation in the specified JSON format."""
        
        return prompt
    
    def get_intelligent_recommendation(self, deployment_name: str, namespace: str,
                                      current_metrics: Dict[str, Any],
                                      forecast: Dict[str, Any],
                                      hpa_status: Optional[Dict[str, Any]] = None,
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

