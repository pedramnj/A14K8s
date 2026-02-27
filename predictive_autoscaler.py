#!/usr/bin/env python3
"""
AI4K8s Predictive Autoscaler
============================

AI-powered predictive autoscaling that uses ML forecasting to proactively
scale resources before demand arrives.

Author: Pedram Nikjooy
Thesis: AI Agent for Kubernetes Management
"""

import logging
import json
import os
import subprocess
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from predictive_monitoring import (
    PredictiveMonitoringSystem,
    ResourceMetrics,
    ForecastResult
)
from autoscaling_engine import HorizontalPodAutoscaler
from llm_autoscaling_advisor import LLMAutoscalingAdvisor
from mcda_optimizer import MCDAAutoscalingOptimizer
from logging_utils import get_app_logger
from scaling_decision import ScalingDecision

logger = get_app_logger(
    __name__,
    level=logging.WARNING,
    log_file=os.path.expanduser("~/ai4k8s/predictive_autoscaler.log"),
)

class PredictiveAutoscaler:
    """Predictive autoscaling based on ML forecasts"""
    
    def __init__(self, monitoring_system: PredictiveMonitoringSystem,
                 hpa_manager: HorizontalPodAutoscaler,
                 vpa_manager=None,  # Optional VPA manager
                 prediction_horizon: int = 6,
                 use_llm: bool = True):
        self.monitoring_system = monitoring_system
        self.hpa_manager = hpa_manager
        self.vpa_manager = vpa_manager  # VPA manager for vertical scaling
        self.prediction_horizon = prediction_horizon  # hours ahead
        self.scaling_history = []
        self.enabled_deployments = {}  # Track enabled deployments: {f"{namespace}/{deployment_name}": {...}}
        
        # Initialize LLM advisor
        self.use_llm = use_llm
        self.llm_advisor = None
        if use_llm:
            try:
                self.llm_advisor = LLMAutoscalingAdvisor()
                if self.llm_advisor.client:
                    logger.info("✅ LLM autoscaling advisor enabled")
                else:
                    logger.warning("⚠️  LLM advisor requested but no API key available")
                    self.use_llm = False
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize LLM advisor: {e}")
                self.use_llm = False
        
        # Scaling thresholds (fallback when LLM is not available)
        self.scale_up_threshold = 75  # CPU or memory > 75%
        self.scale_down_threshold = 25  # CPU and memory < 25%
        self.safety_buffer = 1.2  # 20% buffer for safety

        # MCDA optimizer for formal multi-criteria scaling decisions
        self.mcda_optimizer = MCDAAutoscalingOptimizer(profile='balanced')
        
    def predict_and_scale(self, deployment_name: str, namespace: str = "default",
                         min_replicas: int = 2, max_replicas: int = 10,
                         state_management: Optional[str] = None) -> Dict[str, Any]:
        """Predict future load and scale proactively"""
        try:
            decision_start = time.time()
            stage_timing = {
                'metrics_collection_s': 0.0,
                'forecast_s': 0.0,
                'llm_inference_s': 0.0,
                'mcda_validation_s': 0.0,
                'actuation_s': 0.0,
            }

            def _with_timing(payload: Dict[str, Any]) -> Dict[str, Any]:
                total_decision_s = max(0.0, time.time() - decision_start)
                payload['timing_breakdown'] = {
                    'metrics_collection_s': round(stage_timing['metrics_collection_s'], 4),
                    'forecast_s': round(stage_timing['forecast_s'], 4),
                    'llm_inference_s': round(stage_timing['llm_inference_s'], 4),
                    'mcda_validation_s': round(stage_timing['mcda_validation_s'], 4),
                    'actuation_s': round(stage_timing['actuation_s'], 4),
                    'total_decision_s': round(total_decision_s, 4),
                }
                return payload

            # Trim deployment name and namespace to remove any whitespace
            deployment_name = deployment_name.strip()
            namespace = namespace.strip()
            
            # Mark deployment as enabled in Kubernetes annotations
            enabled_at = datetime.now().isoformat()
            config_json = json.dumps({
                'min_replicas': min_replicas,
                'max_replicas': max_replicas,
                'enabled_at': enabled_at
            })
            
            annotations = {
                'ai4k8s.io/predictive-autoscaling-enabled': 'true',
                'ai4k8s.io/predictive-autoscaling-enabled-at': enabled_at,
                'ai4k8s.io/predictive-autoscaling-config': config_json
            }
            
            # Add state management annotation if provided by user
            if state_management:
                annotations['ai4k8s.io/state-management'] = state_management
                logger.info(f"✅ Setting state-management annotation to: {state_management}")
            else:
                logger.debug(f"ℹ️ No state_management provided, skipping annotation")
            
            labels = {
                'ai4k8s.io/predictive-autoscaling': 'enabled'
            }
            
            # Patch annotations
            annot_result = self.hpa_manager.patch_deployment_annotations(
                deployment_name, namespace, annotations
            )
            if not annot_result.get('success'):
                logger.error(f"❌ Failed to add annotations: {annot_result.get('error')}")
                return _with_timing({
                    'success': False,
                    'error': f"Failed to set predictive autoscaling annotations: {annot_result.get('error')}"
                })
            else:
                logger.info(f"✅ Successfully set annotations: {list(annotations.keys())}")
            
            # Verify annotations were set correctly
            try:
                deployment_status = self.hpa_manager.get_deployment_replicas(deployment_name, namespace)
                if deployment_status.get('success'):
                    verify_annotations = deployment_status.get('annotations', {})
                    verify_config_str = verify_annotations.get('ai4k8s.io/predictive-autoscaling-config', '')
                    if verify_config_str:
                        try:
                            verify_config = json.loads(verify_config_str)
                            expected_config = json.loads(config_json)
                            # Compare key fields instead of exact string match (JSON formatting might differ)
                            if (verify_config.get('min_replicas') == expected_config.get('min_replicas') and
                                verify_config.get('max_replicas') == expected_config.get('max_replicas')):
                                logger.info(f"✅ Verified annotation ai4k8s.io/predictive-autoscaling-config is set correctly (min={verify_config.get('min_replicas')}, max={verify_config.get('max_replicas')})")
                            else:
                                logger.warning(f"⚠️ Annotation verification failed - config mismatch: expected min={expected_config.get('min_replicas')}, max={expected_config.get('max_replicas')}, got min={verify_config.get('min_replicas')}, max={verify_config.get('max_replicas')}")
                        except json.JSONDecodeError:
                            logger.warning(f"⚠️ Annotation verification failed - invalid JSON in annotation")
                    else:
                        logger.warning(f"⚠️ Annotation verification failed - annotation not found after setting")
            except Exception as e:
                logger.warning(f"⚠️ Could not verify annotations: {e}")
            
            # Patch labels
            label_result = self.hpa_manager.patch_deployment_labels(
                deployment_name, namespace, labels
            )
            if not label_result.get('success'):
                logger.warning(f"Failed to add labels: {label_result.get('error')}")
            
            # Also track in memory for backward compatibility
            key = f"{namespace}/{deployment_name}"
            self.enabled_deployments[key] = {
                'deployment_name': deployment_name,
                'namespace': namespace,
                'min_replicas': min_replicas,
                'max_replicas': max_replicas,
                'enabled_at': enabled_at
            }
            
            # Get current deployment status
            deployment_status = self.hpa_manager.get_deployment_replicas(deployment_name, namespace)
            if not deployment_status['success']:
                return _with_timing({
                    'success': False,
                    'error': f'Deployment {deployment_name} not found',
                    'action': 'none'
                })
            
            current_replicas = deployment_status['replicas']
            
            # Get deployment-specific metrics
            metrics_start = time.time()
            deployment_metrics = self._get_deployment_metrics(deployment_name, namespace)
            stage_timing['metrics_collection_s'] = max(0.0, time.time() - metrics_start)
            cpu_current = deployment_metrics.get('cpu_usage', 0)
            memory_current = deployment_metrics.get('memory_usage', 0)
            
            logger.info(f"Deployment metrics for {deployment_name}: CPU={cpu_current}%, Memory={memory_current}%")
            
            # Get forecasts (use deployment-specific current values if available)
            forecast_start = time.time()
            cpu_forecast = self.monitoring_system.forecaster.forecast_cpu_usage(
                hours_ahead=self.prediction_horizon
            )
            memory_forecast = self.monitoring_system.forecaster.forecast_memory_usage(
                hours_ahead=self.prediction_horizon
            )
            stage_timing['forecast_s'] = max(0.0, time.time() - forecast_start)
            
            # Override current values with deployment-specific metrics if available
            if cpu_current > 0:
                cpu_forecast.current_value = cpu_current
            if memory_current > 0:
                memory_forecast.current_value = memory_current
            
            # Find peak predicted usage
            # Use current_value if predicted_values is empty or all zeros
            # Safely handle None predicted_values
            cpu_predicted = cpu_forecast.predicted_values if cpu_forecast.predicted_values is not None else []
            memory_predicted = memory_forecast.predicted_values if memory_forecast.predicted_values is not None else []
            
            if cpu_predicted and any(v > 0 for v in cpu_predicted):
                max_predicted_cpu = max(cpu_predicted)
            else:
                max_predicted_cpu = cpu_forecast.current_value if cpu_forecast.current_value is not None else 0
            
            if memory_predicted and any(v > 0 for v in memory_predicted):
                max_predicted_memory = max(memory_predicted)
            else:
                max_predicted_memory = memory_forecast.current_value if memory_forecast.current_value is not None else 0
            
            logger.info(f"Forecast values - CPU: current={cpu_forecast.current_value}, predicted={cpu_predicted}, max={max_predicted_cpu}")
            logger.info(f"Forecast values - Memory: current={memory_forecast.current_value}, predicted={memory_predicted}, max={max_predicted_memory}")
            
            # Try LLM-based recommendation first (if enabled)
            llm_recommendation = None
            if self.use_llm and self.llm_advisor:
                try:
                    # Get HPA status if exists
                    hpa_name = f"{deployment_name}-hpa"
                    hpa_status = None
                    hpa_result = self.hpa_manager.get_hpa(hpa_name, namespace)
                    if hpa_result.get('success'):
                        hpa_status = hpa_result.get('status', {})
                    
                    # Get VPA status if exists
                    vpa_status = None
                    current_resources = None
                    if self.vpa_manager:
                        vpa_name = f"{deployment_name}-vpa"
                        vpa_result = self.vpa_manager.get_vpa(vpa_name, namespace)
                        if vpa_result.get('success'):
                            vpa_status = vpa_result.get('status', {})
                        
                        # Get current resource requests/limits
                        try:
                            resources_result = self.vpa_manager.get_deployment_resources(deployment_name, namespace)
                            if resources_result.get('success') and resources_result.get('resources'):
                                resources_list = resources_result.get('resources', [])
                                if resources_list and len(resources_list) > 0:
                                    # Use first container's resources
                                    first_container = resources_list[0]
                                    current_resources = {
                                        'cpu_request': first_container.get('cpu_request', 'N/A'),
                                        'memory_request': first_container.get('memory_request', 'N/A'),
                                        'cpu_limit': first_container.get('cpu_limit', 'N/A'),
                                        'memory_limit': first_container.get('memory_limit', 'N/A')
                                    }
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to get deployment resources: {e}")
                            current_resources = None
                    
                    # Prepare metrics
                    current_metrics = {
                        'cpu_usage': cpu_current,
                        'memory_usage': memory_current,
                        'pod_count': current_replicas,
                        'running_pod_count': current_replicas
                    }
                    
                    # Prepare forecast (safely handle None values)
                    forecast_data = {
                        'cpu': {
                            'current': cpu_forecast.current_value if cpu_forecast.current_value is not None else 0,
                            'peak': max_predicted_cpu,
                            'trend': cpu_forecast.trend if cpu_forecast.trend else 'stable',
                            'predictions': cpu_predicted  # Use the safe variable we created above
                        },
                        'memory': {
                            'current': memory_forecast.current_value if memory_forecast.current_value is not None else 0,
                            'peak': max_predicted_memory,
                            'trend': memory_forecast.trend if memory_forecast.trend else 'stable',
                            'predictions': memory_predicted  # Use the safe variable we created above
                        }
                    }
                    
                    # Get LLM recommendation (now includes VPA support and state detection)
                    llm_result = self.llm_advisor.get_intelligent_recommendation(
                        deployment_name=deployment_name,
                        namespace=namespace,
                        current_metrics=current_metrics,
                        forecast=forecast_data,
                        hpa_status=hpa_status,
                        vpa_status=vpa_status,
                        current_resources=current_resources,
                        current_replicas=current_replicas,
                        min_replicas=min_replicas,
                        max_replicas=max_replicas,
                        hpa_manager=self.hpa_manager
                    )
                    llm_timing = llm_result.get('timing_breakdown', {}) if isinstance(llm_result, dict) else {}
                    stage_timing['llm_inference_s'] = float(llm_timing.get('llm_inference_s', 0.0) or 0.0)
                    stage_timing['mcda_validation_s'] = float(llm_timing.get('mcda_validation_s', 0.0) or 0.0)
                    
                    # Additional validation: Ensure LLM recommendation respects min/max replicas
                    if llm_result.get('success') and llm_result.get('recommendation'):
                        llm_rec = llm_result['recommendation']
                        if llm_rec.get('target_replicas') is not None:
                            target = llm_rec['target_replicas']
                            if target > max_replicas:
                                logger.warning(f"⚠️ LLM recommended {target} replicas, capping to max_replicas={max_replicas}")
                                llm_rec['target_replicas'] = max_replicas
                                llm_rec['action'] = 'at_max'
                            elif target < min_replicas:
                                logger.warning(f"⚠️ LLM recommended {target} replicas, setting to min_replicas={min_replicas}")
                                llm_rec['target_replicas'] = min_replicas
                                llm_rec['action'] = 'maintain'
                    
                    if llm_result.get('success'):
                        llm_recommendation = llm_result.get('recommendation', {})
                        scaling_type = llm_recommendation.get('scaling_type', 'hpa')
                        if scaling_type == 'vpa':
                            logger.info(f"✅ Using LLM recommendation for VPA scaling: {llm_recommendation.get('action')} -> CPU: {llm_recommendation.get('target_cpu')}, Memory: {llm_recommendation.get('target_memory')}")
                        else:
                            logger.info(f"✅ Using LLM recommendation for HPA scaling: {llm_recommendation.get('action')} -> {llm_recommendation.get('target_replicas')} replicas")
                    elif llm_result.get('rate_limited'):
                        logger.info(f"⏸️  LLM rate-limited, using rule-based scaling")
                        llm_recommendation = None
                    else:
                        logger.warning(f"⚠️  LLM recommendation failed: {llm_result.get('error')}, using rule-based scaling")
                except Exception as e:
                    logger.warning(f"⚠️  Error getting LLM recommendation: {e}, using rule-based scaling")
            
            # Use LLM recommendation if available, otherwise fallback to rule-based
            if llm_recommendation:
                scaling_type = llm_recommendation.get('scaling_type', 'hpa')  # Default to HPA for backward compatibility
                decision = ScalingDecision(
                    action=llm_recommendation.get('action', 'none'),
                    scaling_type=scaling_type,
                    target_replicas=(
                        llm_recommendation.get('target_replicas', current_replicas)
                        if scaling_type in ['hpa', 'both']
                        else None
                    ),
                    target_cpu=llm_recommendation.get('target_cpu') if scaling_type in ['vpa', 'both'] else None,
                    target_memory=llm_recommendation.get('target_memory') if scaling_type in ['vpa', 'both'] else None,
                    reason=llm_recommendation.get('reasoning', 'LLM-based recommendation'),
                    confidence=llm_recommendation.get('confidence', 0.5),
                    source='llm',
                    metadata={'llm_recommendation': llm_recommendation},
                )
                action = decision.to_dict()
                action['llm_recommendation'] = llm_recommendation
                if scaling_type == 'vpa':
                    logger.info(f"🤖 Using LLM recommendation: {action['action']} (VPA) - CPU: {action.get('target_cpu')}, Memory: {action.get('target_memory')}")
                elif scaling_type == 'both':
                    logger.info(f"🤖 Using LLM recommendation: {action['action']} (HPA+VPA) - Replicas: {action.get('target_replicas')}, CPU: {action.get('target_cpu')}, Memory: {action.get('target_memory')}")
                else:
                    logger.info(f"🤖 Using LLM recommendation: {action['action']} (HPA) to {action['target_replicas']} replicas")
            else:
                # Fallback to rule-based recommendation (HPA only)
                action = self._determine_scaling_action(
                    max_predicted_cpu,
                    max_predicted_memory,
                    current_replicas,
                    min_replicas,
                    max_replicas
                )
                action['scaling_type'] = 'hpa'  # Rule-based defaults to HPA
                logger.info(f"📊 Using rule-based recommendation: {action['action']} to {action['target_replicas']} replicas")
            
            # Handle VPA scaling (vertical - resources per pod)
            # Predictive Autoscaling should patch deployment resources directly (like HPA)
            # We do NOT create VPA resources - Predictive Autoscaling controls scaling independently
            scaling_type = action.get('scaling_type', 'hpa')
            if scaling_type in ['vpa', 'both'] and action.get('target_cpu') and action.get('target_memory'):
                if self.vpa_manager and (action['action'] == 'scale_up' or action['action'] == 'scale_down'):
                    # Check if VPA exists - warn but don't modify it
                    vpa_name = f"{deployment_name}-vpa"
                    vpa_result = self.vpa_manager.get_vpa(vpa_name, namespace)
                    vpa_exists = vpa_result.get('success')
                    
                    if vpa_exists:
                        logger.warning(f"⚠️ VPA {vpa_name} exists for {deployment_name}. "
                                     f"Predictive Autoscaling will patch deployment resources directly, "
                                     f"but VPA may override this.")
                    
                    # Patch deployment resources directly (Predictive Autoscaling, not VPA controller)
                    # Calculate limits (2x CPU, 1.5x Memory for headroom)
                    try:
                        cpu_request_m = int(action['target_cpu'][:-1]) if action['target_cpu'].endswith('m') else int(float(action['target_cpu']) * 1000)
                        cpu_limit_m = min(cpu_request_m * 2, 4000)
                        cpu_limit = f"{cpu_limit_m}m"
                        
                        if action['target_memory'].endswith('Mi'):
                            memory_request_mi = int(action['target_memory'][:-2])
                            memory_limit_mi = min(int(memory_request_mi * 1.5), 4096)
                            memory_limit = f"{memory_limit_mi}Mi"
                        elif action['target_memory'].endswith('Gi'):
                            memory_request_gi = int(action['target_memory'][:-2])
                            memory_limit_gi = min(int(memory_request_gi * 1.5), 4)
                            memory_limit = f"{memory_limit_gi}Gi"
                        else:
                            memory_limit = action['target_memory']
                    except:
                        cpu_limit = action['target_cpu']
                        memory_limit = action['target_memory']
                    
                    vpa_actuation_start = time.time()
                    patch_result = self.vpa_manager.patch_deployment_resources(
                        deployment_name, namespace,
                        cpu_request=action['target_cpu'],
                        memory_request=action['target_memory'],
                        cpu_limit=cpu_limit,
                        memory_limit=memory_limit
                    )
                    stage_timing['actuation_s'] += max(0.0, time.time() - vpa_actuation_start)
                    
                    if patch_result.get('success'):
                        logger.info(f"✅ Predictive Autoscaling patched deployment {deployment_name} resources: CPU={action['target_cpu']}, Memory={action['target_memory']} (direct)")
                    else:
                        logger.warning(f"⚠️ Failed to patch deployment resources: {patch_result.get('error')}")
            
            # Handle HPA scaling (horizontal - replica count)
            # PREDICTIVE AUTOSCALING POLICY: Do NOT create or modify HPAs.
            # Predictive autoscaling scales deployments directly via kubectl scale.
            # If an HPA exists, it may override our scaling, but we don't create/modify it.
            # Only process HPA scaling if scaling_type is 'hpa' or 'both', and we have target_replicas
            if (scaling_type == 'hpa' or scaling_type == 'both') and (action['action'] == 'scale_up' or action['action'] == 'at_max'):
                # Calculate required replicas
                required_replicas = action.get('target_replicas')
                
                # Skip if target_replicas is None (shouldn't happen for HPA, but safety check)
                if required_replicas is None:
                    logger.warning(f"⚠️ Skipping HPA scaling - target_replicas is None for scaling_type={scaling_type}")
                else:
                    # Check if HPA exists - warn but don't create/modify it
                    hpa_name = f"{deployment_name}-hpa"
                    hpa_exists = self.hpa_manager.get_hpa(hpa_name, namespace)['success']
                    
                    if hpa_exists:
                        logger.warning(f"⚠️ HPA {hpa_name} exists for {deployment_name}. "
                                     f"Predictive Autoscaling will scale directly, but HPA may override it.")
                    
                    # Scale directly (Predictive Autoscaling controls scaling independently)
                    actuation_start = time.time()
                    scale_result = self._scale_deployment(deployment_name, namespace, required_replicas)
                    stage_timing['actuation_s'] += max(0.0, time.time() - actuation_start)
                    
                    if not scale_result.get('success'):
                        logger.warning(f"⚠️ Failed to scale up: {scale_result.get('error')}")
                    
                    return _with_timing({
                        'success': True,
                        'action': 'scale_up',
                        'current_replicas': current_replicas,
                        'target_replicas': required_replicas,
                        'reason': f'Predicted CPU: {max_predicted_cpu:.1f}%, Memory: {max_predicted_memory:.1f}%',
                        'forecast': {
                            'cpu': {
                                'current': cpu_forecast.current_value,
                                'peak': max_predicted_cpu,
                                'trend': cpu_forecast.trend
                            },
                            'memory': {
                                'current': memory_forecast.current_value,
                                'peak': max_predicted_memory,
                                'trend': memory_forecast.trend
                            }
                        }
                    })
            
            elif (scaling_type == 'hpa' or scaling_type == 'both') and action['action'] == 'scale_down':
                required_replicas = action.get('target_replicas')
                
                # Skip if target_replicas is None (VPA-only scaling)
                if required_replicas is None:
                    logger.warning(f"⚠️ Skipping HPA scale_down - target_replicas is None for scaling_type={scaling_type}")
                else:
                    # Check if HPA exists - if so, we need to update HPA instead of scaling directly
                    hpa_name = f"{deployment_name}-hpa"
                    hpa_exists = self.hpa_manager.get_hpa(hpa_name, namespace)['success']
                    
                    # Predictive Autoscaling should scale directly WITHOUT modifying HPAs
                    # If HPA exists, warn but scale directly (HPA may override)
                    if hpa_exists:
                        logger.warning(f"⚠️ HPA {hpa_name} exists for {deployment_name}. "
                                     f"Predictive Autoscaling will scale directly, but HPA may override it.")
                    
                    # Scale directly (Predictive Autoscaling controls scaling)
                    actuation_start = time.time()
                    scale_result = self._scale_deployment(deployment_name, namespace, required_replicas)
                    stage_timing['actuation_s'] += max(0.0, time.time() - actuation_start)
                    
                    if not scale_result.get('success'):
                        logger.warning(f"⚠️ Failed to scale down: {scale_result.get('error')}")
                    
                    return _with_timing({
                        'success': True,
                        'action': 'scale_down',
                        'current_replicas': current_replicas,
                        'target_replicas': required_replicas,
                        'reason': f'Predicted CPU: {max_predicted_cpu:.1f}%, Memory: {max_predicted_memory:.1f}%',
                        'forecast': {
                            'cpu': {
                                'current': cpu_forecast.current_value,
                                'peak': max_predicted_cpu,
                                'trend': cpu_forecast.trend
                            },
                            'memory': {
                                'current': memory_forecast.current_value,
                                'peak': max_predicted_memory,
                                'trend': memory_forecast.trend
                            }
                        }
                    })
            
            elif action['action'] == 'at_max':
                # Already at max replicas but high usage detected
                return _with_timing({
                    'success': True,
                    'action': 'at_max',
                    'current_replicas': current_replicas,
                    'target_replicas': action['target_replicas'],
                    'calculated_replicas': action.get('calculated_replicas', current_replicas),
                    'reason': action['reason'],
                    'forecast': {
                        'cpu': {
                            'current': cpu_forecast.current_value,
                            'peak': max_predicted_cpu,
                            'trend': cpu_forecast.trend
                        },
                        'memory': {
                            'current': memory_forecast.current_value,
                            'peak': max_predicted_memory,
                            'trend': memory_forecast.trend
                        }
                    }
                })
            else:
                # CRITICAL: Cap target_replicas to max_replicas even for 'none' action
                target_replicas_capped = min(current_replicas, max_replicas)
                if target_replicas_capped != current_replicas:
                    logger.warning(f"🔍🔍🔍 PREDICT_AND_SCALE: Capping current_replicas={current_replicas} to max={max_replicas} for 'none' action")
                
                return _with_timing({
                    'success': True,
                    'action': 'none',
                    'current_replicas': current_replicas,
                    'target_replicas': target_replicas_capped,  # Capped to max_replicas
                    'reason': action.get('reason', 'No scaling needed based on predictions'),
                    'forecast': {
                        'cpu': {
                            'current': cpu_forecast.current_value,
                            'peak': max_predicted_cpu,
                            'trend': cpu_forecast.trend
                        },
                        'memory': {
                            'current': memory_forecast.current_value,
                            'peak': max_predicted_memory,
                            'trend': memory_forecast.trend
                        }
                    }
                })
                
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Error in predictive scaling: {e}")
            logger.error(f"Traceback: {error_traceback}")
            fallback_result = {
                'success': False,
                'error': str(e),
                'action': 'none'
            }
            if '_with_timing' in locals():
                return _with_timing(fallback_result)
            return fallback_result
    
    def _determine_scaling_action(self, predicted_cpu: float, predicted_memory: float,
                                  current_replicas: int, min_replicas: int, max_replicas: int) -> Dict[str, Any]:
        """
        Determine if scaling is needed using MCDA (Multi-Criteria Decision Analysis).

        Replaces simple threshold heuristics with formal TOPSIS optimization that
        evaluates multiple candidate replica counts across weighted criteria:
        cost, performance, stability, forecast alignment, and response time.

        Falls back to threshold-based logic if MCDA fails.
        """
        try:
            # Build metrics and forecast dicts for MCDA
            metrics = {
                'cpu_percent': predicted_cpu,
                'memory_percent': predicted_memory
            }
            forecast = {
                'predicted_cpu': [predicted_cpu] * 6,  # Use predicted as baseline
                'predicted_memory': [predicted_memory] * 6,
                'cpu_trend': 'increasing' if predicted_cpu > 60 else ('decreasing' if predicted_cpu < 30 else 'stable')
            }

            mcda_result = self.mcda_optimizer.optimize(
                current_replicas=current_replicas,
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                metrics=metrics,
                forecast=forecast
            )

            target_replicas = mcda_result.target_replicas
            action = mcda_result.action

            # Handle at_max case
            if action == 'scale_up' and target_replicas >= max_replicas and current_replicas >= max_replicas:
                action = 'at_max'

            # Map MCDA 'maintain' to legacy 'none' for backward compatibility
            if action == 'maintain':
                action = 'none'

            # CRITICAL: Cap target_replicas to max_replicas
            target_replicas = max(min_replicas, min(target_replicas, max_replicas))

            logger.warning(f"📊 MCDA Decision: {action} → {target_replicas} replicas "
                           f"(score={mcda_result.mcda_score:.4f}, margin={mcda_result.dominance_margin:.4f}, "
                           f"evaluated={mcda_result.alternatives_evaluated} alternatives)")

            return {
                'action': action,
                'target_replicas': target_replicas,
                'calculated_replicas': target_replicas,
                'reason': (f'MCDA optimization: {mcda_result.reasoning}. '
                           f'Predicted usage: CPU={predicted_cpu:.1f}%, Memory={predicted_memory:.1f}%'),
                'mcda_score': mcda_result.mcda_score,
                'mcda_ranking': mcda_result.ranking,
                'mcda_dominance_margin': mcda_result.dominance_margin,
                'mcda_criteria_weights': mcda_result.criteria_weights,
                'mcda_criteria_scores': mcda_result.criteria_scores,
                'decision_method': 'mcda_topsis'
            }

        except Exception as e:
            logger.warning(f"⚠️ MCDA optimization failed, falling back to threshold heuristics: {e}")
            return self._determine_scaling_action_fallback(
                predicted_cpu, predicted_memory, current_replicas, min_replicas, max_replicas)

    def _determine_scaling_action_fallback(self, predicted_cpu: float, predicted_memory: float,
                                            current_replicas: int, min_replicas: int, max_replicas: int) -> Dict[str, Any]:
        """Legacy threshold-based fallback when MCDA is unavailable"""

        # Scale up if predicted usage exceeds threshold
        if predicted_cpu > self.scale_up_threshold or predicted_memory > self.scale_up_threshold:
            cpu_scale = predicted_cpu / 70.0
            mem_scale = predicted_memory / 80.0
            scale_factor = max(cpu_scale, mem_scale) * self.safety_buffer

            target_replicas = int(current_replicas * scale_factor)
            calculated_replicas = target_replicas
            target_replicas = max(min_replicas, min(target_replicas, max_replicas))

            if target_replicas > current_replicas:
                return {
                    'action': 'scale_up',
                    'target_replicas': target_replicas,
                    'calculated_replicas': calculated_replicas,
                    'reason': f'[FALLBACK] Predicted usage exceeds threshold (CPU: {predicted_cpu:.1f}%, Memory: {predicted_memory:.1f}%)',
                    'decision_method': 'threshold_fallback'
                }
            elif calculated_replicas > max_replicas and current_replicas >= max_replicas:
                return {
                    'action': 'at_max',
                    'target_replicas': max_replicas,
                    'calculated_replicas': calculated_replicas,
                    'reason': f'[FALLBACK] High predicted usage but already at max replicas ({max_replicas})',
                    'decision_method': 'threshold_fallback'
                }

        # Scale down if predicted usage is low
        if predicted_cpu < self.scale_down_threshold and predicted_memory < self.scale_down_threshold:
            scale_factor = 0.7
            target_replicas = max(min_replicas, int(current_replicas * scale_factor))

            if target_replicas < current_replicas:
                return {
                    'action': 'scale_down',
                    'target_replicas': target_replicas,
                    'reason': f'[FALLBACK] Predicted usage is low (CPU: {predicted_cpu:.1f}%, Memory: {predicted_memory:.1f}%)',
                    'decision_method': 'threshold_fallback'
                }

        target_replicas = min(current_replicas, max_replicas)
        return {
            'action': 'none',
            'target_replicas': target_replicas,
            'reason': f'[FALLBACK] No scaling needed. CPU: {predicted_cpu:.1f}%, Memory: {predicted_memory:.1f}%',
            'decision_method': 'threshold_fallback'
        }
    
    def _determine_vpa_scaling_action(self, predicted_cpu: float, predicted_memory: float,
                                      current_cpu: float, current_memory: float,
                                      deployment_name: str, namespace: str) -> Dict[str, Any]:
        """Determine VPA scaling action based on predictions and current resources"""
        try:
            # Get current resource requests/limits
            current_resources = None
            if self.vpa_manager:
                try:
                    resources_result = self.vpa_manager.get_deployment_resources(deployment_name, namespace)
                    if resources_result.get('success') and resources_result.get('resources'):
                        resources_list = resources_result.get('resources', [])
                        if resources_list and len(resources_list) > 0:
                            first_container = resources_list[0]
                            current_resources = {
                                'cpu_request': first_container.get('cpu_request', '100m'),
                                'memory_request': first_container.get('memory_request', '128Mi')
                            }
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get current resources: {e}")
            
            # Default current resources if not available
            if not current_resources:
                current_resources = {'cpu_request': '100m', 'memory_request': '128Mi'}
            
            # Parse current CPU (e.g., "100m" -> 100)
            cpu_request_str = current_resources['cpu_request']
            if cpu_request_str.endswith('m'):
                cpu_request_m = int(cpu_request_str[:-1])
            else:
                cpu_request_m = int(float(cpu_request_str) * 1000)
            
            # Parse current Memory (e.g., "128Mi" -> 128)
            memory_request_str = current_resources['memory_request']
            if memory_request_str.endswith('Mi'):
                memory_request_mi = int(memory_request_str[:-2])
            elif memory_request_str.endswith('Gi'):
                memory_request_mi = int(float(memory_request_str[:-2]) * 1024)
            else:
                memory_request_mi = 128  # Default
            
            # Calculate target resources based on predicted usage
            # Scale up if predicted usage exceeds threshold
            if predicted_cpu > self.scale_up_threshold or predicted_memory > self.scale_up_threshold:
                # Calculate scale factor (target 70% CPU, 80% Memory)
                cpu_scale = predicted_cpu / 70.0 if predicted_cpu > 0 else 1.0
                mem_scale = predicted_memory / 80.0 if predicted_memory > 0 else 1.0
                scale_factor = max(cpu_scale, mem_scale) * self.safety_buffer
                
                # Calculate new resources
                target_cpu_m = max(100, int(cpu_request_m * scale_factor))  # Min 100m
                target_cpu_m = min(target_cpu_m, 4000)  # Max 4000m (4 cores)
                target_cpu = f"{target_cpu_m}m"
                
                target_memory_mi = max(128, int(memory_request_mi * scale_factor))  # Min 128Mi
                target_memory_mi = min(target_memory_mi, 4096)  # Max 4Gi
                target_memory = f"{target_memory_mi}Mi"
                
                return {
                    'action': 'scale_up',
                    'target_cpu': target_cpu,
                    'target_memory': target_memory,
                    'target_replicas': None,  # VPA doesn't change replicas
                    'reason': f'Predicted usage exceeds threshold (CPU: {predicted_cpu:.1f}%, Memory: {predicted_memory:.1f}%). Recommend increasing resources per pod.'
                }
            
            # Scale down if predicted usage is low
            if predicted_cpu < self.scale_down_threshold and predicted_memory < self.scale_down_threshold:
                scale_factor = 0.8  # Reduce by 20% (conservative for VPA)
                target_cpu_m = max(100, int(cpu_request_m * scale_factor))  # Min 100m
                target_cpu = f"{target_cpu_m}m"
                
                target_memory_mi = max(128, int(memory_request_mi * scale_factor))  # Min 128Mi
                target_memory = f"{target_memory_mi}Mi"
                
                return {
                    'action': 'scale_down',
                    'target_cpu': target_cpu,
                    'target_memory': target_memory,
                    'target_replicas': None,  # VPA doesn't change replicas
                    'reason': f'Predicted usage is low (CPU: {predicted_cpu:.1f}%, Memory: {predicted_memory:.1f}%). Can reduce resources per pod.'
                }
            
            # No scaling needed
            return {
                'action': 'none',
                'target_cpu': current_resources['cpu_request'],
                'target_memory': current_resources['memory_request'],
                'target_replicas': None,
                'reason': f'No scaling needed. Current usage: CPU: {predicted_cpu:.1f}%, Memory: {predicted_memory:.1f}%'
            }
        except Exception as e:
            logger.error(f"❌ Error in VPA scaling action: {e}")
            # Fallback to safe defaults
            return {
                'action': 'none',
                'target_cpu': '200m',
                'target_memory': '256Mi',
                'target_replicas': None,
                'reason': f'Error calculating VPA recommendation: {e}'
            }
    
    def _get_deployment_metrics(self, deployment_name: str, namespace: str) -> Dict[str, float]:
        """Get deployment-specific CPU and memory usage"""
        try:
            # Get pods for this deployment (use deployment name as selector)
            result = subprocess.run(
                ['kubectl', 'get', 'pods', '-n', namespace,
                 '--selector', f'app={deployment_name}', '-o', 'json'],
                capture_output=True,
                text=True,
                timeout=10,
                env={**os.environ, **({'KUBECONFIG': self.hpa_manager.kubeconfig_path} if self.hpa_manager.kubeconfig_path else {})}
            )
            
            if result.returncode != 0:
                return {'cpu_usage': 0, 'memory_usage': 0}
            
            pods_data = json.loads(result.stdout)
            # Safely handle None items
            pods = pods_data.get('items', []) if pods_data else []
            if pods is None:
                pods = []
            
            if not pods:
                return {'cpu_usage': 0, 'memory_usage': 0}
            
            # Use kubectl top pods with label selector (more efficient than listing all pod names)
            # Reduced timeout for recommendations endpoint (faster response)
            top_result = subprocess.run(
                ['kubectl', 'top', 'pods', '-n', namespace, '-l', f'app={deployment_name}'],
                capture_output=True,
                text=True,
                timeout=5,  # Reduced from 10s to 5s for faster recommendations
                env={**os.environ, **({'KUBECONFIG': self.hpa_manager.kubeconfig_path} if self.hpa_manager.kubeconfig_path else {})}
            )
            
            if top_result.returncode != 0:
                return {'cpu_usage': 0, 'memory_usage': 0}
            
            # Safely handle None or empty stdout
            stdout_text = top_result.stdout.strip() if top_result.stdout else ''
            if not stdout_text:
                return {'cpu_usage': 0, 'memory_usage': 0}
            
            # Parse kubectl top output
            lines = stdout_text.split('\n')
            # Skip header if present, filter out empty lines
            if len(lines) > 1:
                lines = [line for line in lines[1:] if line.strip()]
            else:
                lines = [line for line in lines if line.strip()]
            
            if not lines:
                return {'cpu_usage': 0, 'memory_usage': 0}
            
            total_cpu_m = 0
            total_memory_mi = 0
            pod_count = 0
            
            for line in lines:
                parts = line.split()
                if len(parts) >= 3:
                    cpu_str = parts[1]
                    memory_str = parts[2]
                    
                    # Parse CPU (could be m, cores, etc.)
                    if cpu_str.endswith('m'):
                        cpu_m = int(cpu_str[:-1])
                    elif cpu_str.endswith('n'):
                        cpu_m = int(cpu_str[:-1]) / 1000000  # nano to milli
                    else:
                        cpu_m = float(cpu_str) * 1000  # cores to milli
                    
                    # Parse Memory (could be Mi, Gi, etc.)
                    if memory_str.endswith('Mi'):
                        memory_mi = int(memory_str[:-2])
                    elif memory_str.endswith('Gi'):
                        memory_mi = int(memory_str[:-2]) * 1024
                    elif memory_str.endswith('Ki'):
                        memory_mi = int(memory_str[:-2]) / 1024
                    else:
                        memory_mi = int(memory_str) if memory_str.isdigit() else 0
                    
                    total_cpu_m += cpu_m
                    total_memory_mi += memory_mi
                    pod_count += 1
            
            if pod_count == 0:
                return {'cpu_usage': 0, 'memory_usage': 0}
            
            # Get resource requests from deployment spec
            deployment_result = subprocess.run(
                ['kubectl', 'get', 'deployment', deployment_name, '-n', namespace, '-o', 'json'],
                capture_output=True,
                text=True,
                timeout=10,
                env={**os.environ, **({'KUBECONFIG': self.hpa_manager.kubeconfig_path} if self.hpa_manager.kubeconfig_path else {})}
            )
            
            cpu_request_m = 100  # Default fallback
            memory_request_mi = 128  # Default fallback
            
            if deployment_result.returncode == 0:
                try:
                    deployment_data = json.loads(deployment_result.stdout)
                    containers = deployment_data.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])
                    if containers:
                        resources = containers[0].get('resources', {})
                        requests = resources.get('requests', {})
                        
                        # Parse CPU request
                        cpu_request_str = requests.get('cpu', '100m')
                        if cpu_request_str.endswith('m'):
                            cpu_request_m = int(cpu_request_str[:-1])
                        elif cpu_request_str.endswith('n'):
                            cpu_request_m = int(cpu_request_str[:-1]) / 1000000
                        else:
                            cpu_request_m = float(cpu_request_str) * 1000
                        
                        # Parse Memory request
                        memory_request_str = requests.get('memory', '128Mi')
                        if memory_request_str.endswith('Mi'):
                            memory_request_mi = int(memory_request_str[:-2])
                        elif memory_request_str.endswith('Gi'):
                            memory_request_mi = int(memory_request_str[:-2]) * 1024
                        elif memory_request_str.endswith('Ki'):
                            memory_request_mi = int(memory_request_str[:-2]) / 1024
                        else:
                            memory_request_mi = int(memory_request_str) if memory_request_str.isdigit() else 128
                except Exception as e:
                    logger.warning(f"Failed to parse deployment spec: {e}")
            
            # Calculate average usage
            avg_cpu_m = total_cpu_m / pod_count
            avg_memory_mi = total_memory_mi / pod_count
            
            # Calculate percentage based on actual requests
            cpu_percent = (avg_cpu_m / cpu_request_m * 100) if cpu_request_m > 0 else 0
            memory_percent = (avg_memory_mi / memory_request_mi * 100) if memory_request_mi > 0 else 0
            
            result = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent
            }
            logger.info(f"Calculated metrics for {deployment_name}: CPU={cpu_percent:.1f}%, Memory={memory_percent:.1f}% (avg_cpu_m={avg_cpu_m:.1f}m/{cpu_request_m}m, avg_memory_mi={avg_memory_mi:.1f}Mi/{memory_request_mi}Mi)")
            return result
            
        except Exception as e:
            logger.error(f"Failed to get deployment metrics: {e}", exc_info=True)
            return {'cpu_usage': 0, 'memory_usage': 0}
    
    def list_enabled_deployments(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """List all deployments with predictive autoscaling enabled"""
        enabled_deployments = []
        logger.debug(f"🔍 Listing enabled deployments for namespace: {namespace or 'all'}")
        
        # Method 1: Query Kubernetes for deployments with the label
        result = self.hpa_manager.list_deployments_with_label(
            'ai4k8s.io/predictive-autoscaling=enabled',
            namespace
        )
        
        if result and result.get('success'):
            deployments_list = result.get('deployments')
            if deployments_list is None:
                deployments_list = []
            for deployment in deployments_list:
                metadata = deployment.get('metadata', {})
                annotations = metadata.get('annotations', {})
                
                deployment_name = metadata.get('name')
                deployment_namespace = metadata.get('namespace', 'default')
                
                # CRITICAL: Verify annotation actually exists and is set to 'true'
                # This prevents including deployments that are being disabled
                if annotations.get('ai4k8s.io/predictive-autoscaling-enabled') != 'true':
                    logger.debug(f"Skipping {deployment_namespace}/{deployment_name}: annotation not set to 'true'")
                    continue
                
                # Parse config from annotations
                config_json = annotations.get('ai4k8s.io/predictive-autoscaling-config', '{}')
                try:
                    config = json.loads(config_json)
                except:
                    config = {}
                
                # Get actual replica count from deployment status
                deployment_status = self.hpa_manager.get_deployment_replicas(deployment_name, deployment_namespace)
                actual_replicas = deployment_status.get('replicas', 0) if deployment_status.get('success') else 0
                
                enabled_deployments.append({
                    'deployment_name': deployment_name,
                    'namespace': deployment_namespace,
                    'min_replicas': config.get('min_replicas', 2),
                    'max_replicas': config.get('max_replicas', 10),
                    'replicas': actual_replicas,  # Add actual replica count
                    'enabled_at': annotations.get('ai4k8s.io/predictive-autoscaling-enabled-at', '')
                })
        
        # Method 2: Query all deployments and check annotations (fallback if labels missing)
        # This catches deployments where labels weren't added but annotations exist
        try:
            cmd = "get deployments --all-namespaces -o json" if not namespace else f"get deployments -n {namespace} -o json"
            all_deployments_result = self.hpa_manager._execute_kubectl(cmd)
            
            if all_deployments_result and all_deployments_result.get('success'):
                result_data = all_deployments_result.get('result')
                if result_data is None:
                    result_data = {}
                all_deployments = result_data.get('items')
                if all_deployments is None:
                    all_deployments = []
                for deployment in all_deployments:
                    metadata = deployment.get('metadata', {})
                    annotations = metadata.get('annotations', {})
                    labels = metadata.get('labels', {})
                    deployment_name = metadata.get('name')
                    deployment_namespace = metadata.get('namespace', 'default')
                    
                    # CRITICAL: Check if this deployment has predictive autoscaling annotation set to 'true'
                    # This prevents including deployments that are being disabled or have stale annotations
                    has_annotation = annotations.get('ai4k8s.io/predictive-autoscaling-enabled') == 'true'
                    
                    if not has_annotation:
                        logger.debug(f"Skipping {deployment_namespace}/{deployment_name}: annotation not set to 'true' (value: {annotations.get('ai4k8s.io/predictive-autoscaling-enabled')})")
                        continue
                    
                    # Double-check: verify annotation exists and is set to 'true'
                    # This prevents including deployments that are in the process of being disabled
                    if has_annotation:
                        # Check if already in list
                        found = any(
                            d['deployment_name'] == deployment_name and
                            d['namespace'] == deployment_namespace
                            for d in enabled_deployments
                        )
                        if not found:
                            # Parse config from annotations
                            config_json = annotations.get('ai4k8s.io/predictive-autoscaling-config', '{}')
                            try:
                                config = json.loads(config_json)
                            except:
                                config = {}
                            
                            # Get actual replica count
                            deployment_status = self.hpa_manager.get_deployment_replicas(deployment_name, deployment_namespace)
                            actual_replicas = deployment_status.get('replicas', 0) if deployment_status.get('success') else 0
                            
                            enabled_deployments.append({
                                'deployment_name': deployment_name,
                                'namespace': deployment_namespace,
                                'min_replicas': config.get('min_replicas', 2),
                                'max_replicas': config.get('max_replicas', 10),
                                'replicas': actual_replicas,
                                'enabled_at': annotations.get('ai4k8s.io/predictive-autoscaling-enabled-at', '')
                            })
        except Exception as e:
            logger.warning(f"Failed to query all deployments for predictive autoscaling: {e}")
        
        # Method 3: Merge with in-memory cache (for backward compatibility)
        # Only include deployments that are still enabled (have annotations in Kubernetes)
        for key, deployment in self.enabled_deployments.items():
            # Skip if disabled (shouldn't happen after fix, but safety check)
            if deployment.get('disabled'):
                continue
                
            # Check if already in list
            found = any(
                d['deployment_name'] == deployment['deployment_name'] and
                d['namespace'] == deployment['namespace']
                for d in enabled_deployments
            )
            if not found:
                # Verify deployment still has annotation (double-check)
                deployment_status = self.hpa_manager.get_deployment_replicas(
                    deployment['deployment_name'], deployment['namespace']
                )
                if not deployment_status.get('success'):
                    # Deployment doesn't exist, skip
                    continue
                
                # Get actual replica count for in-memory cache entries
                actual_replicas = deployment_status.get('replicas', 0) if deployment_status.get('success') else 0
                
                # Add to list with replica count
                enabled_deployments.append({
                    'deployment_name': deployment['deployment_name'],
                    'namespace': deployment['namespace'],
                    'min_replicas': deployment.get('min_replicas', 2),
                    'max_replicas': deployment.get('max_replicas', 10),
                    'replicas': actual_replicas,  # Add actual replica count
                    'enabled_at': deployment.get('enabled_at', '')
                })
        
        return {
            'success': True,
            'deployments': enabled_deployments,
            'count': len(enabled_deployments)
        }
    
    def disable_predictive_autoscaling(self, deployment_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Disable predictive autoscaling for a deployment"""
        try:
            import subprocess
            env = os.environ.copy()
            if self.hpa_manager.kubeconfig_path:
                env['KUBECONFIG'] = self.hpa_manager.kubeconfig_path
            
            # Use kubectl patch with JSON patch to remove annotations and labels
            # This is more reliable than annotate/label with - suffix
            patch_operations = [
                {'op': 'remove', 'path': '/metadata/annotations/ai4k8s.io~1predictive-autoscaling-enabled'},
                {'op': 'remove', 'path': '/metadata/annotations/ai4k8s.io~1predictive-autoscaling-enabled-at'},
                {'op': 'remove', 'path': '/metadata/annotations/ai4k8s.io~1predictive-autoscaling-config'},
                {'op': 'remove', 'path': '/metadata/annotations/ai4k8s.io~1state-management'},  # Also remove state-management annotation
                {'op': 'remove', 'path': '/metadata/labels/ai4k8s.io~1predictive-autoscaling'}
            ]
            
            # Filter out operations for annotations/labels that don't exist (to avoid errors)
            # First, get the deployment to see what exists
            get_result = self.hpa_manager._execute_kubectl(
                f"get deployment {deployment_name} -n {namespace} -o json"
            )
            
            if not get_result.get('success'):
                logger.error(f"❌ Failed to get deployment {deployment_name} in {namespace}: {get_result.get('error')}")
                return {
                    'success': False,
                    'error': f'Deployment {deployment_name} not found in namespace {namespace}'
                }
            
            deployment_data = get_result.get('result', {})
            annotations = deployment_data.get('metadata', {}).get('annotations', {})
            labels = deployment_data.get('metadata', {}).get('labels', {})
            
            # Remove annotations using kubectl annotate with - suffix (most reliable method)
            # Use --overwrite flag to ensure removal even if annotation doesn't exist
            annotations_removed = []
            for key in ['ai4k8s.io/predictive-autoscaling-enabled', 
                       'ai4k8s.io/predictive-autoscaling-enabled-at',
                       'ai4k8s.io/predictive-autoscaling-config']:
                if key in annotations:
                    annotate_cmd = [
                        'kubectl', 'annotate', 'deployment', deployment_name,
                        f'-n', namespace, '--overwrite', f'{key}-'
                    ]
                    logger.info(f"🔄 Removing annotation {key} using: {' '.join(annotate_cmd)}")
                    result = subprocess.run(annotate_cmd, capture_output=True, text=True, timeout=10, env=env)
                    logger.info(f"Annotation removal result: returncode={result.returncode}, stdout={result.stdout}, stderr={result.stderr}")
                    if result.returncode == 0:
                        annotations_removed.append(key)
                        logger.info(f"✅ Removed annotation: {key}")
                    else:
                        logger.error(f"❌ Failed to remove annotation {key}: {result.stderr}")
                        # Try alternative method: use kubectl patch with strategic merge
                        try:
                            patch_cmd = [
                                'kubectl', 'patch', 'deployment', deployment_name,
                                f'-n', namespace,
                                '--type', 'merge',
                                '-p', json.dumps({'metadata': {'annotations': {key: None}}})
                            ]
                            logger.info(f"🔄 Trying alternative patch method: {' '.join(patch_cmd)}")
                            patch_result = subprocess.run(patch_cmd, capture_output=True, text=True, timeout=10, env=env)
                            if patch_result.returncode == 0:
                                logger.info(f"✅ Removed annotation {key} using patch method")
                                annotations_removed.append(key)
                            else:
                                logger.error(f"❌ Patch method also failed: {patch_result.stderr}")
                        except Exception as e:
                            logger.error(f"❌ Exception trying patch method: {e}")
            
            # Remove label using kubectl label with - suffix
            if 'ai4k8s.io/predictive-autoscaling' in labels:
                label_cmd = [
                    'kubectl', 'label', 'deployment', deployment_name,
                    f'-n', namespace, '--overwrite', 'ai4k8s.io/predictive-autoscaling-'
                ]
                logger.info(f"🔄 Removing label using: {' '.join(label_cmd)}")
                result = subprocess.run(label_cmd, capture_output=True, text=True, timeout=10, env=env)
                logger.info(f"Label removal result: returncode={result.returncode}, stdout={result.stdout}, stderr={result.stderr}")
                if result.returncode == 0:
                    logger.info(f"✅ Removed label: ai4k8s.io/predictive-autoscaling")
                else:
                    logger.error(f"❌ Failed to remove label: {result.stderr}")
            
            if not annotations_removed and 'ai4k8s.io/predictive-autoscaling' not in labels:
                logger.info(f"ℹ️ No annotations/labels to remove for {namespace}/{deployment_name}")
            
            # If annotate commands failed, try strategic merge patch as fallback
            if len(annotations_removed) < len([k for k in ['ai4k8s.io/predictive-autoscaling-enabled', 
                                                          'ai4k8s.io/predictive-autoscaling-enabled-at',
                                                          'ai4k8s.io/predictive-autoscaling-config'] if k in annotations]):
                logger.warning(f"⚠️ Some annotations were not removed, trying strategic merge patch as fallback")
                # Build patch to remove all annotations and label at once
                patch_dict = {'metadata': {'annotations': {}, 'labels': {}}}
                for key in ['ai4k8s.io/predictive-autoscaling-enabled', 
                           'ai4k8s.io/predictive-autoscaling-enabled-at',
                           'ai4k8s.io/predictive-autoscaling-config',
                           'ai4k8s.io/state-management']:  # Also remove state-management annotation
                    if key in annotations:
                        patch_dict['metadata']['annotations'][key] = None
                if 'ai4k8s.io/predictive-autoscaling' in labels:
                    patch_dict['metadata']['labels']['ai4k8s.io/predictive-autoscaling'] = None
                
                patch_cmd = [
                    'kubectl', 'patch', 'deployment', deployment_name,
                    f'-n', namespace,
                    '--type', 'merge',
                    '-p', json.dumps(patch_dict)
                ]
                logger.info(f"🔄 Trying strategic merge patch: {' '.join(patch_cmd)}")
                patch_result = subprocess.run(patch_cmd, capture_output=True, text=True, timeout=10, env=env)
                logger.info(f"Patch result: returncode={patch_result.returncode}, stdout={patch_result.stdout}, stderr={patch_result.stderr}")
                if patch_result.returncode == 0:
                    logger.info(f"✅ Successfully removed annotations/labels using strategic merge patch")
                else:
                    logger.error(f"❌ Strategic merge patch also failed: {patch_result.stderr}")
            
            # Verify removal with retry (Kubernetes API might have slight delay)
            import time
            max_retries = 5  # Increased retries
            removal_verified = False
            
            for attempt in range(max_retries):
                time.sleep(1)  # Increased delay to 1 second
                verify_result = self.hpa_manager._execute_kubectl(
                    f"get deployment {deployment_name} -n {namespace} -o json"
                )
                if verify_result.get('success'):
                    verify_data = verify_result.get('result', {})
                    verify_annotations = verify_data.get('metadata', {}).get('annotations', {})
                    verify_labels = verify_data.get('metadata', {}).get('labels', {})
                    
                    annotation_exists = verify_annotations.get('ai4k8s.io/predictive-autoscaling-enabled') == 'true'
                    label_exists = verify_labels.get('ai4k8s.io/predictive-autoscaling') == 'enabled'
                    
                    if not annotation_exists and not label_exists:
                        removal_verified = True
                        logger.info(f"✅ Verified removal of annotations/labels for {namespace}/{deployment_name} (attempt {attempt + 1})")
                        break
                    else:
                        logger.warning(f"⚠️ Annotations/labels still exist (attempt {attempt + 1}/{max_retries}): annotation={annotation_exists}, label={label_exists}")
                        # If still exists after 2 attempts, try patch again
                        if attempt == 1:
                            logger.info(f"🔄 Retrying patch removal...")
                            patch_dict = {'metadata': {'annotations': {}, 'labels': {}}}
                            for key in ['ai4k8s.io/predictive-autoscaling-enabled', 
                                       'ai4k8s.io/predictive-autoscaling-enabled-at',
                                       'ai4k8s.io/predictive-autoscaling-config']:
                                if verify_annotations.get(key):
                                    patch_dict['metadata']['annotations'][key] = None
                            if verify_labels.get('ai4k8s.io/predictive-autoscaling'):
                                patch_dict['metadata']['labels']['ai4k8s.io/predictive-autoscaling'] = None
                            
                            retry_patch_cmd = [
                                'kubectl', 'patch', 'deployment', deployment_name,
                                f'-n', namespace,
                                '--type', 'merge',
                                '-p', json.dumps(patch_dict)
                            ]
                            retry_result = subprocess.run(retry_patch_cmd, capture_output=True, text=True, timeout=10, env=env)
                            logger.info(f"Retry patch result: returncode={retry_result.returncode}, stderr={retry_result.stderr}")
                else:
                    logger.warning(f"⚠️ Failed to verify removal (attempt {attempt + 1}/{max_retries}): {verify_result.get('error')}")
            
            if not removal_verified:
                logger.error(f"❌ CRITICAL: Could not verify removal of annotations/labels for {deployment_name} after {max_retries} attempts")
                # Still continue to remove from cache
            
            # Remove from in-memory cache completely
            key = f"{namespace}/{deployment_name}"
            if key in self.enabled_deployments:
                del self.enabled_deployments[key]
                logger.info(f"✅ Removed {key} from enabled_deployments cache")
            else:
                logger.debug(f"ℹ️ {key} was not in enabled_deployments cache")
            
            return {
                'success': True,
                'message': f'Predictive autoscaling disabled for {deployment_name}',
                'replicas': 0,  # Return 0 since it's disabled
                'removal_verified': removal_verified
            }
        except Exception as e:
            logger.error(f"Error disabling predictive autoscaling: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def _scale_deployment(self, deployment_name: str, namespace: str, replicas: int) -> Dict[str, Any]:
        """Scale deployment directly"""
        try:
            env = os.environ.copy()
            if self.hpa_manager.kubeconfig_path:
                env['KUBECONFIG'] = self.hpa_manager.kubeconfig_path
            
            # Trim deployment name to remove any whitespace
            deployment_name = deployment_name.strip()
            
            logger.info(f"🔄 Scaling deployment {namespace}/{deployment_name} to {replicas} replicas")
            
            cmd = ['kubectl', 'scale', 'deployment', deployment_name,
                   f'--replicas={replicas}', '-n', namespace]
            
            logger.debug(f"Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                env=env
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Successfully scaled {namespace}/{deployment_name} to {replicas} replicas")
                logger.debug(f"kubectl output: {result.stdout}")
                return {'success': True, 'message': f'Scaled to {replicas} replicas', 'stdout': result.stdout}
            else:
                error_msg = result.stderr or result.stdout or 'Unknown error'
                logger.error(f"❌ Failed to scale {namespace}/{deployment_name}: {error_msg}")
                return {'success': False, 'error': error_msg, 'stdout': result.stdout, 'stderr': result.stderr}
        except Exception as e:
            logger.error(f"❌ Exception in _scale_deployment: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}
    
    def get_scaling_recommendation(self, deployment_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Get scaling recommendation without executing"""
        try:
            # Trim deployment name and namespace to remove any whitespace
            deployment_name = deployment_name.strip()
            namespace = namespace.strip()
            
            # Get current status
            deployment_status = self.hpa_manager.get_deployment_replicas(deployment_name, namespace)
            if not deployment_status['success']:
                return {
                    'success': False,
                    'error': f'Deployment {deployment_name} not found'
                }
            
            current_replicas = deployment_status['replicas']
            
            # Get min/max replicas from deployment annotations
            min_replicas = 2  # default
            max_replicas = 10  # default
            try:
                deployment_result = self.hpa_manager._execute_kubectl(
                    f"get deployment {deployment_name} -n {namespace} -o json"
                )
                if deployment_result.get('success'):
                    deployment_data = deployment_result.get('result', {})
                    annotations = deployment_data.get('metadata', {}).get('annotations', {})
                    config_json = annotations.get('ai4k8s.io/predictive-autoscaling-config', '{}')
                    try:
                        if config_json and config_json != '{}':
                            config = json.loads(config_json)
                            min_replicas = config.get('min_replicas', 2)
                            max_replicas = config.get('max_replicas', 10)
                            logger.warning(f"📊📊📊 Retrieved min/max replicas from annotation: min={min_replicas}, max={max_replicas}")
                            print(f"📊📊📊 Retrieved min/max replicas from annotation: min={min_replicas}, max={max_replicas}")
                        else:
                            # Annotation is missing or empty - this shouldn't happen if predictive autoscaling is enabled
                            logger.warning(f"⚠️ Annotation ai4k8s.io/predictive-autoscaling-config is missing or empty! Deployment might not be properly enabled.")
                            logger.warning(f"⚠️ Using defaults: min=2, max=10. Please re-enable predictive autoscaling to set correct values.")
                            min_replicas = 2
                            max_replicas = 10
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ Failed to parse config annotation (invalid JSON): {e}, using defaults")
                        min_replicas = 2
                        max_replicas = 10
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to parse config annotation: {e}, using defaults")
                        min_replicas = 2
                        max_replicas = 10
            except Exception as e:
                logger.warning(f"⚠️ Failed to get deployment annotations: {e}, using defaults")
            
            # Try to get deployment-specific metrics (with shorter timeout for recommendations)
            try:
                deployment_metrics = self._get_deployment_metrics(deployment_name, namespace)
                logger.info(f"Getting recommendation for {deployment_name}: metrics={deployment_metrics}")
            except Exception as e:
                logger.warning(f"⚠️ Metrics collection failed/slow: {e}, using forecast values")
                deployment_metrics = {'cpu_usage': 0, 'memory_usage': 0}
            
            # Get forecasts (use deployment-specific current values if available)
            cpu_current_from_metrics = deployment_metrics.get('cpu_usage', 0)
            memory_current_from_metrics = deployment_metrics.get('memory_usage', 0)
            
            cpu_forecast = self.monitoring_system.forecaster.forecast_cpu_usage(
                hours_ahead=self.prediction_horizon
            )
            memory_forecast = self.monitoring_system.forecaster.forecast_memory_usage(
                hours_ahead=self.prediction_horizon
            )
            
            # Override current values with deployment-specific metrics if available
            # CRITICAL: Use forecast's current_value as the source of truth (it's more up-to-date)
            if cpu_current_from_metrics > 0:
                cpu_forecast.current_value = cpu_current_from_metrics
            if memory_current_from_metrics > 0:
                memory_forecast.current_value = memory_current_from_metrics
            
            # Use forecast's current_value as the authoritative source (same as UI displays)
            cpu_current = cpu_forecast.current_value if cpu_forecast.current_value is not None else cpu_current_from_metrics
            memory_current = memory_forecast.current_value if memory_forecast.current_value is not None else memory_current_from_metrics
            
            # Safely format for logging (handle None values)
            cpu_forecast_val = cpu_forecast.current_value if cpu_forecast.current_value is not None else 0
            memory_forecast_val = memory_forecast.current_value if memory_forecast.current_value is not None else 0
            logger.warning(f"🔍🔍🔍 METRICS SOURCE: cpu_from_metrics={cpu_current_from_metrics:.1f}%, cpu_from_forecast={cpu_forecast_val:.1f}%, using={cpu_current:.1f}%")
            logger.warning(f"🔍🔍🔍 METRICS SOURCE: memory_from_metrics={memory_current_from_metrics:.1f}%, memory_from_forecast={memory_forecast_val:.1f}%, using={memory_current:.1f}%")
            
            # Find peak predicted usage
            # Use current_value if predicted_values is empty or all zeros
            # Safely handle None predicted_values
            cpu_predicted = cpu_forecast.predicted_values if cpu_forecast.predicted_values is not None else []
            memory_predicted = memory_forecast.predicted_values if memory_forecast.predicted_values is not None else []
            
            if cpu_predicted and any(v > 0 for v in cpu_predicted):
                max_predicted_cpu = max(cpu_predicted)
            else:
                max_predicted_cpu = cpu_forecast.current_value if cpu_forecast.current_value is not None else 0
            
            if memory_predicted and any(v > 0 for v in memory_predicted):
                max_predicted_memory = max(memory_predicted)
            else:
                max_predicted_memory = memory_forecast.current_value if memory_forecast.current_value is not None else 0
            
            logger.info(f"Recommendation forecast - CPU: current={cpu_forecast.current_value}, predicted={cpu_predicted}, max={max_predicted_cpu}")
            logger.info(f"Recommendation forecast - Memory: current={memory_forecast.current_value}, predicted={memory_predicted}, max={max_predicted_memory}")
            
            # Try LLM-based recommendation first
            llm_recommendation = None
            logger.warning(f"🔍🔍🔍 LLM CHECK: use_llm={self.use_llm}, llm_advisor={self.llm_advisor is not None}, client={self.llm_advisor.client is not None if self.llm_advisor else False}")
            if self.use_llm and self.llm_advisor:
                try:
                    # OPTIMIZATION: Skip slow HPA/VPA lookups for recommendations to save time
                    # LLM can make good decisions without these - they're optional context
                    hpa_status = None
                    vpa_status = None
                    current_resources = None
                    
                    # Skip HPA/VPA checks for recommendations (saves 2-4 seconds)
                    # These are nice-to-have but not critical for LLM decision-making
                    logger.info("⏩ Skipping HPA/VPA lookups for faster recommendations")
                    
                    # Prepare metrics
                    current_metrics = {
                        'cpu_usage': cpu_current,
                        'memory_usage': memory_current,
                        'pod_count': current_replicas,
                        'running_pod_count': current_replicas
                    }
                    
                    # Prepare forecast (safely handle None values)
                    forecast_data = {
                        'cpu': {
                            'current': cpu_forecast.current_value if cpu_forecast.current_value is not None else 0,
                            'peak': max_predicted_cpu,
                            'trend': cpu_forecast.trend if cpu_forecast.trend else 'stable',
                            'predictions': cpu_predicted  # Use the safe variable we created above
                        },
                        'memory': {
                            'current': memory_forecast.current_value if memory_forecast.current_value is not None else 0,
                            'peak': max_predicted_memory,
                            'trend': memory_forecast.trend if memory_forecast.trend else 'stable',
                            'predictions': memory_predicted  # Use the safe variable we created above
                        }
                    }
                    
                    # Get LLM recommendation (now includes VPA support)
                    # Use min/max replicas from deployment annotations (retrieved above)
                    logger.warning(f"📊📊📊 Calling LLM advisor with min_replicas={min_replicas}, max_replicas={max_replicas}")
                    print(f"📊📊📊 Calling LLM advisor with min_replicas={min_replicas}, max_replicas={max_replicas}")
                    
                    # Add timeout wrapper to fail fast if LLM is slow (8s max)
                    import threading
                    import queue
                    llm_result_queue = queue.Queue()
                    llm_error_queue = queue.Queue()
                    
                    def call_llm():
                        try:
                            result = self.llm_advisor.get_intelligent_recommendation(
                                deployment_name=deployment_name,
                                namespace=namespace,
                                current_metrics=current_metrics,
                                forecast=forecast_data,
                                hpa_status=hpa_status,
                                vpa_status=vpa_status,
                                current_resources=current_resources,
                                current_replicas=current_replicas,
                                min_replicas=min_replicas,  # From deployment annotation
                                max_replicas=max_replicas,  # From deployment annotation
                                hpa_manager=self.hpa_manager
                            )
                            llm_result_queue.put(result)
                        except Exception as e:
                            llm_error_queue.put(e)
                    
                    llm_thread = threading.Thread(target=call_llm, daemon=True)
                    llm_thread.start()
                    llm_thread.join(timeout=250.0)  # 250 second timeout (allows Qwen 240s + Groq 15s + buffer)
                    
                    if llm_thread.is_alive():
                        # LLM timed out - use rule-based fallback immediately
                        logger.warning("⏱️ LLM call timed out after 40s, using rule-based fallback")
                        llm_result = {'success': False, 'error': 'LLM timeout'}
                    elif not llm_error_queue.empty():
                        # LLM call failed with exception
                        error = llm_error_queue.get()
                        logger.warning(f"⚠️ LLM call failed: {error}, using rule-based fallback")
                        llm_result = {'success': False, 'error': str(error)}
                    elif not llm_result_queue.empty():
                        # LLM call succeeded
                        llm_result = llm_result_queue.get()
                    else:
                        # Unexpected - no result, no error
                        logger.warning("⚠️ LLM call returned no result, using rule-based fallback")
                        llm_result = {'success': False, 'error': 'No result from LLM'}
                    
                    logger.warning(f"🔍🔍🔍 LLM RESULT: success={llm_result.get('success')}, cached={llm_result.get('cached', False)}, rate_limited={llm_result.get('rate_limited', False)}, error={llm_result.get('error', 'none')}")
                    if llm_result.get('success'):
                        llm_recommendation = llm_result.get('recommendation', {})
                        scaling_type = llm_recommendation.get('scaling_type', 'hpa')
                        logger.warning(f"🔍🔍🔍 LLM RECOMMENDATION RECEIVED: scaling_type={scaling_type}, action={llm_recommendation.get('action')}, target_replicas={llm_recommendation.get('target_replicas')}, cached={llm_result.get('cached', False)}")
                        
                        # CRITICAL: Validate and enforce min/max replica constraints as safety net
                        if scaling_type in ['hpa', 'both'] and 'target_replicas' in llm_recommendation and llm_recommendation['target_replicas'] is not None:
                            target = llm_recommendation['target_replicas']
                            logger.warning(f"🔍🔍🔍 SAFETY VALIDATION CHECK: target={target}, max_replicas={max_replicas}, min_replicas={min_replicas}")
                            if target > max_replicas:
                                logger.error(f"❌❌❌ SAFETY VALIDATION: LLM recommended {target} replicas but max is {max_replicas}. Capping to {max_replicas}.")
                                llm_recommendation['target_replicas'] = max_replicas
                                llm_recommendation['action'] = 'at_max'
                                llm_recommendation['reasoning'] = (llm_recommendation.get('reasoning', '') + 
                                    f" [SAFETY: Capped from {target} to max_replicas={max_replicas}]")
                            elif target < min_replicas:
                                logger.error(f"❌❌❌ SAFETY VALIDATION: LLM recommended {target} replicas but min is {min_replicas}. Setting to {min_replicas}.")
                                llm_recommendation['target_replicas'] = min_replicas
                                llm_recommendation['action'] = 'maintain'
                                llm_recommendation['reasoning'] = (llm_recommendation.get('reasoning', '') + 
                                    f" [SAFETY: Set from {target} to min_replicas={min_replicas}]")
                        
                        if scaling_type == 'vpa':
                            logger.info(f"✅ LLM recommendation: {llm_recommendation.get('action')} (VPA) -> CPU: {llm_recommendation.get('target_cpu')}, Memory: {llm_recommendation.get('target_memory')}")
                        else:
                            logger.info(f"✅ LLM recommendation: {llm_recommendation.get('action')} (HPA) -> {llm_recommendation.get('target_replicas')} replicas (min={min_replicas}, max={max_replicas})")
                    elif llm_result.get('rate_limited'):
                        # Rate-limited, use cached recommendation if available, otherwise fallback
                        logger.info(f"⏸️  LLM rate-limited, using fallback recommendation")
                        llm_recommendation = None
                    else:
                        logger.warning(f"⚠️  LLM recommendation failed: {llm_result.get('error')}, using fallback")
                except Exception as e:
                    logger.warning(f"⚠️  Error getting LLM recommendation: {e}, using fallback")
            
            # Use LLM recommendation if available, otherwise fallback to rule-based
            logger.warning(f"🔍🔍🔍 FINAL CHECK: llm_recommendation={llm_recommendation is not None}, will use {'LLM' if llm_recommendation else 'RULE-BASED FALLBACK'}")
            if llm_recommendation:
                scaling_type = llm_recommendation.get('scaling_type', 'hpa')  # Default to HPA for backward compatibility
                action_type = llm_recommendation.get('action', 'none')
                
                # CRITICAL: For "maintain" or "none" actions, use current_replicas but ensure it's within bounds
                # For other actions, use LLM's target_replicas or default to current_replicas
                if action_type in ['maintain', 'none']:
                    # For maintain actions, check if LLM explicitly set target_replicas
                    llm_target = llm_recommendation.get('target_replicas')
                    if llm_target is not None:
                        # LLM explicitly set target_replicas - use it but cap to max
                        target_replicas_raw = min(llm_target, max_replicas) if scaling_type in ['hpa', 'both'] else None
                        print(f"🔍🔍🔍 MAINTAIN ACTION: LLM set target_replicas={llm_target}, capped to max={max_replicas}, result={target_replicas_raw}")
                        logger.warning(f"🔍🔍🔍 MAINTAIN ACTION: LLM set target_replicas={llm_target}, capped to max={max_replicas}, result={target_replicas_raw}")
                    else:
                        # LLM didn't set target_replicas - use current_replicas but capped to max
                        target_replicas_raw = min(current_replicas, max_replicas) if scaling_type in ['hpa', 'both'] else None
                        print(f"🔍🔍🔍 MAINTAIN ACTION: Using current_replicas={current_replicas} capped to max={max_replicas}, result={target_replicas_raw}")
                        logger.warning(f"🔍🔍🔍 MAINTAIN ACTION: Using current_replicas={current_replicas} capped to max={max_replicas}, result={target_replicas_raw}")
                else:
                    # For scale_up/scale_down, use LLM's target_replicas or default to current_replicas
                    target_replicas_raw = llm_recommendation.get('target_replicas', current_replicas) if scaling_type in ['hpa', 'both'] else None
                
                print(f"🔍🔍🔍 LLM RECOMMENDATION PROCESSING: action={action_type}, target_replicas_raw={target_replicas_raw}, current_replicas={current_replicas}, min={min_replicas}, max={max_replicas}")
                logger.warning(f"🔍🔍🔍 LLM RECOMMENDATION PROCESSING: action={action_type}, target_replicas_raw={target_replicas_raw}, current_replicas={current_replicas}, min={min_replicas}, max={max_replicas}")
                
                # CRITICAL: Final validation - cap target_replicas to min/max
                if target_replicas_raw is not None:
                    if target_replicas_raw > max_replicas:
                        print(f"❌❌❌ FINAL VALIDATION: target_replicas={target_replicas_raw} exceeds max={max_replicas}, capping to {max_replicas}")
                        logger.error(f"❌❌❌ FINAL VALIDATION: target_replicas={target_replicas_raw} exceeds max={max_replicas}, capping to {max_replicas}")
                        target_replicas_raw = max_replicas
                    elif target_replicas_raw < min_replicas:
                        print(f"❌❌❌ FINAL VALIDATION: target_replicas={target_replicas_raw} below min={min_replicas}, setting to {min_replicas}")
                        logger.error(f"❌❌❌ FINAL VALIDATION: target_replicas={target_replicas_raw} below min={min_replicas}, setting to {min_replicas}")
                        target_replicas_raw = min_replicas
                    print(f"🔍🔍🔍 FINAL VALIDATION RESULT: target_replicas={target_replicas_raw} (min={min_replicas}, max={max_replicas})")
                    logger.warning(f"🔍🔍🔍 FINAL VALIDATION: target_replicas={target_replicas_raw} (min={min_replicas}, max={max_replicas})")
                
                decision = ScalingDecision(
                    action=llm_recommendation.get('action', 'none'),
                    scaling_type=scaling_type,
                    target_replicas=target_replicas_raw,
                    target_cpu=llm_recommendation.get('target_cpu') if scaling_type in ['vpa', 'both'] else None,
                    target_memory=llm_recommendation.get('target_memory') if scaling_type in ['vpa', 'both'] else None,
                    reason=llm_recommendation.get('reasoning', 'LLM-based recommendation'),
                    confidence=llm_recommendation.get('confidence', 0.5),
                    source='llm',
                    metadata={'llm_recommendation': llm_recommendation},
                )
                action = decision.to_dict()
                action['llm_recommendation'] = llm_recommendation
            else:
                # Fallback to rule-based recommendation - USE ACTUAL min/max from annotations!
                print(f"🔍🔍🔍 Using RULE-BASED FALLBACK with min={min_replicas}, max={max_replicas}, current_replicas={current_replicas}")
                logger.warning(f"🔍🔍🔍 Using RULE-BASED FALLBACK with min={min_replicas}, max={max_replicas}, current_replicas={current_replicas}")
                
                # Check if VPA exists - if so, provide VPA recommendations instead of HPA
                vpa_exists = False
                if self.vpa_manager:
                    try:
                        vpa_name = f"{deployment_name}-vpa"
                        vpa_result = self.vpa_manager.get_vpa(vpa_name, namespace)
                        vpa_exists = vpa_result.get('success', False)
                        logger.info(f"🔍 VPA check: exists={vpa_exists}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to check VPA: {e}")
                        vpa_exists = False
                
                if vpa_exists:
                    # Provide VPA recommendations (vertical scaling)
                    action = self._determine_vpa_scaling_action(
                        max_predicted_cpu,
                        max_predicted_memory,
                        cpu_current,
                        memory_current,
                        deployment_name,
                        namespace
                    )
                    action['scaling_type'] = 'vpa'
                    logger.info(f"📊 Rule-based VPA recommendation: {action.get('action')}, CPU={action.get('target_cpu')}, Memory={action.get('target_memory')}")
                else:
                    # Provide HPA recommendations (horizontal scaling)
                    action = self._determine_scaling_action(
                        max_predicted_cpu,
                        max_predicted_memory,
                        current_replicas,
                        min_replicas,  # Use actual min from deployment annotation
                        max_replicas   # Use actual max from deployment annotation
                    )
                    action['scaling_type'] = 'hpa'
                    print(f"🔍🔍🔍 RULE-BASED RESULT: action={action.get('action')}, target_replicas={action.get('target_replicas')}")
                    logger.warning(f"🔍🔍🔍 RULE-BASED RESULT: action={action.get('action')}, target_replicas={action.get('target_replicas')}")
                    
                    # CRITICAL: Ensure rule-based fallback also respects max_replicas
                    if action.get('target_replicas') is not None:
                        target = action['target_replicas']
                        if target > max_replicas:
                            print(f"❌❌❌ RULE-BASED VALIDATION: target_replicas={target} exceeds max={max_replicas}, capping to {max_replicas}")
                            logger.error(f"❌❌❌ RULE-BASED VALIDATION: target_replicas={target} exceeds max={max_replicas}, capping to {max_replicas}")
                            action['target_replicas'] = max_replicas
                            action['action'] = 'at_max'
                        elif target < min_replicas:
                            print(f"❌❌❌ RULE-BASED VALIDATION: target_replicas={target} below min={min_replicas}, setting to {min_replicas}")
                            logger.error(f"❌❌❌ RULE-BASED VALIDATION: target_replicas={target} below min={min_replicas}, setting to {min_replicas}")
                            action['target_replicas'] = min_replicas
                            action['action'] = 'maintain'
            
            # Get deployment status for resource stats
            deployment_status = self.hpa_manager.get_deployment_replicas(deployment_name, namespace)
            ready_replicas = deployment_status.get('ready_replicas', 0) if deployment_status.get('success') else 0
            available_replicas = deployment_status.get('available_replicas', 0) if deployment_status.get('success') else 0
            
            # Safely get predicted values (handle None)
            cpu_predicted_safe = cpu_forecast.predicted_values if cpu_forecast.predicted_values is not None else []
            memory_predicted_safe = memory_forecast.predicted_values if memory_forecast.predicted_values is not None else []
            
            # CRITICAL: Final safety check - ensure target_replicas respects min/max before returning
            if action.get('target_replicas') is not None:
                target = action['target_replicas']
                print(f"🔍🔍🔍 RETURN VALIDATION CHECK: target={target}, min={min_replicas}, max={max_replicas}")
                logger.warning(f"🔍🔍🔍 RETURN VALIDATION CHECK: target={target}, min={min_replicas}, max={max_replicas}")
                if target > max_replicas:
                    print(f"❌❌❌ RETURN VALIDATION: target_replicas={target} exceeds max={max_replicas}, capping to {max_replicas}")
                    logger.error(f"❌❌❌ RETURN VALIDATION: target_replicas={target} exceeds max={max_replicas}, capping to {max_replicas}")
                    action['target_replicas'] = max_replicas
                    action['action'] = 'at_max'
                elif target < min_replicas:
                    print(f"❌❌❌ RETURN VALIDATION: target_replicas={target} below min={min_replicas}, setting to {min_replicas}")
                    logger.error(f"❌❌❌ RETURN VALIDATION: target_replicas={target} below min={min_replicas}, setting to {min_replicas}")
                    action['target_replicas'] = min_replicas
                    action['action'] = 'maintain'
                print(f"🔍🔍🔍 RETURN VALIDATION RESULT: Final target_replicas={action['target_replicas']} (min={min_replicas}, max={max_replicas})")
                logger.warning(f"🔍🔍🔍 RETURN VALIDATION: Final target_replicas={action['target_replicas']} (min={min_replicas}, max={max_replicas})")
            else:
                print(f"🔍🔍🔍 RETURN VALIDATION: target_replicas is None, skipping validation")
                logger.warning(f"🔍🔍🔍 RETURN VALIDATION: target_replicas is None, skipping validation")
            
            # ABSOLUTE FINAL CHECK: Ensure target_replicas is ALWAYS within bounds before returning
            if action.get('target_replicas') is not None:
                final_target = action['target_replicas']
                if final_target > max_replicas:
                    print(f"🚨🚨🚨 ABSOLUTE FINAL CHECK: target_replicas={final_target} > max={max_replicas}, FORCING to {max_replicas}")
                    logger.error(f"🚨🚨🚨 ABSOLUTE FINAL CHECK: target_replicas={final_target} > max={max_replicas}, FORCING to {max_replicas}")
                    action['target_replicas'] = max_replicas
                    if action.get('action') in ['maintain', 'none']:
                        action['action'] = 'at_max'
                elif final_target < min_replicas:
                    print(f"🚨🚨🚨 ABSOLUTE FINAL CHECK: target_replicas={final_target} < min={min_replicas}, FORCING to {min_replicas}")
                    logger.error(f"🚨🚨🚨 ABSOLUTE FINAL CHECK: target_replicas={final_target} < min={min_replicas}, FORCING to {min_replicas}")
                    action['target_replicas'] = min_replicas
                    action['action'] = 'maintain'
            
            print(f"🔍🔍🔍 FINAL ACTION BEFORE RETURN: {action}")
            logger.warning(f"🔍🔍🔍 FINAL ACTION BEFORE RETURN: target_replicas={action.get('target_replicas')}, action={action.get('action')}, min={min_replicas}, max={max_replicas}")
            
            # Create a clean message for UI (without full LLM reasoning)
            clean_message = "Predictive autoscaling enabled successfully"
            if action.get('action') == 'scale_up' and action.get('target_replicas'):
                clean_message = f"Predictive autoscaling enabled. Scaling to {action.get('target_replicas')} replicas."
            elif action.get('action') == 'scale_down' and action.get('target_replicas'):
                clean_message = f"Predictive autoscaling enabled. Scaling to {action.get('target_replicas')} replicas."
            elif action.get('action') == 'at_max':
                clean_message = "Predictive autoscaling enabled. Already at max replicas."
            elif action.get('scaling_type') == 'vpa' and (action.get('target_cpu') or action.get('target_memory')):
                clean_message = "Predictive autoscaling enabled. VPA recommendation will be applied."
            elif action.get('action') == 'maintain' or action.get('action') == 'none':
                clean_message = "Predictive autoscaling enabled. No scaling needed at this time."
            
            return {
                'success': True,
                'message': clean_message,  # Clean message for UI alerts
                'recommendation': action,
                'forecast': {
                    'cpu': {
                        'current': cpu_forecast.current_value if cpu_forecast.current_value is not None else 0,
                        'peak': max_predicted_cpu,
                        'trend': cpu_forecast.trend if cpu_forecast.trend else 'stable',
                        'predictions': cpu_predicted_safe
                    },
                    'memory': {
                        'current': memory_forecast.current_value if memory_forecast.current_value is not None else 0,
                        'peak': max_predicted_memory,
                        'trend': memory_forecast.trend if memory_forecast.trend else 'stable',
                        'predictions': memory_predicted_safe
                    }
                },
                'current_replicas': current_replicas,
                'ready_replicas': ready_replicas,
                'available_replicas': available_replicas,
                'current_resources': current_resources,  # Add current resource requests/limits for VPA display
                'current_metrics': {
                    'cpu_usage': cpu_current,
                    'memory_usage': memory_current
                },
                'llm_enabled': self.use_llm and self.llm_advisor is not None,
                'llm_used': llm_recommendation is not None
            }
            
        except Exception as e:
            logger.error(f"Error getting scaling recommendation: {e}")
            return {
                'success': False,
                'error': str(e)
            }

