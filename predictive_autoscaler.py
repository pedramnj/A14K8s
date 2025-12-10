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
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from predictive_monitoring import (
    PredictiveMonitoringSystem,
    ResourceMetrics,
    ForecastResult
)
from autoscaling_engine import HorizontalPodAutoscaler
from llm_autoscaling_advisor import LLMAutoscalingAdvisor

logger = logging.getLogger(__name__)

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
        
    def predict_and_scale(self, deployment_name: str, namespace: str = "default",
                         min_replicas: int = 2, max_replicas: int = 10) -> Dict[str, Any]:
        """Predict future load and scale proactively"""
        try:
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
            
            labels = {
                'ai4k8s.io/predictive-autoscaling': 'enabled'
            }
            
            # Patch annotations
            annot_result = self.hpa_manager.patch_deployment_annotations(
                deployment_name, namespace, annotations
            )
            if not annot_result.get('success'):
                logger.warning(f"Failed to add annotations: {annot_result.get('error')}")
            
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
                return {
                    'success': False,
                    'error': f'Deployment {deployment_name} not found',
                    'action': 'none'
                }
            
            current_replicas = deployment_status['replicas']
            
            # Get deployment-specific metrics
            deployment_metrics = self._get_deployment_metrics(deployment_name, namespace)
            cpu_current = deployment_metrics.get('cpu_usage', 0)
            memory_current = deployment_metrics.get('memory_usage', 0)
            
            logger.info(f"Deployment metrics for {deployment_name}: CPU={cpu_current}%, Memory={memory_current}%")
            
            # Get forecasts (use deployment-specific current values if available)
            cpu_forecast = self.monitoring_system.forecaster.forecast_cpu_usage(
                hours_ahead=self.prediction_horizon
            )
            memory_forecast = self.monitoring_system.forecaster.forecast_memory_usage(
                hours_ahead=self.prediction_horizon
            )
            
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
                    
                    # Get LLM recommendation (now includes VPA support)
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
                        max_replicas=max_replicas
                    )
                    
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
                action = {
                    'action': llm_recommendation.get('action', 'none'),
                    'scaling_type': scaling_type,  # 'hpa', 'vpa', 'both', or 'maintain'
                    'target_replicas': llm_recommendation.get('target_replicas', current_replicas) if scaling_type in ['hpa', 'both'] else None,
                    'target_cpu': llm_recommendation.get('target_cpu') if scaling_type in ['vpa', 'both'] else None,
                    'target_memory': llm_recommendation.get('target_memory') if scaling_type in ['vpa', 'both'] else None,
                    'reason': llm_recommendation.get('reasoning', 'LLM-based recommendation'),
                    'confidence': llm_recommendation.get('confidence', 0.5),
                    'llm_recommendation': llm_recommendation
                }
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
                    
                    patch_result = self.vpa_manager.patch_deployment_resources(
                        deployment_name, namespace,
                        cpu_request=action['target_cpu'],
                        memory_request=action['target_memory'],
                        cpu_limit=cpu_limit,
                        memory_limit=memory_limit
                    )
                    
                    if patch_result.get('success'):
                        logger.info(f"✅ Predictive Autoscaling patched deployment {deployment_name} resources: CPU={action['target_cpu']}, Memory={action['target_memory']} (direct)")
                    else:
                        logger.warning(f"⚠️ Failed to patch deployment resources: {patch_result.get('error')}")
            
            # Handle HPA scaling (horizontal - replica count)
            # Only process HPA scaling if scaling_type is 'hpa' or 'both', and we have target_replicas
            if (scaling_type == 'hpa' or scaling_type == 'both') and (action['action'] == 'scale_up' or action['action'] == 'at_max'):
                # Calculate required replicas
                required_replicas = action.get('target_replicas')
                
                # Skip if target_replicas is None (shouldn't happen for HPA, but safety check)
                if required_replicas is None:
                    logger.warning(f"⚠️ Skipping HPA scaling - target_replicas is None for scaling_type={scaling_type}")
                else:
                    # Check if HPA exists
                    hpa_name = f"{deployment_name}-hpa"
                    hpa_exists = self.hpa_manager.get_hpa(hpa_name, namespace)['success']
                    
                    if not hpa_exists:
                        # Create HPA with predictive settings
                        hpa_result = self.hpa_manager.create_hpa(
                            deployment_name=deployment_name,
                            namespace=namespace,
                            min_replicas=min_replicas,
                            max_replicas=max_replicas,
                            cpu_target=70,
                            memory_target=80
                        )
                        
                        if hpa_result['success']:
                            # Update HPA to desired replicas immediately (proactive scaling)
                            update_result = self.hpa_manager.update_hpa(
                                hpa_name=hpa_name,
                                namespace=namespace,
                                min_replicas=min_replicas,
                                max_replicas=max(required_replicas, max_replicas)
                            )
                            
                            # Also scale deployment directly for immediate effect
                            self._scale_deployment(deployment_name, namespace, required_replicas)
                            
                            return {
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
                                },
                                'hpa_created': True
                            }
                        else:
                            return {
                                'success': False,
                                'error': f'Failed to create HPA: {hpa_result.get("error")}',
                                'action': 'none'
                            }
                    else:
                        # Update existing HPA
                        self._scale_deployment(deployment_name, namespace, required_replicas)
                        
                        return {
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
                            },
                            'hpa_updated': True
                        }
            
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
                    scale_result = self._scale_deployment(deployment_name, namespace, required_replicas)
                    
                    if not scale_result.get('success'):
                        logger.warning(f"⚠️ Failed to scale down: {scale_result.get('error')}")
                    
                    return {
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
                    }
            
            elif action['action'] == 'at_max':
                # Already at max replicas but high usage detected
                return {
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
                }
            else:
                return {
                    'success': True,
                    'action': 'none',
                    'current_replicas': current_replicas,
                    'target_replicas': current_replicas,
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
                }
                
        except Exception as e:
            logger.error(f"Error in predictive scaling: {e}")
            return {
                'success': False,
                'error': str(e),
                'action': 'none'
            }
    
    def _determine_scaling_action(self, predicted_cpu: float, predicted_memory: float,
                                  current_replicas: int, min_replicas: int, max_replicas: int) -> Dict[str, Any]:
        """Determine if scaling is needed based on predictions"""
        
        # Scale up if predicted usage exceeds threshold
        if predicted_cpu > self.scale_up_threshold or predicted_memory > self.scale_up_threshold:
            # Calculate scale factor
            cpu_scale = predicted_cpu / 70.0  # Target 70% CPU
            mem_scale = predicted_memory / 80.0  # Target 80% memory
            scale_factor = max(cpu_scale, mem_scale) * self.safety_buffer
            
            target_replicas = int(current_replicas * scale_factor)
            calculated_replicas = target_replicas  # Store calculated value before capping
            target_replicas = max(min_replicas, min(target_replicas, max_replicas))
            
            if target_replicas > current_replicas:
                return {
                    'action': 'scale_up',
                    'target_replicas': target_replicas,
                    'calculated_replicas': calculated_replicas,
                    'reason': f'Predicted usage exceeds threshold (CPU: {predicted_cpu:.1f}%, Memory: {predicted_memory:.1f}%)'
                }
            elif calculated_replicas > max_replicas and current_replicas >= max_replicas:
                # Already at max, but need more replicas
                return {
                    'action': 'at_max',
                    'target_replicas': max_replicas,
                    'calculated_replicas': calculated_replicas,
                    'reason': f'High predicted usage (CPU: {predicted_cpu:.1f}%, Memory: {predicted_memory:.1f}%) but already at max replicas ({max_replicas}). Consider increasing max_replicas or optimizing application.'
                }
        
        # Scale down if predicted usage is low
        if predicted_cpu < self.scale_down_threshold and predicted_memory < self.scale_down_threshold:
            scale_factor = 0.7  # Reduce by 30%
            target_replicas = max(min_replicas, int(current_replicas * scale_factor))
            
            if target_replicas < current_replicas:
                return {
                    'action': 'scale_down',
                    'target_replicas': target_replicas,
                    'reason': f'Predicted usage is low (CPU: {predicted_cpu:.1f}%, Memory: {predicted_memory:.1f}%)'
                }
        
        return {
            'action': 'none',
            'target_replicas': current_replicas,
            'reason': f'No scaling needed. Current usage: CPU: {predicted_cpu:.1f}%, Memory: {predicted_memory:.1f}%'
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
            pods = pods_data.get('items', [])
            
            if not pods:
                return {'cpu_usage': 0, 'memory_usage': 0}
            
            # Use kubectl top pods with label selector (more efficient than listing all pod names)
            top_result = subprocess.run(
                ['kubectl', 'top', 'pods', '-n', namespace, '-l', f'app={deployment_name}'],
                capture_output=True,
                text=True,
                timeout=10,
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
                    deployment_name = metadata.get('name')
                    deployment_namespace = metadata.get('namespace', 'default')
                    
                    # Check if this deployment has predictive autoscaling annotation
                    if annotations.get('ai4k8s.io/predictive-autoscaling-enabled') == 'true':
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
        # Include both enabled and recently disabled deployments (for replica counting)
        for key, deployment in self.enabled_deployments.items():
            # Check if already in list
            found = any(
                d['deployment_name'] == deployment['deployment_name'] and
                d['namespace'] == deployment['namespace']
                for d in enabled_deployments
            )
            if not found:
                # Get actual replica count for in-memory cache entries
                deployment_status = self.hpa_manager.get_deployment_replicas(
                    deployment['deployment_name'], deployment['namespace']
                )
                actual_replicas = deployment_status.get('replicas', 0) if deployment_status.get('success') else 0
                
                # Use cached replicas if available and deployment is disabled
                if deployment.get('disabled') and 'replicas' in deployment:
                    actual_replicas = deployment.get('replicas', actual_replicas)
                
                # Add to list with replica count (even if disabled, for stat calculation)
                enabled_deployments.append({
                    'deployment_name': deployment['deployment_name'],
                    'namespace': deployment['namespace'],
                    'min_replicas': deployment.get('min_replicas', 2),
                    'max_replicas': deployment.get('max_replicas', 10),
                    'replicas': actual_replicas,  # Add actual replica count
                    'enabled_at': deployment.get('enabled_at', ''),
                    'disabled': deployment.get('disabled', False)  # Track if disabled
                })
        
        return {
            'success': True,
            'deployments': enabled_deployments,
            'count': len(enabled_deployments)
        }
    
    def disable_predictive_autoscaling(self, deployment_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Disable predictive autoscaling for a deployment"""
        try:
            # Use kubectl annotate to remove annotations (using - suffix)
            import subprocess
            env = os.environ.copy()
            if self.hpa_manager.kubeconfig_path:
                env['KUBECONFIG'] = self.hpa_manager.kubeconfig_path
            
            # Remove annotations one by one using kubectl annotate with - suffix
            annotations_to_remove = [
                'ai4k8s.io/predictive-autoscaling-enabled',
                'ai4k8s.io/predictive-autoscaling-enabled-at',
                'ai4k8s.io/predictive-autoscaling-config'
            ]
            
            for key in annotations_to_remove:
                result = subprocess.run(
                    ['kubectl', 'annotate', 'deployment', deployment_name,
                     f'-n', namespace, f'{key}-'],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env=env
                )
                if result.returncode != 0 and 'not found' not in result.stderr.lower():
                    logger.warning(f"Failed to remove annotation {key}: {result.stderr}")
            
            # Remove label using kubectl label with - suffix
            result = subprocess.run(
                ['kubectl', 'label', 'deployment', deployment_name,
                 f'-n', namespace, 'ai4k8s.io/predictive-autoscaling-'],
                capture_output=True,
                text=True,
                timeout=10,
                env=env
            )
            if result.returncode != 0 and 'not found' not in result.stderr.lower():
                logger.warning(f"Failed to remove label: {result.stderr}")
            
            # Get current replica count before disabling (for stat calculation)
            deployment_status = self.hpa_manager.get_deployment_replicas(deployment_name, namespace)
            current_replicas = deployment_status.get('replicas', 0) if deployment_status.get('success') else 0
            
            # Keep in in-memory cache temporarily with a "disabled" flag so we can still count replicas
            # This ensures Total Replicas stat doesn't drop to 0 immediately after disabling
            key = f"{namespace}/{deployment_name}"
            if key in self.enabled_deployments:
                # Mark as disabled but keep replica count
                self.enabled_deployments[key]['disabled'] = True
                self.enabled_deployments[key]['replicas'] = current_replicas
            else:
                # Add to cache with disabled flag (in case it wasn't tracked before)
                self.enabled_deployments[key] = {
                    'deployment_name': deployment_name,
                    'namespace': namespace,
                    'min_replicas': 2,
                    'max_replicas': 10,
                    'replicas': current_replicas,
                    'disabled': True
                }
            
            return {
                'success': True,
                'message': f'Predictive autoscaling disabled for {deployment_name}',
                'replicas': current_replicas  # Return current replicas for UI
            }
        except Exception as e:
            logger.error(f"Error disabling predictive autoscaling: {e}")
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
            
            # Try to get deployment-specific metrics
            deployment_metrics = self._get_deployment_metrics(deployment_name, namespace)
            logger.info(f"Getting recommendation for {deployment_name}: metrics={deployment_metrics}")
            
            # Get forecasts (use deployment-specific current values if available)
            cpu_current = deployment_metrics.get('cpu_usage', 0)
            memory_current = deployment_metrics.get('memory_usage', 0)
            
            cpu_forecast = self.monitoring_system.forecaster.forecast_cpu_usage(
                hours_ahead=self.prediction_horizon
            )
            memory_forecast = self.monitoring_system.forecaster.forecast_memory_usage(
                hours_ahead=self.prediction_horizon
            )
            
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
            
            logger.info(f"Recommendation forecast - CPU: current={cpu_forecast.current_value}, predicted={cpu_predicted}, max={max_predicted_cpu}")
            logger.info(f"Recommendation forecast - Memory: current={memory_forecast.current_value}, predicted={memory_predicted}, max={max_predicted_memory}")
            
            # Try LLM-based recommendation first
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
                    
                    # Get LLM recommendation (now includes VPA support)
                    llm_result = self.llm_advisor.get_intelligent_recommendation(
                        deployment_name=deployment_name,
                        namespace=namespace,
                        current_metrics=current_metrics,
                        forecast=forecast_data,
                        hpa_status=hpa_status,
                        vpa_status=vpa_status,
                        current_resources=current_resources,
                        current_replicas=current_replicas,
                        min_replicas=2,  # default min
                        max_replicas=10  # default max
                    )
                    
                    if llm_result.get('success'):
                        llm_recommendation = llm_result.get('recommendation', {})
                        scaling_type = llm_recommendation.get('scaling_type', 'hpa')
                        if scaling_type == 'vpa':
                            logger.info(f"✅ LLM recommendation: {llm_recommendation.get('action')} (VPA) -> CPU: {llm_recommendation.get('target_cpu')}, Memory: {llm_recommendation.get('target_memory')}")
                        else:
                            logger.info(f"✅ LLM recommendation: {llm_recommendation.get('action')} (HPA) -> {llm_recommendation.get('target_replicas')} replicas")
                    elif llm_result.get('rate_limited'):
                        # Rate-limited, use cached recommendation if available, otherwise fallback
                        logger.info(f"⏸️  LLM rate-limited, using fallback recommendation")
                        llm_recommendation = None
                    else:
                        logger.warning(f"⚠️  LLM recommendation failed: {llm_result.get('error')}, using fallback")
                except Exception as e:
                    logger.warning(f"⚠️  Error getting LLM recommendation: {e}, using fallback")
            
            # Use LLM recommendation if available, otherwise fallback to rule-based
            if llm_recommendation:
                scaling_type = llm_recommendation.get('scaling_type', 'hpa')  # Default to HPA for backward compatibility
                action = {
                    'action': llm_recommendation.get('action', 'none'),
                    'scaling_type': scaling_type,  # 'hpa', 'vpa', 'both', or 'maintain'
                    'target_replicas': llm_recommendation.get('target_replicas', current_replicas) if scaling_type in ['hpa', 'both'] else None,
                    'target_cpu': llm_recommendation.get('target_cpu') if scaling_type in ['vpa', 'both'] else None,
                    'target_memory': llm_recommendation.get('target_memory') if scaling_type in ['vpa', 'both'] else None,
                    'reason': llm_recommendation.get('reasoning', 'LLM-based recommendation'),
                    'confidence': llm_recommendation.get('confidence', 0.5),
                    'llm_recommendation': llm_recommendation
                }
            else:
                # Fallback to rule-based recommendation
                action = self._determine_scaling_action(
                    max_predicted_cpu,
                    max_predicted_memory,
                    current_replicas,
                    2,  # default min
                    10  # default max
                )
            
            # Get deployment status for resource stats
            deployment_status = self.hpa_manager.get_deployment_replicas(deployment_name, namespace)
            ready_replicas = deployment_status.get('ready_replicas', 0) if deployment_status.get('success') else 0
            available_replicas = deployment_status.get('available_replicas', 0) if deployment_status.get('success') else 0
            
            # Safely get predicted values (handle None)
            cpu_predicted_safe = cpu_forecast.predicted_values if cpu_forecast.predicted_values is not None else []
            memory_predicted_safe = memory_forecast.predicted_values if memory_forecast.predicted_values is not None else []
            
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

