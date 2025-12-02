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
                 prediction_horizon: int = 6,
                 use_llm: bool = True):
        self.monitoring_system = monitoring_system
        self.hpa_manager = hpa_manager
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
            if cpu_forecast.predicted_values and any(v > 0 for v in cpu_forecast.predicted_values):
                max_predicted_cpu = max(cpu_forecast.predicted_values)
            else:
                max_predicted_cpu = cpu_forecast.current_value
            
            if memory_forecast.predicted_values and any(v > 0 for v in memory_forecast.predicted_values):
                max_predicted_memory = max(memory_forecast.predicted_values)
            else:
                max_predicted_memory = memory_forecast.current_value
            
            logger.info(f"Forecast values - CPU: current={cpu_forecast.current_value}, predicted={cpu_forecast.predicted_values}, max={max_predicted_cpu}")
            logger.info(f"Forecast values - Memory: current={memory_forecast.current_value}, predicted={memory_forecast.predicted_values}, max={max_predicted_memory}")
            
            # Determine scaling action
            action = self._determine_scaling_action(
                max_predicted_cpu,
                max_predicted_memory,
                current_replicas,
                min_replicas,
                max_replicas
            )
            
            if action['action'] == 'scale_up' or action['action'] == 'at_max':
                # Calculate required replicas
                required_replicas = action['target_replicas']
                
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
            
            elif action['action'] == 'scale_down':
                required_replicas = action['target_replicas']
                self._scale_deployment(deployment_name, namespace, required_replicas)
                
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
            
            # Parse kubectl top output
            lines = top_result.stdout.strip().split('\n')[1:]  # Skip header
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
        # Query Kubernetes for deployments with the label
        result = self.hpa_manager.list_deployments_with_label(
            'ai4k8s.io/predictive-autoscaling=enabled',
            namespace
        )
        
        enabled_deployments = []
        
        if result.get('success'):
            for deployment in result.get('deployments', []):
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
                
                enabled_deployments.append({
                    'deployment_name': deployment_name,
                    'namespace': deployment_namespace,
                    'min_replicas': config.get('min_replicas', 2),
                    'max_replicas': config.get('max_replicas', 10),
                    'enabled_at': annotations.get('ai4k8s.io/predictive-autoscaling-enabled-at', '')
                })
        
        # Merge with in-memory cache (for backward compatibility)
        for key, deployment in self.enabled_deployments.items():
            # Check if already in list
            found = any(
                d['deployment_name'] == deployment['deployment_name'] and
                d['namespace'] == deployment['namespace']
                for d in enabled_deployments
            )
            if not found:
                enabled_deployments.append(deployment)
        
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
            
            # Remove from in-memory cache
            key = f"{namespace}/{deployment_name}"
            if key in self.enabled_deployments:
                del self.enabled_deployments[key]
            
            return {
                'success': True,
                'message': f'Predictive autoscaling disabled for {deployment_name}'
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
            
            result = subprocess.run(
                ['kubectl', 'scale', 'deployment', deployment_name,
                 f'--replicas={replicas}', '-n', namespace],
                capture_output=True,
                text=True,
                timeout=30,
                env=env
            )
            
            if result.returncode == 0:
                return {'success': True, 'message': f'Scaled to {replicas} replicas'}
            else:
                return {'success': False, 'error': result.stderr}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_scaling_recommendation(self, deployment_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Get scaling recommendation without executing"""
        try:
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
            if cpu_forecast.predicted_values and any(v > 0 for v in cpu_forecast.predicted_values):
                max_predicted_cpu = max(cpu_forecast.predicted_values)
            else:
                max_predicted_cpu = cpu_forecast.current_value
            
            if memory_forecast.predicted_values and any(v > 0 for v in memory_forecast.predicted_values):
                max_predicted_memory = max(memory_forecast.predicted_values)
            else:
                max_predicted_memory = memory_forecast.current_value
            
            logger.info(f"Recommendation forecast - CPU: current={cpu_forecast.current_value}, predicted={cpu_forecast.predicted_values}, max={max_predicted_cpu}")
            logger.info(f"Recommendation forecast - Memory: current={memory_forecast.current_value}, predicted={memory_forecast.predicted_values}, max={max_predicted_memory}")
            
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
                    
                    # Prepare metrics
                    current_metrics = {
                        'cpu_usage': cpu_current,
                        'memory_usage': memory_current,
                        'pod_count': current_replicas,
                        'running_pod_count': current_replicas
                    }
                    
                    # Prepare forecast
                    forecast_data = {
                        'cpu': {
                            'current': cpu_forecast.current_value,
                            'peak': max_predicted_cpu,
                            'trend': cpu_forecast.trend,
                            'predictions': cpu_forecast.predicted_values
                        },
                        'memory': {
                            'current': memory_forecast.current_value,
                            'peak': max_predicted_memory,
                            'trend': memory_forecast.trend,
                            'predictions': memory_forecast.predicted_values
                        }
                    }
                    
                    # Get LLM recommendation
                    llm_result = self.llm_advisor.get_intelligent_recommendation(
                        deployment_name=deployment_name,
                        namespace=namespace,
                        current_metrics=current_metrics,
                        forecast=forecast_data,
                        hpa_status=hpa_status,
                        current_replicas=current_replicas,
                        min_replicas=2,  # default min
                        max_replicas=10  # default max
                    )
                    
                    if llm_result.get('success'):
                        llm_recommendation = llm_result.get('recommendation', {})
                        logger.info(f"✅ LLM recommendation: {llm_recommendation.get('action')} -> {llm_recommendation.get('target_replicas')} replicas")
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
                action = {
                    'action': llm_recommendation.get('action', 'none'),
                    'target_replicas': llm_recommendation.get('target_replicas', current_replicas),
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
            
            return {
                'success': True,
                'recommendation': action,
                'forecast': {
                    'cpu': {
                        'current': cpu_forecast.current_value,
                        'peak': max_predicted_cpu,
                        'trend': cpu_forecast.trend,
                        'predictions': cpu_forecast.predicted_values
                    },
                    'memory': {
                        'current': memory_forecast.current_value,
                        'peak': max_predicted_memory,
                        'trend': memory_forecast.trend,
                        'predictions': memory_forecast.predicted_values
                    }
                },
                'current_replicas': current_replicas,
                'llm_enabled': self.use_llm and self.llm_advisor is not None,
                'llm_used': llm_recommendation is not None
            }
            
        except Exception as e:
            logger.error(f"Error getting scaling recommendation: {e}")
            return {
                'success': False,
                'error': str(e)
            }

