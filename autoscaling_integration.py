#!/usr/bin/env python3
"""
AI4K8s Autoscaling Integration
==============================

Integrates all autoscaling components (HPA, Predictive, Scheduled)
with the web application.

Author: Pedram Nikjooy
Thesis: AI Agent for Kubernetes Management
"""

import logging
import threading
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from autoscaling_engine import HorizontalPodAutoscaler
from vpa_engine import VerticalPodAutoscaler
from predictive_autoscaler import PredictiveAutoscaler
from scheduled_autoscaler import ScheduledAutoscaler
from predictive_monitoring import PredictiveMonitoringSystem
from ai_monitoring_integration import AIMonitoringIntegration
from llm_autoscaling_advisor import LLMAutoscalingAdvisor

logger = logging.getLogger(__name__)

class AutoscalingIntegration:
    """Main integration class for all autoscaling features"""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        self.kubeconfig_path = kubeconfig_path
        
        # Initialize components
        self.hpa_manager = HorizontalPodAutoscaler(kubeconfig_path)
        self.vpa_manager = VerticalPodAutoscaler(kubeconfig_path)  # VPA manager for vertical scaling
        
        # Get monitoring system from AI integration
        self.ai_integration = None
        try:
            ai_integration = AIMonitoringIntegration(kubeconfig_path)
            self.monitoring_system = ai_integration.monitoring_system
            self.ai_integration = ai_integration  # Store reference for getting metrics
        except:
            # Fallback if monitoring system not available
            from predictive_monitoring import PredictiveMonitoringSystem
            self.monitoring_system = PredictiveMonitoringSystem()
            self.ai_integration = None
        
        # Initialize LLM advisor
        self.llm_advisor = LLMAutoscalingAdvisor()
        
        # Initialize autoscalers (with LLM support and VPA manager)
        self.predictive_autoscaler = PredictiveAutoscaler(
            self.monitoring_system,
            self.hpa_manager,
            vpa_manager=self.vpa_manager,  # Pass VPA manager for vertical scaling support
            use_llm=True  # Enable LLM-based recommendations
        )
        self.scheduled_autoscaler = ScheduledAutoscaler(self.monitoring_system, self.hpa_manager)
        
        # Start background thread for periodic predictive scaling
        self.predictive_scaling_thread = None
        self.predictive_scaling_interval = 300  # 5 minutes (matches LLM rate limit)
        self.predictive_scaling_running = False
        self._start_predictive_scaling_loop()
        
        logger.info("Autoscaling integration initialized")
    
    def get_autoscaling_status(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive autoscaling status"""
        try:
            # Get all HPAs from all namespaces (pass None to list all)
            hpas = self.hpa_manager.list_hpas(namespace)
            
            # Check if HPA listing failed due to cluster disconnection
            if not hpas.get('success') and 'error' in hpas:
                error_msg = str(hpas.get('error', ''))
                if any(phrase in error_msg.lower() for phrase in ["no such host", "connection refused", "timeout", "unable to connect", "dial tcp", "name or service not known", "name resolution"]):
                    return {
                        'success': False,
                        'error': error_msg,
                        'cluster_disconnected': True,
                        'timestamp': datetime.now().isoformat(),
                        'hpas': [],
                        'hpa_count': 0,
                        'schedules': [],
                        'schedule_count': 0,
                        'current_metrics': {},
                        'forecasts': {}
                    }
            
            # Get schedules (from Kubernetes)
            schedules = self.scheduled_autoscaler.list_schedules(namespace)
            
            # Get VPAs (from Kubernetes) - wrap in try-except in case VPA API is not available
            vpa_list = []
            vpa_count = 0
            try:
                vpas = self.vpa_manager.list_vpas(namespace)
                vpa_list = vpas.get('vpas', []) if vpas.get('success') else []
                vpa_count = vpas.get('count', 0) if vpas.get('success') else 0
            except Exception as e:
                logger.warning(f"⚠️ Failed to list VPAs (VPA API may not be available): {e}")
                vpa_list = []
                vpa_count = 0
            
            # Get enabled predictive autoscaling deployments (from Kubernetes)
            try:
                predictive_deployments = self.predictive_autoscaler.list_enabled_deployments(namespace)
                if not isinstance(predictive_deployments, dict):
                    predictive_deployments = {'success': False, 'deployments': [], 'count': 0}
            except Exception as e:
                logger.warning(f"⚠️ Failed to list enabled deployments: {e}")
                import traceback
                logger.error(traceback.format_exc())
                predictive_deployments = {'success': False, 'deployments': [], 'count': 0}
            
            # Get current metrics for recommendations
            current_analysis = {}
            try:
                # Try to get from AI monitoring integration first (has real-time metrics)
                if self.ai_integration and hasattr(self.ai_integration, 'get_current_analysis'):
                    current_analysis = self.ai_integration.get_current_analysis()
                    if not current_analysis:
                        # If no cached analysis, try to get metrics directly
                        if hasattr(self.ai_integration, 'metrics_collector'):
                            metrics_data = self.ai_integration.metrics_collector.get_aggregated_metrics()
                            if metrics_data.get('success') and metrics_data.get('metrics'):
                                metrics = metrics_data['metrics']
                                current_analysis = {
                                    'current_metrics': {
                                        'cpu_usage': metrics.get('cpu_usage_percent', 0),
                                        'memory_usage': metrics.get('memory_usage_percent', 0),
                                        'pod_count': metrics.get('pod_count', 0),
                                        'running_pod_count': metrics.get('running_pod_count', 0),
                                        'node_count': metrics.get('node_count', 0)
                                    }
                                }
                # Fallback to monitoring system analyze
                elif hasattr(self.monitoring_system, 'analyze'):
                    current_analysis = self.monitoring_system.analyze()
            except Exception as e:
                logger.warning(f"⚠️ Failed to get current metrics: {e}")
                current_analysis = {}
            
            # Ensure current_metrics exists and has values
            if not current_analysis.get('current_metrics'):
                current_analysis['current_metrics'] = {}
            
            # Calculate total replicas: sum of HPA current replicas + predictive deployments without HPA
            hpa_list = hpas.get('hpas', []) if hpas.get('success') else []
            # Safely handle predictive_deployments - it might be None or not have expected structure
            if predictive_deployments and isinstance(predictive_deployments, dict):
                predictive_list = predictive_deployments.get('deployments', []) if predictive_deployments.get('success') else []
                predictive_count = predictive_deployments.get('count', 0) if predictive_deployments.get('success') else 0
            else:
                predictive_list = []
                predictive_count = 0
            
            # Sum replicas from HPAs
            total_replicas = sum(hpa.get('current_replicas', 0) for hpa in hpa_list)
            
            # Track which deployments are managed by HPAs to avoid double-counting
            hpa_targets = {f"{hpa.get('namespace', 'default')}/{hpa.get('target', '')}" for hpa in hpa_list}
            
            # Add replicas from predictive deployments that don't have an HPA
            if isinstance(predictive_list, list):
                for pred_deployment in predictive_list:
                    if not isinstance(pred_deployment, dict):
                        continue
                    pred_key = f"{pred_deployment.get('namespace', 'default')}/{pred_deployment.get('deployment_name', '')}"
                    if pred_key not in hpa_targets:
                        # This deployment has predictive autoscaling but no HPA, get its replicas
                        deployment_replicas = pred_deployment.get('replicas', 0)
                        total_replicas += deployment_replicas
                        logger.info(f"Added {deployment_replicas} replicas from predictive deployment {pred_key} (total now: {total_replicas})")
            
            # Fallback: Check in-memory cache (includes both enabled and disabled deployments)
            # This handles cases where labels weren't added or predictive autoscaling was just disabled
            if len(self.predictive_autoscaler.enabled_deployments) > 0:
                logger.info(f"Fallback: Checking in-memory cache for {len(self.predictive_autoscaler.enabled_deployments)} deployments")
                for key, deployment in self.predictive_autoscaler.enabled_deployments.items():
                    # Skip if already counted in predictive_list
                    pred_key = f"{deployment.get('namespace', 'default')}/{deployment.get('deployment_name', '')}"
                    already_counted = any(
                        d.get('deployment_name') == deployment.get('deployment_name') and
                        d.get('namespace') == deployment.get('namespace')
                        for d in (predictive_list if isinstance(predictive_list, list) else [])
                    )
                    
                    if not already_counted:
                        # Get actual replicas for this deployment
                        deployment_status = self.hpa_manager.get_deployment_replicas(
                            deployment['deployment_name'], deployment['namespace']
                        )
                        if deployment_status.get('success'):
                            replicas = deployment_status.get('replicas', 0)
                            # Use cached replicas if deployment is disabled and cache has it
                            if deployment.get('disabled') and 'replicas' in deployment:
                                replicas = deployment.get('replicas', replicas)
                            total_replicas += replicas
                            logger.info(f"Fallback: Added {replicas} replicas from in-memory cache {key} (disabled: {deployment.get('disabled', False)}, total now: {total_replicas})")
            
            # Additional fallback: If still 0, check for deployments that were recently managed
            # This handles the case where predictive autoscaling was just disabled but deployment still exists
            if total_replicas == 0 and len(hpa_list) == 0:
                # Query all deployments and check if any have the annotation (even if label was removed)
                try:
                    cmd = "get deployments --all-namespaces -o json"
                    all_deployments_result = self.hpa_manager._execute_kubectl(cmd)
                    if all_deployments_result.get('success'):
                        all_deployments = all_deployments_result.get('result', {}).get('items', [])
                        for deployment in all_deployments:
                            metadata = deployment.get('metadata', {})
                            annotations = metadata.get('annotations', {})
                            # Check if it has predictive annotation (even if label was removed during disable)
                            if annotations.get('ai4k8s.io/predictive-autoscaling-enabled') == 'true':
                                deployment_name = metadata.get('name')
                                deployment_namespace = metadata.get('namespace', 'default')
                                deployment_status = self.hpa_manager.get_deployment_replicas(deployment_name, deployment_namespace)
                                if deployment_status.get('success'):
                                    replicas = deployment_status.get('replicas', 0)
                                    total_replicas += replicas
                                    logger.info(f"Fallback annotation check: Added {replicas} replicas from {deployment_namespace}/{deployment_name}")
                except Exception as e:
                    logger.warning(f"Failed to check all deployments for replicas: {e}")
            
            logger.info(f"Final total_replicas calculation: {total_replicas} (HPAs: {len(hpa_list)}, Predictive: {len(predictive_list)}, In-memory: {len(self.predictive_autoscaler.enabled_deployments)})")
            
            return {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'hpas': hpa_list,
                'hpa_count': len(hpa_list),
                'vpas': vpa_list,
                'vpa_count': vpa_count,
                'schedules': schedules.get('schedules', []) if schedules.get('success') else [],
                'schedule_count': schedules.get('count', 0) if schedules.get('success') else 0,
                'predictive_deployments': predictive_list,
                'predictive_count': len(predictive_list),
                'total_replicas': total_replicas,  # Add total_replicas to response
                'current_metrics': current_analysis.get('current_metrics', {}),
                'forecasts': current_analysis.get('forecasts', {})
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error getting autoscaling status: {e}")
            # Check if it's a connection error
            if any(phrase in error_msg.lower() for phrase in ["no such host", "connection refused", "timeout", "unable to connect", "dial tcp", "name or service not known", "name resolution"]):
                return {
                    'success': False,
                    'error': error_msg,
                    'cluster_disconnected': True,
                    'timestamp': datetime.now().isoformat(),
                    'hpas': [],
                    'hpa_count': 0,
                    'schedules': [],
                    'schedule_count': 0,
                    'predictive_deployments': [],
                    'predictive_count': 0,
                    'current_metrics': {},
                    'forecasts': {}
                }
            return {
                'success': False,
                'error': error_msg
            }
    
    def enable_predictive_autoscaling(self, deployment_name: str, namespace: str = "default",
                                     min_replicas: int = 2, max_replicas: int = 10,
                                     state_management: Optional[str] = None) -> Dict[str, Any]:
        """Enable predictive autoscaling for a deployment"""
        try:
            # Trim deployment name and namespace to remove any whitespace
            deployment_name = deployment_name.strip()
            namespace = namespace.strip()
            
            # Get scaling recommendation
            recommendation = self.predictive_autoscaler.get_scaling_recommendation(
                deployment_name, namespace
            )
            
            if not recommendation['success']:
                return recommendation
            
            # Execute predictive scaling immediately (this will also mark it as enabled)
            result = self.predictive_autoscaler.predict_and_scale(
                deployment_name, namespace, min_replicas, max_replicas, state_management
            )
            
            # The periodic loop will continue to execute scaling every 5 minutes
            logger.info(f"✅ Predictive autoscaling enabled for {deployment_name}. Periodic scaling will run every {self.predictive_scaling_interval}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error enabling predictive autoscaling: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_hpa(self, deployment_name: str, namespace: str = "default",
                  min_replicas: int = 2, max_replicas: int = 10,
                  cpu_target: int = 70, memory_target: int = 80) -> Dict[str, Any]:
        """Create HPA for a deployment"""
        return self.hpa_manager.create_hpa(
            deployment_name, namespace, min_replicas, max_replicas,
            cpu_target, memory_target
        )
    
    def delete_hpa(self, hpa_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Delete HPA"""
        return self.hpa_manager.delete_hpa(hpa_name, namespace)
    
    def disable_predictive_autoscaling(self, deployment_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Disable predictive autoscaling for a deployment"""
        return self.predictive_autoscaler.disable_predictive_autoscaling(deployment_name, namespace)
    
    def delete_schedule(self, deployment_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Delete scheduled autoscaling"""
        return self.scheduled_autoscaler.delete_schedule(deployment_name, namespace)
    
    def create_schedule(self, deployment_name: str, namespace: str = "default",
                       schedule_rules: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create scheduled autoscaling"""
        return self.scheduled_autoscaler.create_schedule(
            deployment_name, namespace, schedule_rules
        )

    def apply_predictive_target(self, deployment_name: str, namespace: str,
                                target_replicas: Optional[int] = None,
                                target_cpu: Optional[str] = None,
                                target_memory: Optional[str] = None,
                                scaling_type: str = 'hpa') -> Dict[str, Any]:
        """
        Force-apply a specific predictive target (HPA or VPA).
        This bypasses recommendation logic and directly applies the scaling.
        
        Args:
            deployment_name: Name of the deployment
            namespace: Kubernetes namespace
            target_replicas: Target replica count (for HPA)
            target_cpu: Target CPU request/limit (for VPA, e.g., "200m")
            target_memory: Target Memory request/limit (for VPA, e.g., "256Mi")
            scaling_type: 'hpa', 'vpa', or 'both'
        """
        try:
            if scaling_type in ['hpa', 'both']:
                if target_replicas is None or target_replicas < 0:
                    return {'success': False, 'error': 'target_replicas must be >= 0 for HPA scaling'}

                # Check if HPA exists for this deployment
                hpa_name = f"{deployment_name}-hpa"
                hpa_result = self.hpa_manager.get_hpa(hpa_name, namespace)
                hpa_exists = hpa_result.get('success')
                
                if hpa_exists:
                    # HPA exists - Predictive Autoscaling conflicts with HPA
                    # We'll scale directly, but warn the user that HPA may override
                    # IMPORTANT: We do NOT modify the HPA - Predictive Autoscaling scales independently
                    logger.warning(f"⚠️ HPA {hpa_name} exists for {deployment_name}. "
                                 f"Predictive Autoscaling will scale directly to {target_replicas} replicas, "
                                 f"but HPA may override this. Consider deleting HPA if using Predictive Autoscaling exclusively.")
                    
                    # Check if target is outside HPA range - warn but don't patch
                    hpa_status = hpa_result.get('status', {})
                    current_min = hpa_status.get('min_replicas', 1)
                    current_max = hpa_status.get('max_replicas', 10)
                    
                    if target_replicas < current_min or target_replicas > current_max:
                        logger.warning(f"⚠️ Target {target_replicas} is outside HPA range ({current_min}-{current_max}). "
                                     f"HPA may prevent this scaling. Consider deleting HPA or adjusting its range manually.")

                # Scale the deployment directly (Predictive Autoscaling controls this, NOT HPA)
                # We do NOT create or modify HPAs - Predictive Autoscaling is independent
                result = self.predictive_autoscaler._scale_deployment(
                    deployment_name,
                    namespace,
                    target_replicas
                )

                if not result.get('success'):
                    return {
                        'success': False,
                        'error': result.get('error', 'Failed to scale deployment')
                    }
                
                # Build response message
                message = f'Scaled to {target_replicas} replicas via Predictive Autoscaling'
                if hpa_exists:
                    message += f'. (HPA exists and may override - consider deleting HPA if using Predictive Autoscaling exclusively)'
                
                logger.info(f"✅ Force-applied Predictive Autoscaling target: {deployment_name} -> {target_replicas} replicas (HPA not modified)")
                
                # Return early for HPA-only scaling to avoid duplicate return
                hpa_response = {
                    'success': True,
                    'action': 'force_apply',
                    'scaling_type': scaling_type,
                    'target_replicas': target_replicas,
                    'message': message,
                    'hpa_exists': hpa_exists,
                    'hpa_modified': False  # We don't modify HPAs for Predictive Autoscaling
                }
                
                # If only HPA scaling, return now
                if scaling_type == 'hpa':
                    return hpa_response
            
            if scaling_type in ['vpa', 'both']:
                if not target_cpu or not target_memory:
                    return {'success': False, 'error': 'target_cpu and target_memory required for VPA scaling'}
                
                # Predictive Autoscaling should patch deployment resources directly (like HPA)
                # We do NOT create VPA resources - Predictive Autoscaling controls scaling independently
                # Check if VPA exists - warn but don't modify it
                vpa_name = f"{deployment_name}-vpa"
                vpa_result = self.vpa_manager.get_vpa(vpa_name, namespace)
                vpa_exists = vpa_result.get('success')
                
                if vpa_exists:
                    logger.warning(f"⚠️ VPA {vpa_name} exists for {deployment_name}. "
                                 f"Predictive Autoscaling will patch deployment resources directly, "
                                 f"but VPA may override this. Consider deleting VPA if using Predictive Autoscaling exclusively.")
                
                # Patch deployment resources directly (Predictive Autoscaling, not VPA controller)
                # Use target values for both requests and limits (or calculate limits as 2x requests)
                try:
                    # Parse CPU to calculate limit (2x request for headroom)
                    cpu_request_m = int(target_cpu[:-1]) if target_cpu.endswith('m') else int(float(target_cpu) * 1000)
                    cpu_limit_m = min(cpu_request_m * 2, 4000)  # Cap at 4000m
                    cpu_limit = f"{cpu_limit_m}m"
                    
                    # Parse Memory to calculate limit (1.5x request for headroom)
                    if target_memory.endswith('Mi'):
                        memory_request_mi = int(target_memory[:-2])
                        memory_limit_mi = min(int(memory_request_mi * 1.5), 4096)  # Cap at 4Gi
                        memory_limit = f"{memory_limit_mi}Mi"
                    elif target_memory.endswith('Gi'):
                        memory_request_gi = int(target_memory[:-2])
                        memory_limit_gi = min(int(memory_request_gi * 1.5), 4)
                        memory_limit = f"{memory_limit_gi}Gi"
                    else:
                        memory_limit = target_memory  # Use as-is if format unknown
                except:
                    # Fallback: use target values for both requests and limits
                    cpu_limit = target_cpu
                    memory_limit = target_memory
                
                patch_result = self.vpa_manager.patch_deployment_resources(
                    deployment_name, namespace,
                    cpu_request=target_cpu,
                    memory_request=target_memory,
                    cpu_limit=cpu_limit,
                    memory_limit=memory_limit
                )
                
                if not patch_result.get('success'):
                    return {
                        'success': False,
                        'error': f"Failed to patch deployment resources: {patch_result.get('error')}"
                    }
                
                logger.info(f"✅ Force-applied Predictive Autoscaling VPA target: {deployment_name} -> CPU: {target_cpu}, Memory: {target_memory} (direct patch)")
                
                # Build response message
                vpa_message = f'Updated resources to CPU: {target_cpu}, Memory: {target_memory} via Predictive Autoscaling'
                if vpa_exists:
                    vpa_message += f'. (VPA exists and may override - consider deleting VPA if using Predictive Autoscaling exclusively)'
                
                # Return early for VPA-only scaling to avoid duplicate return
                vpa_response = {
                    'success': True,
                    'action': 'force_apply',
                    'scaling_type': scaling_type,
                    'target_cpu': target_cpu,
                    'target_memory': target_memory,
                    'message': vpa_message,
                    'vpa_exists': vpa_exists,
                    'vpa_modified': False  # We don't modify VPAs for Predictive Autoscaling
                }
                
                # If only VPA scaling, return now
                if scaling_type == 'vpa':
                    return vpa_response
            
            # Return combined response for 'both' or VPA-only (HPA-only already returned above)
            if scaling_type == 'both':
                return {
                    'success': True,
                    'action': 'force_apply',
                    'scaling_type': scaling_type,
                    'target_replicas': target_replicas,
                    'target_cpu': target_cpu,
                    'target_memory': target_memory,
                    'message': message if 'message' in locals() else f'Applied {scaling_type.upper()} scaling for {deployment_name}',
                    'hpa_exists': hpa_exists if 'hpa_exists' in locals() else False,
                    'hpa_modified': False  # We don't modify HPAs for Predictive Autoscaling
                }
            elif scaling_type == 'vpa':
                return {
                    'success': True,
                    'action': 'force_apply',
                    'scaling_type': scaling_type,
                    'target_cpu': target_cpu,
                    'target_memory': target_memory,
                    'message': f'Applied VPA scaling for {deployment_name}'
                }
            # HPA-only already returned above
        except Exception as e:
            logger.error(f"Error in apply_predictive_target: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}
    
    def create_vpa(self, deployment_name: str, namespace: str = "default",
                   min_cpu: str = "100m", max_cpu: str = "1000m",
                   min_memory: str = "128Mi", max_memory: str = "512Mi",
                   update_mode: str = "Auto") -> Dict[str, Any]:
        """Create VPA for a deployment"""
        return self.vpa_manager.create_vpa(
            deployment_name, namespace, min_cpu, max_cpu,
            min_memory, max_memory, update_mode
        )
    
    def delete_vpa(self, vpa_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Delete VPA"""
        return self.vpa_manager.delete_vpa(vpa_name, namespace)
    
    def get_scaling_recommendations(self, deployment_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Get scaling recommendations from all sources including LLM"""
        try:
            recommendations = {
                'predictive': None,
                'reactive': None,
                'scheduled': None,
                'llm': None
            }
            
            # Predictive recommendation (may include LLM if enabled)
            try:
                print(f"🔍🔍🔍 INTEGRATION: Calling get_scaling_recommendation for {deployment_name} in {namespace}")
                logger.warning(f"🔍🔍🔍 INTEGRATION: Calling get_scaling_recommendation for {deployment_name} in {namespace}")
                pred_rec = self.predictive_autoscaler.get_scaling_recommendation(
                    deployment_name, namespace
                )
                if pred_rec['success']:
                    rec = pred_rec.get('recommendation', {})
                    target_replicas = rec.get('target_replicas')
                    print(f"🔍🔍🔍 INTEGRATION RESULT: target_replicas={target_replicas}, action={rec.get('action')}, scaling_type={rec.get('scaling_type')}")
                    logger.warning(f"🔍🔍🔍 INTEGRATION RESULT: target_replicas={target_replicas}, action={rec.get('action')}, scaling_type={rec.get('scaling_type')}")
                    
                    # CRITICAL: Final validation at integration layer - ensure target_replicas respects bounds
                    # Get min/max from the recommendation if available, or from deployment
                    if target_replicas is not None:
                        # Try to get min/max from deployment annotation
                        try:
                            deployment_status = self.hpa_manager.get_deployment_replicas(deployment_name, namespace)
                            if deployment_status.get('success'):
                                annotations = deployment_status.get('annotations', {})
                                config_str = annotations.get('ai4k8s.io/predictive-autoscaling-config', '{}')
                                import json
                                config = json.loads(config_str) if config_str else {}
                                min_replicas = config.get('min_replicas', 2)
                                max_replicas = config.get('max_replicas', 10)
                                
                                if target_replicas > max_replicas:
                                    print(f"🚨🚨🚨 INTEGRATION LAYER: target_replicas={target_replicas} > max={max_replicas}, FORCING to {max_replicas}")
                                    logger.error(f"🚨🚨🚨 INTEGRATION LAYER: target_replicas={target_replicas} > max={max_replicas}, FORCING to {max_replicas}")
                                    rec['target_replicas'] = max_replicas
                                    pred_rec['recommendation'] = rec
                                elif target_replicas < min_replicas:
                                    print(f"🚨🚨🚨 INTEGRATION LAYER: target_replicas={target_replicas} < min={min_replicas}, FORCING to {min_replicas}")
                                    logger.error(f"🚨🚨🚨 INTEGRATION LAYER: target_replicas={target_replicas} < min={min_replicas}, FORCING to {min_replicas}")
                                    rec['target_replicas'] = min_replicas
                                    pred_rec['recommendation'] = rec
                        except Exception as e:
                            logger.warning(f"⚠️ Could not validate target_replicas at integration layer: {e}")
                    
                    recommendations['predictive'] = pred_rec
                    
                    # Extract LLM recommendation if present
                    if pred_rec.get('llm_used') and pred_rec.get('recommendation', {}).get('llm_recommendation'):
                        recommendations['llm'] = {
                            'success': True,
                            'recommendation': pred_rec['recommendation']['llm_recommendation'],
                            'source': 'predictive_with_llm'
                        }
            except Exception as e:
                logger.warning(f"Failed to get predictive recommendation: {e}")
            
            # Reactive recommendation (from HPA status)
            try:
                hpa_name = f"{deployment_name}-hpa"
                hpa_status = self.hpa_manager.get_hpa(hpa_name, namespace)
                if hpa_status['success']:
                    recommendations['reactive'] = {
                        'success': True,
                        'hpa_status': hpa_status['status']
                    }
            except Exception as e:
                logger.warning(f"Failed to get reactive recommendation: {e}")
            
            # Scheduled recommendation
            try:
                schedule = self.scheduled_autoscaler.get_schedule(deployment_name, namespace)
                if schedule['success']:
                    recommendations['scheduled'] = schedule
            except Exception as e:
                logger.warning(f"Failed to get scheduled recommendation: {e}")
            
            # Standalone LLM recommendation (if not already included in predictive)
            if not recommendations.get('llm') and self.llm_advisor and self.llm_advisor.client:
                try:
                    # Get current metrics
                    deployment_status = self.hpa_manager.get_deployment_replicas(deployment_name, namespace)
                    if deployment_status.get('success'):
                        current_replicas = deployment_status['replicas']
                        
                        # Get metrics from monitoring system
                        current_analysis = self.monitoring_system.analyze() if hasattr(self.monitoring_system, 'analyze') else {}
                        current_metrics = current_analysis.get('current_metrics', {})
                        forecasts = current_analysis.get('forecasts', {})
                        
                        # Get HPA status
                        hpa_name = f"{deployment_name}-hpa"
                        hpa_result = self.hpa_manager.get_hpa(hpa_name, namespace)
                        hpa_status = hpa_result.get('status') if hpa_result.get('success') else None
                        
                        # Get LLM recommendation
                        llm_result = self.llm_advisor.get_intelligent_recommendation(
                            deployment_name=deployment_name,
                            namespace=namespace,
                            current_metrics=current_metrics,
                            forecast=forecasts,
                            hpa_status=hpa_status,
                            current_replicas=current_replicas,
                            min_replicas=2,
                            max_replicas=10,
                            hpa_manager=self.hpa_manager  # Pass hpa_manager so state detection can read annotations
                        )
                        
                        if llm_result.get('success'):
                            recommendations['llm'] = {
                                'success': True,
                                'recommendation': llm_result.get('recommendation', {}),
                                'source': 'standalone_llm',
                                'model': llm_result.get('llm_model')
                            }
                except Exception as e:
                    logger.warning(f"Failed to get standalone LLM recommendation: {e}")
            
            return {
                'success': True,
                'recommendations': recommendations,
                'deployment': deployment_name,
                'namespace': namespace
            }
            
        except Exception as e:
            logger.error(f"Error getting scaling recommendations: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def analyze_patterns(self, deployment_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Analyze historical patterns for schedule suggestions"""
        return self.scheduled_autoscaler.analyze_historical_patterns(
            deployment_name, namespace
        )
    
    def _start_predictive_scaling_loop(self):
        """Start background thread to periodically execute predictive scaling for enabled deployments"""
        if self.predictive_scaling_running:
            return
        
        self.predictive_scaling_running = True
        
        def scaling_loop():
            while self.predictive_scaling_running:
                try:
                    # Get all enabled predictive deployments
                    enabled_deployments = self.predictive_autoscaler.list_enabled_deployments()
                    
                    if enabled_deployments.get('success') and enabled_deployments.get('count', 0) > 0:
                        logger.info(f"🔄 Periodic predictive scaling: Checking {enabled_deployments.get('count')} enabled deployments")
                        
                        for deployment in enabled_deployments.get('deployments', []):
                            # Skip disabled deployments
                            if deployment.get('disabled'):
                                continue
                            
                            deployment_name = deployment.get('deployment_name')
                            namespace = deployment.get('namespace', 'default')
                            min_replicas = deployment.get('min_replicas', 2)
                            max_replicas = deployment.get('max_replicas', 10)
                            
                            try:
                                # Execute predictive scaling (this will use LLM recommendations)
                                result = self.predictive_autoscaler.predict_and_scale(
                                    deployment_name, namespace, min_replicas, max_replicas
                                )
                                
                                if result.get('success'):
                                    action = result.get('action', 'none')
                                    target_replicas = result.get('target_replicas', 0)
                                    if action != 'none':
                                        logger.info(f"✅ Predictive scaling executed: {deployment_name} -> {action} to {target_replicas} replicas")
                                else:
                                    logger.warning(f"⚠️ Predictive scaling failed for {deployment_name}: {result.get('error')}")
                            except Exception as e:
                                logger.error(f"❌ Error in periodic predictive scaling for {deployment_name}: {e}")
                    
                    # Sleep for the interval
                    time.sleep(self.predictive_scaling_interval)
                except Exception as e:
                    logger.error(f"❌ Error in predictive scaling loop: {e}")
                    time.sleep(self.predictive_scaling_interval)
        
        self.predictive_scaling_thread = threading.Thread(target=scaling_loop, daemon=True)
        self.predictive_scaling_thread.start()
        logger.info(f"✅ Started predictive autoscaling background thread (interval: {self.predictive_scaling_interval}s)")
    
    def stop_predictive_scaling_loop(self):
        """Stop the background predictive scaling thread"""
        self.predictive_scaling_running = False
        if self.predictive_scaling_thread:
            self.predictive_scaling_thread.join(timeout=5)
        logger.info("🛑 Stopped predictive autoscaling background thread")

