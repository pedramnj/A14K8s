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
from typing import Dict, Any, Optional, List
from datetime import datetime
from autoscaling_engine import HorizontalPodAutoscaler
from predictive_autoscaler import PredictiveAutoscaler
from scheduled_autoscaler import ScheduledAutoscaler
from predictive_monitoring import PredictiveMonitoringSystem
from ai_monitoring_integration import AIMonitoringIntegration

logger = logging.getLogger(__name__)

class AutoscalingIntegration:
    """Main integration class for all autoscaling features"""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        self.kubeconfig_path = kubeconfig_path
        
        # Initialize components
        self.hpa_manager = HorizontalPodAutoscaler(kubeconfig_path)
        
        # Get monitoring system from AI integration
        try:
            ai_integration = AIMonitoringIntegration(kubeconfig_path)
            self.monitoring_system = ai_integration.monitoring_system
        except:
            # Fallback if monitoring system not available
            from predictive_monitoring import PredictiveMonitoringSystem
            self.monitoring_system = PredictiveMonitoringSystem()
        
        # Initialize autoscalers
        self.predictive_autoscaler = PredictiveAutoscaler(
            self.monitoring_system,
            self.hpa_manager
        )
        self.scheduled_autoscaler = ScheduledAutoscaler(self.monitoring_system, self.hpa_manager)
        
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
            
            # Get enabled predictive autoscaling deployments (from Kubernetes)
            predictive_deployments = self.predictive_autoscaler.list_enabled_deployments(namespace)
            
            # Get current metrics for recommendations
            current_analysis = self.monitoring_system.analyze() if hasattr(self.monitoring_system, 'analyze') else {}
            
            return {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'hpas': hpas.get('hpas', []) if hpas.get('success') else [],
                'hpa_count': hpas.get('count', 0) if hpas.get('success') else 0,
                'schedules': schedules.get('schedules', []) if schedules.get('success') else [],
                'schedule_count': schedules.get('count', 0) if schedules.get('success') else 0,
                'predictive_deployments': predictive_deployments.get('deployments', []) if predictive_deployments.get('success') else [],
                'predictive_count': predictive_deployments.get('count', 0) if predictive_deployments.get('success') else 0,
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
                                     min_replicas: int = 2, max_replicas: int = 10) -> Dict[str, Any]:
        """Enable predictive autoscaling for a deployment"""
        try:
            # Get scaling recommendation
            recommendation = self.predictive_autoscaler.get_scaling_recommendation(
                deployment_name, namespace
            )
            
            if not recommendation['success']:
                return recommendation
            
            # Execute predictive scaling
            result = self.predictive_autoscaler.predict_and_scale(
                deployment_name, namespace, min_replicas, max_replicas
            )
            
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
    
    def create_schedule(self, deployment_name: str, namespace: str = "default",
                       schedule_rules: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create scheduled autoscaling"""
        return self.scheduled_autoscaler.create_schedule(
            deployment_name, namespace, schedule_rules
        )
    
    def get_scaling_recommendations(self, deployment_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Get scaling recommendations from all sources"""
        try:
            recommendations = {
                'predictive': None,
                'reactive': None,
                'scheduled': None
            }
            
            # Predictive recommendation
            try:
                pred_rec = self.predictive_autoscaler.get_scaling_recommendation(
                    deployment_name, namespace
                )
                if pred_rec['success']:
                    recommendations['predictive'] = pred_rec
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

