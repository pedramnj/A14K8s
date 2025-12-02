#!/usr/bin/env python3
"""
AI4K8s Scheduled Autoscaler
===========================

Time-based autoscaling that scales resources based on schedules.
Uses historical patterns and forecasting to suggest optimal schedules.

Author: Pedram Nikjooy
Thesis: AI Agent for Kubernetes Management
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, time
import subprocess
import json
from predictive_monitoring import PredictiveMonitoringSystem, ResourceMetrics

logger = logging.getLogger(__name__)

class ScheduledAutoscaler:
    """Scheduled autoscaling based on time patterns"""
    
    def __init__(self, monitoring_system: PredictiveMonitoringSystem, hpa_manager=None):
        self.monitoring_system = monitoring_system
        self.hpa_manager = hpa_manager  # For Kubernetes operations
        self.schedules = {}  # {deployment_name: [schedule_rules]} - kept for backward compatibility
        
    def create_schedule(self, deployment_name: str, namespace: str = "default",
                       schedule_rules: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create scheduled autoscaling rules"""
        try:
            # Validate schedule rules
            if not schedule_rules:
                return {
                    'success': False,
                    'error': 'Schedule rules are required'
                }
            
            # Store schedule in Kubernetes annotations
            created_at = datetime.now().isoformat()
            schedule_json = json.dumps({
                'rules': schedule_rules,
                'created_at': created_at
            })
            
            annotations = {
                'ai4k8s.io/scheduled-autoscaling-enabled': 'true',
                'ai4k8s.io/scheduled-autoscaling-enabled-at': created_at,
                'ai4k8s.io/scheduled-autoscaling-config': schedule_json
            }
            
            labels = {
                'ai4k8s.io/scheduled-autoscaling': 'enabled'
            }
            
            # Store in Kubernetes if hpa_manager is available
            if self.hpa_manager:
                annot_result = self.hpa_manager.patch_deployment_annotations(
                    deployment_name, namespace, annotations
                )
                if not annot_result.get('success'):
                    logger.warning(f"Failed to add annotations: {annot_result.get('error')}")
                
                label_result = self.hpa_manager.patch_deployment_labels(
                    deployment_name, namespace, labels
                )
                if not label_result.get('success'):
                    logger.warning(f"Failed to add labels: {label_result.get('error')}")
            
            # Also store in memory for backward compatibility
            key = f"{namespace}/{deployment_name}"
            self.schedules[key] = {
                'deployment_name': deployment_name,
                'namespace': namespace,
                'rules': schedule_rules,
                'created_at': created_at
            }
            
            return {
                'success': True,
                'message': f'Schedule created for {deployment_name}',
                'schedule': self.schedules[key],
                'rules_count': len(schedule_rules)
            }
            
        except Exception as e:
            logger.error(f"Error creating schedule: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def analyze_historical_patterns(self, deployment_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Analyze historical patterns to suggest schedules"""
        try:
            # Get historical metrics
            history = self.monitoring_system.metrics_history
            
            if len(history) < 24:
                return {
                    'success': False,
                    'error': 'Insufficient historical data (need at least 24 hours)'
                }
            
            # Analyze daily patterns
            hourly_usage = {}
            for metrics in history[-168:]:  # Last 7 days
                hour = metrics.timestamp.hour
                if hour not in hourly_usage:
                    hourly_usage[hour] = []
                hourly_usage[hour].append((metrics.cpu_usage + metrics.memory_usage) / 2)
            
            # Calculate average usage per hour
            avg_hourly_usage = {
                hour: sum(usage) / len(usage)
                for hour, usage in hourly_usage.items()
            }
            
            # Identify peak and off-peak hours
            sorted_hours = sorted(avg_hourly_usage.items(), key=lambda x: x[1], reverse=True)
            peak_hours = [h for h, u in sorted_hours[:8] if u > 50]  # Top 8 hours with >50% usage
            off_peak_hours = [h for h, u in sorted_hours[-8:] if u < 30]  # Bottom 8 hours with <30% usage
            
            # Generate schedule recommendations
            recommendations = []
            
            # Peak hours schedule
            if peak_hours:
                peak_start = min(peak_hours)
                peak_end = max(peak_hours)
                recommendations.append({
                    'time': f'0 {peak_start} * * 1-5',  # Weekdays at peak start
                    'replicas': 5,
                    'reason': f'Peak hours detected: {peak_start}:00-{peak_end}:00',
                    'type': 'peak'
                })
            
            # Off-peak hours schedule
            if off_peak_hours:
                off_peak_start = min(off_peak_hours)
                off_peak_end = max(off_peak_hours)
                recommendations.append({
                    'time': f'0 {off_peak_start} * * *',  # Daily at off-peak start
                    'replicas': 2,
                    'reason': f'Off-peak hours detected: {off_peak_start}:00-{off_peak_end}:00',
                    'type': 'off_peak'
                })
            
            # Business hours schedule (9 AM - 5 PM weekdays)
            recommendations.append({
                'time': '0 9 * * 1-5',
                'replicas': 4,
                'reason': 'Business hours (9 AM - 5 PM weekdays)',
                'type': 'business_hours'
            })
            
            recommendations.append({
                'time': '0 17 * * 1-5',
                'replicas': 2,
                'reason': 'End of business hours',
                'type': 'business_hours'
            })
            
            return {
                'success': True,
                'patterns': {
                    'peak_hours': peak_hours,
                    'off_peak_hours': off_peak_hours,
                    'avg_hourly_usage': avg_hourly_usage
                },
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def apply_schedule(self, deployment_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Apply current schedule rules"""
        try:
            key = f"{namespace}/{deployment_name}"
            if key not in self.schedules:
                return {
                    'success': False,
                    'error': f'No schedule found for {deployment_name}'
                }
            
            schedule = self.schedules[key]
            current_time = datetime.now()
            current_hour = current_time.hour
            current_weekday = current_time.weekday()  # 0=Monday, 6=Sunday
            
            # Find matching rule
            matching_rule = None
            for rule in schedule['rules']:
                # Parse cron expression (simplified)
                cron_parts = rule['time'].split()
                if len(cron_parts) >= 2:
                    rule_hour = int(cron_parts[1])
                    rule_days = cron_parts[4] if len(cron_parts) > 4 else '*'
                    
                    # Check if hour matches
                    if rule_hour == current_hour:
                        # Check if day matches
                        if rule_days == '*' or self._day_matches(rule_days, current_weekday):
                            matching_rule = rule
                            break
            
            if matching_rule:
                # Scale deployment
                replicas = matching_rule['replicas']
                result = subprocess.run(
                    ['kubectl', 'scale', 'deployment', deployment_name,
                     f'--replicas={replicas}', '-n', namespace],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    return {
                        'success': True,
                        'action': 'scaled',
                        'replicas': replicas,
                        'rule': matching_rule,
                        'message': f'Scaled to {replicas} replicas based on schedule'
                    }
                else:
                    return {
                        'success': False,
                        'error': result.stderr
                    }
            else:
                return {
                    'success': True,
                    'action': 'no_match',
                    'message': 'No schedule rule matches current time'
                }
                
        except Exception as e:
            logger.error(f"Error applying schedule: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _day_matches(self, cron_days: str, current_weekday: int) -> bool:
        """Check if current weekday matches cron day expression"""
        if cron_days == '*':
            return True
        
        # Handle ranges like "1-5" (Monday-Friday)
        if '-' in cron_days:
            start, end = map(int, cron_days.split('-'))
            return start <= current_weekday <= end
        
        # Handle lists like "1,3,5"
        if ',' in cron_days:
            days = [int(d) for d in cron_days.split(',')]
            return current_weekday in days
        
        # Single day
        return int(cron_days) == current_weekday
    
    def get_schedule(self, deployment_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Get current schedule for deployment"""
        key = f"{namespace}/{deployment_name}"
        if key in self.schedules:
            return {
                'success': True,
                'schedule': self.schedules[key]
            }
        else:
            return {
                'success': False,
                'error': f'No schedule found for {deployment_name}'
            }
    
    def delete_schedule(self, deployment_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Delete schedule for deployment"""
        key = f"{namespace}/{deployment_name}"
        if key in self.schedules:
            del self.schedules[key]
            return {
                'success': True,
                'message': f'Schedule deleted for {deployment_name}'
            }
        else:
            return {
                'success': False,
                'error': f'No schedule found for {deployment_name}'
            }
    
    def list_schedules(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """List all schedules from Kubernetes annotations"""
        schedules_list = []
        
        # Query Kubernetes for deployments with scheduled autoscaling label
        if self.hpa_manager:
            result = self.hpa_manager.list_deployments_with_label(
                'ai4k8s.io/scheduled-autoscaling=enabled',
                namespace
            )
            
            if result.get('success'):
                for deployment in result.get('deployments', []):
                    metadata = deployment.get('metadata', {})
                    annotations = metadata.get('annotations', {})
                    
                    deployment_name = metadata.get('name')
                    deployment_namespace = metadata.get('namespace', 'default')
                    
                    # Parse schedule config from annotations
                    config_json = annotations.get('ai4k8s.io/scheduled-autoscaling-config', '{}')
                    try:
                        config = json.loads(config_json)
                        schedule = {
                            'deployment_name': deployment_name,
                            'namespace': deployment_namespace,
                            'rules': config.get('rules', []),
                            'created_at': config.get('created_at', ''),
                            'rules_count': len(config.get('rules', []))
                        }
                        schedules_list.append(schedule)
                    except Exception as e:
                        logger.warning(f"Failed to parse schedule config for {deployment_name}: {e}")
        
        # Merge with in-memory cache (for backward compatibility)
        for schedule in self.schedules.values():
            schedule_with_count = schedule.copy()
            schedule_with_count['rules_count'] = len(schedule.get('rules', []))
            
            # Check if already in list
            found = any(
                s['deployment_name'] == schedule['deployment_name'] and
                s['namespace'] == schedule['namespace']
                for s in schedules_list
            )
            if not found:
                schedules_list.append(schedule_with_count)
        
        return {
            'success': True,
            'schedules': schedules_list,
            'count': len(schedules_list)
        }

