#!/usr/bin/env python3
"""
AI4K8s Autoscaling Engine
==========================

Core autoscaling engine for managing HPA (Horizontal Pod Autoscaler) resources.
Provides baseline reactive autoscaling capabilities.

Author: Pedram Nikjooy
Thesis: AI Agent for Kubernetes Management
"""

import subprocess
import json
import yaml
import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class HorizontalPodAutoscaler:
    """Manage Horizontal Pod Autoscaler (HPA) resources"""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        self.kubeconfig_path = kubeconfig_path
        self.kubectl_env = {}
        if kubeconfig_path:
            self.kubectl_env['KUBECONFIG'] = kubeconfig_path
    
    def _execute_kubectl(self, command: str) -> Dict[str, Any]:
        """Execute kubectl command"""
        try:
            cmd = ['kubectl'] + command.split()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, **self.kubectl_env} if self.kubectl_env else None
            )
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'output': result.stdout,
                    'result': self._parse_output(result.stdout)
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr or 'Command failed',
                    'output': result.stdout
                }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Command timed out'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _parse_output(self, output: str) -> Any:
        """Parse kubectl output"""
        try:
            # Try to parse as JSON
            return json.loads(output)
        except:
            # Return raw output if not JSON
            return output
    
    def create_hpa(self, deployment_name: str, namespace: str = "default",
                   min_replicas: int = 2, max_replicas: int = 10,
                   cpu_target: int = 70, memory_target: int = 80,
                   custom_metrics: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Create HPA resource"""
        # Trim deployment name and namespace to remove any whitespace
        deployment_name = deployment_name.strip()
        namespace = namespace.strip()
        
        hpa_name = f"{deployment_name}-hpa"
        
        # Build HPA YAML
        hpa_spec = {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': hpa_name,
                'namespace': namespace,
                'labels': {
                    'app': deployment_name,
                    'managed-by': 'ai4k8s',
                    'created-at': datetime.now().strftime('%Y-%m-%d-%H-%M-%S').replace(':', '-')
                }
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': deployment_name
                },
                'minReplicas': min_replicas,
                'maxReplicas': max_replicas,
                'metrics': [
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'cpu',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': cpu_target
                            }
                        }
                    },
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'memory',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': memory_target
                            }
                        }
                    }
                ],
                'behavior': {
                    'scaleDown': {
                        'stabilizationWindowSeconds': 300,
                        'policies': [
                            {
                                'type': 'Percent',
                                'value': 50,
                                'periodSeconds': 15
                            }
                        ]
                    },
                    'scaleUp': {
                        'stabilizationWindowSeconds': 0,
                        'policies': [
                            {
                                'type': 'Percent',
                                'value': 100,
                                'periodSeconds': 15
                            },
                            {
                                'type': 'Pods',
                                'value': 4,
                                'periodSeconds': 15
                            }
                        ],
                        'selectPolicy': 'Max'
                    }
                }
            }
        }
        
        # Add custom metrics if provided
        if custom_metrics:
            hpa_spec['spec']['metrics'].extend(custom_metrics)
        
        # Convert to YAML
        hpa_yaml = yaml.dump(hpa_spec, default_flow_style=False)
        
        # Write YAML to temp file and apply via kubectl
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(hpa_yaml)
            temp_file = f.name
        
        try:
            apply_result = self._execute_kubectl(
                f"apply -f {temp_file} -n {namespace}"
            )
            
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass
            
            if apply_result['success']:
                return {
                    'success': True,
                    'hpa_name': hpa_name,
                    'namespace': namespace,
                    'message': f'HPA {hpa_name} created successfully',
                    'spec': hpa_spec
                }
            else:
                return apply_result
        except Exception as e:
            # Clean up temp file on error
            try:
                os.unlink(temp_file)
            except:
                pass
            return {
                'success': False,
                'error': f'Failed to apply HPA: {str(e)}'
            }
    
    def get_hpa(self, hpa_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Get HPA resource"""
        # Trim names to remove any whitespace
        hpa_name = hpa_name.strip()
        namespace = namespace.strip()
        
        result = self._execute_kubectl(
            f"get hpa {hpa_name} -n {namespace} -o json"
        )
        
        if result['success']:
            hpa_data = result['result']
            return {
                'success': True,
                'hpa': hpa_data,
                'status': self._parse_hpa_status(hpa_data)
            }
        else:
            return result
    
    def list_hpas(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """List all HPA resources"""
        cmd = "get hpa --all-namespaces -o json"
        if namespace:
            cmd = f"get hpa -n {namespace} -o json"
        
        result = self._execute_kubectl(cmd)
        
        if result['success']:
            hpa_list = result['result']
            items = hpa_list.get('items', [])
            return {
                'success': True,
                'hpas': [
                    {
                        'name': item['metadata']['name'],
                        'namespace': item['metadata']['namespace'],
                        'target': item['spec']['scaleTargetRef']['name'],
                        'min_replicas': item['spec']['minReplicas'],
                        'max_replicas': item['spec']['maxReplicas'],
                        'current_replicas': item['status'].get('currentReplicas', 0),
                        'desired_replicas': item['status'].get('desiredReplicas', 0),
                        'status': self._parse_hpa_status(item)
                    }
                    for item in items
                ],
                'count': len(items)
            }
        else:
            return result
    
    def update_hpa(self, hpa_name: str, namespace: str = "default",
                   min_replicas: Optional[int] = None,
                   max_replicas: Optional[int] = None,
                   cpu_target: Optional[int] = None,
                   memory_target: Optional[int] = None) -> Dict[str, Any]:
        """Update HPA resource"""
        # Get current HPA
        current = self.get_hpa(hpa_name, namespace)
        if not current['success']:
            return current
        
        hpa = current['hpa']
        
        # Update fields
        if min_replicas is not None:
            hpa['spec']['minReplicas'] = min_replicas
        if max_replicas is not None:
            hpa['spec']['maxReplicas'] = max_replicas
        if cpu_target is not None:
            for metric in hpa['spec']['metrics']:
                if metric['type'] == 'Resource' and metric['resource']['name'] == 'cpu':
                    metric['resource']['target']['averageUtilization'] = cpu_target
        if memory_target is not None:
            for metric in hpa['spec']['metrics']:
                if metric['type'] == 'Resource' and metric['resource']['name'] == 'memory':
                    metric['resource']['target']['averageUtilization'] = memory_target
        
        # Apply update
        hpa_yaml = yaml.dump(hpa, default_flow_style=False)
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(hpa_yaml)
            temp_file = f.name
        
        try:
            result = self._execute_kubectl(
                f"apply -f {temp_file} -n {namespace}"
            )
            
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass
            
            if result['success']:
                return {
                    'success': True,
                    'message': f'HPA {hpa_name} updated successfully',
                    'hpa': hpa
                }
            else:
                return result
        except Exception as e:
            # Clean up temp file on error
            try:
                os.unlink(temp_file)
            except:
                pass
            return {
                'success': False,
                'error': f'Failed to update HPA: {str(e)}'
            }
    
    def patch_hpa_replicas(self, hpa_name: str, namespace: str, min_replicas: int, max_replicas: int) -> Dict[str, Any]:
        """Patch HPA min/max replicas to allow predictive scaling"""
        try:
            # Use kubectl patch to update min/max replicas
            env = os.environ.copy()
            if self.kubeconfig_path:
                env['KUBECONFIG'] = self.kubeconfig_path
            
            # Patch minReplicas
            result_min = subprocess.run(
                ['kubectl', 'patch', 'hpa', hpa_name, '-n', namespace,
                 '--type=json', '-p', f'[{{"op": "replace", "path": "/spec/minReplicas", "value": {min_replicas}}}]'],
                capture_output=True,
                text=True,
                timeout=30,
                env=env
            )
            
            # Patch maxReplicas
            result_max = subprocess.run(
                ['kubectl', 'patch', 'hpa', hpa_name, '-n', namespace,
                 '--type=json', '-p', f'[{{"op": "replace", "path": "/spec/maxReplicas", "value": {max_replicas}}}]'],
                capture_output=True,
                text=True,
                timeout=30,
                env=env
            )
            
            if result_min.returncode == 0 and result_max.returncode == 0:
                return {
                    'success': True,
                    'message': f'HPA {hpa_name} updated: min={min_replicas}, max={max_replicas}'
                }
            else:
                error_msg = result_min.stderr if result_min.returncode != 0 else result_max.stderr
                return {
                    'success': False,
                    'error': f'Failed to patch HPA: {error_msg}'
                }
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to patch HPA: {str(e)}'
            }
    
    def delete_hpa(self, hpa_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Delete HPA resource"""
        result = self._execute_kubectl(
            f"delete hpa {hpa_name} -n {namespace}"
        )
        
        if result['success']:
            return {
                'success': True,
                'message': f'HPA {hpa_name} deleted successfully'
            }
        else:
            return result
    
    def _parse_hpa_status(self, hpa_data: Dict) -> Dict[str, Any]:
        """Parse HPA status information"""
        status = hpa_data.get('status', {})
        spec = hpa_data.get('spec', {})
        
        current_replicas = status.get('currentReplicas', 0)
        desired_replicas = status.get('desiredReplicas', 0)
        min_replicas = spec.get('minReplicas', 1)
        max_replicas = spec.get('maxReplicas', 10)
        
        # Get metrics
        current_metrics = {}
        for condition in status.get('conditions', []):
            if condition['type'] == 'AbleToScale':
                current_metrics['able_to_scale'] = condition['status'] == 'True'
            elif condition['type'] == 'ScalingActive':
                current_metrics['scaling_active'] = condition['status'] == 'True'
        
        # Get CPU and memory metrics
        for metric in status.get('currentMetrics', []):
            if metric['type'] == 'Resource':
                resource_name = metric['resource']['name']
                current_value = metric['resource'].get('current', {}).get('averageUtilization', 0)
                current_metrics[f'{resource_name}_usage'] = current_value
        
        return {
            'current_replicas': current_replicas,
            'desired_replicas': desired_replicas,
            'min_replicas': min_replicas,
            'max_replicas': max_replicas,
            'metrics': current_metrics,
            'scaling_status': 'scaling_up' if desired_replicas > current_replicas else 
                             'scaling_down' if desired_replicas < current_replicas else 'stable'
        }
    
    def get_deployment_replicas(self, deployment_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Get current deployment replica count"""
        result = self._execute_kubectl(
            f"get deployment {deployment_name} -n {namespace} -o json"
        )
        
        if result['success']:
            deployment = result['result']
            return {
                'success': True,
                'replicas': deployment['spec'].get('replicas', 0),
                'ready_replicas': deployment['status'].get('readyReplicas', 0),
                'available_replicas': deployment['status'].get('availableReplicas', 0)
            }
        else:
            return result
    
    def patch_deployment_annotations(self, deployment_name: str, namespace: str, 
                                     annotations: Dict[str, str]) -> Dict[str, Any]:
        """Patch deployment annotations"""
        import json
        # Build patch JSON
        patch = {
            'metadata': {
                'annotations': annotations
            }
        }
        patch_json = json.dumps(patch)
        
        result = self._execute_kubectl(
            f"patch deployment {deployment_name} -n {namespace} --type=merge -p '{patch_json}'"
        )
        return result
    
    def patch_deployment_labels(self, deployment_name: str, namespace: str,
                                labels: Dict[str, str]) -> Dict[str, Any]:
        """Patch deployment labels"""
        import json
        # Build patch JSON
        patch = {
            'metadata': {
                'labels': labels
            }
        }
        patch_json = json.dumps(patch)
        
        result = self._execute_kubectl(
            f"patch deployment {deployment_name} -n {namespace} --type=merge -p '{patch_json}'"
        )
        return result
    
    def get_deployment_annotations(self, deployment_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Get deployment annotations"""
        result = self._execute_kubectl(
            f"get deployment {deployment_name} -n {namespace} -o jsonpath='{{.metadata.annotations}}'"
        )
        
        if result['success']:
            # Parse annotations (kubectl returns them as key=value pairs)
            annotations = {}
            if result.get('output'):
                import shlex
                # Parse the output which might be in format: key1=value1 key2=value2
                parts = shlex.split(result['output'])
                for part in parts:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        annotations[key] = value
            return {
                'success': True,
                'annotations': annotations
            }
        else:
            return result
    
    def list_deployments_with_label(self, label_selector: str, namespace: Optional[str] = None) -> Dict[str, Any]:
        """List deployments matching a label selector"""
        cmd = f"get deployments -l {label_selector} -o json"
        if namespace:
            cmd = f"get deployments -n {namespace} -l {label_selector} -o json"
        else:
            cmd = f"get deployments --all-namespaces -l {label_selector} -o json"
        
        result = self._execute_kubectl(cmd)
        
        if result.get('success'):
            deployments = result.get('result')
            if deployments is None:
                deployments = {}
            items = deployments.get('items')
            if items is None:
                items = []
            return {
                'success': True,
                'deployments': items,
                'count': len(items)
            }
        else:
            return result


