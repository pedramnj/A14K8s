#!/usr/bin/env python3
"""
AI4K8s VPA Engine
==================

Core VPA (Vertical Pod Autoscaler) engine for managing vertical scaling resources.
VPA adjusts CPU/Memory requests and limits per pod (vertical scaling).

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

class VerticalPodAutoscaler:
    """Manage Vertical Pod Autoscaler (VPA) resources"""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        self.kubeconfig_path = kubeconfig_path
        self.kubectl_env = {}
        if kubeconfig_path:
            self.kubectl_env['KUBECONFIG'] = kubeconfig_path
    
    def _execute_kubectl(self, command: str, stdin_input: Optional[str] = None) -> Dict[str, Any]:
        """Execute kubectl command"""
        try:
            cmd = ['kubectl'] + command.split()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                input=stdin_input,
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
            return json.loads(output)
        except:
            return output
    
    def check_vpa_available(self) -> Dict[str, Any]:
        """Check if VPA CRD is installed in the cluster"""
        try:
            # Method 1: Check api-resources for VPA
            result = self._execute_kubectl("api-resources --api-group=autoscaling.k8s.io -o name")
            if result['success']:
                output = result.get('output', '')
                if 'verticalpodautoscaler' in output.lower() or 'vpa' in output.lower():
                    return {'success': True, 'available': True}
            
            # Method 2: Try to get CRD directly
            result2 = self._execute_kubectl("get crd verticalpodautoscalers.autoscaling.k8s.io")
            if result2['success']:
                return {'success': True, 'available': True}
            
            # Method 3: Try to list VPAs (will fail gracefully if CRD not installed)
            result3 = self._execute_kubectl("get vpa --all-namespaces -o json")
            if result3['success']:
                # Even if list is empty, CRD exists
                return {'success': True, 'available': True}
            
            # If we get here, VPA is likely not installed
            return {'success': True, 'available': False, 'error': 'VPA CRD not installed'}
        except Exception as e:
            logger.warning(f"⚠️ Error checking VPA availability: {e}")
            return {'success': False, 'available': False, 'error': str(e)}
    
    def create_vpa(self, deployment_name: str, namespace: str = "default",
                   min_cpu: str = "100m", max_cpu: str = "1000m",
                   min_memory: str = "128Mi", max_memory: str = "512Mi",
                   update_mode: str = "Auto") -> Dict[str, Any]:
        """
        Create VPA resource
        
        Args:
            deployment_name: Name of the deployment to autoscale
            namespace: Kubernetes namespace
            min_cpu: Minimum CPU request (e.g., "100m")
            max_cpu: Maximum CPU limit (e.g., "1000m")
            min_memory: Minimum Memory request (e.g., "128Mi")
            max_memory: Maximum Memory limit (e.g., "512Mi")
            update_mode: VPA update mode - "Off", "Initial", "Auto", or "Recreate"
        """
        # Trim deployment name and namespace to remove any whitespace
        deployment_name = deployment_name.strip()
        namespace = namespace.strip()
        
        vpa_name = f"{deployment_name}-vpa"
        
        # Check if VPA CRD is installed first
        vpa_check = self.check_vpa_available()
        if not vpa_check.get('available', False):
            error_msg = vpa_check.get('error', 'VPA CRD not installed')
            logger.error(f"❌ VPA not available: {error_msg}")
            return {
                'success': False,
                'error': f'VPA is not installed in this cluster. {error_msg}. Please install the VPA controller first. See: https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler',
                'vpa_not_installed': True
            }
        
        # Build VPA YAML
        vpa_spec = {
            'apiVersion': 'autoscaling.k8s.io/v1',
            'kind': 'VerticalPodAutoscaler',
            'metadata': {
                'name': vpa_name,
                'namespace': namespace,
                'labels': {
                    'app': deployment_name,
                    'managed-by': 'ai4k8s',
                    'created-at': datetime.now().strftime('%Y-%m-%d-%H-%M-%S').replace(':', '-')
                }
            },
            'spec': {
                'targetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': deployment_name
                },
                'updatePolicy': {
                    'updateMode': update_mode
                },
                'resourcePolicy': {
                    'containerPolicies': [
                        {
                            'containerName': '*',  # Apply to all containers
                            'minAllowed': {
                                'cpu': min_cpu,
                                'memory': min_memory
                            },
                            'maxAllowed': {
                                'cpu': max_cpu,
                                'memory': max_memory
                            }
                        }
                    ]
                }
            }
        }
        
        try:
            # Write to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(vpa_spec, f)
                temp_file = f.name
            
            # Apply VPA
            result = self._execute_kubectl(f"apply -f {temp_file}")
            
            # Clean up
            os.unlink(temp_file)
            
            if result['success']:
                logger.info(f"✅ Created VPA {vpa_name} for {namespace}/{deployment_name}")
                return {
                    'success': True,
                    'vpa_name': vpa_name,
                    'message': f'VPA {vpa_name} created successfully'
                }
            else:
                error_msg = result.get('error', 'Failed to create VPA')
                # Check for CRD-related errors
                if any(phrase in error_msg.lower() for phrase in ['no matches for kind', 'crd', 'resource mapping not found', 'ensure crds are installed']):
                    logger.error(f"❌ VPA CRD not installed: {error_msg}")
                    return {
                        'success': False,
                        'error': f'VPA is not installed in this cluster. The VerticalPodAutoscaler CRD is missing. Please install the VPA controller first. See: https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler',
                        'vpa_not_installed': True
                    }
                return result
        except Exception as e:
            logger.error(f"❌ Error creating VPA: {e}")
            error_str = str(e)
            if any(phrase in error_str.lower() for phrase in ['no matches for kind', 'crd', 'resource mapping not found', 'ensure crds are installed']):
                return {
                    'success': False,
                    'error': f'VPA is not installed in this cluster. The VerticalPodAutoscaler CRD is missing. Please install the VPA controller first. See: https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler',
                    'vpa_not_installed': True
                }
            return {
                'success': False,
                'error': error_str
            }
    
    def get_vpa(self, vpa_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Get VPA resource"""
        result = self._execute_kubectl(
            f"get vpa {vpa_name} -n {namespace} -o json"
        )
        
        if result['success']:
            vpa_data = result['result']
            return {
                'success': True,
                'vpa': vpa_data,
                'status': self._parse_vpa_status(vpa_data)
            }
        else:
            return result
    
    def _parse_vpa_status(self, vpa_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse VPA status"""
        status = vpa_data.get('status', {})
        recommendation = status.get('recommendation', {})
        container_recommendations = recommendation.get('containerRecommendations', [])
        
        parsed_status = {
            'update_mode': vpa_data.get('spec', {}).get('updatePolicy', {}).get('updateMode', 'Unknown'),
            'target': vpa_data.get('spec', {}).get('targetRef', {}).get('name', ''),
            'recommendations': []
        }
        
        for container_rec in container_recommendations:
            target = container_rec.get('target', {})
            lower_bound = container_rec.get('lowerBound', {})
            upper_bound = container_rec.get('upperBound', {})
            
            parsed_status['recommendations'].append({
                'container_name': container_rec.get('containerName', ''),
                'target_cpu': target.get('cpu', 'N/A'),
                'target_memory': target.get('memory', 'N/A'),
                'lower_bound_cpu': lower_bound.get('cpu', 'N/A'),
                'lower_bound_memory': lower_bound.get('memory', 'N/A'),
                'upper_bound_cpu': upper_bound.get('cpu', 'N/A'),
                'upper_bound_memory': upper_bound.get('memory', 'N/A')
            })
        
        return parsed_status
    
    def list_vpas(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """List all VPA resources"""
        try:
            cmd = "get vpa --all-namespaces -o json"
            if namespace:
                cmd = f"get vpa -n {namespace} -o json"
            
            result = self._execute_kubectl(cmd)
            
            # Check if VPA API is not available (common if VPA controller not installed)
            if not result['success']:
                error_msg = result.get('error', '')
                # If error indicates VPA API doesn't exist, return empty list (not an error)
                if 'the server doesn\'t have a resource type "vpa"' in error_msg.lower() or \
                   'no matches for kind "vpa"' in error_msg.lower() or \
                   'unknown (get vpa' in error_msg.lower():
                    logger.debug("VPA API not available (VPA controller likely not installed)")
                    return {
                        'success': True,
                        'vpas': [],
                        'count': 0,
                        'vpa_api_unavailable': True
                    }
                return result
            
            vpas = []
            items = result.get('result', {}).get('items', []) if result.get('result') else []
            
            for item in items:
                try:
                    vpa_status = self._parse_vpa_status(item)
                    vpas.append({
                        'name': item.get('metadata', {}).get('name', ''),
                        'namespace': item.get('metadata', {}).get('namespace', 'default'),
                        'target': vpa_status.get('target', ''),
                        'update_mode': vpa_status.get('update_mode', 'Unknown'),
                        'recommendations': vpa_status.get('recommendations', [])
                    })
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse VPA item: {e}")
                    continue
            
            return {
                'success': True,
                'vpas': vpas,
                'count': len(vpas)
            }
        except Exception as e:
            logger.error(f"❌ Error listing VPAs: {e}")
            return {
                'success': False,
                'error': str(e),
                'vpas': [],
                'count': 0
            }
    
    def delete_vpa(self, vpa_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Delete VPA resource"""
        result = self._execute_kubectl(f"delete vpa {vpa_name} -n {namespace}")
        
        if result['success']:
            logger.info(f"✅ Deleted VPA {vpa_name} from {namespace}")
            return {
                'success': True,
                'message': f'VPA {vpa_name} deleted successfully'
            }
        else:
            return result
    
    def patch_vpa_resources(self, vpa_name: str, namespace: str,
                           min_cpu: Optional[str] = None, max_cpu: Optional[str] = None,
                           min_memory: Optional[str] = None, max_memory: Optional[str] = None) -> Dict[str, Any]:
        """Patch VPA resource limits"""
        # Get current VPA
        vpa_result = self.get_vpa(vpa_name, namespace)
        if not vpa_result['success']:
            return vpa_result
        
        vpa = vpa_result['vpa']
        spec = vpa.get('spec', {})
        resource_policy = spec.get('resourcePolicy', {})
        container_policies = resource_policy.get('containerPolicies', [])
        
        if not container_policies:
            container_policies = [{'containerName': '*'}]
        
        # Update resource limits
        for policy in container_policies:
            if min_cpu or max_cpu or min_memory or max_memory:
                if 'minAllowed' not in policy:
                    policy['minAllowed'] = {}
                if 'maxAllowed' not in policy:
                    policy['maxAllowed'] = {}
                
                if min_cpu:
                    policy['minAllowed']['cpu'] = min_cpu
                if max_cpu:
                    policy['maxAllowed']['cpu'] = max_cpu
                if min_memory:
                    policy['minAllowed']['memory'] = min_memory
                if max_memory:
                    policy['maxAllowed']['memory'] = max_memory
        
        # Update VPA spec
        spec['resourcePolicy'] = {'containerPolicies': container_policies}
        vpa['spec'] = spec
        
        # Apply updated VPA
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(vpa, f)
                temp_file = f.name
            
            result = self._execute_kubectl(f"apply -f {temp_file}")
            os.unlink(temp_file)
            
            if result['success']:
                logger.info(f"✅ Patched VPA {vpa_name} resource limits")
                return {
                    'success': True,
                    'message': f'VPA {vpa_name} resource limits updated'
                }
            return result
        except Exception as e:
            logger.error(f"❌ Error patching VPA: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def patch_deployment_resources(self, deployment_name: str, namespace: str,
                                   cpu_request: Optional[str] = None,
                                   memory_request: Optional[str] = None,
                                   cpu_limit: Optional[str] = None,
                                   memory_limit: Optional[str] = None) -> Dict[str, Any]:
        """
        Patch deployment resource requests/limits directly (for Predictive Autoscaling)
        This bypasses VPA and directly updates the deployment spec.
        
        Args:
            deployment_name: Name of the deployment
            namespace: Kubernetes namespace
            cpu_request: CPU request (e.g., "200m")
            memory_request: Memory request (e.g., "256Mi")
            cpu_limit: CPU limit (e.g., "500m")
            memory_limit: Memory limit (e.g., "512Mi")
        """
        deployment_name = deployment_name.strip()
        namespace = namespace.strip()
        
        try:
            # Get current deployment to find container name
            get_result = self._execute_kubectl(f"get deployment {deployment_name} -n {namespace} -o json")
            if not get_result.get('success'):
                return {
                    'success': False,
                    'error': f'Deployment {deployment_name} not found: {get_result.get("error")}'
                }
            
            deployment = get_result.get('result', {})
            containers = deployment.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])
            
            if not containers:
                return {
                    'success': False,
                    'error': f'No containers found in deployment {deployment_name}'
                }
            
            # Use strategic merge patch (simpler and more reliable than JSON patch)
            # Build patch for all containers - preserve all existing fields, only update resources
            import json
            containers_patch = []
            for container in containers:
                container_name = container.get('name', '')
                if not container_name:
                    continue
                
                # Start with existing container data to preserve all fields (image, ports, env, etc.)
                container_patch = container.copy()
                
                # Build resources patch
                resources_patch = {}
                if cpu_request or memory_request:
                    resources_patch['requests'] = {}
                    if cpu_request:
                        resources_patch['requests']['cpu'] = cpu_request
                    if memory_request:
                        resources_patch['requests']['memory'] = memory_request
                
                if cpu_limit or memory_limit:
                    resources_patch['limits'] = {}
                    if cpu_limit:
                        resources_patch['limits']['cpu'] = cpu_limit
                    if memory_limit:
                        resources_patch['limits']['memory'] = memory_limit
                
                if resources_patch:
                    # Merge with existing resources if any
                    if 'resources' in container_patch and container_patch['resources']:
                        existing_resources = container_patch['resources']
                        if 'requests' in existing_resources:
                            resources_patch['requests'] = {**existing_resources['requests'], **resources_patch.get('requests', {})}
                        if 'limits' in existing_resources:
                            resources_patch['limits'] = {**existing_resources['limits'], **resources_patch.get('limits', {})}
                    
                    container_patch['resources'] = resources_patch
                    containers_patch.append(container_patch)
            
            if not containers_patch:
                return {
                    'success': False,
                    'error': 'No resource values provided to patch'
                }
            
            # Build strategic merge patch structure
            patch = {
                'spec': {
                    'template': {
                        'spec': {
                            'containers': containers_patch
                        }
                    }
                }
            }
            
            # Apply patches using strategic merge patch (more reliable)
            # Use -p with JSON string directly (most compatible method)
            patch_json = json.dumps(patch)
            # Escape single quotes for shell safety
            patch_json_escaped = patch_json.replace("'", "'\"'\"'")
            
            # Try --patch-file first (kubectl 1.18+), fallback to -p with escaped JSON
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(patch_json)
                temp_file = f.name
            
            try:
                # Try --patch-file first
                result = self._execute_kubectl(
                    f"patch deployment {deployment_name} -n {namespace} --type=merge --patch-file={temp_file}"
                )
                
                # If --patch-file not supported, fallback to -p with JSON string
                if not result['success'] and ('unknown flag' in result.get('error', '').lower() or '--patch-file' in result.get('error', '')):
                    logger.debug("--patch-file not supported, using -p with JSON string")
                    result = self._execute_kubectl(
                        f"patch deployment {deployment_name} -n {namespace} --type=merge -p '{patch_json_escaped}'"
                    )
                
                os.unlink(temp_file)
            except Exception as e:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                raise
            
            if result['success']:
                logger.info(f"✅ Patched deployment {deployment_name} resources directly: CPU={cpu_request}, Memory={memory_request}")
                return {
                    'success': True,
                    'message': f'Updated resources for {deployment_name}'
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"❌ Error patching deployment resources: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }
    
    def _patch_deployment_resources_strategic(self, deployment_name: str, namespace: str,
                                             containers: List[Dict],
                                             cpu_request: Optional[str],
                                             memory_request: Optional[str],
                                             cpu_limit: Optional[str],
                                             memory_limit: Optional[str]) -> Dict[str, Any]:
        """Alternative method using strategic merge patch"""
        try:
            import json
            # Build patch for all containers
            containers_patch = []
            for container in containers:
                container_patch = {'name': container.get('name')}
                resources_patch = {}
                if cpu_request or memory_request:
                    resources_patch['requests'] = {}
                    if cpu_request:
                        resources_patch['requests']['cpu'] = cpu_request
                    if memory_request:
                        resources_patch['requests']['memory'] = memory_request
                if cpu_limit or memory_limit:
                    resources_patch['limits'] = {}
                    if cpu_limit:
                        resources_patch['limits']['cpu'] = cpu_limit
                    if memory_limit:
                        resources_patch['limits']['memory'] = memory_limit
                if resources_patch:
                    container_patch['resources'] = resources_patch
                containers_patch.append(container_patch)
            
            patch = {
                'spec': {
                    'template': {
                        'spec': {
                            'containers': containers_patch
                        }
                    }
                }
            }
            
            # Write to temp file and use stdin to avoid shell escaping issues
            import tempfile
            patch_json = json.dumps(patch)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(patch_json)
                temp_file = f.name
            
            try:
                # Use --patch-file flag (most reliable method)
                result = self._execute_kubectl(
                    f"patch deployment {deployment_name} -n {namespace} --type=merge --patch-file={temp_file}"
                )
                os.unlink(temp_file)
            except Exception as e:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                raise
            
            if result['success']:
                logger.info(f"✅ Patched deployment {deployment_name} resources using strategic merge")
                return {
                    'success': True,
                    'message': f'Updated resources for {deployment_name}'
                }
            return result
        except Exception as e:
            logger.error(f"❌ Error in strategic merge patch: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_deployment_resources(self, deployment_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Get current resource requests/limits from deployment"""
        result = self._execute_kubectl(
            f"get deployment {deployment_name} -n {namespace} -o json"
        )
        
        if not result['success']:
            return result
        
        deployment = result.get('result', {})
        if not deployment:
            return {
                'success': False,
                'error': 'Deployment not found',
                'resources': []
            }
        
        containers = deployment.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])
        if not containers:
            containers = []
        
        resources = []
        for container in containers:
            container_resources = container.get('resources', {})
            requests = container_resources.get('requests', {})
            limits = container_resources.get('limits', {})
            
            resources.append({
                'container_name': container.get('name', ''),
                'cpu_request': requests.get('cpu', 'N/A'),
                'memory_request': requests.get('memory', 'N/A'),
                'cpu_limit': limits.get('cpu', 'N/A'),
                'memory_limit': limits.get('memory', 'N/A')
            })
        
        return {
            'success': True,
            'resources': resources
        }

