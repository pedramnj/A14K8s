#!/usr/bin/env python3
"""
Kubernetes Metrics Collector
============================

This module collects real-time metrics from Kubernetes clusters for use with
the predictive monitoring system.

Author: Pedram Nikjooy
Thesis: AI Agent for Kubernetes Management
"""

import subprocess
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import time
import re
from dataclasses import dataclass
import requests
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NodeMetrics:
    """Data class for node-level metrics"""
    name: str
    cpu_usage_percent: float
    memory_usage_percent: float
    cpu_capacity: str
    memory_capacity: str
    pod_count: int

@dataclass
class PodMetrics:
    """Data class for pod-level metrics"""
    name: str
    namespace: str
    cpu_usage: str
    memory_usage: str
    cpu_limit: str
    memory_limit: str
    status: str

class KubernetesMetricsCollector:
    """Collects metrics from Kubernetes clusters"""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        self.kubeconfig_path = kubeconfig_path
        self.k8s_client = None
        self.metrics_server_available = False
        
        # Initialize Kubernetes client
        self._initialize_k8s_client()
        
        # Check if metrics server is available
        self._check_metrics_server()
    
    def _initialize_k8s_client(self):
        """Initialize Kubernetes client"""
        try:
            if self.kubeconfig_path:
                logger.info(f"Loading kubeconfig from: {self.kubeconfig_path}")
                config.load_kube_config(config_file=self.kubeconfig_path)
            else:
                logger.info("No kubeconfig path provided, trying default locations")
                # Try in-cluster config first (for pods running in cluster)
                try:
                    config.load_incluster_config()
                    logger.info("Using in-cluster config")
                except:
                    # Fallback to default kubeconfig
                    try:
                        config.load_kube_config()
                        logger.info("Using default kubeconfig")
                    except Exception as e:
                        logger.warning(f"Could not load default kubeconfig: {e}")
                        raise
            
            self.k8s_client = client.ApiClient()
            logger.info("Kubernetes client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            logger.warning("Will fall back to demo mode for metrics collection")
            self.k8s_client = None
    
    def _check_metrics_server(self):
        """Check if metrics server is available"""
        try:
            # Try to get node metrics
            result = subprocess.run(
                ['kubectl', 'top', 'nodes'],
                capture_output=True,
                text=True,
                timeout=10
            )
            self.metrics_server_available = result.returncode == 0
            if self.metrics_server_available:
                logger.info("Metrics server is available")
            else:
                logger.warning("Metrics server is not available - using alternative methods")
        except Exception as e:
            logger.warning(f"Could not check metrics server: {e}")
            self.metrics_server_available = False
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get basic cluster information"""
        if not self.k8s_client:
            logger.warning("Kubernetes client not available, returning demo data")
            return {
                "node_count": 0,
                "namespace_count": 0,
                "pod_count": 0,
                "service_count": 0,
                "timestamp": datetime.now().isoformat(),
                "demo_mode": True
            }
        
        try:
            v1 = client.CoreV1Api(self.k8s_client)
            
            # Get nodes
            nodes = v1.list_node()
            node_count = len(nodes.items)
            
            # Get namespaces
            namespaces = v1.list_namespace()
            namespace_count = len(namespaces.items)
            
            # Get pods
            pods = v1.list_pod_for_all_namespaces()
            pod_count = len(pods.items)
            
            # Get services
            services = v1.list_service_for_all_namespaces()
            service_count = len(services.items)
            
            return {
                "node_count": node_count,
                "namespace_count": namespace_count,
                "pod_count": pod_count,
                "service_count": service_count,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get cluster info: {e}")
            return {"error": str(e)}
    
    def get_node_metrics(self) -> List[NodeMetrics]:
        """Get node-level metrics"""
        node_metrics = []
        
        try:
            if self.metrics_server_available:
                # Use kubectl top nodes
                result = subprocess.run(
                    ['kubectl', 'top', 'nodes'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 4:  # NAME, CPU(cores), CPU(%), MEMORY(bytes), MEMORY(%)
                            try:
                                name = parts[0]
                                cpu_cores = parts[1]
                                cpu_percent = float(parts[2].rstrip('%'))  # CPU percentage (remove %)
                                memory_bytes = parts[3]
                                memory_percent = float(parts[4].rstrip('%'))  # Memory percentage (remove %)
                                
                                # Get pod count for this node
                                pod_count = self._get_pod_count_for_node(name)
                                
                                node_metrics.append(NodeMetrics(
                                    name=name,
                                    cpu_usage_percent=cpu_percent,
                                    memory_usage_percent=memory_percent,
                                    cpu_capacity="unknown",
                                    memory_capacity="unknown",
                                    pod_count=pod_count
                                ))
                                
                                logger.info(f"✅ Parsed node {name}: CPU={cpu_percent}%, Memory={memory_percent}%, Pods={pod_count}")
                            except (ValueError, IndexError) as e:
                                logger.error(f"Failed to parse node metrics line: {line} - Error: {e}")
                                continue
            else:
                # Fallback: get basic node info without metrics
                v1 = client.CoreV1Api(self.k8s_client)
                nodes = v1.list_node()
                
                for node in nodes.items:
                    pod_count = self._get_pod_count_for_node(node.metadata.name)
                    node_metrics.append(NodeMetrics(
                        name=node.metadata.name,
                        cpu_usage_percent=0.0,  # Unknown without metrics server
                        memory_usage_percent=0.0,
                        cpu_capacity=node.status.capacity.get('cpu', 'unknown'),
                        memory_capacity=node.status.capacity.get('memory', 'unknown'),
                        pod_count=pod_count
                    ))
        
        except Exception as e:
            logger.error(f"Failed to get node metrics: {e}")
        
        return node_metrics
    
    def get_pod_metrics(self, namespace: str = "default") -> List[PodMetrics]:
        """Get pod-level metrics"""
        pod_metrics = []
        
        try:
            if self.metrics_server_available:
                # Use kubectl top pods
                cmd = ['kubectl', 'top', 'pods', '-n', namespace]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 3:
                            name = parts[0]
                            cpu_usage = parts[1]
                            memory_usage = parts[2]
                            
                            # Get pod details
                            pod_details = self._get_pod_details(name, namespace)
                            
                            pod_metrics.append(PodMetrics(
                                name=name,
                                namespace=namespace,
                                cpu_usage=cpu_usage,
                                memory_usage=memory_usage,
                                cpu_limit=pod_details.get('cpu_limit', 'unknown'),
                                memory_limit=pod_details.get('memory_limit', 'unknown'),
                                status=pod_details.get('status', 'unknown')
                            ))
            else:
                # Fallback: get basic pod info
                v1 = client.CoreV1Api(self.k8s_client)
                pods = v1.list_namespaced_pod(namespace=namespace)
                
                for pod in pods.items:
                    pod_metrics.append(PodMetrics(
                        name=pod.metadata.name,
                        namespace=namespace,
                        cpu_usage="unknown",
                        memory_usage="unknown",
                        cpu_limit="unknown",
                        memory_limit="unknown",
                        status=pod.status.phase
                    ))
        
        except Exception as e:
            logger.error(f"Failed to get pod metrics: {e}")
        
        return pod_metrics
    
    def _parse_cpu_usage(self, cpu_str: str) -> float:
        """Parse CPU usage string to percentage"""
        try:
            if cpu_str.endswith('m'):
                # Millicores (e.g., "150m")
                millicores = float(cpu_str[:-1])
                # Assume 1 core = 1000m, so percentage = (millicores / 1000) * 100
                return (millicores / 1000) * 100
            elif cpu_str.endswith('n'):
                # Nanocores (e.g., "150000000n")
                nanocores = float(cpu_str[:-1])
                return (nanocores / 1000000000) * 100
            else:
                # Assume it's already a percentage or cores
                return float(cpu_str)
        except:
            return 0.0
    
    def _parse_memory_usage(self, memory_str: str) -> float:
        """Parse memory usage string to percentage"""
        try:
            # This is a simplified parser - in reality, you'd need to know the node's total memory
            # For now, we'll return a placeholder value
            if memory_str.endswith('Gi'):
                return float(memory_str[:-2]) * 10  # Rough estimate
            elif memory_str.endswith('Mi'):
                return float(memory_str[:-2]) * 0.1  # Rough estimate
            elif memory_str.endswith('Ki'):
                return float(memory_str[:-2]) * 0.0001  # Rough estimate
            else:
                return float(memory_str)
        except:
            return 0.0
    
    def _get_pod_count_for_node(self, node_name: str) -> int:
        """Get pod count for a specific node"""
        try:
            v1 = client.CoreV1Api(self.k8s_client)
            pods = v1.list_pod_for_all_namespaces(field_selector=f"spec.nodeName={node_name}")
            return len(pods.items)
        except:
            return 0
    
    def _get_pod_details(self, pod_name: str, namespace: str) -> Dict[str, str]:
        """Get detailed pod information"""
        try:
            v1 = client.CoreV1Api(self.k8s_client)
            pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
            
            cpu_limit = "unknown"
            memory_limit = "unknown"
            
            if pod.spec.containers:
                container = pod.spec.containers[0]
                if container.resources and container.resources.limits:
                    cpu_limit = container.resources.limits.get('cpu', 'unknown')
                    memory_limit = container.resources.limits.get('memory', 'unknown')
            
            return {
                'cpu_limit': cpu_limit,
                'memory_limit': memory_limit,
                'status': pod.status.phase
            }
        except:
            return {
                'cpu_limit': 'unknown',
                'memory_limit': 'unknown',
                'status': 'unknown'
            }
    
    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get aggregated cluster metrics"""
        if not self.k8s_client:
            logger.warning("Kubernetes client not available, returning demo metrics")
            return {
                "timestamp": datetime.now().isoformat(),
                "cluster_info": {
                    "node_count": 0,
                    "namespace_count": 0,
                    "pod_count": 0,
                    "service_count": 0,
                    "demo_mode": True
                },
                "aggregated_metrics": {
                    "cpu_usage_percent": 0,
                    "memory_usage_percent": 0,
                    "network_io_mbps": 0,
                    "disk_io_mbps": 0,
                    "pod_count": 0,
                    "node_count": 0
                },
                "node_metrics": [],
                "pod_count_by_namespace": {},
                "demo_mode": True
            }
        
        try:
            cluster_info = self.get_cluster_info()
            node_metrics = self.get_node_metrics()
            pod_metrics = self.get_pod_metrics()
            
            # Calculate aggregated values
            total_cpu_usage = sum(node.cpu_usage_percent for node in node_metrics) / len(node_metrics) if node_metrics else 0
            total_memory_usage = sum(node.memory_usage_percent for node in node_metrics) / len(node_metrics) if node_metrics else 0
            total_pod_count = sum(node.pod_count for node in node_metrics)
            
            # Debug logging
            logger.info(f"📊 Collected metrics: CPU={total_cpu_usage:.1f}%, Memory={total_memory_usage:.1f}%, Pods={total_pod_count}")
            for node in node_metrics:
                logger.info(f"  Node {node.name}: CPU={node.cpu_usage_percent:.1f}%, Memory={node.memory_usage_percent:.1f}%, Pods={node.pod_count}")
            
            # Calculate network and disk I/O (simplified)
            network_io = len(pod_metrics) * 10  # Rough estimate
            disk_io = len(pod_metrics) * 5  # Rough estimate
            
            return {
                "timestamp": datetime.now().isoformat(),
                "cluster_info": cluster_info,
                "aggregated_metrics": {
                    "cpu_usage_percent": total_cpu_usage,
                    "memory_usage_percent": total_memory_usage,
                    "network_io_mbps": network_io,
                    "disk_io_mbps": disk_io,
                    "pod_count": total_pod_count,
                    "node_count": cluster_info.get("node_count", 0)
                },
                "node_metrics": [
                    {
                        "name": node.name,
                        "cpu_usage_percent": node.cpu_usage_percent,
                        "memory_usage_percent": node.memory_usage_percent,
                        "pod_count": node.pod_count
                    } for node in node_metrics
                ],
                "pod_count_by_namespace": self._get_pod_count_by_namespace()
            }
        except Exception as e:
            logger.error(f"Failed to get aggregated metrics: {e}")
            return {"error": str(e)}
    
    def _get_pod_count_by_namespace(self) -> Dict[str, int]:
        """Get pod count by namespace"""
        try:
            v1 = client.CoreV1Api(self.k8s_client)
            namespaces = v1.list_namespace()
            
            namespace_counts = {}
            for ns in namespaces.items:
                pods = v1.list_namespaced_pod(namespace=ns.metadata.name)
                namespace_counts[ns.metadata.name] = len(pods.items)
            
            return namespace_counts
        except:
            return {}

# Example usage
if __name__ == "__main__":
    collector = KubernetesMetricsCollector()
    
    print("=== Kubernetes Metrics Collection ===")
    
    # Get cluster info
    cluster_info = collector.get_cluster_info()
    print(f"Cluster Info: {json.dumps(cluster_info, indent=2)}")
    
    # Get aggregated metrics
    metrics = collector.get_aggregated_metrics()
    print(f"Aggregated Metrics: {json.dumps(metrics, indent=2)}")
