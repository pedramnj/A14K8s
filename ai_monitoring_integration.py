#!/usr/bin/env python3
"""
AI Monitoring Integration
=========================

This module integrates the predictive monitoring system with the web application,
providing AI-powered insights and recommendations through the existing MCP interface.

Author: Pedram Nikjooy
Thesis: AI Agent for Kubernetes Management
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
import time
from dataclasses import asdict
from collections import deque

try:  # pragma: no cover - optional dependency for type conversion
    import numpy as np  # type: ignore
    NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMPY_AVAILABLE = False

from predictive_monitoring import (
    PredictiveMonitoringSystem, 
    ResourceMetrics, 
    AnomalyResult,
    ForecastResult
)
from k8s_metrics_collector import KubernetesMetricsCollector
from kubernetes_rag import KubernetesRAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIMonitoringIntegration:
    """Integrates AI monitoring with the web application"""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        self.metrics_history: deque = deque(maxlen=288)
        self._load_env_file()

        try:
            print(f"🔧 Initializing AI monitoring integration...")
            self.monitoring_system = PredictiveMonitoringSystem()
            print(f"✅ Predictive monitoring system initialized")
            
            self.metrics_collector = KubernetesMetricsCollector(kubeconfig_path)
            print(f"✅ Kubernetes metrics collector initialized")
            
            # Initialize RAG system for intelligent recommendations
            self.rag_system = KubernetesRAG()
            print(f"✅ RAG system initialized for intelligent monitoring")
            
            self.is_running = False
            self.collection_thread = None
            self.collection_interval = 300  # 5 minutes
            self.last_analysis = None
            print(f"✅ AI monitoring integration initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize AI monitoring integration: {e}")
            # Create a minimal fallback
            self.monitoring_system = None
            self.metrics_collector = None
            self.is_running = False
            self.collection_thread = None
            self.collection_interval = 300
            self.last_analysis = None
            self.metrics_history = deque(maxlen=288)

    def _load_env_file(self):
        """Load environment variables from .env-style files for remote services."""
        env_paths = [
            'client/.env',
            '.env',
            os.path.expanduser('~/.env'),
            os.path.join(os.path.dirname(__file__), '.env'),
        ]

        for env_path in env_paths:
            if os.path.exists(env_path):
                try:
                    with open(env_path, 'r') as fp:
                        for line in fp:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                os.environ[key.strip()] = value.strip().strip('\'"')
                    logger.info(f"✅ Loaded environment from {env_path}")
                    return
                except Exception as exc:
                    logger.warning(f"⚠️  Failed to load {env_path}: {exc}")
        logger.warning("⚠️  No .env file found in expected locations")

    def _to_json_safe(self, value: Any) -> Any:
        """Recursively convert values to JSON-serializable primitives."""
        if isinstance(value, dict):
            return {str(k): self._to_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._to_json_safe(v) for v in value]
        if isinstance(value, datetime):
            return value.isoformat()
        if NUMPY_AVAILABLE:
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, np.floating):
                return float(value)
            if isinstance(value, np.bool_):
                return bool(value)
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, (int, float, str)) or value is None:
            return value
        return str(value)
        
    def start_monitoring(self, interval_seconds: int = 300):
        """Start continuous monitoring"""
        if self.is_running:
            logger.warning("Monitoring is already running")
            return
        
        self.collection_interval = interval_seconds
        self.is_running = True
        
        # Start collection thread
        self.collection_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.collection_thread.start()
        
        logger.info(f"AI monitoring started with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.is_running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("AI monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Collect metrics
                metrics_data = self.metrics_collector.get_aggregated_metrics()
                
                if "error" not in metrics_data:
                    # Convert to ResourceMetrics format
                    agg_metrics = metrics_data["aggregated_metrics"]
                    resource_metrics = ResourceMetrics(
                        timestamp=datetime.now(),
                        cpu_usage=agg_metrics["cpu_usage_percent"],
                        memory_usage=agg_metrics["memory_usage_percent"],
                        network_io=agg_metrics["network_io_mbps"],
                        disk_io=agg_metrics["disk_io_mbps"],
                        pod_count=agg_metrics["pod_count"],
                        node_count=agg_metrics["node_count"]
                    )
                    
                    # Add to monitoring system
                    self.monitoring_system.add_metrics(resource_metrics)
                    
                    # Perform analysis
                    self.last_analysis = self.monitoring_system.analyze()
                    try:
                        self.metrics_history.append({
                            "timestamp": resource_metrics.timestamp,
                            "cpu_usage": resource_metrics.cpu_usage,
                            "memory_usage": resource_metrics.memory_usage
                        })
                    except Exception:
                        pass
                    
                    logger.info("Metrics collected and analyzed successfully")
                else:
                    logger.error(f"Failed to collect metrics: {metrics_data['error']}")
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Wait for next collection
            time.sleep(self.collection_interval)
    
    def get_current_analysis(self) -> Dict[str, Any]:
        """Get the latest analysis results"""
        # Check if components are available
        if not self.monitoring_system or not self.metrics_collector:
            print("⚠️  AI monitoring components not available, using demo mode")
            self.last_analysis = None  # Clear stale cache
            return self._generate_demo_analysis()
        
        # Perform one-time analysis if no continuous monitoring
        try:
            metrics_data = self.metrics_collector.get_aggregated_metrics()
            
            # Check for connection errors (cluster disconnected)
            if "error" in metrics_data:
                error_msg = str(metrics_data.get("error", ""))
                # Detect common disconnection errors
                if any(phrase in error_msg.lower() for phrase in ["no such host", "connection refused", "timeout", "unable to connect", "dial tcp"]):
                    logger.warning(f"Cluster appears to be disconnected: {error_msg}")
                    self.last_analysis = None  # Clear stale cache
                    return self._generate_cluster_disconnected_analysis(error_msg)
            
            # If collector indicates demo mode, return synthetic demo analysis instead of zeros
            if metrics_data.get("demo_mode"):
                logger.info("Metrics collector in demo mode; generating synthetic analysis")
                analysis_result = self._generate_demo_analysis()
                self.last_analysis = analysis_result
                return analysis_result
            if "error" not in metrics_data and "aggregated_metrics" in metrics_data:
                agg_metrics = metrics_data["aggregated_metrics"]
                # Guard: if everything is zero (no metrics), fall back to demo
                if (
                    agg_metrics.get("cpu_usage_percent", 0) == 0 and
                    agg_metrics.get("memory_usage_percent", 0) == 0 and
                    agg_metrics.get("pod_count", 0) == 0 and
                    agg_metrics.get("node_count", 0) == 0
                ):
                    logger.info("Aggregated metrics are all zero; generating synthetic analysis")
                    analysis_result = self._generate_demo_analysis()
                    self.last_analysis = analysis_result
                    return analysis_result
                resource_metrics = ResourceMetrics(
                    timestamp=datetime.now(),
                    cpu_usage=agg_metrics["cpu_usage_percent"],
                    memory_usage=agg_metrics["memory_usage_percent"],
                    network_io=agg_metrics["network_io_mbps"],
                    disk_io=agg_metrics["disk_io_mbps"],
                    pod_count=agg_metrics["pod_count"],
                    node_count=agg_metrics["node_count"],
                    running_pod_count=agg_metrics.get("running_pod_count", agg_metrics["pod_count"])
                )
                
                self.monitoring_system.add_metrics(resource_metrics)
                try:
                    self.metrics_history.append({
                        "timestamp": resource_metrics.timestamp,
                        "cpu_usage": resource_metrics.cpu_usage,
                        "memory_usage": resource_metrics.memory_usage
                    })
                except Exception:
                    pass
                analysis_result = self.monitoring_system.analyze()
                if "error" not in analysis_result:
                    self.last_analysis = analysis_result
                    return analysis_result
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to get current analysis: {e}")
            # Check if it's a connection error
            if any(phrase in error_msg.lower() for phrase in ["no such host", "connection refused", "timeout", "unable to connect", "dial tcp"]):
                logger.warning(f"Cluster connection error detected: {error_msg}")
                self.last_analysis = None  # Clear stale cache
                return self._generate_cluster_disconnected_analysis(error_msg)
        
        # Fallback: Generate synthetic data for demonstration
        logger.info("Using demo mode for AI monitoring")
        self.last_analysis = None  # Clear stale cache
        return self._generate_demo_analysis()
    
    def _generate_demo_analysis(self) -> Dict[str, Any]:
        """Generate demo analysis when Kubernetes is not available"""
        import random
        
        # First, check if cluster is actually disconnected (not just unavailable)
        cluster_disconnected = False
        error_msg = ""
        
        # Try to get real metrics using MCP tools
        try:
            import asyncio
            from mcp_client import call_mcp_tool
            
            # Get real pod count using MCP tools (revert to original approach)
            pod_count = 0  # Start with 0, only use real data if available
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(call_mcp_tool("pods_list", {"namespace": "all"}))
                loop.close()
                
                # Check if MCP call failed due to cluster disconnection
                if not result.get("success"):
                    error_text = str(result.get("error", ""))
                    if any(phrase in error_text.lower() for phrase in ["no such host", "connection refused", "timeout", "unable to connect", "dial tcp"]):
                        cluster_disconnected = True
                        error_msg = error_text
                        logger.warning(f"Cluster disconnected detected via MCP: {error_msg}")
                        return self._generate_cluster_disconnected_analysis(error_msg)
                
                if result.get("success"):
                    pod_data = result.get("result", {})
                    if isinstance(pod_data, dict) and "content" in pod_data:
                        content = pod_data["content"]
                        if isinstance(content, list) and len(content) > 0:
                            pod_text = content[0].get("text", "")
                            # Check for connection errors in the text
                            if any(phrase in pod_text.lower() for phrase in ["no such host", "connection refused", "unable to connect", "dial tcp"]):
                                cluster_disconnected = True
                                error_msg = pod_text
                                logger.warning(f"Cluster disconnected detected in MCP response: {error_msg}")
                                return self._generate_cluster_disconnected_analysis(error_msg)
                            # Count lines that look like pod entries (skip header)
                            pod_lines = [line for line in pod_text.split('\n') if line.strip() and not line.startswith('NAME') and 'Running' in line]
                            pod_count = len(pod_lines)
                            if pod_count > 0:
                                logger.info(f"✅ Got real pod count via MCP: {pod_count}")
            except Exception as e:
                error_msg = str(e)
                # Check if exception indicates cluster disconnection
                if any(phrase in error_msg.lower() for phrase in ["no such host", "connection refused", "timeout", "unable to connect", "dial tcp"]):
                    cluster_disconnected = True
                    logger.warning(f"Cluster disconnected detected via MCP exception: {error_msg}")
                    return self._generate_cluster_disconnected_analysis(error_msg)
                logger.warning(f"Failed to get pod count via MCP: {e}")
            
            # Get real CPU and memory usage using MCP tools
            cpu_usage = 0  # Start with 0, only use real data if available
            memory_usage = 0  # Start with 0, only use real data if available
            
            try:
                # Try to get top pods data
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(call_mcp_tool("pods_top", {"namespace": "all"}))
                loop.close()
                
                # Check if MCP call failed due to cluster disconnection
                if not result.get("success"):
                    error_text = str(result.get("error", ""))
                    if any(phrase in error_text.lower() for phrase in ["no such host", "connection refused", "timeout", "unable to connect", "dial tcp"]):
                        cluster_disconnected = True
                        error_msg = error_text
                        logger.warning(f"Cluster disconnected detected via MCP pods_top: {error_msg}")
                        return self._generate_cluster_disconnected_analysis(error_msg)
                
                if result.get("success"):
                    top_data = result.get("result", {})
                    if isinstance(top_data, dict) and "content" in top_data:
                        content = top_data["content"]
                        if isinstance(content, list) and len(content) > 0:
                            top_text = content[0].get("text", "")
                            # Parse CPU and memory from kubectl top output
                            lines = top_text.split('\n')
                            total_cpu_m = 0
                            total_memory_mi = 0
                            pod_count_from_top = 0
                            
                            for line in lines:
                                if line.strip() and not line.startswith('NAMESPACE') and 'm' in line:
                                    parts = line.split()
                                    if len(parts) >= 4:
                                        try:
                                            cpu_part = parts[2]  # CPU(cores) column
                                            mem_part = parts[3]  # MEMORY(bytes) column
                                            
                                            # Parse CPU (format: "69m")
                                            if 'm' in cpu_part:
                                                cpu_val = int(cpu_part.replace('m', ''))
                                                total_cpu_m += cpu_val
                                            
                                            # Parse Memory (format: "556Mi")
                                            if 'Mi' in mem_part:
                                                mem_val = int(mem_part.replace('Mi', ''))
                                                total_memory_mi += mem_val
                                            
                                            pod_count_from_top += 1
                                        except (ValueError, IndexError):
                                            continue
                            
                            # Convert to percentages (rough estimates)
                            if pod_count_from_top > 0:
                                # Assume a reasonable baseline for calculations
                                cpu_usage = min(90, max(5, (total_cpu_m / pod_count_from_top) * 2))  # Rough conversion
                                memory_usage = min(95, max(10, (total_memory_mi / pod_count_from_top) * 1.5))  # Rough conversion
                                logger.info(f"✅ Got real resource usage via MCP: CPU={cpu_usage:.1f}%, Memory={memory_usage:.1f}%")
                                
                                # Update pod count if we got it from top command
                                if pod_count_from_top > 0:
                                    pod_count = pod_count_from_top
                                    
            except Exception as e:
                error_msg = str(e)
                # Check if exception indicates cluster disconnection
                if any(phrase in error_msg.lower() for phrase in ["no such host", "connection refused", "timeout", "unable to connect", "dial tcp"]):
                    cluster_disconnected = True
                    logger.warning(f"Cluster disconnected detected via MCP pods_top exception: {error_msg}")
                    return self._generate_cluster_disconnected_analysis(error_msg)
                logger.warning(f"Failed to get top data via MCP: {e}")
                
        except Exception as e:
            error_msg = str(e)
            # Check if exception indicates cluster disconnection
            if any(phrase in error_msg.lower() for phrase in ["no such host", "connection refused", "timeout", "unable to connect", "dial tcp"]):
                cluster_disconnected = True
                logger.warning(f"Cluster disconnected detected via MCP exception: {error_msg}")
                return self._generate_cluster_disconnected_analysis(error_msg)
            logger.warning(f"Failed to get real metrics via MCP: {e}")
        
        # If we couldn't get any real data and cluster appears disconnected, show disconnected message
        if pod_count == 0 and cpu_usage == 0 and memory_usage == 0:
            return self._generate_cluster_disconnected_analysis("Unable to connect to Kubernetes cluster - no metrics available")
        
        # Generate realistic demo metrics
        current_time = datetime.now()
        # Use the real values we got from MCP tools above
        network_io = random.uniform(100, 500)
        disk_io = random.uniform(50, 200)
        node_count = 1
        
        # Create demo metrics
        demo_metrics = ResourceMetrics(
            timestamp=current_time,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            network_io=network_io,
            disk_io=disk_io,
            pod_count=pod_count,
            node_count=node_count
        )
        try:
            self.metrics_history.append({
                "timestamp": demo_metrics.timestamp,
                "cpu_usage": demo_metrics.cpu_usage,
                "memory_usage": demo_metrics.memory_usage
            })
        except Exception:
            pass
        
        # Add to monitoring system if available
        if self.monitoring_system:
            self.monitoring_system.add_metrics(demo_metrics)
            
            # Generate analysis with error handling
            try:
                analysis = self.monitoring_system.analyze()
                if "error" in analysis:
                    # If analysis fails, create a basic demo response
                    analysis = self._create_basic_demo_analysis(demo_metrics)
            except Exception as e:
                logger.error(f"Demo analysis failed: {e}")
                analysis = self._create_basic_demo_analysis(demo_metrics)
        else:
            # If monitoring system is not available, create basic demo response
            analysis = self._create_basic_demo_analysis(demo_metrics)
        
        # Add demo indicator - check if we have real data
        has_real_data = False
        # Don't test MCP if we're in disconnected state - it will give false positives
        if not (analysis.get("error") or (analysis.get("demo_message", "").lower().find("disconnected") >= 0)):
            try:
                import asyncio
                from mcp_client import call_mcp_tool
                # Try to verify we can get real data
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                test_result = loop.run_until_complete(call_mcp_tool("pods_list", {"namespace": "all"}))
                loop.close()
                # Check if MCP result actually contains connection errors
                if test_result.get("success"):
                    result_text = str(test_result.get("result", "")).lower()
                    # If result contains connection errors, it's not real data
                    if any(phrase in result_text for phrase in ["no such host", "connection refused", "timeout", "unable to connect", "dial tcp"]):
                        has_real_data = False
                    else:
                        has_real_data = True
                else:
                    has_real_data = False
                logger.info(f"MCP connection test result: {has_real_data}")
            except Exception as e:
                logger.warning(f"MCP connection test failed: {e}")
                has_real_data = False
            
        if has_real_data:
            analysis["demo_mode"] = False
            analysis["demo_message"] = "Real-time monitoring active with live cluster data"
        else:
            analysis["demo_mode"] = True
            analysis["demo_message"] = "Demo mode: Using synthetic data for demonstration purposes"
        
        return analysis
    
    def _generate_cluster_disconnected_analysis(self, error_msg: str) -> Dict[str, Any]:
        """Generate analysis when cluster is disconnected"""
        return {
            "timestamp": datetime.now().isoformat(),
            "current_metrics": {
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "network_io": 0.0,
                "disk_io": 0.0,
                "pod_count": 0,
                "running_pod_count": 0,
                "node_count": 0
            },
            "health_score": {
                "score": 0,
                "status": "disconnected",
                "message": "Cluster is not accessible"
            },
            "anomaly_detection": {
                "anomaly_detected": False,
                "severity": "none",
                "message": "No data available - cluster disconnected"
            },
            "forecasts": {
                "cpu_forecast": {
                    "current_value": 0.0,
                    "predicted_values": [],
                    "trend": "unknown",
                    "message": "No forecast available - cluster disconnected"
                },
                "memory_forecast": {
                    "current_value": 0.0,
                    "predicted_values": [],
                    "trend": "unknown",
                    "message": "No forecast available - cluster disconnected"
                }
            },
            "recommendations": [
                {
                    "type": "cluster_disconnected",
                    "priority": "high",
                    "message": f"Kubernetes cluster is not accessible. Error: {error_msg}",
                    "action": "Please verify the cluster is running and the kubeconfig is correct."
                }
            ],
            "summary": "Cluster Disconnected - Unable to connect to Kubernetes cluster",
            "demo_mode": True,
            "demo_message": f"⚠️ Cluster Disconnected: {error_msg}. Please check your cluster configuration and ensure the cluster is running.",
            "error": error_msg
        }
    
    def _create_basic_demo_analysis(self, metrics: ResourceMetrics) -> Dict[str, Any]:
        """Create a basic demo analysis when ML models fail"""
        return {
            "timestamp": metrics.timestamp.isoformat(),
            "current_metrics": {
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "network_io": metrics.network_io,
                "disk_io": metrics.disk_io,
                "pod_count": metrics.pod_count,
                "running_pod_count": metrics.running_pod_count,
                "node_count": metrics.node_count
            },
            "health_score": {
                "score": 85,
                "status": "healthy",
                "message": "Demo cluster is running normally"
            },
            "anomaly_detection": {
                "anomaly_detected": False,
                "severity": "none",
                "message": "No anomalies detected in demo data"
            },
            "forecasts": {
                "cpu_forecast": {
                    "trend": "stable",
                    "predicted_values": [metrics.cpu_usage] * 6,
                    "recommendation": "CPU usage is stable"
                },
                "memory_forecast": {
                    "trend": "stable", 
                    "predicted_values": [metrics.memory_usage] * 6,
                    "recommendation": "Memory usage is stable"
                }
            },
            "recommendations": [
                {
                    "type": "demo",
                    "priority": "low",
                    "recommendation": "This is demo data. Connect to a real Kubernetes cluster for live monitoring."
                }
            ]
        }
    
    def get_anomaly_alerts(self) -> List[Dict[str, Any]]:
        """Get current anomaly alerts"""
        try:
            analysis = self.get_current_analysis()
            
            alerts = []
            if "anomaly_detection" in analysis and analysis["anomaly_detection"].get("is_anomaly", False):
                anomaly = analysis["anomaly_detection"]
                alerts.append({
                    "type": "anomaly",
                    "severity": anomaly.get("severity", "low"),
                    "message": f"Anomaly detected: {anomaly.get('anomaly_type', 'unknown')}",
                    "details": anomaly,
                    "timestamp": analysis.get("timestamp", datetime.now().isoformat())
                })
            
            return self._to_json_safe(alerts)
        except Exception as e:
            logger.error(f"Failed to get anomaly alerts: {e}")
            return []
    
    def get_performance_recommendations(self) -> List[Dict[str, Any]]:
        """Get performance optimization recommendations"""
        try:
            analysis = self.get_current_analysis()
            
            recommendations = []
            
            # Add performance optimization recommendations
            if "performance_optimization" in analysis:
                perf_recs = analysis["performance_optimization"].get("recommendations", [])
                for rec in perf_recs:
                    recommendations.append({
                        "type": "performance",
                        "priority": rec.get("priority", "low"),
                        "message": rec.get("recommendation", "Performance optimization available"),
                        "action": rec.get("action", "monitor"),
                        "details": rec
                    })
            
            # Add capacity planning recommendations
            if "capacity_planning" in analysis:
                cap_recs = analysis["capacity_planning"].get("recommendations", [])
                for rec in cap_recs:
                    recommendations.append({
                        "type": "capacity",
                        "priority": rec.get("urgency", "low"),
                        "message": rec.get("recommendation", "Capacity planning recommendation"),
                        "action": rec.get("action", "monitor"),
                        "details": rec
                    })
            
            # Add demo recommendations if in demo mode
            if analysis.get("demo_mode", False) and "recommendations" in analysis:
                for rec in analysis["recommendations"]:
                    recommendations.append({
                        "type": "demo",
                        "priority": rec.get("priority", "low"),
                        "message": rec.get("recommendation", "Demo recommendation"),
                        "action": "demo",
                        "details": rec
                    })
            
            # If no recommendations found, provide some basic ones based on current metrics
            if not recommendations:
                current_metrics = analysis.get("current_metrics", {})
                cpu_usage = current_metrics.get("cpu_usage", 0)
                memory_usage = current_metrics.get("memory_usage", 0)
                
                # CPU-based recommendations
                if cpu_usage > 80:
                    recommendations.append({
                        "type": "performance",
                        "priority": "high",
                        "message": f"High CPU usage detected ({cpu_usage:.1f}%). Consider scaling up or optimizing resource-intensive pods.",
                        "action": "scale",
                        "details": {"metric": "cpu", "value": cpu_usage}
                    })
                elif cpu_usage < 20:
                    recommendations.append({
                        "type": "optimization",
                        "priority": "low",
                        "message": f"Low CPU usage ({cpu_usage:.1f}%). Consider right-sizing or consolidating resources.",
                        "action": "optimize",
                        "details": {"metric": "cpu", "value": cpu_usage}
                    })
                
                # Memory-based recommendations
                if memory_usage > 80:
                    recommendations.append({
                        "type": "performance",
                        "priority": "high",
                        "message": f"High memory usage detected ({memory_usage:.1f}%). Monitor memory-intensive applications.",
                        "action": "monitor",
                        "details": {"metric": "memory", "value": memory_usage}
                    })
                elif memory_usage < 30:
                    recommendations.append({
                        "type": "optimization",
                        "priority": "low",
                        "message": f"Low memory usage ({memory_usage:.1f}%). System is efficiently utilizing memory resources.",
                        "action": "monitor",
                        "details": {"metric": "memory", "value": memory_usage}
                    })
                
                # General recommendations
                if not recommendations:
                    recommendations.append({
                        "type": "general",
                        "priority": "low",
                        "message": "System is running optimally. Continue monitoring for any changes in resource usage patterns.",
                        "action": "monitor",
                        "details": {"status": "healthy"}
                    })
            
            return self._to_json_safe(recommendations)
        except Exception as e:
            logger.error(f"Failed to get performance recommendations: {e}")
            return []
    
    def get_rag_enhanced_recommendations(self) -> List[Dict[str, Any]]:
        """Get RAG-enhanced intelligent recommendations"""
        try:
            if not self.rag_system:
                return []
            
            analysis = self.get_current_analysis()
            if "error" in analysis:
                return []
            
            current_metrics = analysis.get("current_metrics", {})
            
            # Get RAG-enhanced recommendations
            rag_data = self.rag_system.get_monitoring_recommendations(current_metrics)
            
            # Generate intelligent recommendations based on RAG context
            recommendations = []
            
            # CPU-based recommendations with RAG context
            cpu_usage = current_metrics.get("cpu_usage", 0)
            if cpu_usage > 80:
                recommendations.append({
                    "type": "performance",
                    "priority": "high",
                    "message": f"High CPU usage ({cpu_usage}%) - scale up immediately based on monitoring best practices",
                    "action": "scale_up_cpu",
                    "details": {
                        "type": "cpu_optimization",
                        "priority": "high",
                        "current_value": cpu_usage,
                        "recommendation": "Scale up CPU resources by 50% based on HPA best practices",
                        "action": "scale_up_cpu",
                        "rag_context": "Based on monitoring guidance: CPU > 80% requires immediate scaling"
                    }
                })
            elif cpu_usage < 30:
                recommendations.append({
                    "type": "performance",
                    "priority": "medium",
                    "message": f"Low CPU usage ({cpu_usage}%) - consider scaling down to save costs",
                    "action": "scale_down_cpu",
                    "details": {
                        "type": "cpu_optimization",
                        "priority": "medium",
                        "current_value": cpu_usage,
                        "recommendation": "Scale down CPU resources by 30% to optimize costs",
                        "action": "scale_down_cpu",
                        "rag_context": "Based on monitoring guidance: CPU < 30% indicates underutilization"
                    }
                })
            
            # Memory-based recommendations with RAG context
            memory_usage = current_metrics.get("memory_usage", 0)
            if memory_usage > 85:
                recommendations.append({
                    "type": "performance",
                    "priority": "critical",
                    "message": f"Critical memory usage ({memory_usage}%) - immediate action required",
                    "action": "scale_up_memory",
                    "details": {
                        "type": "memory_optimization",
                        "priority": "critical",
                        "current_value": memory_usage,
                        "recommendation": "Scale up memory resources immediately to prevent OOM kills",
                        "action": "scale_up_memory",
                        "rag_context": "Based on monitoring guidance: Memory > 85% is critical threshold"
                    }
                })
            elif memory_usage < 40:
                recommendations.append({
                    "type": "performance",
                    "priority": "low",
                    "message": f"Low memory usage ({memory_usage}%) - consider scaling down",
                    "action": "scale_down_memory",
                    "details": {
                        "type": "memory_optimization",
                        "priority": "low",
                        "current_value": memory_usage,
                        "recommendation": "Consider reducing memory allocation to optimize costs",
                        "action": "scale_down_memory",
                        "rag_context": "Based on monitoring guidance: Memory < 40% indicates underutilization"
                    }
                })
            
            # Add RAG-enhanced insights
            if rag_data.get("relevant_docs"):
                recommendations.append({
                    "type": "insight",
                    "priority": "low",
                    "message": "RAG-enhanced monitoring insights available",
                    "action": "view_insights",
                    "details": {
                        "type": "rag_insights",
                        "priority": "low",
                        "recommendation": "View detailed monitoring insights based on Kubernetes best practices",
                        "action": "view_insights",
                        "rag_context": f"Retrieved {len(rag_data['relevant_docs'])} relevant monitoring guidance documents"
                    }
                })
            
            return self._to_json_safe(recommendations)
            
        except Exception as e:
            logger.error(f"Error getting RAG-enhanced recommendations: {e}")
            return []

    def _build_llm_fallback(self, current_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return deterministic guidance when LLM is unavailable."""
        cpu = current_metrics.get("cpu_usage", 0)
        memory = current_metrics.get("memory_usage", 0)
        pod_count = current_metrics.get("pod_count", 0)

        fallback: List[Dict[str, Any]] = []

        if cpu < 30:
            fallback.append({
                "type": "llm_fallback",
                "priority": "medium",
                "message": f"CPU usage is {cpu:.1f}% (below 30%). Consider scaling down workloads or node counts to save cost.",
                "action": "scale_down_cpu",
                "details": {
                    "reason": "LLM disabled - rule-based guidance",
                    "metric": "cpu_usage",
                    "metric_value": cpu
                }
            })
        elif cpu > 80:
            fallback.append({
                "type": "llm_fallback",
                "priority": "high",
                "message": f"CPU usage is {cpu:.1f}% (above 80%). Scale up or optimize CPU-intensive workloads.",
                "action": "scale_up_cpu",
                "details": {
                    "reason": "LLM disabled - rule-based guidance",
                    "metric": "cpu_usage",
                    "metric_value": cpu
                }
            })

        if memory < 40:
            fallback.append({
                "type": "llm_fallback",
                "priority": "low",
                "message": f"Memory usage is {memory:.1f}% (below 40%). Right-size memory allocations to reduce waste.",
                "action": "scale_down_memory",
                "details": {
                    "reason": "LLM disabled - rule-based guidance",
                    "metric": "memory_usage",
                    "metric_value": memory
                }
            })
        elif memory > 85:
            fallback.append({
                "type": "llm_fallback",
                "priority": "high",
                "message": f"Memory usage is {memory:.1f}% (above 85%). Increase limits or investigate high-memory workloads.",
                "action": "scale_up_memory",
                "details": {
                    "reason": "LLM disabled - rule-based guidance",
                    "metric": "memory_usage",
                    "metric_value": memory
                }
            })

        if pod_count > 60:
            fallback.append({
                "type": "llm_fallback",
                "priority": "medium",
                "message": f"Cluster is running {pod_count} pods. Review autoscaling policies and retire unused workloads.",
                "action": "review_autoscaling",
                "details": {
                    "reason": "LLM disabled - rule-based guidance",
                    "metric": "pod_count",
                    "metric_value": pod_count
                }
            })

        if not fallback:
            fallback.append({
                "type": "llm_fallback",
                "priority": "low",
                "message": "Cluster metrics look healthy. Continue monitoring and keep alerts enabled.",
                "action": "monitor",
                "details": {"reason": "LLM disabled - rule-based guidance"}
            })

        return self._to_json_safe(fallback)
    
    def get_llm_recommendations(self, current_metrics: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Return LLM recommendations, falling back to heuristic guidance if LLM is disabled."""
        try:
            if current_metrics is None:
                metrics_data = self.metrics_collector.get_aggregated_metrics()
                if "error" in metrics_data or "aggregated_metrics" not in metrics_data:
                    return []
                agg = metrics_data["aggregated_metrics"]
                current_metrics = {
                    "cpu_usage": agg.get("cpu_usage_percent", 0),
                    "memory_usage": agg.get("memory_usage_percent", 0),
                    "pod_count": agg.get("pod_count", 0),
                    "running_pod_count": agg.get("running_pod_count", 0),
                    "node_count": agg.get("node_count", 0)
                }

            llm_enabled = os.environ.get('LLM_ENABLED', '')
            groq_key = os.environ.get('GROQ_API_KEY', '')

            if not groq_key or llm_enabled not in ('1', 'true', 'True'):
                return self._build_llm_fallback(current_metrics)

            from groq import Groq  # type: ignore
        except ImportError:
            logger.warning("Groq client not installed; using fallback recommendations.")
            return self._build_llm_fallback(current_metrics or {})
        except Exception as exc:
            logger.warning(f"Failed to prepare metrics for LLM: {exc}")
            return self._build_llm_fallback(current_metrics or {})

        try:
            client = Groq(api_key=groq_key)

            prompt = (
                "You are a Kubernetes SRE. Given current cluster metrics, produce up to 3 actionable "
                "recommendations as JSON array of objects: "
                "[{\"action\": str, \"message\": str, \"priority\": \"low|medium|high|critical\", \"details\": {}}]. "
                "Return ONLY JSON.\n\n"
                f"CURRENT_METRICS: {json.dumps(current_metrics)}"
            )

            response = client.chat.completions.create(
                model=os.environ.get('GROQ_MODEL', 'llama-3.1-8b-instant'),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=600
            )
            text = response.choices[0].message.content if response and response.choices else "[]"
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError("LLM response is not a list")

            normalized: List[Dict[str, Any]] = []
            for rec in parsed[:5]:
                normalized.append({
                    "type": "llm",
                    "priority": (rec.get("priority") or "medium").lower(),
                    "message": rec.get("message", "Recommendation"),
                    "action": rec.get("action", "review"),
                    "details": rec.get("details", {})
                })
            if not normalized:
                return self._build_llm_fallback(current_metrics)
            return self._to_json_safe(normalized)
        except Exception as exc:
            logger.warning(f"Groq call failed: {exc}")
            return self._build_llm_fallback(current_metrics)

    def get_forecast_summary(self) -> Dict[str, Any]:
        """Get a summary of forecasts"""
        try:
            analysis = self.get_current_analysis()
            
            if "forecasts" not in analysis:
                return {"error": "No forecast data available"}
            
            forecasts = analysis["forecasts"]
            # Support both shapes: {'cpu': {...}, 'memory': {...}} and {'cpu_forecast': {...}, 'memory_forecast': {...}}
            cpu_src = forecasts.get("cpu") or forecasts.get("cpu_forecast") or {}
            mem_src = forecasts.get("memory") or forecasts.get("memory_forecast") or {}
            # Normalize predicted values arrays
            cpu_pred = cpu_src.get("predicted_values") or cpu_src.get("predicted") or []
            mem_pred = mem_src.get("predicted_values") or mem_src.get("predicted") or []
            summary = {
                "timestamp": analysis.get("timestamp", datetime.now().isoformat()),
                "cpu_forecast": {
                    "current": cpu_src.get("current", 0),
                    "trend": cpu_src.get("trend", "unknown"),
                    "next_hour_prediction": (cpu_pred[0] if cpu_pred else cpu_src.get("next_hour_prediction")),
                    "recommendation": cpu_src.get("recommendation", "No recommendation available")
                },
                "memory_forecast": {
                    "current": mem_src.get("current", 0),
                    "trend": mem_src.get("trend", "unknown"),
                    "next_hour_prediction": (mem_pred[0] if mem_pred else mem_src.get("next_hour_prediction")),
                    "recommendation": mem_src.get("recommendation", "No recommendation available")
                }
            }
            
            return self._to_json_safe(summary)
        except Exception as e:
            logger.error(f"Failed to get forecast summary: {e}")
            return {"error": f"Failed to get forecast summary: {str(e)}"}

    def get_trends_24h(self) -> Dict[str, Any]:
        """Return CPU/memory usage trend data for the last 24 hours."""
        try:
            cutoff = datetime.utcnow() - timedelta(hours=24)
            cpu_series: List[Dict[str, Any]] = []
            mem_series: List[Dict[str, Any]] = []

            for item in list(self.metrics_history):
                ts = item.get("timestamp")
                if isinstance(ts, datetime):
                    if ts < cutoff:
                        continue
                    ts_iso = ts.isoformat()
                else:
                    ts_iso = str(ts) if ts else datetime.utcnow().isoformat()

                cpu_series.append({"t": ts_iso, "v": item.get("cpu_usage", 0)})
                mem_series.append({"t": ts_iso, "v": item.get("memory_usage", 0)})

            return self._to_json_safe({"cpu": cpu_series, "memory": mem_series})
        except Exception as exc:
            logger.warning(f"Failed to build 24h trends: {exc}")
            return {"cpu": [], "memory": []}
    
    def get_health_score(self) -> Dict[str, Any]:
        """Calculate overall cluster health score"""
        try:
            analysis = self.get_current_analysis()
            
            if "error" in analysis:
                return {"error": "Cannot calculate health score"}
            
            current_metrics = analysis.get("current_metrics", {})
            
            # Calculate health score (0-100)
            cpu_usage = current_metrics.get("cpu_usage", 0)
            memory_usage = current_metrics.get("memory_usage", 0)
            cpu_health = max(0, 100 - cpu_usage)
            memory_health = max(0, 100 - memory_usage)
            
            # Penalize for anomalies
            anomaly_penalty = 0
            anomaly_detection = analysis.get("anomaly_detection", {})
            if anomaly_detection.get("is_anomaly", False):
                severity_penalties = {"low": 5, "medium": 15, "high": 30, "critical": 50}
                anomaly_penalty = severity_penalties.get(anomaly_detection.get("severity", "low"), 10)
            
            # Calculate overall score
            overall_score = (cpu_health + memory_health) / 2 - anomaly_penalty
            overall_score = max(0, min(100, overall_score))
            
            # Determine health status
            if overall_score >= 80:
                status = "excellent"
            elif overall_score >= 60:
                status = "good"
            elif overall_score >= 40:
                status = "fair"
            elif overall_score >= 20:
                status = "poor"
            else:
                status = "critical"
            
            return {
                "overall_score": round(overall_score, 1),
                "status": status,
                "cpu_health": round(cpu_health, 1),
                "memory_health": round(memory_health, 1),
                "anomaly_penalty": anomaly_penalty,
                "timestamp": analysis.get("timestamp", datetime.now().isoformat())
            }
        except Exception as e:
            logger.error(f"Failed to get health score: {e}")
            return {"error": f"Failed to get health score: {str(e)}"}
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        analysis = self.get_current_analysis()
        
        # If analysis has an error field (cluster disconnected), return it directly
        if "error" in analysis:
            dashboard = {
                "timestamp": analysis.get("timestamp", datetime.now().isoformat()),
                "current_metrics": analysis.get("current_metrics", {}),
                "health_score": {"score": 0, "status": "disconnected", "message": "Cluster is not accessible"},
                "forecasts": analysis.get("forecasts", {}),
                "alerts": analysis.get("anomaly_detection", {}),
                "recommendations": analysis.get("recommendations", []),
                "rag_recommendations": [],
                "trends": [],
                "summary": analysis.get("summary", "Cluster Disconnected"),
                "demo_mode": True,
                "demo_message": analysis.get("demo_message", analysis.get("error", "Cluster is not accessible")),
                "error": analysis.get("error", "Cluster is not accessible")
            }
            return self._to_json_safe(dashboard)
        
        dashboard = {
            "timestamp": analysis.get("timestamp", datetime.now().isoformat()),
            "current_metrics": analysis.get("current_metrics", {}),
            "health_score": self.get_health_score(),
            "forecasts": self.get_forecast_summary(),
            "alerts": self.get_anomaly_alerts(),
            "recommendations": self.get_performance_recommendations(),
            "rag_recommendations": self.get_rag_enhanced_recommendations(),
            "trends": self.get_trends_24h(),
            "summary": analysis.get("summary", "AI monitoring analysis"),
            "demo_mode": analysis.get("demo_mode", False),
            "demo_message": analysis.get("demo_message", "")
        }
        # Include error field if present
        if "error" in analysis:
            dashboard["error"] = analysis["error"]
        return self._to_json_safe(dashboard)

# MCP Tool Integration Functions
def get_ai_insights() -> Dict[str, Any]:
    """MCP tool function to get AI insights"""
    try:
        integration = AIMonitoringIntegration()
        return integration.get_dashboard_data()
    except Exception as e:
        return {"error": f"Failed to get AI insights: {str(e)}"}

def get_anomaly_alerts() -> Dict[str, Any]:
    """MCP tool function to get anomaly alerts"""
    try:
        integration = AIMonitoringIntegration()
        alerts = integration.get_anomaly_alerts()
        return {
            "alerts": alerts,
            "count": len(alerts),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": f"Failed to get anomaly alerts: {str(e)}"}

def get_performance_recommendations() -> Dict[str, Any]:
    """MCP tool function to get performance recommendations"""
    try:
        integration = AIMonitoringIntegration()
        recommendations = integration.get_performance_recommendations()
        return {
            "recommendations": recommendations,
            "count": len(recommendations),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": f"Failed to get performance recommendations: {str(e)}"}

def get_capacity_forecast() -> Dict[str, Any]:
    """MCP tool function to get capacity forecasts"""
    try:
        integration = AIMonitoringIntegration()
        forecast = integration.get_forecast_summary()
        return forecast
    except Exception as e:
        return {"error": f"Failed to get capacity forecast: {str(e)}"}

def get_cluster_health() -> Dict[str, Any]:
    """MCP tool function to get cluster health score"""
    try:
        integration = AIMonitoringIntegration()
        health = integration.get_health_score()
        return health
    except Exception as e:
        return {"error": f"Failed to get cluster health: {str(e)}"}

def get_rag_recommendations() -> Dict[str, Any]:
    """MCP tool function to get RAG-enhanced recommendations"""
    try:
        integration = AIMonitoringIntegration()
        recommendations = integration.get_rag_enhanced_recommendations()
        return {
            "rag_recommendations": recommendations,
            "count": len(recommendations),
            "timestamp": datetime.now().isoformat(),
            "description": "RAG-enhanced intelligent recommendations based on Kubernetes best practices"
        }
    except Exception as e:
        return {"error": f"Failed to get RAG recommendations: {str(e)}"}

# Example usage
if __name__ == "__main__":
    integration = AIMonitoringIntegration()
    
    print("=== AI Monitoring Integration Test ===")
    
    # Get dashboard data
    dashboard_data = integration.get_dashboard_data()
    print(f"Dashboard Data: {json.dumps(dashboard_data, indent=2, default=str)}")
    
    # Test MCP functions
    print("\n=== MCP Tool Functions ===")
    print(f"AI Insights: {json.dumps(get_ai_insights(), indent=2, default=str)}")
    print(f"Anomaly Alerts: {json.dumps(get_anomaly_alerts(), indent=2, default=str)}")
    print(f"Performance Recommendations: {json.dumps(get_performance_recommendations(), indent=2, default=str)}")
    print(f"Capacity Forecast: {json.dumps(get_capacity_forecast(), indent=2, default=str)}")
    print(f"Cluster Health: {json.dumps(get_cluster_health(), indent=2, default=str)}")
