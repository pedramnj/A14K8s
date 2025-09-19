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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
import time
from dataclasses import asdict

from predictive_monitoring import (
    PredictiveMonitoringSystem, 
    ResourceMetrics, 
    AnomalyResult,
    ForecastResult
)
from k8s_metrics_collector import KubernetesMetricsCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIMonitoringIntegration:
    """Integrates AI monitoring with the web application"""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        self.monitoring_system = PredictiveMonitoringSystem()
        self.metrics_collector = KubernetesMetricsCollector(kubeconfig_path)
        self.is_running = False
        self.collection_thread = None
        self.collection_interval = 300  # 5 minutes
        self.last_analysis = None
        
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
                    
                    logger.info("Metrics collected and analyzed successfully")
                else:
                    logger.error(f"Failed to collect metrics: {metrics_data['error']}")
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Wait for next collection
            time.sleep(self.collection_interval)
    
    def get_current_analysis(self) -> Dict[str, Any]:
        """Get the latest analysis results"""
        if self.last_analysis:
            return self.last_analysis
        
        # Perform one-time analysis if no continuous monitoring
        try:
            metrics_data = self.metrics_collector.get_aggregated_metrics()
            if "error" not in metrics_data:
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
                
                self.monitoring_system.add_metrics(resource_metrics)
                return self.monitoring_system.analyze()
        except Exception as e:
            logger.error(f"Failed to get current analysis: {e}")
        
        return {"error": "No analysis available"}
    
    def get_anomaly_alerts(self) -> List[Dict[str, Any]]:
        """Get current anomaly alerts"""
        analysis = self.get_current_analysis()
        
        alerts = []
        if "anomaly_detection" in analysis and analysis["anomaly_detection"]["is_anomaly"]:
            anomaly = analysis["anomaly_detection"]
            alerts.append({
                "type": "anomaly",
                "severity": anomaly["severity"],
                "message": f"Anomaly detected: {anomaly['anomaly_type']}",
                "details": anomaly,
                "timestamp": analysis["timestamp"]
            })
        
        return alerts
    
    def get_performance_recommendations(self) -> List[Dict[str, Any]]:
        """Get performance optimization recommendations"""
        analysis = self.get_current_analysis()
        
        recommendations = []
        
        # Add performance optimization recommendations
        if "performance_optimization" in analysis:
            perf_recs = analysis["performance_optimization"].get("recommendations", [])
            for rec in perf_recs:
                recommendations.append({
                    "type": "performance",
                    "priority": rec["priority"],
                    "message": rec["recommendation"],
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
                    "message": rec["recommendation"],
                    "action": rec["action"],
                    "details": rec
                })
        
        return recommendations
    
    def get_forecast_summary(self) -> Dict[str, Any]:
        """Get a summary of forecasts"""
        analysis = self.get_current_analysis()
        
        if "forecasts" not in analysis:
            return {"error": "No forecast data available"}
        
        forecasts = analysis["forecasts"]
        summary = {
            "timestamp": analysis["timestamp"],
            "cpu_forecast": {
                "current": forecasts["cpu"]["current"],
                "trend": forecasts["cpu"]["trend"],
                "next_hour_prediction": forecasts["cpu"]["predicted"][0] if forecasts["cpu"]["predicted"] else None,
                "recommendation": forecasts["cpu"]["recommendation"]
            },
            "memory_forecast": {
                "current": forecasts["memory"]["current"],
                "trend": forecasts["memory"]["trend"],
                "next_hour_prediction": forecasts["memory"]["predicted"][0] if forecasts["memory"]["predicted"] else None,
                "recommendation": forecasts["memory"]["recommendation"]
            }
        }
        
        return summary
    
    def get_health_score(self) -> Dict[str, Any]:
        """Calculate overall cluster health score"""
        analysis = self.get_current_analysis()
        
        if "error" in analysis:
            return {"error": "Cannot calculate health score"}
        
        current_metrics = analysis["current_metrics"]
        
        # Calculate health score (0-100)
        cpu_health = max(0, 100 - current_metrics["cpu_usage"])
        memory_health = max(0, 100 - current_metrics["memory_usage"])
        
        # Penalize for anomalies
        anomaly_penalty = 0
        if analysis["anomaly_detection"]["is_anomaly"]:
            severity_penalties = {"low": 5, "medium": 15, "high": 30, "critical": 50}
            anomaly_penalty = severity_penalties.get(analysis["anomaly_detection"]["severity"], 10)
        
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
            "timestamp": analysis["timestamp"]
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        analysis = self.get_current_analysis()
        
        if "error" in analysis:
            return {"error": "No analysis data available"}
        
        return {
            "timestamp": analysis["timestamp"],
            "current_metrics": analysis["current_metrics"],
            "health_score": self.get_health_score(),
            "forecasts": self.get_forecast_summary(),
            "alerts": self.get_anomaly_alerts(),
            "recommendations": self.get_performance_recommendations(),
            "summary": analysis["summary"]
        }

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
