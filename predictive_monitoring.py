#!/usr/bin/env python3
"""
AI4K8s Predictive Monitoring System
===================================

This module implements AI-driven predictive monitoring capabilities including:
- Time series forecasting for resource usage patterns
- Anomaly detection using ML models
- Performance optimization recommendations
- Capacity planning and predictive scaling

Author: Pedram Nikjooy
Thesis: AI Agent for Kubernetes Management
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResourceMetrics:
    """Data class for resource metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    network_io: float
    disk_io: float
    pod_count: int
    node_count: int
    namespace: str = "default"

@dataclass
class AnomalyResult:
    """Data class for anomaly detection results"""
    timestamp: datetime
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: str
    affected_metrics: List[str]
    severity: str  # low, medium, high, critical
    recommendation: str

@dataclass
class ForecastResult:
    """Data class for forecasting results"""
    metric_name: str
    current_value: float
    predicted_values: List[float]
    confidence_intervals: List[Tuple[float, float]]
    trend: str  # increasing, decreasing, stable
    recommendation: str

class TimeSeriesForecaster:
    """Time series forecasting for resource usage patterns"""
    
    def __init__(self, window_size: int = 24):
        self.window_size = window_size
        self.history = []
        
    def add_data_point(self, metrics: ResourceMetrics):
        """Add a new data point to the time series"""
        self.history.append(metrics)
        # Keep only recent data
        if len(self.history) > self.window_size * 2:
            self.history = self.history[-self.window_size * 2:]
    
    def forecast_cpu_usage(self, hours_ahead: int = 6) -> ForecastResult:
        """Forecast CPU usage for the next N hours"""
        if len(self.history) < 10:
            return ForecastResult(
                metric_name="cpu_usage",
                current_value=0.0,
                predicted_values=[0.0] * hours_ahead,
                confidence_intervals=[(0.0, 0.0)] * hours_ahead,
                trend="insufficient_data",
                recommendation="Collect more data for accurate forecasting"
            )

        # Extract CPU usage data
        cpu_data = [m.cpu_usage for m in self.history[-self.window_size:]]
        current_cpu = cpu_data[-1]
        
        # Simple linear trend + seasonal component
        x = np.arange(len(cpu_data))
        trend = np.polyfit(x, cpu_data, 1)[0]
        
        # Calculate seasonal component (daily pattern)
        if len(cpu_data) >= 24:
            daily_pattern = np.mean([cpu_data[i::24] for i in range(24)], axis=1)
        else:
            daily_pattern = np.array(cpu_data)
        
        # Generate predictions
        predictions = []
        confidence_intervals = []
        
        for h in range(1, hours_ahead + 1):
            # Linear trend component
            trend_value = current_cpu + trend * h
            
            # Seasonal component
            hour_of_day = (datetime.now().hour + h) % 24
            if len(daily_pattern) > hour_of_day:
                seasonal_value = daily_pattern[hour_of_day]
            else:
                seasonal_value = np.mean(daily_pattern)
            
            # Combine trend and seasonal
            prediction = 0.7 * trend_value + 0.3 * seasonal_value
            prediction = max(0, min(100, prediction))  # Clamp to 0-100%
            
            predictions.append(prediction)
            
            # Simple confidence interval based on historical variance
            variance = np.var(cpu_data)
            confidence = 1.96 * np.sqrt(variance)  # 95% confidence
            confidence_intervals.append((max(0, prediction - confidence), 
                                       min(100, prediction + confidence)))
        
        # Determine trend
        if trend > 0.5:
            trend_str = "increasing"
            recommendation = "Consider scaling up resources or optimizing applications"
        elif trend < -0.5:
            trend_str = "decreasing"
            recommendation = "Resources may be over-provisioned"
        else:
            trend_str = "stable"
            recommendation = "Current resource allocation appears adequate"
        
        return ForecastResult(
            metric_name="cpu_usage",
            current_value=current_cpu,
            predicted_values=predictions,
            confidence_intervals=confidence_intervals,
            trend=trend_str,
            recommendation=recommendation
        )
    
    def forecast_memory_usage(self, hours_ahead: int = 6) -> ForecastResult:
        """Forecast memory usage for the next N hours"""
        if len(self.history) < 10:
            return ForecastResult(
                metric_name="memory_usage",
                current_value=0.0,
                predicted_values=[0.0] * hours_ahead,
                confidence_intervals=[(0.0, 0.0)] * hours_ahead,
                trend="insufficient_data",
                recommendation="Collect more data for accurate forecasting"
            )
        
        # Extract memory usage data
        memory_data = [m.memory_usage for m in self.history[-self.window_size:]]
        current_memory = memory_data[-1]
        
        # Simple exponential smoothing
        alpha = 0.3
        predictions = []
        confidence_intervals = []
        
        last_value = current_memory
        for h in range(1, hours_ahead + 1):
            # Exponential smoothing with trend
            trend = np.mean(np.diff(memory_data[-5:])) if len(memory_data) >= 5 else 0
            prediction = alpha * last_value + (1 - alpha) * (last_value + trend)
            prediction = max(0, min(100, prediction))
            
            predictions.append(prediction)
            last_value = prediction
            
            # Confidence interval
            variance = np.var(memory_data)
            confidence = 1.96 * np.sqrt(variance)
            confidence_intervals.append((max(0, prediction - confidence), 
                                       min(100, prediction + confidence)))
        
        # Determine trend
        recent_trend = np.mean(np.diff(memory_data[-5:])) if len(memory_data) >= 5 else 0
        if recent_trend > 1:
            trend_str = "increasing"
            recommendation = "Memory usage is growing - consider memory optimization or scaling"
        elif recent_trend < -1:
            trend_str = "decreasing"
            recommendation = "Memory usage is declining - resources may be over-provisioned"
        else:
            trend_str = "stable"
            recommendation = "Memory usage is stable"
        
        return ForecastResult(
            metric_name="memory_usage",
            current_value=current_memory,
            predicted_values=predictions,
            confidence_intervals=confidence_intervals,
            trend=trend_str,
            recommendation=recommendation
        )

class AnomalyDetector:
    """ML-based anomaly detection for Kubernetes metrics"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, metrics_history: List[ResourceMetrics]):
        """Train the anomaly detection models"""
        if len(metrics_history) < 20:
            logger.warning("Insufficient data for training anomaly detection models")
            return
        
        # Prepare feature matrix
        features = []
        for metrics in metrics_history:
            feature_vector = [
                metrics.cpu_usage,
                metrics.memory_usage,
                metrics.network_io,
                metrics.disk_io,
                metrics.pod_count,
                metrics.node_count
            ]
            features.append(feature_vector)
        
        features = np.array(features)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train models
        self.isolation_forest.fit(features_scaled)
        self.dbscan.fit(features_scaled)
        
        self.is_trained = True
        logger.info("Anomaly detection models trained successfully")
    
    def detect_anomaly(self, metrics: ResourceMetrics) -> AnomalyResult:
        """Detect anomalies in current metrics"""
        if not self.is_trained:
            return AnomalyResult(
                timestamp=metrics.timestamp,
                is_anomaly=False,
                anomaly_score=0.0,
                anomaly_type="model_not_trained",
                affected_metrics=[],
                severity="low",
                recommendation="Train anomaly detection models with historical data"
            )
        
        # Prepare feature vector
        feature_vector = np.array([[
            metrics.cpu_usage,
            metrics.memory_usage,
            metrics.network_io,
            metrics.disk_io,
            metrics.pod_count,
            metrics.node_count
        ]])
        
        # Scale features
        feature_scaled = self.scaler.transform(feature_vector)
        
        # Isolation Forest prediction
        isolation_prediction = self.isolation_forest.predict(feature_scaled)[0]
        isolation_score = self.isolation_forest.score_samples(feature_scaled)[0]
        
        # DBSCAN clustering
        cluster_label = self.dbscan.fit_predict(feature_scaled)[0]
        
        # Determine if anomaly
        is_anomaly = isolation_prediction == -1 or cluster_label == -1
        anomaly_score = abs(isolation_score)
        
        # Determine anomaly type and severity
        anomaly_type = "unknown"
        severity = "low"
        affected_metrics = []
        
        if is_anomaly:
            # Check which metrics are unusual
            if metrics.cpu_usage > 90:
                affected_metrics.append("cpu_usage")
                anomaly_type = "high_cpu_usage"
                severity = "high"
            elif metrics.cpu_usage < 5:
                affected_metrics.append("cpu_usage")
                anomaly_type = "low_cpu_usage"
                severity = "medium"
            
            if metrics.memory_usage > 90:
                affected_metrics.append("memory_usage")
                anomaly_type = "high_memory_usage"
                severity = "high"
            elif metrics.memory_usage < 5:
                affected_metrics.append("memory_usage")
                anomaly_type = "low_memory_usage"
                severity = "medium"
            
            if metrics.network_io > 1000:  # Assuming MB/s
                affected_metrics.append("network_io")
                anomaly_type = "high_network_io"
                severity = "medium"
            
            if metrics.disk_io > 1000:  # Assuming MB/s
                affected_metrics.append("disk_io")
                anomaly_type = "high_disk_io"
                severity = "medium"
            
            if not affected_metrics:
                anomaly_type = "statistical_anomaly"
                severity = "low"
                affected_metrics = ["multiple_metrics"]
        
        # Generate recommendation
        recommendation = self._generate_anomaly_recommendation(anomaly_type, severity, affected_metrics)
        
        return AnomalyResult(
            timestamp=metrics.timestamp,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomaly_type=anomaly_type,
            affected_metrics=affected_metrics,
            severity=severity,
            recommendation=recommendation
        )
    
    def _generate_anomaly_recommendation(self, anomaly_type: str, severity: str, affected_metrics: List[str]) -> str:
        """Generate recommendations based on anomaly type"""
        recommendations = {
            "high_cpu_usage": "Consider scaling up CPU resources or optimizing application performance",
            "low_cpu_usage": "CPU resources may be over-provisioned - consider right-sizing",
            "high_memory_usage": "Memory usage is critical - check for memory leaks or scale up memory",
            "low_memory_usage": "Memory resources may be over-provisioned",
            "high_network_io": "High network traffic detected - monitor for potential DDoS or data transfer issues",
            "high_disk_io": "High disk I/O detected - check for storage bottlenecks or I/O intensive operations",
            "statistical_anomaly": "Unusual pattern detected - investigate system behavior and monitor closely"
        }
        
        return recommendations.get(anomaly_type, "Monitor the system closely and investigate unusual behavior")

class PerformanceOptimizer:
    """AI-driven performance optimization recommendations"""
    
    def __init__(self):
        self.optimization_rules = {
            "cpu_optimization": {
                "high_usage_threshold": 80,
                "low_usage_threshold": 20,
                "recommendations": {
                    "high": "Scale up CPU resources or optimize application code",
                    "low": "Consider reducing CPU allocation to save costs"
                }
            },
            "memory_optimization": {
                "high_usage_threshold": 85,
                "low_usage_threshold": 15,
                "recommendations": {
                    "high": "Increase memory allocation or investigate memory leaks",
                    "low": "Reduce memory allocation to optimize resource usage"
                }
            },
            "network_optimization": {
                "high_usage_threshold": 800,
                "low_usage_threshold": 50,
                "recommendations": {
                    "high": "Optimize network configuration or implement traffic shaping",
                    "low": "Network usage is efficient"
                }
            }
        }
    
    def analyze_performance(self, metrics: ResourceMetrics, forecast: ForecastResult) -> Dict[str, Any]:
        """Analyze performance and provide optimization recommendations"""
        recommendations = []
        priority = "low"
        
        # CPU analysis
        if metrics.cpu_usage > self.optimization_rules["cpu_optimization"]["high_usage_threshold"]:
            recommendations.append({
                "type": "cpu_optimization",
                "priority": "high",
                "current_value": metrics.cpu_usage,
                "recommendation": self.optimization_rules["cpu_optimization"]["recommendations"]["high"],
                "action": "scale_up_cpu"
            })
            priority = "high"
        elif metrics.cpu_usage < self.optimization_rules["cpu_optimization"]["low_usage_threshold"]:
            recommendations.append({
                "type": "cpu_optimization",
                "priority": "medium",
                "current_value": metrics.cpu_usage,
                "recommendation": self.optimization_rules["cpu_optimization"]["recommendations"]["low"],
                "action": "scale_down_cpu"
            })
            if priority != "high":
                priority = "medium"
        
        # Memory analysis
        if metrics.memory_usage > self.optimization_rules["memory_optimization"]["high_usage_threshold"]:
            recommendations.append({
                "type": "memory_optimization",
                "priority": "high",
                "current_value": metrics.memory_usage,
                "recommendation": self.optimization_rules["memory_optimization"]["recommendations"]["high"],
                "action": "scale_up_memory"
            })
            priority = "high"
        elif metrics.memory_usage < self.optimization_rules["memory_optimization"]["low_usage_threshold"]:
            recommendations.append({
                "type": "memory_optimization",
                "priority": "medium",
                "current_value": metrics.memory_usage,
                "recommendation": self.optimization_rules["memory_optimization"]["recommendations"]["low"],
                "action": "scale_down_memory"
            })
            if priority != "high":
                priority = "medium"
        
        # Network analysis
        if metrics.network_io > self.optimization_rules["network_optimization"]["high_usage_threshold"]:
            recommendations.append({
                "type": "network_optimization",
                "priority": "medium",
                "current_value": metrics.network_io,
                "recommendation": self.optimization_rules["network_optimization"]["recommendations"]["high"],
                "action": "optimize_network"
            })
            if priority != "high":
                priority = "medium"
        
        # Forecast-based recommendations
        if forecast.trend == "increasing" and forecast.metric_name == "cpu_usage":
            recommendations.append({
                "type": "predictive_scaling",
                "priority": "medium",
                "current_value": forecast.current_value,
                "recommendation": f"CPU usage is trending upward. {forecast.recommendation}",
                "action": "prepare_for_scaling"
            })
        
        return {
            "timestamp": metrics.timestamp,
            "overall_priority": priority,
            "recommendations": recommendations,
            "summary": f"Found {len(recommendations)} optimization opportunities"
        }

class CapacityPlanner:
    """Predictive capacity planning and scaling recommendations"""
    
    def __init__(self):
        self.scaling_thresholds = {
            "cpu_scale_up": 75,
            "cpu_scale_down": 25,
            "memory_scale_up": 80,
            "memory_scale_down": 20,
            "pod_scale_up": 0.8,  # 80% of current capacity
            "pod_scale_down": 0.3  # 30% of current capacity
        }
    
    def plan_capacity(self, current_metrics: ResourceMetrics, 
                     cpu_forecast: ForecastResult, 
                     memory_forecast: ForecastResult) -> Dict[str, Any]:
        """Generate capacity planning recommendations"""
        
        recommendations = []
        
        # CPU capacity planning
        max_predicted_cpu = max(cpu_forecast.predicted_values) if cpu_forecast.predicted_values else current_metrics.cpu_usage
        
        if max_predicted_cpu > self.scaling_thresholds["cpu_scale_up"]:
            scale_factor = max_predicted_cpu / self.scaling_thresholds["cpu_scale_up"]
            recommendations.append({
                "resource": "cpu",
                "action": "scale_up",
                "current_usage": current_metrics.cpu_usage,
                "predicted_peak": max_predicted_cpu,
                "scale_factor": scale_factor,
                "recommendation": f"Scale up CPU resources by {scale_factor:.1f}x to handle predicted peak usage",
                "urgency": "high" if max_predicted_cpu > 90 else "medium"
            })
        elif max_predicted_cpu < self.scaling_thresholds["cpu_scale_down"] and max_predicted_cpu > 0:
            scale_factor = self.scaling_thresholds["cpu_scale_down"] / max_predicted_cpu
            recommendations.append({
                "resource": "cpu",
                "action": "scale_down",
                "current_usage": current_metrics.cpu_usage,
                "predicted_peak": max_predicted_cpu,
                "scale_factor": scale_factor,
                "recommendation": f"Consider scaling down CPU resources by {scale_factor:.1f}x to optimize costs",
                "urgency": "low"
            })
        
        # Memory capacity planning
        max_predicted_memory = max(memory_forecast.predicted_values) if memory_forecast.predicted_values else current_metrics.memory_usage
        
        if max_predicted_memory > self.scaling_thresholds["memory_scale_up"]:
            scale_factor = max_predicted_memory / self.scaling_thresholds["memory_scale_up"]
            recommendations.append({
                "resource": "memory",
                "action": "scale_up",
                "current_usage": current_metrics.memory_usage,
                "predicted_peak": max_predicted_memory,
                "scale_factor": scale_factor,
                "recommendation": f"Scale up memory resources by {scale_factor:.1f}x to handle predicted peak usage",
                "urgency": "high" if max_predicted_memory > 95 else "medium"
            })
        elif max_predicted_memory < self.scaling_thresholds["memory_scale_down"] and max_predicted_memory > 0:
            scale_factor = self.scaling_thresholds["memory_scale_down"] / max_predicted_memory
            recommendations.append({
                "resource": "memory",
                "action": "scale_down",
                "current_usage": current_metrics.memory_usage,
                "predicted_peak": max_predicted_memory,
                "scale_factor": scale_factor,
                "recommendation": f"Consider scaling down memory resources by {scale_factor:.1f}x to optimize costs",
                "urgency": "low"
            })
        
        # Pod scaling recommendations
        if current_metrics.pod_count > 0:
            pod_utilization = (current_metrics.cpu_usage + current_metrics.memory_usage) / 2
            
            if pod_utilization > 80:
                new_pod_count = int(current_metrics.pod_count * 1.5)
                recommendations.append({
                    "resource": "pods",
                    "action": "scale_up",
                    "current_count": current_metrics.pod_count,
                    "recommended_count": new_pod_count,
                    "recommendation": f"Increase pod count from {current_metrics.pod_count} to {new_pod_count} to handle current load",
                    "urgency": "high"
                })
            elif pod_utilization < 30:
                new_pod_count = max(1, int(current_metrics.pod_count * 0.7))
                recommendations.append({
                    "resource": "pods",
                    "action": "scale_down",
                    "current_count": current_metrics.pod_count,
                    "recommended_count": new_pod_count,
                    "recommendation": f"Consider reducing pod count from {current_metrics.pod_count} to {new_pod_count} to optimize resources",
                    "urgency": "low"
                })
        
        return {
            "timestamp": current_metrics.timestamp,
            "recommendations": recommendations,
            "summary": f"Generated {len(recommendations)} capacity planning recommendations",
            "overall_urgency": "high" if any(r.get("urgency") == "high" for r in recommendations) else "medium" if recommendations else "low"
        }

class PredictiveMonitoringSystem:
    """Main predictive monitoring system that orchestrates all components"""
    
    def __init__(self):
        self.forecaster = TimeSeriesForecaster()
        self.anomaly_detector = AnomalyDetector()
        self.performance_optimizer = PerformanceOptimizer()
        self.capacity_planner = CapacityPlanner()
        self.metrics_history = []
        
    def add_metrics(self, metrics: ResourceMetrics):
        """Add new metrics and update all models"""
        self.metrics_history.append(metrics)
        self.forecaster.add_data_point(metrics)
        
        # Train anomaly detector if we have enough data
        if len(self.metrics_history) >= 20 and not self.anomaly_detector.is_trained:
            self.anomaly_detector.train(self.metrics_history)
    
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive analysis of current metrics"""
        if not self.metrics_history:
            return {"error": "No metrics data available"}
        
        current_metrics = self.metrics_history[-1]
        
        # Generate forecasts
        cpu_forecast = self.forecaster.forecast_cpu_usage()
        memory_forecast = self.forecaster.forecast_memory_usage()
        
        # Detect anomalies
        anomaly_result = self.anomaly_detector.detect_anomaly(current_metrics)
        
        # Performance optimization
        performance_analysis = self.performance_optimizer.analyze_performance(current_metrics, cpu_forecast)
        
        # Capacity planning
        capacity_plan = self.capacity_planner.plan_capacity(current_metrics, cpu_forecast, memory_forecast)
        
        return {
            "timestamp": current_metrics.timestamp,
            "current_metrics": {
                "cpu_usage": current_metrics.cpu_usage,
                "memory_usage": current_metrics.memory_usage,
                "network_io": current_metrics.network_io,
                "disk_io": current_metrics.disk_io,
                "pod_count": current_metrics.pod_count,
                "node_count": current_metrics.node_count
            },
            "forecasts": {
                "cpu": {
                    "current": cpu_forecast.current_value,
                    "predicted": cpu_forecast.predicted_values,
                    "trend": cpu_forecast.trend,
                    "recommendation": cpu_forecast.recommendation
                },
                "memory": {
                    "current": memory_forecast.current_value,
                    "predicted": memory_forecast.predicted_values,
                    "trend": memory_forecast.trend,
                    "recommendation": memory_forecast.recommendation
                }
            },
            "anomaly_detection": {
                "is_anomaly": anomaly_result.is_anomaly,
                "anomaly_score": anomaly_result.anomaly_score,
                "anomaly_type": anomaly_result.anomaly_type,
                "severity": anomaly_result.severity,
                "affected_metrics": anomaly_result.affected_metrics,
                "recommendation": anomaly_result.recommendation
            },
            "performance_optimization": performance_analysis,
            "capacity_planning": capacity_plan,
            "summary": {
                "total_recommendations": len(performance_analysis.get("recommendations", [])) + len(capacity_plan.get("recommendations", [])),
                "has_anomalies": anomaly_result.is_anomaly,
                "overall_priority": max(performance_analysis.get("overall_priority", "low"), 
                                      capacity_plan.get("overall_urgency", "low"))
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Create monitoring system
    monitoring = PredictiveMonitoringSystem()
    
    # Simulate some metrics data
    import random
    from datetime import datetime, timedelta
    
    base_time = datetime.now()
    for i in range(30):
        metrics = ResourceMetrics(
            timestamp=base_time + timedelta(hours=i),
            cpu_usage=random.uniform(20, 80),
            memory_usage=random.uniform(30, 70),
            network_io=random.uniform(100, 500),
            disk_io=random.uniform(50, 200),
            pod_count=random.randint(5, 15),
            node_count=3
        )
        monitoring.add_metrics(metrics)
    
    # Perform analysis
    analysis = monitoring.analyze()
    
    print("=== AI4K8s Predictive Monitoring Analysis ===")
    print(json.dumps(analysis, indent=2, default=str))
