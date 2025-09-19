#!/usr/bin/env python3
"""
Test Script for Predictive Monitoring System
===========================================

This script tests the AI-powered predictive monitoring capabilities.

Author: Pedram Nikjooy
Thesis: AI Agent for Kubernetes Management
"""

import sys
import json
import logging
from datetime import datetime, timedelta
import random
import time

# Add current directory to path for imports
sys.path.append('.')

from predictive_monitoring import (
    PredictiveMonitoringSystem, 
    ResourceMetrics,
    TimeSeriesForecaster,
    AnomalyDetector,
    PerformanceOptimizer,
    CapacityPlanner
)
from ai_monitoring_integration import AIMonitoringIntegration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_metrics(num_points: int = 50) -> list:
    """Generate synthetic metrics data for testing"""
    metrics = []
    base_time = datetime.now() - timedelta(hours=num_points)
    
    for i in range(num_points):
        # Generate realistic patterns
        hour = (base_time + timedelta(hours=i)).hour
        
        # CPU usage with daily pattern (higher during business hours)
        if 9 <= hour <= 17:
            cpu_base = 60
            cpu_variance = 20
        else:
            cpu_base = 30
            cpu_variance = 15
        
        cpu_usage = max(0, min(100, cpu_base + random.uniform(-cpu_variance, cpu_variance)))
        
        # Memory usage with gradual increase
        memory_base = 40 + (i * 0.5)  # Gradual increase
        memory_usage = max(0, min(100, memory_base + random.uniform(-10, 10)))
        
        # Network I/O with some spikes
        network_io = random.uniform(100, 500)
        if random.random() < 0.1:  # 10% chance of spike
            network_io *= 3
        
        # Disk I/O
        disk_io = random.uniform(50, 200)
        
        # Pod count with some variation
        pod_count = random.randint(8, 15)
        
        metric = ResourceMetrics(
            timestamp=base_time + timedelta(hours=i),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            network_io=network_io,
            disk_io=disk_io,
            pod_count=pod_count,
            node_count=3
        )
        metrics.append(metric)
    
    return metrics

def test_time_series_forecasting():
    """Test time series forecasting capabilities"""
    logger.info("=== Testing Time Series Forecasting ===")
    
    forecaster = TimeSeriesForecaster()
    
    # Generate synthetic data
    metrics = generate_synthetic_metrics(30)
    
    # Add data to forecaster
    for metric in metrics:
        forecaster.add_data_point(metric)
    
    # Test CPU forecasting
    cpu_forecast = forecaster.forecast_cpu_usage(hours_ahead=6)
    logger.info(f"CPU Forecast: {cpu_forecast.trend} trend")
    logger.info(f"Current CPU: {cpu_forecast.current_value:.1f}%")
    logger.info(f"Next 6 hours: {[f'{v:.1f}%' for v in cpu_forecast.predicted_values]}")
    logger.info(f"Recommendation: {cpu_forecast.recommendation}")
    
    # Test Memory forecasting
    memory_forecast = forecaster.forecast_memory_usage(hours_ahead=6)
    logger.info(f"Memory Forecast: {memory_forecast.trend} trend")
    logger.info(f"Current Memory: {memory_forecast.current_value:.1f}%")
    logger.info(f"Next 6 hours: {[f'{v:.1f}%' for v in memory_forecast.predicted_values]}")
    logger.info(f"Recommendation: {memory_forecast.recommendation}")
    
    return cpu_forecast, memory_forecast

def test_anomaly_detection():
    """Test anomaly detection capabilities"""
    logger.info("=== Testing Anomaly Detection ===")
    
    detector = AnomalyDetector()
    
    # Generate training data
    training_metrics = generate_synthetic_metrics(25)
    
    # Train the detector
    detector.train(training_metrics)
    logger.info("Anomaly detector trained successfully")
    
    # Test normal metrics
    normal_metric = ResourceMetrics(
        timestamp=datetime.now(),
        cpu_usage=45.0,
        memory_usage=55.0,
        network_io=300.0,
        disk_io=150.0,
        pod_count=12,
        node_count=3
    )
    
    normal_result = detector.detect_anomaly(normal_metric)
    logger.info(f"Normal metric - Anomaly: {normal_result.is_anomaly}, Score: {normal_result.anomaly_score:.3f}")
    
    # Test anomalous metrics
    anomalous_metric = ResourceMetrics(
        timestamp=datetime.now(),
        cpu_usage=95.0,  # Very high CPU
        memory_usage=90.0,  # Very high memory
        network_io=2000.0,  # Very high network I/O
        disk_io=1000.0,  # Very high disk I/O
        pod_count=12,
        node_count=3
    )
    
    anomaly_result = detector.detect_anomaly(anomalous_metric)
    logger.info(f"Anomalous metric - Anomaly: {anomaly_result.is_anomaly}, Score: {anomaly_result.anomaly_score:.3f}")
    logger.info(f"Anomaly Type: {anomaly_result.anomaly_type}, Severity: {anomaly_result.severity}")
    logger.info(f"Affected Metrics: {anomaly_result.affected_metrics}")
    logger.info(f"Recommendation: {anomaly_result.recommendation}")
    
    return normal_result, anomaly_result

def test_performance_optimization():
    """Test performance optimization recommendations"""
    logger.info("=== Testing Performance Optimization ===")
    
    optimizer = PerformanceOptimizer()
    
    # Test high CPU scenario
    high_cpu_metric = ResourceMetrics(
        timestamp=datetime.now(),
        cpu_usage=85.0,
        memory_usage=60.0,
        network_io=400.0,
        disk_io=200.0,
        pod_count=15,
        node_count=3
    )
    
    # Create a mock forecast
    from predictive_monitoring import ForecastResult
    mock_forecast = ForecastResult(
        metric_name="cpu_usage",
        current_value=85.0,
        predicted_values=[87.0, 89.0, 91.0],
        confidence_intervals=[(80.0, 95.0), (82.0, 97.0), (84.0, 99.0)],
        trend="increasing",
        recommendation="Consider scaling up CPU resources"
    )
    
    optimization_result = optimizer.analyze_performance(high_cpu_metric, mock_forecast)
    logger.info(f"Performance Analysis Priority: {optimization_result['overall_priority']}")
    logger.info(f"Number of Recommendations: {len(optimization_result['recommendations'])}")
    
    for rec in optimization_result['recommendations']:
        logger.info(f"- {rec['type']}: {rec['recommendation']} (Priority: {rec['priority']})")
    
    return optimization_result

def test_capacity_planning():
    """Test capacity planning recommendations"""
    logger.info("=== Testing Capacity Planning ===")
    
    planner = CapacityPlanner()
    
    # Test scenario with high predicted usage
    current_metric = ResourceMetrics(
        timestamp=datetime.now(),
        cpu_usage=70.0,
        memory_usage=75.0,
        network_io=500.0,
        disk_io=250.0,
        pod_count=12,
        node_count=3
    )
    
    # Create forecasts with increasing trends
    from predictive_monitoring import ForecastResult
    cpu_forecast = ForecastResult(
        metric_name="cpu_usage",
        current_value=70.0,
        predicted_values=[75.0, 80.0, 85.0, 90.0, 95.0, 98.0],
        confidence_intervals=[(70.0, 80.0), (75.0, 85.0), (80.0, 90.0), (85.0, 95.0), (90.0, 100.0), (95.0, 100.0)],
        trend="increasing",
        recommendation="Scale up CPU resources"
    )
    
    memory_forecast = ForecastResult(
        metric_name="memory_usage",
        current_value=75.0,
        predicted_values=[78.0, 82.0, 86.0, 90.0, 94.0, 97.0],
        confidence_intervals=[(70.0, 85.0), (75.0, 90.0), (80.0, 95.0), (85.0, 100.0), (90.0, 100.0), (95.0, 100.0)],
        trend="increasing",
        recommendation="Scale up memory resources"
    )
    
    capacity_plan = planner.plan_capacity(current_metric, cpu_forecast, memory_forecast)
    logger.info(f"Capacity Planning Urgency: {capacity_plan['overall_urgency']}")
    logger.info(f"Number of Recommendations: {len(capacity_plan['recommendations'])}")
    
    for rec in capacity_plan['recommendations']:
        logger.info(f"- {rec['resource']} {rec['action']}: {rec['recommendation']} (Urgency: {rec['urgency']})")
    
    return capacity_plan

def test_full_monitoring_system():
    """Test the complete monitoring system"""
    logger.info("=== Testing Full Monitoring System ===")
    
    monitoring = PredictiveMonitoringSystem()
    
    # Add synthetic data
    metrics = generate_synthetic_metrics(40)
    for metric in metrics:
        monitoring.add_metrics(metric)
    
    # Perform comprehensive analysis
    analysis = monitoring.analyze()
    
    logger.info("=== Comprehensive Analysis Results ===")
    logger.info(f"Current CPU: {analysis['current_metrics']['cpu_usage']:.1f}%")
    logger.info(f"Current Memory: {analysis['current_metrics']['memory_usage']:.1f}%")
    logger.info(f"Pod Count: {analysis['current_metrics']['pod_count']}")
    
    logger.info(f"CPU Trend: {analysis['forecasts']['cpu']['trend']}")
    logger.info(f"Memory Trend: {analysis['forecasts']['memory']['trend']}")
    
    logger.info(f"Anomaly Detected: {analysis['anomaly_detection']['is_anomaly']}")
    if analysis['anomaly_detection']['is_anomaly']:
        logger.info(f"Anomaly Type: {analysis['anomaly_detection']['anomaly_type']}")
        logger.info(f"Severity: {analysis['anomaly_detection']['severity']}")
    
    logger.info(f"Performance Priority: {analysis['performance_optimization']['overall_priority']}")
    logger.info(f"Capacity Urgency: {analysis['capacity_planning']['overall_urgency']}")
    
    logger.info(f"Total Recommendations: {analysis['summary']['total_recommendations']}")
    
    return analysis

def test_ai_integration():
    """Test AI monitoring integration"""
    logger.info("=== Testing AI Monitoring Integration ===")
    
    try:
        integration = AIMonitoringIntegration()
        
        # Test dashboard data (this will try to connect to real K8s cluster)
        logger.info("Attempting to get dashboard data...")
        dashboard_data = integration.get_dashboard_data()
        
        if "error" in dashboard_data:
            logger.warning(f"Could not get real cluster data: {dashboard_data['error']}")
            logger.info("This is expected if not running in a Kubernetes cluster")
        else:
            logger.info("Successfully retrieved dashboard data from real cluster")
            logger.info(f"Health Score: {dashboard_data['health_score']['overall_score']}")
            logger.info(f"Status: {dashboard_data['health_score']['status']}")
        
        # Test MCP functions
        logger.info("Testing MCP tool functions...")
        
        insights = integration.get_current_analysis()
        logger.info(f"AI Insights available: {'error' not in insights}")
        
        alerts = integration.get_anomaly_alerts()
        logger.info(f"Anomaly alerts: {len(alerts)}")
        
        recommendations = integration.get_performance_recommendations()
        logger.info(f"Performance recommendations: {len(recommendations)}")
        
        health = integration.get_health_score()
        logger.info(f"Health score available: {'error' not in health}")
        
    except Exception as e:
        logger.error(f"AI integration test failed: {e}")
        logger.info("This is expected if not running in a Kubernetes cluster")

def main():
    """Run all tests"""
    logger.info("Starting Predictive Monitoring System Tests")
    logger.info("=" * 60)
    
    try:
        # Test individual components
        test_time_series_forecasting()
        print()
        
        test_anomaly_detection()
        print()
        
        test_performance_optimization()
        print()
        
        test_capacity_planning()
        print()
        
        test_full_monitoring_system()
        print()
        
        test_ai_integration()
        print()
        
        logger.info("=" * 60)
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
