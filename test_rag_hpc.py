#!/usr/bin/env python3
"""
Test RAG system on HPC with live EKS data
"""

from ai_monitoring_integration import AIMonitoringIntegration
import json

def test_rag_system():
    # Initialize monitoring integration with RAG
    integration = AIMonitoringIntegration()

    print("Testing RAG-enhanced recommendations with live EKS data...")
    rag_recommendations = integration.get_rag_enhanced_recommendations()

    print(f"RAG recommendations count: {len(rag_recommendations)}")
    for i, rec in enumerate(rag_recommendations):
        message = rec.get("message", "No message")
        priority = rec.get("priority", "unknown")
        print(f"{i+1}. {message} (Priority: {priority})")
        
        details = rec.get("details", {})
        if "rag_context" in details:
            rag_context = details["rag_context"]
            print(f"   RAG Context: {rag_context}")

    print("\nTesting dashboard data with RAG...")
    dashboard_data = integration.get_dashboard_data()
    has_rag = "rag_recommendations" in dashboard_data
    print(f"Dashboard includes RAG recommendations: {has_rag}")
    
    if has_rag:
        rag_count = len(dashboard_data["rag_recommendations"])
        print(f"RAG recommendations in dashboard: {rag_count}")
        current_metrics = dashboard_data.get("current_metrics", {})
        print(f"Current metrics: {current_metrics}")

if __name__ == "__main__":
    test_rag_system()
