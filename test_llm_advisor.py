#!/usr/bin/env python3
"""Test LLM Autoscaling Advisor"""

import sys
import os
sys.path.insert(0, '.')

from autoscaling_integration import AutoscalingIntegration

# Set up environment
kubeconfig_path = os.path.expanduser('~/crownlabs-k3s.yaml')
os.environ['KUBECONFIG'] = kubeconfig_path
groq_key = os.getenv('GROQ_API_KEY', 'sk_Yoif9bMnfFo6JVGioRtGWGdyb3FYhYujAFLl7fyjQBEXJWkEiW6A6A')
os.environ['GROQ_API_KEY'] = groq_key

print("🧪 Testing LLM Autoscaling Advisor...")
print(f"Kubeconfig: {kubeconfig_path}")
print(f"GROQ_API_KEY: {'Set' if groq_key else 'Not set'}")

try:
    integration = AutoscalingIntegration(kubeconfig_path)
    print("✅ Autoscaling integration initialized")
    print(f"LLM Advisor available: {integration.llm_advisor.client is not None if integration.llm_advisor else False}")
    
    # Test getting LLM recommendation
    print("\n📊 Getting scaling recommendations for test-app-autoscaling...")
    result = integration.get_scaling_recommendations('test-app-autoscaling', 'default')
    
    if result.get('success'):
        recommendations = result.get('recommendations', {})
        
        # Check LLM recommendation
        llm_rec = recommendations.get('llm')
        if llm_rec and llm_rec.get('success'):
            rec = llm_rec.get('recommendation', {})
            print("\n✅ LLM Recommendation Received!")
            print(f"  Action: {rec.get('action')}")
            print(f"  Target Replicas: {rec.get('target_replicas')}")
            print(f"  Confidence: {rec.get('confidence', 0)*100:.0f}%")
            print(f"  Risk: {rec.get('risk_assessment', 'unknown')}")
            print(f"  Cost Impact: {rec.get('cost_impact', 'unknown')}")
            print(f"  Performance Impact: {rec.get('performance_impact', 'unknown')}")
            reasoning = rec.get('reasoning', '')
            if reasoning:
                print(f"\n  Reasoning:\n  {reasoning[:300]}...")
        else:
            # Check if LLM was used in predictive
            pred = recommendations.get('predictive')
            if pred and pred.get('llm_used'):
                rec = pred.get('recommendation', {})
                llm_rec_data = rec.get('llm_recommendation', {})
                if llm_rec_data:
                    print("\n✅ LLM Recommendation (via Predictive Autoscaler)!")
                    print(f"  Action: {llm_rec_data.get('action')}")
                    print(f"  Target Replicas: {llm_rec_data.get('target_replicas')}")
                    print(f"  Confidence: {llm_rec_data.get('confidence', 0)*100:.0f}%")
                    reasoning = llm_rec_data.get('reasoning', '')
                    if reasoning:
                        print(f"\n  Reasoning:\n  {reasoning[:300]}...")
                else:
                    print("\n⚠️  Predictive recommendation available but no LLM data")
            else:
                print("\n⚠️  Using rule-based recommendations")
                print("   (LLM may be processing or unavailable)")
    else:
        print(f"\n❌ Error: {result.get('error')}")
        
except Exception as e:
    import traceback
    print(f"\n❌ Error: {e}")
    traceback.print_exc()

