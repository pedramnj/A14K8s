#!/usr/bin/env python3
"""Test script to directly check API response for target_replicas by calling Python functions"""

import sys
import os
import json

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from autoscaling_integration import AutoscalingIntegration
    
    deployment_name = "test-app-autoscaling"
    namespace = "ai4k8s-test"
    
    print("=" * 80)
    print(f"Testing direct API call for: {deployment_name} in {namespace}")
    print("=" * 80)
    
    # Initialize autoscaling integration
    print("Initializing AutoscalingIntegration...")
    autoscaling = AutoscalingIntegration()
    
    if not autoscaling:
        print("❌ Failed to initialize AutoscalingIntegration")
        sys.exit(1)
    
    print("✅ AutoscalingIntegration initialized")
    print("=" * 80)
    
    # Get scaling recommendations
    print(f"Calling get_scaling_recommendations for {deployment_name}...")
    
    # Also try calling get_scaling_recommendation directly
    print(f"\nAlso calling get_scaling_recommendation directly...")
    direct_result = autoscaling.predictive_autoscaler.get_scaling_recommendation(deployment_name, namespace)
    print(f"Direct result success: {direct_result.get('success')}")
    if not direct_result.get('success'):
        print(f"Direct result error: {direct_result.get('error')}")
    else:
        print(f"Direct result recommendation: {json.dumps(direct_result.get('recommendation', {}), indent=2, default=str)}")
    
    result = autoscaling.get_scaling_recommendations(deployment_name, namespace)
    
    print("\n" + "=" * 80)
    print("RAW API RESPONSE:")
    print("=" * 80)
    print(json.dumps(result, indent=2, default=str))
    
    if result.get('predictive') and result['predictive'].get('recommendation'):
        rec = result['predictive']['recommendation']
        target_replicas = rec.get('target_replicas')
        action = rec.get('action')
        scaling_type = rec.get('scaling_type')
        
        print("\n" + "=" * 80)
        print("RECOMMENDATION SUMMARY:")
        print("=" * 80)
        print(f"  Target Replicas: {target_replicas}")
        print(f"  Action: {action}")
        print(f"  Scaling Type: {scaling_type}")
        print("=" * 80)
        
        # Check if target_replicas exceeds max (assuming max is 5 based on UI)
        if target_replicas is not None:
            if target_replicas > 5:
                print(f"\n❌ ERROR: target_replicas={target_replicas} exceeds expected max=5")
                print("   This indicates the validation is not working correctly!")
            else:
                print(f"\n✅ OK: target_replicas={target_replicas} is within expected bounds (<=5)")
        else:
            print("\n⚠️  WARNING: target_replicas is None")
    else:
        print("\n⚠️  No recommendation found in response")
        if result.get('error'):
            print(f"Error: {result['error']}")
    
    print("\n" + "=" * 80)
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running this from the ai4k8s directory")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

