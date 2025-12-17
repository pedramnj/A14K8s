#!/usr/bin/env python3
"""Test script to check API response for target_replicas"""

import requests
import json
import sys
import os

# Test the API endpoint
deployment_name = "test-app-autoscaling"
namespace = "ai4k8s-test"
server_id = 3  # Adjust based on your server ID

# Try to get the port from environment or use default
port = os.environ.get('FLASK_PORT', '5000')
url = f"http://localhost:{port}/api/autoscaling/recommendations/{server_id}?deployment={deployment_name}&namespace={namespace}"

print(f"Testing API endpoint: {url}")
print("=" * 80)

try:
    # Create a session to handle cookies if needed
    session = requests.Session()
    
    # Try to get a session first (login if needed)
    # For testing, we'll try without auth first
    response = session.get(url, timeout=30)
    
    print(f"Response Status Code: {response.status_code}")
    print("=" * 80)
    
    if response.status_code == 200:
        data = response.json()
        print("✅ API Response:")
        print(json.dumps(data, indent=2))
        
        if data.get('predictive') and data['predictive'].get('recommendation'):
            rec = data['predictive']['recommendation']
            target_replicas = rec.get('target_replicas')
            action = rec.get('action')
            scaling_type = rec.get('scaling_type')
            min_replicas = rec.get('min_replicas')
            max_replicas = rec.get('max_replicas')
            
            print("\n" + "=" * 80)
            print("RECOMMENDATION SUMMARY:")
            print(f"  Target Replicas: {target_replicas}")
            print(f"  Action: {action}")
            print(f"  Scaling Type: {scaling_type}")
            if min_replicas:
                print(f"  Min Replicas: {min_replicas}")
            if max_replicas:
                print(f"  Max Replicas: {max_replicas}")
            print("=" * 80)
            
            # Check if target_replicas exceeds max (assuming max is 5 based on UI)
            if target_replicas is not None:
                if max_replicas and target_replicas > max_replicas:
                    print(f"❌ ERROR: target_replicas={target_replicas} exceeds max_replicas={max_replicas}")
                elif target_replicas > 5:
                    print(f"⚠️  WARNING: target_replicas={target_replicas} might exceed expected max=5 (max_replicas not in response)")
                else:
                    print(f"✅ OK: target_replicas={target_replicas} is within bounds")
            else:
                print("⚠️  WARNING: target_replicas is None")
        else:
            print("⚠️  No recommendation found in response")
            if data.get('error'):
                print(f"Error: {data['error']}")
    elif response.status_code == 401:
        print("❌ Unauthorized - API requires authentication")
        print("Response:", response.text)
    else:
        print(f"❌ API Error: {response.status_code}")
        print("Response:", response.text)
        
except requests.exceptions.ConnectionError as e:
    print(f"❌ Connection Error: {e}")
    print("Make sure the Flask app is running on the specified port")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

