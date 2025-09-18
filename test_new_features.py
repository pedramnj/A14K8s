#!/usr/bin/env python3
"""
Test script for the new pod features
"""

import requests
import json

def test_pod_features():
    """Test the new pod features"""
    base_url = "http://localhost:5001/api/chat"
    
    # Test cases
    test_cases = [
        {
            "name": "Pod Top - All Pods",
            "message": "Show me resource usage for all pods",
            "expected_keywords": ["top", "resource", "cpu", "memory"]
        },
        {
            "name": "Pod Exec",
            "message": "Execute 'echo hello world' in the nginx pod",
            "expected_keywords": ["exec", "echo", "hello"]
        },
        {
            "name": "Run Container",
            "message": "Run a busybox container with name test-busybox",
            "expected_keywords": ["run", "busybox", "container"]
        }
    ]
    
    print("🧪 Testing New Pod Features")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Message: {test_case['message']}")
        
        try:
            response = requests.post(
                base_url,
                json={"message": test_case["message"]},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Status: {result.get('status', 'unknown')}")
                print(f"   📝 Response: {result.get('response', 'No response')[:100]}...")
                
                # Check for expected keywords
                response_text = result.get('response', '').lower()
                found_keywords = [kw for kw in test_case['expected_keywords'] if kw in response_text]
                print(f"   🔍 Found keywords: {found_keywords}")
                
            else:
                print(f"   ❌ Error: HTTP {response.status_code}")
                print(f"   📝 Response: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🏁 Testing Complete")

if __name__ == "__main__":
    test_pod_features()
