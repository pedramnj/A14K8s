#!/usr/bin/env python3
"""
Test the AI-powered processing
"""

import requests
import json

# Test the intelligent AI processing
def test_ai_processing():
    print("🧪 Testing AI-Powered Natural Language Processing\n")
    
    # Test queries that should now work with AI
    test_queries = [
        "can you today show me how is my cluster?",
        "what pods are running?",
        "show me the health of my cluster",
        "create a pod name it test-ai-pod",
        "delete the test-ai-pod pod"
    ]
    
    for query in test_queries:
        print(f"Query: '{query}'")
        try:
            response = requests.post(
                'http://localhost:5003/api/chat/2',
                json={'message': query},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Status: {result.get('status', 'unknown')}")
                print(f"🤖 AI Processed: {result.get('ai_processed', False)}")
                print(f"📝 Response: {result.get('response', 'No response')[:200]}...")
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 80)

if __name__ == "__main__":
    test_ai_processing()