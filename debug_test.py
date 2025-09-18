#!/usr/bin/env python3
"""Debug test to see what's happening."""

print("Starting debug test...")

try:
    import requests
    print("✅ requests imported")
    
    print("Testing server...")
    response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
    print(f"Health status: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ Server is healthy")
        
        print("Testing t2t endpoint...")
        response = requests.post(
            "http://localhost:8000/api/v1/t2t",
            json={"text_data": "Hello test"},
            timeout=10
        )
        print(f"T2T status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result.get('text', 'No text')[:100]}...")
            print("✅ T2T endpoint working")
        else:
            print(f"T2T error: {response.text}")
    else:
        print("❌ Server not healthy")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("Debug test completed!")
