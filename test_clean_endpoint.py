#!/usr/bin/env python3
"""
Test the clean endpoint to clear cache and GPU memory.
"""

import requests
import time

def test_clean_endpoint():
    """Test the clean endpoint."""
    print("🧹 Testing Clean Endpoint")
    print("=" * 30)
    
    try:
        # Test GET request
        print("1. Testing GET /api/v1/clean...")
        response = requests.get("http://localhost:8000/api/v1/clean", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result}")
        else:
            print(f"Error: {response.text}")
        
        # Test POST request
        print("\n2. Testing POST /api/v1/clean...")
        response = requests.post("http://localhost:8000/api/v1/clean", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result}")
        else:
            print(f"Error: {response.text}")
        
        print("\n✅ Clean endpoint test completed!")
        
    except Exception as e:
        print(f"❌ Error testing clean endpoint: {e}")

if __name__ == "__main__":
    test_clean_endpoint()
