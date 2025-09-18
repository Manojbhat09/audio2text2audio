#!/usr/bin/env python3
"""
Simple script to check server status and identify issues.
"""

import requests
import time

def check_server():
    """Check if server is responding and identify issues."""
    print("🔍 Checking server status...")
    
    # Try different endpoints
    endpoints = [
        "http://localhost:8000/",
        "http://localhost:8000/api/v1/health",
        "http://localhost:8000/api/v1/ping",
        "http://localhost:8000/api/v1/test"
    ]
    
    for endpoint in endpoints:
        print(f"\nTesting {endpoint}...")
        try:
            response = requests.get(endpoint, timeout=5)
            print(f"✅ Status: {response.status_code}")
            try:
                print(f"Response: {response.json()}")
            except:
                print(f"Response: {response.text[:200]}...")
        except requests.exceptions.ConnectionError:
            print("❌ Connection refused - server not running")
        except requests.exceptions.Timeout:
            print("❌ Timeout - server might be loading models")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Try a simple POST request
    print(f"\nTesting POST to /api/v1/t2t...")
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/t2t",
            json={"text_data": "test"},
            timeout=10
        )
        print(f"✅ POST Status: {response.status_code}")
        try:
            print(f"Response: {response.json()}")
        except:
            print(f"Response: {response.text[:200]}...")
    except Exception as e:
        print(f"❌ POST Error: {e}")

if __name__ == "__main__":
    check_server()
