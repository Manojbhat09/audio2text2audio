#!/usr/bin/env python3
import requests
import time

def test_server():
    print("Testing server connection...")
    
    # Test health
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=10)
        print(f"Health status: {response.status_code}")
        print(f"Health response: {response.json()}")
    except Exception as e:
        print(f"Health error: {e}")
    
    # Test t2t endpoint
    try:
        print("\nTesting t2t endpoint...")
        response = requests.post(
            "http://localhost:8000/api/v1/t2t",
            json={"text_data": "Hello test"},
            timeout=60
        )
        print(f"T2T status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"T2T response: {result.get('text', 'No text')[:100]}...")
        else:
            print(f"T2T error: {response.text}")
    except Exception as e:
        print(f"T2T error: {e}")

if __name__ == "__main__":
    test_server()
