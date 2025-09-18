#!/usr/bin/env python3
"""
Simple VoiceBench test to debug issues.
"""

import requests
import json
from datasets import load_dataset

def test_server_connection():
    """Test if server is running."""
    print("Testing server connection...")
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        print(f"Server status: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"Server error: {e}")
        return False

def test_dataset_loading():
    """Test loading a small sample from VoiceBench."""
    print("Testing dataset loading...")
    try:
        # Load just one sample
        dataset = load_dataset('hlt-lab/voicebench', 'commoneval', split='test', streaming=True)
        print("Dataset loaded successfully")
        
        # Get first item
        item = next(iter(dataset))
        print(f"Sample keys: {list(item.keys())}")
        print(f"Prompt: {item.get('prompt', 'No prompt')[:100]}...")
        print(f"Output: {item.get('output', 'No output')[:100]}...")
        
        return True
    except Exception as e:
        print(f"Dataset error: {e}")
        return False

def test_single_request():
    """Test a single request to the server."""
    print("Testing single request...")
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/t2t",
            json={"text_data": "Hello, this is a test."},
            timeout=10
        )
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result.get('text', 'No text')[:100]}...")
            return True
        else:
            print(f"Error response: {response.text}")
            return False
    except Exception as e:
        print(f"Request error: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Simple VoiceBench Test")
    print("=" * 30)
    
    # Test server
    server_ok = test_server_connection()
    
    # Test dataset
    dataset_ok = test_dataset_loading()
    
    # Test request
    request_ok = test_single_request()
    
    print(f"\n📊 Test Results:")
    print(f"  Server: {'✅' if server_ok else '❌'}")
    print(f"  Dataset: {'✅' if dataset_ok else '❌'}")
    print(f"  Request: {'✅' if request_ok else '❌'}")
    
    if server_ok and dataset_ok and request_ok:
        print("\n🎉 All tests passed! Ready for full VoiceBench test.")
    else:
        print("\n⚠️ Some tests failed. Check the issues above.")

if __name__ == "__main__":
    main()
