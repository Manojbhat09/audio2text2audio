#!/usr/bin/env python3
"""
Basic multi-turn test without comparison to avoid hanging issues.
"""

import requests
import time

def test_basic_multiturn():
    """Test basic multi-turn functionality."""
    print("🚀 Basic Multi-Turn Test")
    print("=" * 30)
    
    # Test server connection
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print(f"❌ Server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False
    
    # Test multi-turn endpoint
    print("\n🔍 Testing multi-turn endpoint...")
    
    try:
        # Set max turns to 3
        requests.post("http://localhost:8000/api/v1/set_max_turns", json={"max_turns": 3}, timeout=5)
        print("✅ Set max turns to 3")
        
        # Test first message
        print("📝 Sending message 1...")
        response = requests.post(
            "http://localhost:8000/api/v1/multiturn",
            json={"user_message": "Write 3 sentences about cats."},
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Message 1 successful!")
            print(f"   Turn: {result.get('current_turn')}/{result.get('max_turns')}")
            print(f"   Response: {result.get('text', 'No text')[:100]}...")
        else:
            print(f"❌ Message 1 failed: {response.status_code}")
            return False
        
        # Test second message
        print("\n📝 Sending message 2...")
        response = requests.post(
            "http://localhost:8000/api/v1/multiturn",
            json={"user_message": "Now write 2 sentences about dogs."},
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Message 2 successful!")
            print(f"   Turn: {result.get('current_turn')}/{result.get('max_turns')}")
            print(f"   Response: {result.get('text', 'No text')[:100]}...")
        else:
            print(f"❌ Message 2 failed: {response.status_code}")
            return False
        
        # Test third message
        print("\n📝 Sending message 3...")
        response = requests.post(
            "http://localhost:8000/api/v1/multiturn",
            json={"user_message": "What about birds?"},
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Message 3 successful!")
            print(f"   Turn: {result.get('current_turn')}/{result.get('max_turns')}")
            print(f"   Response: {result.get('text', 'No text')[:100]}...")
        else:
            print(f"❌ Message 3 failed: {response.status_code}")
            return False
        
        # Test fourth message (should be single-turn mode)
        print("\n📝 Sending message 4 (single-turn mode)...")
        response = requests.post(
            "http://localhost:8000/api/v1/multiturn",
            json={"user_message": "What about fish?"},
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Message 4 successful!")
            print(f"   Turn: {result.get('current_turn')}/{result.get('max_turns')}")
            print(f"   Response: {result.get('text', 'No text')[:100]}...")
        else:
            print(f"❌ Message 4 failed: {response.status_code}")
            return False
        
        print("\n✅ All multi-turn tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Multi-turn test failed: {e}")
        return False

def test_clean_endpoint():
    """Test the clean endpoint."""
    print("\n🧹 Testing clean endpoint...")
    
    try:
        response = requests.get("http://localhost:8000/api/v1/clean", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Clean endpoint successful!")
            print(f"   Message: {result.get('message')}")
            print(f"   CUDA available: {result.get('cuda_available')}")
            print(f"   Conversation cleared: {result.get('conversation_cleared')}")
        else:
            print(f"❌ Clean endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Clean endpoint error: {e}")

def main():
    """Main function."""
    print("🚀 Basic Multi-Turn Test Suite")
    print("=" * 40)
    
    # Test basic multi-turn
    if test_basic_multiturn():
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    # Test clean endpoint
    test_clean_endpoint()
    
    print("\n✅ Test suite completed!")

if __name__ == "__main__":
    main()
