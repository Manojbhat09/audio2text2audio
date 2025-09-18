#!/usr/bin/env python3
"""
Debug script to test server functionality and find issues.
"""

import requests
import json
import time

def test_server_connection():
    """Test basic server connectivity."""
    print("🔍 Testing server connection...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        print(f"✅ Health check: {response.status_code}")
        print(f"Response: {response.json()}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_single_turn():
    """Test single-turn endpoint."""
    print("\n🔍 Testing single-turn endpoint...")
    
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/t2t",
            json={"text_data": "Hello, this is a test message."},
            timeout=10
        )
        print(f"✅ Single-turn: {response.status_code}")
        result = response.json()
        print(f"Response: {result.get('text', 'No text')[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Single-turn error: {e}")
        return False

def test_multi_turn():
    """Test multi-turn endpoint."""
    print("\n🔍 Testing multi-turn endpoint...")
    
    try:
        # First set max turns
        response = requests.post(
            "http://localhost:8000/api/v1/set_max_turns",
            json={"max_turns": 3},
            timeout=5
        )
        print(f"✅ Set max turns: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Test first message
        response = requests.post(
            "http://localhost:8000/api/v1/multiturn",
            json={"user_message": "Write 3 sentences about cats."},
            timeout=10
        )
        print(f"✅ Multi-turn message 1: {response.status_code}")
        result = response.json()
        print(f"Turn: {result.get('current_turn')}/{result.get('max_turns')}")
        print(f"Response: {result.get('text', 'No text')[:100]}...")
        
        # Test second message
        response = requests.post(
            "http://localhost:8000/api/v1/multiturn",
            json={"user_message": "Now write 2 sentences about dogs."},
            timeout=10
        )
        print(f"✅ Multi-turn message 2: {response.status_code}")
        result = response.json()
        print(f"Turn: {result.get('current_turn')}/{result.get('max_turns')}")
        print(f"Response: {result.get('text', 'No text')[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Multi-turn error: {e}")
        return False

def test_conversation_status():
    """Test conversation status endpoint."""
    print("\n🔍 Testing conversation status...")
    
    try:
        response = requests.get("http://localhost:8000/api/v1/conversation_status", timeout=5)
        print(f"✅ Status check: {response.status_code}")
        result = response.json()
        print(f"Current turn: {result.get('current_turn')}")
        print(f"Conversation length: {result.get('conversation_length')}")
        return True
    except Exception as e:
        print(f"❌ Status error: {e}")
        return False

def main():
    """Main debug function."""
    print("🚀 Server Debug Test")
    print("=" * 30)
    
    # Test server connection
    if not test_server_connection():
        print("\n❌ Server is not accessible. Please check:")
        print("1. Is the server running?")
        print("2. Is it running on the correct port (8000)?")
        print("3. Are you in the correct environment?")
        return
    
    # Test single-turn
    test_single_turn()
    
    # Test multi-turn
    test_multi_turn()
    
    # Test status
    test_conversation_status()
    
    print("\n✅ Debug test completed!")

if __name__ == "__main__":
    main()
