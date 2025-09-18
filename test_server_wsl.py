#!/usr/bin/env python3
"""
Test script to run in WSL environment where server is running.
"""

import requests
import json
import time

def test_server():
    """Test the server functionality."""
    print("🚀 Testing Server in WSL Environment")
    print("=" * 40)
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=10)
        print(f"✅ Health: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health error: {e}")
        return
    
    # Test single-turn
    print("\n2. Testing single-turn endpoint...")
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/t2t",
            json={"text_data": "Hello, this is a test."},
            timeout=15
        )
        print(f"✅ Single-turn: {response.status_code}")
        result = response.json()
        print(f"Response: {result.get('text', 'No text')[:100]}...")
    except Exception as e:
        print(f"❌ Single-turn error: {e}")
        return
    
    # Test multi-turn setup
    print("\n3. Testing multi-turn setup...")
    try:
        # Set max turns
        response = requests.post(
            "http://localhost:8000/api/v1/set_max_turns",
            json={"max_turns": 3},
            timeout=5
        )
        print(f"✅ Set max turns: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Test first multi-turn message
        response = requests.post(
            "http://localhost:8000/api/v1/multiturn",
            json={"user_message": "Write 2 sentences about cats."},
            timeout=15
        )
        print(f"✅ Multi-turn 1: {response.status_code}")
        result = response.json()
        print(f"Turn: {result.get('current_turn')}/{result.get('max_turns')}")
        print(f"Response: {result.get('text', 'No text')[:100]}...")
        
        # Test second multi-turn message
        response = requests.post(
            "http://localhost:8000/api/v1/multiturn",
            json={"user_message": "Now write 2 sentences about dogs."},
            timeout=15
        )
        print(f"✅ Multi-turn 2: {response.status_code}")
        result = response.json()
        print(f"Turn: {result.get('current_turn')}/{result.get('max_turns')}")
        print(f"Response: {result.get('text', 'No text')[:100]}...")
        
        # Test third message (should still be multi-turn)
        response = requests.post(
            "http://localhost:8000/api/v1/multiturn",
            json={"user_message": "What about birds?"},
            timeout=15
        )
        print(f"✅ Multi-turn 3: {response.status_code}")
        result = response.json()
        print(f"Turn: {result.get('current_turn')}/{result.get('max_turns')}")
        print(f"Turns remaining: {result.get('turns_remaining')}")
        print(f"Response: {result.get('text', 'No text')[:100]}...")
        
        # Test fourth message (should be single-turn now)
        response = requests.post(
            "http://localhost:8000/api/v1/multiturn",
            json={"user_message": "What about fish?"},
            timeout=15
        )
        print(f"✅ Multi-turn 4 (single-turn mode): {response.status_code}")
        result = response.json()
        print(f"Turn: {result.get('current_turn')}/{result.get('max_turns')}")
        print(f"Turns remaining: {result.get('turns_remaining')}")
        print(f"Response: {result.get('text', 'No text')[:100]}...")
        
    except Exception as e:
        print(f"❌ Multi-turn error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test conversation status
    print("\n4. Testing conversation status...")
    try:
        response = requests.get("http://localhost:8000/api/v1/conversation_status", timeout=5)
        print(f"✅ Status: {response.status_code}")
        result = response.json()
        print(f"Current turn: {result.get('current_turn')}")
        print(f"Conversation length: {result.get('conversation_length')}")
    except Exception as e:
        print(f"❌ Status error: {e}")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    test_server()
