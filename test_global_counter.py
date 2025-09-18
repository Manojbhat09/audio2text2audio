#!/usr/bin/env python3
"""
Test script for global conversation counter functionality.

This script tests that the global counter increments correctly and resets after max_turns.
"""

import requests
import json
import time

def test_global_counter():
    """Test the global counter functionality."""
    
    api_url = "http://localhost:8000/api/v1/multiturn"
    status_url = "http://localhost:8000/api/v1/conversation_status"
    reset_url = "http://localhost:8000/api/v1/reset_conversation"
    set_turns_url = "http://localhost:8000/api/v1/set_max_turns"
    
    print("Testing Global Conversation Counter")
    print("=" * 40)
    
    # Reset conversation first
    print("1. Resetting conversation...")
    response = requests.post(reset_url)
    print(f"Reset response: {response.json()}")
    
    # Set max turns to 3
    print("2. Setting max turns to 3...")
    response = requests.post(set_turns_url, json={"max_turns": 3})
    print(f"Set turns response: {response.json()}")
    
    # Test multiple turns
    print("\n3. Testing multiple turns...")
    
    for i in range(5):  # Test more than max_turns to see single-turn mode
        print(f"\n--- Turn {i+1} ---")
        
        # Send message
        response = requests.post(
            api_url,
            json={
                "user_message": f"Hello, this is message {i+1}"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result['text'][:100]}...")
            print(f"Current turn: {result['current_turn']}")
            print(f"Max turns: {result['max_turns']}")
            print(f"Turns remaining: {result['turns_remaining']}")
            print(f"Conversation reset: {result['conversation_reset']}")
            
            # Check if we're in single-turn mode
            if result['turns_remaining'] == 0:
                print("🔄 Now in SINGLE-TURN mode!")
        else:
            print(f"Error: {response.status_code} - {response.text}")
        
        # Check status
        status_response = requests.get(status_url)
        if status_response.status_code == 200:
            status = status_response.json()
            print(f"Global counter: {status['current_turn']}")
            print(f"Conversation length: {status['conversation_length']}")
        
        time.sleep(1)  # Small delay between requests
    
    print("\n" + "=" * 40)
    print("Test completed!")

def test_conversation_reset():
    """Test that conversation resets after max_turns."""
    
    api_url = "http://localhost:8000/api/v1/multiturn"
    reset_url = "http://localhost:8000/api/v1/reset_conversation"
    
    print("\nTesting Conversation Reset")
    print("=" * 30)
    
    # Reset first
    requests.post(reset_url)
    
    # Send exactly max_turns messages
    max_turns = 3
    for i in range(max_turns):
        response = requests.post(
            api_url,
            json={
                "user_message": f"Test message {i+1}",
                "max_turns": max_turns
            }
        )
        
        result = response.json()
        print(f"Turn {i+1}: Current={result['current_turn']}, Reset={result['conversation_reset']}")
    
    # Send one more message - should trigger reset
    response = requests.post(
        api_url,
        json={
            "user_message": "This should trigger reset",
            "max_turns": max_turns
        }
    )
    
    result = response.json()
    print(f"Turn {max_turns+1}: Current={result['current_turn']}, Reset={result['conversation_reset']}")
    
    if result['conversation_reset']:
        print("✅ Conversation reset correctly triggered!")
    else:
        print("❌ Conversation reset failed!")

def main():
    """Main test function."""
    print("Global Counter Test")
    print("=" * 20)
    
    try:
        test_global_counter()
        test_conversation_reset()
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    main()
