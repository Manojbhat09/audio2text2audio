#!/usr/bin/env python3
"""
Simple test script to verify multi-turn conversation functionality.

This script tests the multi-turn endpoint with a simple conversation.
"""

import requests
import json
import time

def test_multiturn_endpoint():
    """Test the multi-turn endpoint with a simple conversation."""
    
    api_url = "http://localhost:8000/api/v1/multiturn"
    
    print("Testing multi-turn endpoint...")
    print(f"API URL: {api_url}")
    
    try:
        # First, set max turns to 3 for testing
        set_turns_url = "http://localhost:8000/api/v1/set_max_turns"
        requests.post(set_turns_url, json={"max_turns": 3})
        print("Set max turns to 3")
        
        # Test first message
        response = requests.post(
            api_url,
            json={
                "user_message": "Write exactly 3 sentences about cats."
            },
            timeout=60
        )
        
        response.raise_for_status()
        result1 = response.json()
        print("✅ First message successful!")
        print(f"Response: {result1.get('text', 'No text')}...")
        print(f"Turn: {result1.get('current_turn')}/{result1.get('max_turns')}")
        print(f"Turns remaining: {result1.get('turns_remaining')}")
        
        # Test second message (should build on context)
        response = requests.post(
            api_url,
            json={
                "user_message": "Now write exactly 2 sentences about dogs."
            },
            timeout=60
        )
        
        response.raise_for_status()
        result2 = response.json()
        
        print("✅ Second message successful!")
        print(f"Response: {result2.get('text', 'No text')}...")
        print(f"Turn: {result2.get('current_turn')}/{result2.get('max_turns')}")
        print(f"Turns remaining: {result2.get('turns_remaining')}")
        print(f"Conversation reset: {result2.get('conversation_reset')}")
        
        # Test third message (should still be multi-turn)
        response = requests.post(
            api_url,
            json={
                "user_message": "What about birds?"
            },
            timeout=60
        )
        
        response.raise_for_status()
        result3 = response.json()
        print("✅ Third message successful!")
        print(f"Response: {result3.get('text', 'No text')}...")
        print(f"Turn: {result3.get('current_turn')}/{result3.get('max_turns')}")
        print(f"Turns remaining: {result3.get('turns_remaining')}")
        
        # Test fourth message (should be single-turn now)
        response = requests.post(
            api_url,
            json={
                "user_message": "What about fish?"
            },
            timeout=60
        )
        
        response.raise_for_status()
        result4 = response.json()
        print("✅ Fourth message (single-turn mode):")
        print(f"Response: {result4.get('text', 'No text')}...")
        print(f"Turn: {result4.get('current_turn')}/{result4.get('max_turns')}")
        print(f"Turns remaining: {result4.get('turns_remaining')}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_single_turn_comparison():
    """Test single-turn vs multi-turn for comparison."""
    
    print("\n" + "="*50)
    print("SINGLE-TURN vs MULTI-TURN COMPARISON")
    print("="*50)
    
    # Single-turn test
    print("\n1. Single-turn test:")
    single_turn_url = "http://localhost:8000/api/v1/t2t"
    
    try:
        print("   Sending single-turn request...")
        single_response = requests.post(
            single_turn_url,
            json={"text_data": "Write exactly 2 sentences about dogs."},
            timeout=15  # Reduced timeout
        )
        print(f"   Response status: {single_response.status_code}")
        single_response.raise_for_status()
        single_result = single_response.json()
        print(f"Single-turn response: {single_result.get('text', 'No text')[:100]}...")
        
    except requests.exceptions.Timeout:
        print("   Single-turn timed out - using fallback")
        single_result = {"text": "Timeout - server may be overloaded"}
    except Exception as e:
        print(f"   Single-turn failed: {e}")
        single_result = {"text": "Error"}
    
    # Multi-turn test
    print("\n2. Multi-turn test:")
    multi_turn_url = "http://localhost:8000/api/v1/multiturn"
    
    try:
        print("   Setting up conversation context...")
        # First, set up the conversation context by sending previous messages
        # This simulates the multi-turn context
        print("   Sending message 1...")
        requests.post(multi_turn_url, json={"user_message": "Write exactly 3 sentences about cats."}, timeout=15)
        print("   Sending message 2...")
        requests.post(multi_turn_url, json={"user_message": "Now write exactly 2 sentences about dogs."}, timeout=15)
        
        print("   Sending final message...")
        # Now test the multi-turn response
        multi_response = requests.post(
            multi_turn_url,
            json={"user_message": "What about birds?"},
            timeout=15
        )
        print(f"   Response status: {multi_response.status_code}")
        multi_response.raise_for_status()
        multi_result = multi_response.json()
        print(f"Multi-turn response: {multi_result.get('text', 'No text')[:100]}...")
        
    except requests.exceptions.Timeout:
        print("   Multi-turn timed out - using fallback")
        multi_result = {"text": "Timeout - server may be overloaded"}
    except Exception as e:
        print(f"   Multi-turn failed: {e}")
        multi_result = {"text": "Error"}
    
    # Compare responses
    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    print(f"Single-turn length: {len(single_result.get('text', ''))}")
    print(f"Multi-turn length: {len(multi_result.get('text', ''))}")
    
    # Check if multi-turn response shows context awareness
    multi_text = multi_result.get('text', '').lower()
    single_text = single_result.get('text', '').lower()
    
    # Check for context awareness indicators
    context_indicators = ['birds', 'fascinating', 'diversity', 'ecosystems', 'adaptations']
    has_context = any(indicator in multi_text for indicator in context_indicators)
    
    if has_context and len(multi_text) > len(single_text) * 1.2:
        print("✅ Multi-turn response shows awareness of instruction context")
    else:
        print("❌ Multi-turn response doesn't show clear instruction awareness")
    
    return single_result, multi_result

def main():
    """Main test function."""
    print("Multi-Turn Conversation Test")
    print("=" * 40)
    
    # Test basic functionality
    if not test_multiturn_endpoint():
        print("❌ Multi-turn endpoint test failed. Make sure server is running.")
        return
    
    # Test comparison (with timeout protection)
    try:
        print("\n🔄 Starting comparison test...")
        single_result, multi_result = test_single_turn_comparison()
        print("\n✅ All tests completed!")
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Comparison test failed: {e}")
        print("✅ Basic multi-turn test completed successfully!")

if __name__ == "__main__":
    main()
