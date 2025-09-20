#!/usr/bin/env python3
"""
Test script for Ollama judge integration with qwen3 think tag support.
"""

import json
import time
from comprehensive_voicebench_scoring import OllamaJudgeClient

def test_ollama_judge():
    """Test the Ollama judge with qwen3 think tag parsing."""
    print("🧪 Testing Ollama Judge with Qwen3 Think Tag Support")
    print("=" * 60)
    
    # Initialize Ollama judge client
    try:
        judge = OllamaJudgeClient(model="qwen3:14b", ollama_url="http://localhost:11434")
        print("✅ Ollama judge client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Ollama judge: {e}")
        return False
    
    # Test evaluation
    test_prompt = "What is the capital of France?"
    test_response = "The capital of France is Paris. It is located in the north-central part of the country and is known for its rich history, culture, and landmarks like the Eiffel Tower."
    
    print(f"\n📝 Testing evaluation:")
    print(f"Prompt: {test_prompt}")
    print(f"Response: {test_response}")
    
    try:
        result = judge.evaluate_response(
            prompt=test_prompt,
            response=test_response,
            reference="Paris",
            dataset_type="commoneval"
        )
        
        print(f"\n📊 Evaluation Result:")
        print(f"Score: {result['score']}")
        print(f"Model used: {result['model_used']}")
        print(f"Evaluation time: {result['evaluation_time']:.2f}s")
        
        if result.get('thinking'):
            print(f"\n🧠 Thinking Process:")
            print(result['thinking'])
        
        if result.get('final_response'):
            print(f"\n🎯 Final Response:")
            print(result['final_response'])
        
        print(f"\nRaw LLM Response:")
        print(result['llm_raw_response'])
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False

def test_think_tag_parsing():
    """Test the think tag parsing functionality."""
    print("\n🔍 Testing Think Tag Parsing")
    print("=" * 40)
    
    # Test response with think tags
    test_response_with_think = """<think>
The user is asking about the capital of France. This is a straightforward factual question. The capital of France is Paris, which is a well-known fact. I should provide a clear, accurate answer.
</think>
The capital of France is Paris."""
    
    judge = OllamaJudgeClient(model="qwen3:14b", ollama_url="http://localhost:11434")
    thinking, final_response = judge._parse_qwen_response(test_response_with_think)
    
    print(f"Original response:")
    print(test_response_with_think)
    print(f"\nExtracted thinking:")
    print(thinking)
    print(f"\nFinal response:")
    print(final_response)
    
    # Test response without think tags
    test_response_without_think = "The capital of France is Paris."
    thinking2, final_response2 = judge._parse_qwen_response(test_response_without_think)
    
    print(f"\n\nResponse without think tags:")
    print(test_response_without_think)
    print(f"\nExtracted thinking: '{thinking2}'")
    print(f"Final response: '{final_response2}'")

if __name__ == "__main__":
    print("🚀 Starting Ollama Judge Tests")
    print("=" * 60)
    
    # Test think tag parsing first
    test_think_tag_parsing()
    
    # Test full evaluation
    success = test_ollama_judge()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")


