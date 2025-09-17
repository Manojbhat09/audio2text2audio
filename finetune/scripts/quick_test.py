#!/usr/bin/env python3
"""
Quick test script to verify the trained model is working.
This script does a simple inference test without full dataset evaluation.
"""

import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def quick_test(model_path: str = "/home/mbhat/omegalabs-anytoany-bittensor/finetune/models/final_model"):
    """Quick test of the trained model."""
    print("🚀 Quick Model Test")
    print("=" * 50)
    
    try:
        # Load model and tokenizer
        print("📥 Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ Model loaded successfully!")
        
        # Test prompts
        test_prompts = [
            "Classify this audio transcript as one of these datasets: ifeval, commoneval, wildvoice. Transcript: This is a test audio for ifeval dataset.",
            "Classify this audio transcript as one of these datasets: ifeval, commoneval, wildvoice. Transcript: This is a common evaluation task.",
            "Classify this audio transcript as one of these datasets: ifeval, commoneval, wildvoice. Transcript: This is a wild voice sample."
        ]
        
        print("\n🧪 Testing inference...")
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i} ---")
            print(f"Prompt: {prompt[:80]}...")
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            prediction = response[len(prompt):].strip()
            
            print(f"Response: {prediction}")
            
            # Check if response contains expected keywords
            response_lower = prediction.lower()
            if any(keyword in response_lower for keyword in ["ifeval", "commoneval", "wildvoice"]):
                print("✅ Model generated relevant response")
            else:
                print("⚠️  Model response doesn't contain expected keywords")
        
        print("\n🎉 Quick test completed!")
        print("✅ Model is working and can generate responses")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
