#!/usr/bin/env python3
"""
Safe model testing script with numerical stability fixes.
"""

import sys
import os
# Enable synchronous CUDA execution for better debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import json
import time
from collections import Counter
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_generate(model, tokenizer, prompt, max_new_tokens=20, temperature=0.7, top_p=0.9, top_k=40):
    """
    Safely generate text with numerical stability checks.
    
    Args:
        model: The model to use for generation
        tokenizer: Tokenizer for the model
        prompt: Input prompt
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        
    Returns:
        Generated text or error message
    """
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Test forward pass first
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Check for invalid values in logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logger.warning("Invalid values detected in logits during generation")
                return "ERROR: Invalid model outputs"
            
            # Apply temperature scaling with clamping
            if temperature > 0:
                logits = logits / max(temperature, 1e-8)  # Prevent division by zero
                logits = torch.clamp(logits, min=-50.0, max=50.0)
            
            # Generate with safer parameters
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.1),  # Minimum temperature of 0.1
                top_p=min(max(top_p, 0.1), 1.0),    # Clamp top_p between 0.1 and 1.0
                top_k=max(top_k, 1),                # Minimum top_k of 1
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Add repetition penalty
                no_repeat_ngram_size=2,  # Prevent repetition
            )
            
            # Decode and return
            response = tokenizer.decode(generated[0], skip_special_tokens=True)
            return response
            
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return f"ERROR: {str(e)}"

def test_model_safely(model_path, num_tests=10):
    """Test the model with safe generation parameters."""
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        logger.info("✓ Model loaded successfully")
        
        # Test prompts
        test_prompts = [
            "Hello, how are you?",
            "The weather is",
            "What is the capital of France?",
            "Tell me a story about",
            "Classify this audio transcript as one of these datasets: ifeval, commoneval, wildvoice. Transcript: This is a test audio for ifeval dataset.",
            "The answer is",
            "In the beginning",
            "Once upon a time",
            "The problem is",
            "The solution is"
        ]
        
        results = []
        successful_tests = 0
        
        for i, prompt in enumerate(test_prompts[:num_tests]):
            logger.info(f"Test {i+1}/{num_tests}: {prompt}")
            
            # Test with different temperature settings
            for temp in [0.1, 0.7, 1.0]:
                try:
                    response = safe_generate(
                        model, tokenizer, prompt, 
                        max_new_tokens=20, 
                        temperature=temp,
                        top_p=0.9,
                        top_k=40
                    )
                    
                    if not response.startswith("ERROR"):
                        successful_tests += 1
                        logger.info(f"  Temperature {temp}: {response}")
                        break
                    else:
                        logger.warning(f"  Temperature {temp}: {response}")
                        
                except Exception as e:
                    logger.error(f"  Temperature {temp} failed: {e}")
            
            results.append({
                'prompt': prompt,
                'successful': successful_tests > 0
            })
        
        # Test classification specifically
        logger.info("Testing classification task...")
        classification_prompt = "Classify this audio transcript as one of these datasets: ifeval, commoneval, wildvoice. Transcript: This is a test audio for ifeval dataset."
        
        classification_response = safe_generate(
            model, tokenizer, classification_prompt,
            max_new_tokens=30,
            temperature=0.1,  # Low temperature for more deterministic output
            top_p=0.9,
            top_k=40
        )
        
        logger.info(f"Classification test: {classification_response}")
        
        # Summary
        success_rate = successful_tests / num_tests if num_tests > 0 else 0
        logger.info(f"Success rate: {success_rate:.2%} ({successful_tests}/{num_tests})")
        
        return {
            'success_rate': success_rate,
            'successful_tests': successful_tests,
            'total_tests': num_tests,
            'classification_response': classification_response,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Model testing failed: {e}")
        return None

def main():
    """Main function."""
    model_path = "/home/mbhat/omegalabs-anytoany-bittensor/finetune/models/final_model"
    
    logger.info("🧪 Starting safe model testing...")
    
    results = test_model_safely(model_path, num_tests=10)
    
    if results:
        logger.info("✅ Model testing completed successfully!")
        logger.info(f"Success rate: {results['success_rate']:.2%}")
        logger.info(f"Classification response: {results['classification_response']}")
        
        # Save results
        with open("safe_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to safe_test_results.json")
    else:
        logger.error("❌ Model testing failed")

if __name__ == "__main__":
    main()
