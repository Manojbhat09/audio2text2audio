#!/usr/bin/env python3
"""
Test script to evaluate the quality of model generation.
"""

import sys
import os
# Enable synchronous CUDA execution for better debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_generation():
    """Test basic text generation capabilities."""
    logger.info("🧪 Testing basic generation capabilities...")
    
    model_path = "/home/mbhat/omegalabs-anytoany-bittensor/finetune/models/final_model"
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    logger.info("✅ Model loaded successfully")
    
    # Test prompts
    test_prompts = [
        "Hello, how are you?",
        "The weather is",
        "What is the capital of France?",
        "Tell me a story about",
        "The answer is"
    ]
    
    for i, prompt in enumerate(test_prompts):
        logger.info(f"\n--- Test {i+1}: '{prompt}' ---")
        
        try:
            # Method 1: Deterministic generation
            inputs = tokenizer(prompt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                response = tokenizer.decode(generated[0], skip_special_tokens=True)
                logger.info(f"Deterministic: {response}")
                
        except Exception as e:
            logger.error(f"Deterministic generation failed: {e}")
        
        try:
            # Method 2: Sampling generation
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                response = tokenizer.decode(generated[0], skip_special_tokens=True)
                logger.info(f"Sampling: {response}")
                
        except Exception as e:
            logger.error(f"Sampling generation failed: {e}")

def test_classification_prompts():
    """Test classification-specific prompts."""
    logger.info("\n🎯 Testing classification prompts...")
    
    model_path = "/home/mbhat/omegalabs-anytoany-bittensor/finetune/models/final_model"
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    # Classification prompts
    classification_prompts = [
        "Classify this audio transcript as one of these datasets: ifeval, commoneval, wildvoice. Transcript: This is a test audio for ifeval dataset.",
        "Which dataset does this belong to: ifeval, commoneval, wildvoice? Audio: Common evaluation task.",
        "Dataset classification: ifeval, commoneval, wildvoice. Input: Wild voice sample with natural speech."
    ]
    
    for i, prompt in enumerate(classification_prompts):
        logger.info(f"\n--- Classification Test {i+1} ---")
        logger.info(f"Prompt: {prompt[:100]}...")
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                response = tokenizer.decode(generated[0], skip_special_tokens=True)
                logger.info(f"Response: {response}")
                
        except Exception as e:
            logger.error(f"Classification generation failed: {e}")

def main():
    """Main test function."""
    logger.info("🚀 Starting generation quality tests...")
    
    # Test basic generation
    test_basic_generation()
    
    # Test classification prompts
    test_classification_prompts()
    
    logger.info("\n✅ Generation quality tests completed!")

if __name__ == "__main__":
    main()

