#!/usr/bin/env python3
"""
Debug script to test model loading and generation step by step.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test model loading step by step."""
    model_path = "/home/mbhat/omegalabs-anytoany-bittensor/finetune/models/final_model"
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"✓ Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    logger.info(f"✓ Model loaded. Device: {next(model.parameters()).device}")
    logger.info(f"Model config: {model.config}")
    
    return model, tokenizer

def test_simple_generation(model, tokenizer):
    """Test simple generation without complex parameters."""
    logger.info("Testing simple generation...")
    
    # Simple test prompt
    prompt = "Hello, how are you?"
    logger.info(f"Prompt: {prompt}")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    logger.info(f"Input shape: {inputs['input_ids'].shape}")
    
    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logger.info(f"Inputs moved to device: {device}")
    
    # Test forward pass first
    logger.info("Testing forward pass...")
    with torch.no_grad():
        outputs = model(**inputs)
        logger.info(f"✓ Forward pass successful. Output shape: {outputs.logits.shape}")
    
    # Test generation with minimal parameters
    logger.info("Testing generation with minimal parameters...")
    try:
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,  # Use greedy decoding
                pad_token_id=tokenizer.eos_token_id
            )
        logger.info(f"✓ Generation successful. Shape: {generated.shape}")
        
        # Decode
        response = tokenizer.decode(generated[0], skip_special_tokens=True)
        logger.info(f"Response: {response}")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return False
    
    return True

def test_with_sampling(model, tokenizer):
    """Test generation with sampling parameters."""
    logger.info("Testing generation with sampling...")
    
    prompt = "The weather is"
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    try:
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(generated[0], skip_special_tokens=True)
        logger.info(f"✓ Sampling generation successful: {response}")
        return True
        
    except Exception as e:
        logger.error(f"Sampling generation failed: {e}")
        return False

def main():
    """Main debug function."""
    try:
        # Test model loading
        model, tokenizer = test_model_loading()
        
        # Test simple generation
        if test_simple_generation(model, tokenizer):
            logger.info("✓ Simple generation works")
        else:
            logger.error("✗ Simple generation failed")
            return
        
        # Test sampling generation
        if test_with_sampling(model, tokenizer):
            logger.info("✓ Sampling generation works")
        else:
            logger.error("✗ Sampling generation failed")
            return
        
        logger.info("🎉 All tests passed!")
        
    except Exception as e:
        logger.error(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


