#!/usr/bin/env python3
"""
Test if the model can learn a basic classification task.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import logging

# Enable synchronous CUDA execution for better debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_learning():
    """Test if the model can learn a basic classification task."""
    
    # Load model
    model_path = "/home/mbhat/omegalabs-anytoany-bittensor/finetune/models/final_model"
    logger.info(f"Loading model from {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Test simple prompts
    simple_prompts = [
        "What is 2+2? Answer:",
        "What color is the sky? Answer:",
        "What is the capital of France? Answer:",
        "What is 5*3? Answer:",
    ]
    
    logger.info("Testing basic reasoning capabilities...")
    
    for i, prompt in enumerate(simple_prompts, 1):
        logger.info(f"\n=== Test {i} ===")
        logger.info(f"Prompt: {prompt}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        try:
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )
                
                response = tokenizer.decode(generated[0], skip_special_tokens=True)
                assistant_response = response[len(prompt):].strip()
                logger.info(f"Response: '{assistant_response}'")
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
    
    # Test if the model can generate the dataset names
    logger.info("\n=== Testing dataset name generation ===")
    
    dataset_prompts = [
        "The dataset name is:",
        "Answer: ifeval",
        "Answer: commoneval", 
        "Answer: wildvoice",
    ]
    
    for i, prompt in enumerate(dataset_prompts, 1):
        logger.info(f"\nTest {i}: {prompt}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        try:
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )
                
                response = tokenizer.decode(generated[0], skip_special_tokens=True)
                assistant_response = response[len(prompt):].strip()
                logger.info(f"Response: '{assistant_response}'")
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")

if __name__ == "__main__":
    test_basic_learning()
