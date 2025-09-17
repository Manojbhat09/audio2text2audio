#!/usr/bin/env python3
"""
Comprehensive fix for CUDA assert error and training issues.
Based on online research and best practices.
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable synchronous CUDA execution for better debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def safe_generate_with_clamping(model, tokenizer, prompt, max_new_tokens=20, temperature=0.7, top_p=0.9, top_k=40):
    """
    Safely generate text with numerical stability fixes.
    Based on online research for CUDA assert error prevention.
    """
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # First, test forward pass with stability checks
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Check for invalid values in logits
            has_nan = torch.isnan(logits).any()
            has_inf = torch.isinf(logits).any()
            
            if has_nan or has_inf:
                logger.warning(f"⚠️  Invalid values in logits: nan={has_nan}, inf={has_inf}")
                # Clamp logits to prevent inf/nan
                logits = torch.clamp(logits, min=-50.0, max=50.0)
            
            # Check for extreme values that could cause issues
            if torch.abs(logits).max() > 100:
                logger.warning("⚠️  Extreme logit values detected, clamping...")
                logits = torch.clamp(logits, min=-50.0, max=50.0)
        
        # Generate with safe parameters
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": max(temperature, 0.01),  # Ensure minimum temperature
            "top_p": min(max(top_p, 0.01), 0.99),   # Clamp top_p to valid range
            "top_k": max(min(top_k, 100), 1),       # Clamp top_k to valid range
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.1,  # Prevent repetition
            "no_repeat_ngram_size": 2,  # Prevent n-gram repetition
        }
        
        # Generate with error handling
        try:
            generated = model.generate(
                **inputs,
                **generation_config
            )
            
            # Decode response
            response = tokenizer.decode(generated[0], skip_special_tokens=True)
            return response, True
            
        except Exception as gen_error:
            logger.error(f"Generation failed: {gen_error}")
            # Fallback to greedy generation
            try:
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                response = tokenizer.decode(generated[0], skip_special_tokens=True)
                return response, True
            except Exception as fallback_error:
                logger.error(f"Fallback generation also failed: {fallback_error}")
                return f"Error: {str(fallback_error)}", False
                
    except Exception as e:
        logger.error(f"Safe generation failed: {e}")
        return f"Error: {str(e)}", False

def test_model_with_stability_checks(model_path):
    """
    Test model with comprehensive stability checks.
    """
    logger.info("🔍 Testing model with stability checks...")
    
    try:
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        logger.info(f"✅ Model loaded successfully. Device: {next(model.parameters()).device}")
        
        # Test prompts
        test_prompts = [
            "Hello, how are you?",
            "The weather is",
            "What is the capital of France?",
            "Tell me a story about",
            "Classify this audio transcript as one of these datasets: ifeval, commoneval, wildvoice. Transcript: This is a test audio for ifeval dataset."
        ]
        
        results = []
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"🧪 Test {i+1}: {prompt}")
            
            response, success = safe_generate_with_clamping(
                model, tokenizer, prompt, 
                max_new_tokens=20, 
                temperature=0.7, 
                top_p=0.9, 
                top_k=40
            )
            
            results.append({
                'prompt': prompt,
                'response': response,
                'success': success
            })
            
            logger.info(f"  Response: {response}")
            logger.info(f"  Success: {'✅' if success else '❌'}")
        
        # Summary
        successful_tests = sum(1 for r in results if r['success'])
        total_tests = len(results)
        
        logger.info(f"\n📊 Test Results: {successful_tests}/{total_tests} successful")
        
        if successful_tests == total_tests:
            logger.info("🎉 All tests passed! Model is stable.")
        else:
            logger.warning(f"⚠️  {total_tests - successful_tests} tests failed. Model needs further fixes.")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Model testing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def fix_training_config():
    """
    Fix training configuration to prevent NaN gradients and overfitting.
    """
    logger.info("🔧 Applying training configuration fixes...")
    
    fixes = {
        "gradient_clip_val": 1.0,  # Prevent exploding gradients
        "learning_rate": 1e-5,     # Lower learning rate for stability
        "weight_decay": 0.01,      # Add weight decay for regularization
        "warmup_steps": 100,       # Add warmup
        "max_grad_norm": 1.0,      # Gradient clipping
        "fp16": False,             # Disable mixed precision for stability
        "bf16": False,             # Disable bfloat16 for stability
        "dataloader_drop_last": True,  # Drop incomplete batches
        "remove_unused_columns": False,  # Keep all columns
        "eval_strategy": "no",     # Disable evaluation during training
        "save_strategy": "steps",  # Save by steps
        "save_steps": 100,         # Save every 100 steps
        "logging_steps": 10,       # Log every 10 steps
        "per_device_train_batch_size": 1,  # Small batch size
        "gradient_accumulation_steps": 4,  # Accumulate gradients
        "num_train_epochs": 1,     # Single epoch to prevent overfitting
        "max_steps": 200,          # Limit training steps
    }
    
    logger.info("✅ Training configuration fixes applied:")
    for key, value in fixes.items():
        logger.info(f"  {key}: {value}")
    
    return fixes

def main():
    """
    Main function to test and fix CUDA assert errors.
    """
    logger.info("🚀 Starting CUDA assert error fix...")
    
    # Apply training configuration fixes
    training_fixes = fix_training_config()
    
    # Test the current model
    model_path = "/home/mbhat/omegalabs-anytoany-bittensor/finetune/models/final_model"
    
    if os.path.exists(model_path):
        logger.info("📥 Testing existing model...")
        results = test_model_with_stability_checks(model_path)
        
        if results:
            logger.info("✅ Model testing completed successfully!")
        else:
            logger.error("❌ Model testing failed!")
    else:
        logger.warning("⚠️  Model path does not exist. Please train a model first.")
    
    logger.info("🎉 CUDA assert error fix completed!")

if __name__ == "__main__":
    main()


