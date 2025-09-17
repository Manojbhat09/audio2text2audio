#!/usr/bin/env python3
"""
Debug script to identify and fix NaN values in the training pipeline.
"""

import sys
import os
# Enable synchronous CUDA execution for better debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Enable PyTorch anomaly detection
import torch
torch.autograd.set_detect_anomaly(True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_model_weights(model):
    """Check model weights for NaN or infinite values."""
    logger.info("Checking model weights...")
    
    nan_params = []
    inf_params = []
    
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_params.append(name)
            logger.warning(f"NaN detected in {name}")
        
        if torch.isinf(param).any():
            inf_params.append(name)
            logger.warning(f"Inf detected in {name}")
    
    logger.info(f"Found {len(nan_params)} parameters with NaN values")
    logger.info(f"Found {len(inf_params)} parameters with Inf values")
    
    return nan_params, inf_params

def check_input_data(inputs):
    """Check input data for NaN or infinite values."""
    logger.info("Checking input data...")
    
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                logger.warning(f"NaN detected in input {key}")
            if torch.isinf(value).any():
                logger.warning(f"Inf detected in input {key}")

def test_forward_pass(model, tokenizer, test_prompt="Hello, how are you?"):
    """Test forward pass to identify where NaN values appear."""
    logger.info("Testing forward pass...")
    
    # Tokenize input
    inputs = tokenizer(test_prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Check input data
    check_input_data(inputs)
    
    # Forward pass with monitoring
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            logger.info(f"Logits shape: {logits.shape}")
            logger.info(f"Logits min: {logits.min().item():.6f}")
            logger.info(f"Logits max: {logits.max().item():.6f}")
            logger.info(f"Logits mean: {logits.mean().item():.6f}")
            logger.info(f"Logits std: {logits.std().item():.6f}")
            
            # Check for NaN/Inf
            if torch.isnan(logits).any():
                logger.error("NaN detected in logits!")
                nan_mask = torch.isnan(logits)
                logger.error(f"NaN count: {nan_mask.sum().item()}")
                logger.error(f"NaN percentage: {nan_mask.float().mean().item() * 100:.2f}%")
            
            if torch.isinf(logits).any():
                logger.error("Inf detected in logits!")
                inf_mask = torch.isinf(logits)
                logger.error(f"Inf count: {inf_mask.sum().item()}")
                logger.error(f"Inf percentage: {inf_mask.float().mean().item() * 100:.2f}%")
            
            return logits
            
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        return None

def test_loss_computation(model, tokenizer, test_prompt="Hello, how are you?"):
    """Test loss computation to identify where NaN values appear."""
    logger.info("Testing loss computation...")
    
    # Tokenize input
    inputs = tokenizer(test_prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Add labels for loss computation
    inputs["labels"] = inputs["input_ids"].clone()
    
    try:
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Compute loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        logger.info(f"Loss value: {loss.item():.6f}")
        
        if torch.isnan(loss):
            logger.error("NaN loss detected!")
        if torch.isinf(loss):
            logger.error("Inf loss detected!")
        
        return loss
        
    except Exception as e:
        logger.error(f"Loss computation failed: {e}")
        return None

def main():
    """Main debugging function."""
    model_path = "/home/mbhat/omegalabs-anytoany-bittensor/finetune/models/final_model"
    
    logger.info("🔍 Starting NaN debugging...")
    
    try:
        # Load model
        logger.info("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        logger.info("✅ Model loaded successfully")
        
        # Check model weights
        nan_params, inf_params = check_model_weights(model)
        
        # Test forward pass
        logits = test_forward_pass(model, tokenizer)
        
        # Test loss computation
        loss = test_loss_computation(model, tokenizer)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("🔍 NaN DEBUGGING SUMMARY")
        logger.info("="*60)
        logger.info(f"Model weights with NaN: {len(nan_params)}")
        logger.info(f"Model weights with Inf: {len(inf_params)}")
        logger.info(f"Forward pass successful: {logits is not None}")
        logger.info(f"Loss computation successful: {loss is not None}")
        
        if logits is not None:
            logger.info(f"Logits contain NaN: {torch.isnan(logits).any().item()}")
            logger.info(f"Logits contain Inf: {torch.isinf(logits).any().item()}")
        
        if loss is not None:
            logger.info(f"Loss is NaN: {torch.isnan(loss).item()}")
            logger.info(f"Loss is Inf: {torch.isinf(loss).item()}")
        
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Debugging failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


