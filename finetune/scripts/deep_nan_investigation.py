#!/usr/bin/env python3
"""
Deep investigation into NaN values in the model.
This script will test the model step by step to identify the exact source.
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

def test_model_step_by_step():
    """Test the model step by step to identify where NaN values appear."""
    logger.info("🔍 Starting step-by-step model investigation...")
    
    # Load fresh model
    model_name = "unsloth/gemma-3-270m-it"
    logger.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    logger.info("✅ Model loaded successfully")
    
    # Test with simple input
    test_prompt = "Hello"
    logger.info(f"Testing with simple prompt: '{test_prompt}'")
    
    # Tokenize
    inputs = tokenizer(test_prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    logger.info(f"Input shape: {inputs['input_ids'].shape}")
    logger.info(f"Input tokens: {inputs['input_ids']}")
    
    # Test embedding layer
    logger.info("Testing embedding layer...")
    try:
        with torch.no_grad():
            embeddings = model.model.embed_tokens(inputs['input_ids'])
            logger.info(f"Embeddings shape: {embeddings.shape}")
            logger.info(f"Embeddings min: {embeddings.min().item():.6f}")
            logger.info(f"Embeddings max: {embeddings.max().item():.6f}")
            logger.info(f"Embeddings mean: {embeddings.mean().item():.6f}")
            
            if torch.isnan(embeddings).any():
                logger.error("❌ NaN detected in embeddings!")
                return False
            else:
                logger.info("✅ Embeddings are clean")
    except Exception as e:
        logger.error(f"❌ Embedding test failed: {e}")
        return False
    
    # Test first layer
    logger.info("Testing first transformer layer...")
    try:
        with torch.no_grad():
            hidden_states = embeddings
            layer = model.model.layers[0]
            
            # Test attention
            logger.info("  Testing attention...")
            attn_output = layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False
            )
            
            if torch.isnan(attn_output[0]).any():
                logger.error("❌ NaN detected in attention output!")
                return False
            else:
                logger.info("✅ Attention output is clean")
            
            # Test MLP
            logger.info("  Testing MLP...")
            mlp_output = layer.mlp(hidden_states)
            
            if torch.isnan(mlp_output).any():
                logger.error("❌ NaN detected in MLP output!")
                return False
            else:
                logger.info("✅ MLP output is clean")
                
    except Exception as e:
        logger.error(f"❌ Layer test failed: {e}")
        return False
    
    # Test full forward pass
    logger.info("Testing full forward pass...")
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            logger.info(f"Logits shape: {logits.shape}")
            logger.info(f"Logits min: {logits.min().item():.6f}")
            logger.info(f"Logits max: {logits.max().item():.6f}")
            logger.info(f"Logits mean: {logits.mean().item():.6f}")
            
            if torch.isnan(logits).any():
                logger.error("❌ NaN detected in logits!")
                return False
            else:
                logger.info("✅ Logits are clean")
                
    except Exception as e:
        logger.error(f"❌ Full forward pass failed: {e}")
        return False
    
    return True

def test_with_different_dtypes():
    """Test with different data types to see if that's the issue."""
    logger.info("🔍 Testing with different data types...")
    
    model_name = "unsloth/gemma-3-270m-it"
    
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        logger.info(f"Testing with dtype: {dtype}")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto"
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            inputs = tokenizer("Hello", return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                if torch.isnan(logits).any():
                    logger.error(f"❌ NaN detected with {dtype}")
                else:
                    logger.info(f"✅ Clean with {dtype}")
                    
        except Exception as e:
            logger.error(f"❌ Failed with {dtype}: {e}")

def test_with_different_models():
    """Test with different models to see if it's model-specific."""
    logger.info("🔍 Testing with different models...")
    
    models_to_test = [
        "microsoft/DialoGPT-small",
        "gpt2",
        "distilgpt2"
    ]
    
    for model_name in models_to_test:
        logger.info(f"Testing model: {model_name}")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            inputs = tokenizer("Hello", return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                if torch.isnan(logits).any():
                    logger.error(f"❌ NaN detected with {model_name}")
                else:
                    logger.info(f"✅ Clean with {model_name}")
                    
        except Exception as e:
            logger.error(f"❌ Failed with {model_name}: {e}")

def main():
    """Main investigation function."""
    logger.info("🚀 Starting deep NaN investigation...")
    
    # Test 1: Step by step
    if test_model_step_by_step():
        logger.info("✅ Step-by-step test passed")
    else:
        logger.error("❌ Step-by-step test failed")
    
    # Test 2: Different dtypes
    test_with_different_dtypes()
    
    # Test 3: Different models
    test_with_different_models()
    
    logger.info("🔍 Investigation completed")

if __name__ == "__main__":
    main()


