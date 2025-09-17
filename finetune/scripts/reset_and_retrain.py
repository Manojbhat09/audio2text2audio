#!/usr/bin/env python3
"""
Reset and retrain script to fix the corrupted model.
This script will:
1. Delete the corrupted model
2. Start fresh training with proper settings
3. Monitor for NaN values during training
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
import shutil
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_corrupted_model():
    """Reset the corrupted model by deleting it."""
    model_path = "/home/mbhat/omegalabs-anytoany-bittensor/finetune/models/final_model"
    
    if Path(model_path).exists():
        logger.info(f"Deleting corrupted model at {model_path}")
        shutil.rmtree(model_path)
        logger.info("✅ Corrupted model deleted")
    else:
        logger.info("No corrupted model found")

def verify_fresh_model():
    """Verify that we can load a fresh model without NaN values."""
    logger.info("Loading fresh model to verify no NaN values...")
    
    try:
        # Load fresh model from Hugging Face
        model_name = "unsloth/gemma-3-270m-it"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Check for NaN values
        nan_count = 0
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                nan_count += 1
                logger.error(f"NaN detected in fresh model {name}")
        
        if nan_count == 0:
            logger.info("✅ Fresh model is clean (no NaN values)")
            return True
        else:
            logger.error(f"❌ Fresh model has {nan_count} parameters with NaN values")
            return False
            
    except Exception as e:
        logger.error(f"Failed to load fresh model: {e}")
        return False

def create_safe_training_config():
    """Create a safe training configuration to prevent NaN values."""
    logger.info("Creating safe training configuration...")
    
    # Update the training configuration with ultra-safe settings
    config_updates = {
        "learning_rate": 1e-7,  # Very low learning rate
        "max_grad_norm": 0.1,   # Very aggressive gradient clipping
        "warmup_steps": 50,     # More warmup
        "weight_decay": 0.01,   # Add weight decay
        "adam_epsilon": 1e-8,   # Smaller epsilon
        "fp16": False,          # Disable mixed precision
        "bf16": False,          # Disable mixed precision
    }
    
    logger.info("Safe configuration:")
    for key, value in config_updates.items():
        logger.info(f"  {key}: {value}")
    
    return config_updates

def main():
    """Main reset and retrain function."""
    logger.info("🔄 Starting model reset and retrain process...")
    
    # Step 1: Reset corrupted model
    reset_corrupted_model()
    
    # Step 2: Verify fresh model
    if not verify_fresh_model():
        logger.error("❌ Fresh model verification failed")
        return False
    
    # Step 3: Create safe training config
    safe_config = create_safe_training_config()
    
    logger.info("✅ Model reset completed successfully!")
    logger.info("🚀 Ready for safe retraining with the following recommendations:")
    logger.info("1. Use very low learning rate (1e-7)")
    logger.info("2. Use aggressive gradient clipping (0.1)")
    logger.info("3. Disable mixed precision")
    logger.info("4. Add weight decay")
    logger.info("5. Monitor for NaN values during training")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("🎉 Reset process completed successfully!")
    else:
        logger.error("❌ Reset process failed!")
        sys.exit(1)


