#!/usr/bin/env python3
"""
Improve training performance by optimizing hyperparameters and configuration.
NO DATA CHANGES - only training improvements.
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from grpo_trainer import DatasetClassificationTrainer
from configs.training_config import TrainingConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_optimized_config():
    """Create optimized training configuration with better hyperparameters."""
    config = TrainingConfig()
    
    # Better model - use larger DialoGPT
    config.model_name = "microsoft/DialoGPT-medium"  # 345M parameters vs 117M
    
    # Optimized training parameters
    config.learning_rate = 2e-5  # Higher learning rate for better learning
    config.max_steps = 2000  # More training steps
    config.samples_per_dataset = 200  # More data per dataset
    config.batch_size = 8  # Larger batch size for better gradients
    config.gradient_accumulation_steps = 2  # Effective batch size = 16
    
    # Better generation parameters
    config.max_prompt_length = 512  # Longer context
    config.max_completion_length = 64  # Longer responses
    config.temperature = 0.1  # Much lower temperature for focused generation
    config.top_p = 0.95  # Higher top_p for better diversity
    config.top_k = 40  # Lower top_k for more focused generation
    
    # Improved evaluation and saving
    config.eval_steps = 100
    config.save_steps = 100
    config.logging_steps = 10
    
    # Better regularization
    config.weight_decay = 0.01
    config.warmup_steps = 200
    config.lr_scheduler_type = "cosine"  # Better learning rate schedule
    
    # Disable wandb to avoid API issues
    config.use_wandb = False
    
    return config

def main():
    """Main function for optimized training."""
    parser = argparse.ArgumentParser(description="Optimize training hyperparameters")
    parser.add_argument("--datasets", nargs="+", 
                       default=["ifeval", "commoneval", "wildvoice"],
                       help="Datasets to train on")
    parser.add_argument("--samples-per-dataset", type=int, default=200,
                       help="Samples per dataset")
    parser.add_argument("--max-steps", type=int, default=2000,
                       help="Maximum training steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--model-name", type=str, 
                       default="microsoft/DialoGPT-medium",
                       help="Model name")
    parser.add_argument("--output-dir", type=str, default="optimized_models",
                       help="Output directory")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    try:
        logger.info("🚀 Starting optimized training (NO DATA CHANGES)...")
        
        # Create optimized config
        config = create_optimized_config()
        config.model_name = args.model_name
        config.samples_per_dataset = args.samples_per_dataset
        config.max_steps = args.max_steps
        config.learning_rate = args.learning_rate
        config.batch_size = args.batch_size
        config.output_dir = args.output_dir
        
        if args.verbose:
            config.logging_steps = 1
        
        logger.info(f"Optimized Configuration:")
        logger.info(f"  Model: {config.model_name}")
        logger.info(f"  Learning rate: {config.learning_rate}")
        logger.info(f"  Max steps: {config.max_steps}")
        logger.info(f"  Batch size: {config.batch_size}")
        logger.info(f"  Temperature: {config.temperature}")
        logger.info(f"  Top-p: {config.top_p}")
        logger.info(f"  Top-k: {config.top_k}")
        logger.info(f"  Samples per dataset: {config.samples_per_dataset}")
        
        # Initialize trainer
        trainer = DatasetClassificationTrainer(config)
        
        # Setup model and data
        logger.info("Setting up model...")
        trainer.setup_model()
        
        logger.info("Setting up data...")
        trainer.setup_data()
        
        # Train
        logger.info("Starting optimized training...")
        trainer.train()
        
        logger.info("🎉 Optimized training completed!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
