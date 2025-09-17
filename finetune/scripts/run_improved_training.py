#!/usr/bin/env python3
"""
Run improved training with better configuration and parameters.
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

def create_improved_config():
    """Create improved training configuration."""
    config = TrainingConfig()
    
    # Better model settings
    config.model_name = "microsoft/DialoGPT-medium"  # Larger model
    config.whisper_model_path = "openai/whisper-small"  # Better transcription
    
    # Improved training parameters
    config.learning_rate = 1e-5  # Slightly higher learning rate
    config.max_steps = 1000  # More training steps
    config.samples_per_dataset = 200  # More data per dataset
    config.batch_size = 4  # Larger batch size
    config.gradient_accumulation_steps = 4  # Effective batch size = 16
    
    # Better generation parameters
    config.max_prompt_length = 256
    config.max_completion_length = 32  # Shorter, more focused responses
    config.temperature = 0.3  # Lower temperature for more focused generation
    config.top_p = 0.9
    config.top_k = 50
    
    # Improved evaluation
    config.eval_steps = 100
    config.save_steps = 100
    config.logging_steps = 10
    
    # Better regularization
    config.weight_decay = 0.01
    config.warmup_steps = 100
    
    # Wandb settings
    config.wandb_project = "grpo-dataset-classification-improved"
    config.wandb_run_name = "improved-training"
    
    return config

def main():
    """Main function for improved training."""
    parser = argparse.ArgumentParser(description="Run improved GRPO training")
    parser.add_argument("--datasets", nargs="+", 
                       default=["ifeval", "commoneval", "wildvoice"],
                       help="Datasets to train on")
    parser.add_argument("--samples-per-dataset", type=int, default=200,
                       help="Samples per dataset")
    parser.add_argument("--max-steps", type=int, default=1000,
                       help="Maximum training steps")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--model-name", type=str, 
                       default="microsoft/DialoGPT-medium",
                       help="Model name")
    parser.add_argument("--output-dir", type=str, default="improved_models",
                       help="Output directory")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    try:
        logger.info("🚀 Starting improved training...")
        
        # Create improved config
        config = create_improved_config()
        config.model_name = args.model_name
        config.samples_per_dataset = args.samples_per_dataset
        config.max_steps = args.max_steps
        config.learning_rate = args.learning_rate
        config.batch_size = args.batch_size
        config.output_dir = args.output_dir
        
        if args.verbose:
            config.logging_steps = 1
        
        logger.info(f"Configuration:")
        logger.info(f"  Model: {config.model_name}")
        logger.info(f"  Samples per dataset: {config.samples_per_dataset}")
        logger.info(f"  Max steps: {config.max_steps}")
        logger.info(f"  Learning rate: {config.learning_rate}")
        logger.info(f"  Batch size: {config.batch_size}")
        logger.info(f"  Output dir: {config.output_dir}")
        
        # Initialize trainer
        trainer = DatasetClassificationTrainer(config)
        
        # Setup model and data
        logger.info("Setting up model...")
        trainer.setup_model()
        
        logger.info("Setting up data...")
        trainer.setup_data()
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        logger.info("🎉 Improved training completed!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
