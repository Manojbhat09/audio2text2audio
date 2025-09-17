#!/usr/bin/env python3
"""
Main Training Script for Dataset Classification

Trains a Gemma model to classify VoiceBench datasets using GRPO reinforcement learning.
This script handles the complete pipeline from data loading to model training.

python scripts/train_dataset_classifier.py     --model-name microsoft/DialoGPT-small     --samples-per-dataset 200     --max-steps 200     --learning-rate 3e-6     --datasets ifeval commoneval wildvoice --verbose
"""

from __future__ import annotations

import sys
import logging
import argparse
import os
from pathlib import Path

# Enable synchronous CUDA execution for better debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Enable PyTorch anomaly detection to catch NaN values
import torch
torch.autograd.set_detect_anomaly(True)

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.grpo_trainer import create_trainer
from configs.training_config import TrainingConfig, get_default_config, get_small_config, get_large_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train dataset classifier with GRPO")
    
    # Model configuration
    parser.add_argument("--model-name", type=str, default="gpt2",
                       help="Model name to use")
    parser.add_argument("--max-seq-length", type=int, default=1024,
                       help="Maximum sequence length")
    
    # Training configuration
    parser.add_argument("--samples-per-dataset", type=int, default=100,
                       help="Number of samples per dataset")
    parser.add_argument("--max-steps", type=int, default=100,
                       help="Maximum training steps")
    parser.add_argument("--learning-rate", type=float, default=5e-6,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Training batch size")
    
    # Dataset configuration
    parser.add_argument("--datasets", nargs="+", 
                       choices=["ifeval", "commoneval", "wildvoice"],
                       default=["ifeval", "commoneval", "wildvoice"],
                       help="Datasets to use for training")
    parser.add_argument("--sampling-method", type=str, 
                       choices=["random", "first", "last"],
                       default="random",
                       help="Sampling method for dataset")
    
    # Paths
    parser.add_argument("--whisper-model-path", type=str,
                       default="/home/mbhat/omegalabs-anytoany-bittensor/elephant-04/models/wpt/wpt.pt",
                       help="Path to Whisper model")
    parser.add_argument("--output-dir", type=str,
                       default="/home/mbhat/omegalabs-anytoany-bittensor/finetune/outputs",
                       help="Output directory")
    parser.add_argument("--model-dir", type=str,
                       default="/home/mbhat/omegalabs-anytoany-bittensor/finetune/models",
                       help="Model directory")
    
    # Training options
    parser.add_argument("--config", type=str, choices=["default", "small", "large"],
                       default="default",
                       help="Predefined configuration")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience")
    parser.add_argument("--min-delta", type=float, default=0.01,
                       help="Early stopping minimum delta")
    
    # Logging
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    return parser.parse_args()


def create_config_from_args(args) -> TrainingConfig:
    """Create training configuration from command line arguments."""
    
    # Start with base configuration
    if args.config == "small":
        config = get_small_config()
    elif args.config == "large":
        config = get_large_config()
    else:
        config = get_default_config()
    
    # Override with command line arguments
    config.model_name = args.model_name
    config.max_seq_length = args.max_seq_length
    config.samples_per_dataset = args.samples_per_dataset
    config.max_steps = args.max_steps
    config.learning_rate = args.learning_rate
    config.per_device_train_batch_size = args.batch_size
    config.target_datasets = args.datasets
    config.sampling_method = args.sampling_method
    config.whisper_model_path = args.whisper_model_path
    config.output_dir = args.output_dir
    config.model_dir = args.model_dir
    config.patience = args.patience
    config.min_delta = args.min_delta
    
    return config


def setup_logging(verbose: bool):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Update root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Update specific loggers
    for logger_name in ["src.whisper_transcriber", "src.dataset_loader", "src.grpo_trainer"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)


def print_config_summary(config: TrainingConfig):
    """Print a summary of the training configuration."""
    logger.info("=" * 60)
    logger.info("DATASET CLASSIFICATION TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Target datasets: {config.target_datasets}")
    logger.info(f"Samples per dataset: {config.samples_per_dataset}")
    logger.info(f"Sampling method: {config.sampling_method}")
    logger.info(f"Max steps: {config.max_steps}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Batch size: {config.per_device_train_batch_size}")
    logger.info(f"Whisper model: {config.whisper_model_path}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Model directory: {config.model_dir}")
    logger.info(f"Early stopping patience: {config.patience}")
    logger.info("=" * 60)


def main():
    """Main training function."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Setup logging
        setup_logging(args.verbose)
        
        # Create configuration
        config = create_config_from_args(args)
        
        # Print configuration summary
        print_config_summary(config)
        
        # Validate configuration
        if not Path(config.whisper_model_path).exists():
            logger.error(f"Whisper model not found at {config.whisper_model_path}")
            return 1
        
        # Create trainer
        logger.info("Initializing trainer...")
        trainer = create_trainer(config)
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        logger.info("🎉 Training completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
