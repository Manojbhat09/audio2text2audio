#!/usr/bin/env python3
"""
Improved training script with better configuration and data handling.
This script addresses the issues found in the model testing.
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import json

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

class ImprovedTrainingConfig(TrainingConfig):
    """Improved training configuration with better settings."""
    
    def __init__(self):
        super().__init__()
        
        # Improved model settings
        self.model_name = "microsoft/DialoGPT-medium"  # Larger model
        self.whisper_model_path = "openai/whisper-small"  # Better transcription
        
        # Improved training parameters
        self.learning_rate = 1e-5  # Slightly higher learning rate
        self.max_steps = 1000  # More training steps
        self.samples_per_dataset = 200  # More data per dataset
        self.batch_size = 4  # Larger batch size
        self.gradient_accumulation_steps = 4  # Effective batch size = 16
        
        # Better generation parameters
        self.max_prompt_length = 256
        self.max_completion_length = 64  # Shorter, more focused responses
        self.temperature = 0.3  # Lower temperature for more focused generation
        self.top_p = 0.9
        self.top_k = 50
        
        # Improved evaluation
        self.eval_steps = 100
        self.save_steps = 100
        self.logging_steps = 10
        
        # Better regularization
        self.weight_decay = 0.01
        self.warmup_steps = 100
        
        # Wandb settings
        self.wandb_project = "grpo-dataset-classification-improved"
        self.wandb_run_name = "improved-training"

def create_improved_prompts():
    """Create better training prompts for dataset classification."""
    
    # More diverse and specific prompts
    prompt_templates = [
        "Classify this audio transcript as one of these datasets: ifeval, commoneval, wildvoice. Transcript: {transcript}",
        "Which dataset does this audio belong to? Options: ifeval, commoneval, wildvoice. Audio: {transcript}",
        "Dataset classification task. Choose from: ifeval, commoneval, wildvoice. Input: {transcript}",
        "Identify the dataset for this audio transcript. Categories: ifeval, commoneval, wildvoice. Text: {transcript}",
        "Classify the following audio as ifeval, commoneval, or wildvoice: {transcript}"
    ]
    
    # Better response templates
    response_templates = {
        "ifeval": [
            "ifeval",
            "This belongs to the ifeval dataset.",
            "Dataset: ifeval",
            "Classification: ifeval",
            "ifeval dataset"
        ],
        "commoneval": [
            "commoneval", 
            "This belongs to the commoneval dataset.",
            "Dataset: commoneval",
            "Classification: commoneval",
            "commoneval dataset"
        ],
        "wildvoice": [
            "wildvoice",
            "This belongs to the wildvoice dataset.", 
            "Dataset: wildvoice",
            "Classification: wildvoice",
            "wildvoice dataset"
        ]
    }
    
    return prompt_templates, response_templates

def create_balanced_dataset():
    """Create a more balanced and diverse training dataset."""
    
    # Sample transcripts for each dataset (more diverse)
    sample_transcripts = {
        "ifeval": [
            "Follow these instructions step by step to complete the task.",
            "Given the following instructions, provide a detailed response.",
            "Execute the following command and explain your reasoning.",
            "Please follow the instructions carefully and show your work.",
            "Complete this instruction following task with detailed steps.",
            "Given these instructions, what would you do?",
            "Follow the step-by-step instructions provided.",
            "Execute the given instructions and explain your approach.",
            "Please complete this instruction following evaluation.",
            "Given the instructions, provide a comprehensive answer."
        ],
        "commoneval": [
            "This is a common evaluation benchmark for language understanding.",
            "Standard evaluation task for natural language processing models.",
            "Common benchmark dataset for evaluating language models.",
            "This is a typical evaluation example from CommonEval.",
            "Standard NLP evaluation task with multiple choice questions.",
            "Common evaluation benchmark for assessing model performance.",
            "Typical evaluation example from the CommonEval dataset.",
            "Standard benchmark task for language model evaluation.",
            "Common evaluation dataset for testing model capabilities.",
            "This is a standard evaluation example from CommonEval."
        ],
        "wildvoice": [
            "This is a natural conversation with diverse speaking styles.",
            "Wild voice data with various accents and speaking patterns.",
            "Natural speech sample with conversational characteristics.",
            "Diverse voice data with different speaking styles and accents.",
            "Wild voice sample with natural conversational patterns.",
            "This audio contains natural speech with various characteristics.",
            "Wild voice data with diverse speaking patterns and accents.",
            "Natural conversation sample with varied speaking styles.",
            "This is a wild voice sample with conversational characteristics.",
            "Diverse voice data with natural speaking patterns and accents."
        ]
    }
    
    return sample_transcripts

def main():
    """Main function for improved training."""
    parser = argparse.ArgumentParser(description="Improved GRPO training")
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
    
    args = parser.parse_args()
    
    try:
        logger.info("🚀 Starting improved training...")
        
        # Create improved config
        config = ImprovedTrainingConfig()
        config.model_name = args.model_name
        config.samples_per_dataset = args.samples_per_dataset
        config.max_steps = args.max_steps
        config.learning_rate = args.learning_rate
        config.batch_size = args.batch_size
        config.output_dir = args.output_dir
        
        # Initialize trainer
        trainer = DatasetClassificationTrainer(config)
        
        # Setup model and data
        trainer.setup_model()
        trainer.setup_data()
        
        # Train
        trainer.train()
        
        logger.info("🎉 Improved training completed!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
