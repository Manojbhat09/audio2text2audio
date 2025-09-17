#!/usr/bin/env python3
"""
Simple text-based training script for Qwen model using VoiceBench text data for dataset classification.
This avoids audio processing complexity and focuses on text-based classification.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
import json
from typing import Dict, List, Any
import numpy as np
from datasets import load_dataset

# Enable synchronous CUDA execution for better debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VoiceBenchTextDataset(Dataset):
    """Dataset for VoiceBench text-based classification training."""
    
    def __init__(self, tokenizer, max_length=512, samples_per_dataset=50):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples_per_dataset = samples_per_dataset
        
        # Load VoiceBench dataset with specific configs
        self.dataset = {}
        configs = ['ifeval', 'commoneval', 'wildvoice']
        
        for config in configs:
            try:
                self.dataset[config] = load_dataset("hlt-lab/voicebench", config)
                # VoiceBench only has 'test' splits, use that for training
                if 'test' in self.dataset[config]:
                    logger.info(f"Loaded {config} config with {len(self.dataset[config]['test'])} test samples")
                else:
                    logger.warning(f"No test split found for {config}")
                    self.dataset[config] = {'test': []}
            except Exception as e:
                logger.warning(f"Failed to load {config}: {e}")
                # Create empty dataset
                self.dataset[config] = {'test': []}
        
        # Prepare samples
        self.samples = self._prepare_samples()
        
        logger.info(f"Prepared {len(self.samples)} training samples")
    
    def _prepare_samples(self):
        """Prepare training samples from VoiceBench dataset."""
        samples = []
        
        # Get samples from each config
        for config_name, config_data in self.dataset.items():
            if 'test' in config_data and len(config_data['test']) > 0:
                test_data = config_data['test']
                logger.info(f"Processing {config_name} config with {len(test_data)} samples")
                
                # Take up to samples_per_dataset from each config
                selected_samples = test_data[:self.samples_per_dataset]
                logger.info(f"Selected {len(selected_samples)} samples from {config_name}")
                
                for idx, sample in enumerate(selected_samples):
                    samples.append({
                        'dataset_name': config_name,
                        'sample': sample,
                        'split': 'test'
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a training sample."""
        sample_data = self.samples[idx]
        dataset_name = sample_data['dataset_name']
        sample = sample_data['sample']
        
        try:
            # Extract text from the sample
            # VoiceBench samples have different structures, try to get text
            text = ""
            if 'text' in sample:
                text = sample['text']
            elif 'instruction' in sample:
                text = sample['instruction']
            elif 'prompt' in sample:
                text = sample['prompt']
            elif 'question' in sample:
                text = sample['question']
            else:
                # Try to find any text field
                for key, value in sample.items():
                    if isinstance(value, str) and len(value) > 10:
                        text = value
                        break
            
            if not text:
                text = f"Sample from {dataset_name} dataset"
            
            # Create training prompt
            prompt = self._create_training_prompt(text, dataset_name)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Create labels (same as input_ids for causal LM)
            labels = inputs['input_ids'].clone()
            
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': labels.squeeze(),
                'dataset_name': dataset_name,
                'text': text
            }
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            # Return a dummy sample
            dummy_text = f"This is a sample from {dataset_name} dataset."
            prompt = self._create_training_prompt(dummy_text, dataset_name)
            inputs = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': inputs['input_ids'].squeeze(),
                'dataset_name': dataset_name,
                'text': dummy_text
            }
    
    def _create_training_prompt(self, text, dataset_name):
        """Create training prompt for dataset classification."""
        return f"""You are an expert at analyzing text and identifying which type of evaluation dataset it comes from.

Your task is to classify text into one of these categories:
- ifeval: Instruction following evaluation with complex reasoning tasks
- commoneval: Common evaluation benchmark for natural language processing  
- wildvoice: Wild voice data with diverse speaking styles and accents

Analyze the content, style, and structure of the text to determine the most likely source dataset.

Text: "{text}"

Which dataset category does this text most likely come from? Answer with only the dataset name: ifeval, commoneval, or wildvoice.

Answer: {dataset_name}"""

def main():
    parser = argparse.ArgumentParser(description='Train Qwen model on VoiceBench text data')
    parser.add_argument('--model-name', type=str, default='unsloth/Qwen2.5-3B-Instruct', 
                       help='Model name to use')
    parser.add_argument('--samples-per-dataset', type=int, default=50,
                       help='Number of samples per dataset type')
    parser.add_argument('--max-steps', type=int, default=100,
                       help='Maximum training steps')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Training batch size')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--output-dir', type=str, default='./qwen_text_outputs',
                       help='Output directory')
    parser.add_argument('--model-dir', type=str, default='./qwen_text_models',
                       help='Model save directory')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("QWEN VOICEBENCH TEXT TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Samples per dataset: {args.samples_per_dataset}")
    logger.info(f"Max steps: {args.max_steps}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Model directory: {args.model_dir}")
    logger.info("=" * 60)
    
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("✓ Model and tokenizer loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Create dataset
    logger.info("Creating VoiceBench text dataset...")
    try:
        dataset = VoiceBenchTextDataset(
            tokenizer=tokenizer,
            max_length=args.max_length,
            samples_per_dataset=args.samples_per_dataset
        )
        logger.info(f"✓ Dataset created with {len(dataset)} samples")
        
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        return
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        fp16=False,  # Disable mixed precision for stability
        bf16=False,
        gradient_accumulation_steps=1,
        warmup_steps=10,
        weight_decay=0.01,
        max_grad_norm=1.0,
        report_to=[],  # Disable wandb for now
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("✓ Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return
    
    # Save model
    model_save_path = Path(args.model_dir) / "final_qwen_text_model"
    logger.info(f"Saving model to {model_save_path}")
    try:
        trainer.save_model(str(model_save_path))
        tokenizer.save_pretrained(str(model_save_path))
        logger.info("✓ Model saved successfully")
        
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return
    
    # Test the model
    logger.info("Testing the trained model...")
    test_prompts = [
        "Text: 'Please follow these instructions step by step.' Answer:",
        "Text: 'This is a common evaluation task.' Answer:",
        "Text: 'Hello, this is a natural conversation.' Answer:",
    ]
    
    model.eval()
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts, 1):
            logger.info(f"\nTest {i}: {prompt}")
            
            inputs = tokenizer(prompt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            generated = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            response = tokenizer.decode(generated[0], skip_special_tokens=True)
            assistant_response = response[len(prompt):].strip()
            logger.info(f"Response: '{assistant_response}'")
    
    logger.info("🎉 Training and testing completed successfully!")

if __name__ == "__main__":
    main()
