#!/usr/bin/env python3
"""
Direct classification training using a simpler approach based on online best practices.
This uses a more direct prompt format and focuses on learning the exact labels.
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
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
import json
from typing import Dict, List, Any
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DirectClassificationDataset(Dataset):
    """Direct classification dataset with simpler, more focused prompts."""
    
    def __init__(self, tokenizer, max_length=128, samples_per_dataset=50):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples_per_dataset = samples_per_dataset
        
        # Create focused training data
        self.samples = self._create_focused_training_data()
        
        logger.info(f"Prepared {len(self.samples)} training samples")
    
    def _create_focused_training_data(self):
        """Create focused training data with very clear examples."""
        samples = []
        
        # ifeval samples - instruction following
        ifeval_samples = [
            "Follow these instructions exactly: Write 3 sentences about cats.",
            "Your task: Create a list with exactly 5 items about weather.",
            "Instruction: Write a paragraph that contains the word 'example' three times.",
            "Complete this: Write exactly 2 sentences, each starting with 'The'.",
            "Follow this format: Write 4 sentences about food, numbered 1-4.",
            "Your assignment: Write a response that ends with 'Task completed.'",
            "Instruction: Create a response with exactly 6 words about nature.",
            "Follow these steps: 1) Write about trees 2) Write about flowers 3) Write about animals",
            "Your task: Write a sentence that contains both 'blue' and 'sky'.",
            "Complete this instruction: Write a paragraph about cars in exactly 3 sentences.",
            "Follow this format: Write 5 sentences about music, each starting with a different letter.",
            "Your assignment: Write a response that contains the phrase 'as requested'.",
            "Instruction: Write exactly 4 words about the ocean.",
            "Follow these steps: Write about the sun, then the moon, then the stars.",
            "Your task: Create a response that starts with 'According to' and ends with 'the end.'",
            "Complete this: Write 3 sentences about books, each containing the word 'read'.",
            "Instruction: Write a response with exactly 8 words about technology.",
            "Follow this format: Write 2 sentences about sports, numbered 1 and 2.",
            "Your assignment: Write a paragraph that contains 'important' twice.",
            "Complete this instruction: Write exactly 5 words about friendship."
        ]
        
        # commoneval samples - factual questions
        commoneval_samples = [
            "What is the capital of France?",
            "How many days are in a week?",
            "What color is the sun?",
            "What is 2 plus 2?",
            "What is the largest planet?",
            "What do plants need to grow?",
            "What is the opposite of hot?",
            "What do we use to write?",
            "What is the name of our planet?",
            "What do we call water when it's frozen?",
            "What is the smallest unit of matter?",
            "What do we call a baby cat?",
            "What is the fastest land animal?",
            "What do we use to measure temperature?",
            "What is the study of living things called?",
            "What do we call the process of water turning to vapor?",
            "What is the name of the star closest to Earth?",
            "What do we call a group of fish?",
            "What is the hardest natural substance?",
            "What do we call the study of numbers?"
        ]
        
        # wildvoice samples - conversational
        wildvoice_samples = [
            "Hey, how are you doing today?",
            "So what do you think about this idea?",
            "You know, I was just thinking about that.",
            "Wow, that's really interesting! Tell me more.",
            "Hmm, I'm not sure about that approach.",
            "That sounds like a great plan to me.",
            "I'm really excited about this project!",
            "Oh, I see what you mean now.",
            "You're absolutely right about that point.",
            "Well, that's one way to look at it.",
            "I'm really looking forward to hearing more.",
            "That's a good question, let me think.",
            "I've been working on this for a while.",
            "What do you think we should do next?",
            "I'm not an expert, but I think it's good.",
            "That's really cool! How did you do that?",
            "I think we're on the right track here.",
            "You know, I hadn't considered that before.",
            "I'm really glad we had this conversation.",
            "Oh wow, that's incredible! I can't believe it."
        ]
        
        # Create samples for each dataset type with very direct prompts
        for dataset_name, samples_list in [
            ('ifeval', ifeval_samples),
            ('commoneval', commoneval_samples), 
            ('wildvoice', wildvoice_samples)
        ]:
            selected_samples = samples_list[:self.samples_per_dataset]
            logger.info(f"Selected {len(selected_samples)} samples from {dataset_name}")
            
            for text in selected_samples:
                # Very direct prompt format
                prompt = f"Text: {text}\nLabel: {dataset_name}"
                
                samples.append({
                    'text': text,
                    'label': dataset_name,
                    'prompt': prompt
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a training sample."""
        sample = self.samples[idx]
        prompt = sample['prompt']
        
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
            'labels': labels.squeeze()
        }

def main():
    parser = argparse.ArgumentParser(description='Train Qwen model with direct classification approach')
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct', 
                       help='Model name to use')
    parser.add_argument('--samples-per-dataset', type=int, default=50,
                       help='Number of samples per dataset type')
    parser.add_argument('--max-steps', type=int, default=200,
                       help='Maximum training steps')
    parser.add_argument('--learning-rate', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Training batch size')
    parser.add_argument('--max-length', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--output-dir', type=str, default='./direct_classification_outputs',
                       help='Output directory')
    parser.add_argument('--model-dir', type=str, default='./direct_classification_models',
                       help='Model save directory')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("DIRECT CLASSIFICATION TRAINING CONFIGURATION")
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
            torch_dtype=torch.float16,
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
    
    # Configure LoRA
    logger.info("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Lower rank for more focused learning
        lora_alpha=16,  # LoRA alpha
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Create dataset
    logger.info("Creating direct classification dataset...")
    try:
        dataset = DirectClassificationDataset(
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
        fp16=True,
        gradient_accumulation_steps=1,
        warmup_steps=20,
        weight_decay=0.01,
        max_grad_norm=1.0,
        report_to=[],
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
    logger.info("Starting direct classification training...")
    try:
        trainer.train()
        logger.info("✓ Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return
    
    # Save model
    model_save_path = Path(args.model_dir) / "final_direct_classification_model"
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
        "Text: Follow these instructions exactly: Write 3 sentences about cats.\nLabel:",
        "Text: What is the capital of France?\nLabel:",
        "Text: Hey, how are you doing today?\nLabel:",
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
                temperature=0.1,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            response = tokenizer.decode(generated[0], skip_special_tokens=True)
            assistant_response = response[len(prompt):].strip()
            logger.info(f"Response: '{assistant_response}'")
    
    logger.info("🎉 Direct classification training and testing completed successfully!")

if __name__ == "__main__":
    main()
