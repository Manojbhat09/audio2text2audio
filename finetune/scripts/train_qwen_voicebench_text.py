#!/usr/bin/env python3
"""
Train Qwen model using VoiceBench text data for dataset classification.
Uses existing Whisper model and VoiceBench text prompts without audio processing.
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

# Add src to path for whisper transcriber
sys.path.append(str(Path(__file__).parent.parent / "src"))

from whisper_transcriber import WhisperTranscriber

# Enable synchronous CUDA execution for better debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VoiceBenchTextDataset(Dataset):
    """Dataset for VoiceBench text-based classification training using existing infrastructure."""
    
    def __init__(self, tokenizer, max_length=512, samples_per_dataset=50):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples_per_dataset = samples_per_dataset
        
        # Initialize Whisper transcriber (we have this working)
        whisper_model_path = "/home/mbhat/omegalabs-anytoany-bittensor/elephant-04/models/wpt/wpt.pt"
        self.transcriber = WhisperTranscriber(whisper_model_path)
        
        # Create synthetic VoiceBench-style data based on the patterns we know
        self.samples = self._create_synthetic_voicebench_data()
        
        logger.info(f"Prepared {len(self.samples)} training samples")
    
    def _create_synthetic_voicebench_data(self):
        """Create synthetic VoiceBench-style data based on known patterns."""
        samples = []
        
        # ifeval samples - instruction following with complex reasoning
        ifeval_samples = [
            "Please follow these instructions step by step: First, identify the main topic. Then, provide three supporting arguments. Finally, conclude with a summary.",
            "Your task is to analyze the following text and extract the key information. Make sure to follow the format exactly as specified.",
            "Complete this instruction: Write a response that contains exactly 5 sentences, each starting with a different letter of the alphabet.",
            "Follow this instruction precisely: Create a list of 10 items, but do not use any commas in your response.",
            "Your assignment is to write a paragraph that contains at least 3 placeholders in square brackets, like [this].",
            "Please respond to this question: What are the main components of a successful project? Use exactly 50 words.",
            "Follow these steps: 1) Read the question carefully 2) Think about the answer 3) Write your response 4) Check for accuracy",
            "Your task is to create a response that follows this exact format: SECTION 1: Introduction, SECTION 2: Main points, SECTION 3: Conclusion",
            "Please answer this question in all lowercase letters only, without any capital letters.",
            "Complete this instruction: Write a response that ends with the exact phrase 'This completes the task.'",
            "Your assignment is to create a JSON response with the following structure: {\"answer\": \"your response here\", \"confidence\": 7}",
            "Please follow this instruction: Write exactly 3 sentences, each containing the word 'analysis'.",
            "Your task is to respond without using any of these forbidden words: good, bad, nice, terrible.",
            "Complete this instruction: Write a response in ALL CAPITAL LETTERS that explains the concept of machine learning.",
            "Please create a response that contains at least 5 bullet points, each starting with a dash.",
            "Your assignment is to write a paragraph that rephrases the following text: 'The weather is very nice today.'",
            "Follow this instruction: Create a response that contains exactly 2 sections, each starting with 'SECTION X:'",
            "Please answer this question: What is the capital of France? Your response should be exactly one word.",
            "Your task is to write a response that contains the keywords: technology, innovation, future, development.",
            "Complete this instruction: Write a response that is exactly 25 words long, no more, no less."
        ]
        
        # commoneval samples - common evaluation benchmark tasks
        commoneval_samples = [
            "What is the capital city of Australia?",
            "Explain the concept of photosynthesis in simple terms.",
            "List three benefits of regular exercise.",
            "What are the main causes of climate change?",
            "Describe the process of making bread from flour.",
            "What is the difference between a virus and bacteria?",
            "Explain why the sky appears blue during the day.",
            "What are the primary colors and how do they mix?",
            "Describe the water cycle in nature.",
            "What is the purpose of the United Nations?",
            "Explain the concept of gravity and its effects.",
            "What are the main components of a computer?",
            "Describe the process of digestion in humans.",
            "What is the difference between renewable and non-renewable energy?",
            "Explain how a rainbow is formed.",
            "What are the benefits of reading books regularly?",
            "Describe the structure of an atom.",
            "What is the importance of biodiversity in ecosystems?",
            "Explain the concept of supply and demand in economics.",
            "What are the main stages of human development?"
        ]
        
        # wildvoice samples - diverse speaking styles and natural conversation
        wildvoice_samples = [
            "Hey there! How's it going today? I'm doing pretty well, thanks for asking.",
            "So I was thinking about what we discussed earlier, and I have some ideas to share.",
            "You know what? That's actually a really interesting point you made there.",
            "I'm not sure about that, but I think we should consider all the options first.",
            "Wow, that's amazing! I never thought about it that way before.",
            "Hmm, let me think about this for a moment. What do you think we should do?",
            "That sounds like a great plan! When do you want to get started?",
            "I'm really excited about this project. It's going to be so much fun!",
            "Oh, I see what you mean now. That makes a lot more sense.",
            "You're absolutely right about that. I completely agree with your assessment.",
            "Well, that's one way to look at it, but I think there might be other approaches too.",
            "I'm really looking forward to hearing more about your thoughts on this topic.",
            "That's a good question. Let me think about how to explain this clearly.",
            "I've been working on this for a while now, and I think I'm making good progress.",
            "What do you think about trying a different approach to solve this problem?",
            "I'm not an expert on this, but from what I understand, it seems like a good idea.",
            "That's really cool! I'd love to learn more about how you did that.",
            "I think we're on the right track here. This is looking promising.",
            "You know, I hadn't considered that perspective before. It's quite insightful.",
            "I'm really glad we had this conversation. It's been very helpful."
        ]
        
        # Create samples for each dataset type
        for dataset_name, samples_list in [
            ('ifeval', ifeval_samples),
            ('commoneval', commoneval_samples), 
            ('wildvoice', wildvoice_samples)
        ]:
            # Take up to samples_per_dataset from each type
            selected_samples = samples_list[:self.samples_per_dataset]
            logger.info(f"Selected {len(selected_samples)} samples from {dataset_name}")
            
            for text in selected_samples:
                samples.append({
                    'dataset_name': dataset_name,
                    'text': text,
                    'split': 'train'
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a training sample."""
        sample_data = self.samples[idx]
        dataset_name = sample_data['dataset_name']
        text = sample_data['text']
        
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
            'labels': labels.squeeze()
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
    parser = argparse.ArgumentParser(description='Train Qwen model on VoiceBench-style text data')
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
    parser.add_argument('--output-dir', type=str, default='./qwen_voicebench_outputs',
                       help='Output directory')
    parser.add_argument('--model-dir', type=str, default='./qwen_voicebench_models',
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
    logger.info("Creating VoiceBench-style text dataset...")
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
    model_save_path = Path(args.model_dir) / "final_qwen_voicebench_model"
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
        "Text: 'What is the capital of France?' Answer:",
        "Text: 'Hey there! How's it going today?' Answer:",
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
