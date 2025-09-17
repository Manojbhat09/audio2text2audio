#!/usr/bin/env python3
"""
Test simple fine-tuning to see if the model can learn basic tasks.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import logging
from torch.utils.data import Dataset

# Enable synchronous CUDA execution for better debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDataset(Dataset):
    """Simple dataset for testing."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.examples = [
            "What is 2+2? Answer: 4",
            "What is 3+3? Answer: 6", 
            "What is 4+4? Answer: 8",
            "What is 5+5? Answer: 10",
            "What is 6+6? Answer: 12",
        ]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Create labels (same as input_ids for causal LM)
        inputs["labels"] = inputs["input_ids"].clone()
        
        return {k: v.squeeze() for k, v in inputs.items()}

def test_simple_finetuning():
    """Test simple fine-tuning to see if the model can learn."""
    
    # Load model
    model_path = "/home/mbhat/omegalabs-anytoany-bittensor/finetune/models/final_model"
    logger.info(f"Loading model from {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Test before training
    logger.info("=== Testing before training ===")
    test_prompt = "What is 7+7? Answer:"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(generated[0], skip_special_tokens=True)
        logger.info(f"Before training: {response}")
    
    # Create simple dataset
    dataset = SimpleDataset(tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./test_output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=5e-5,
        logging_steps=1,
        save_steps=1000,
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("=== Training ===")
    trainer.train()
    
    # Test after training
    logger.info("=== Testing after training ===")
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(generated[0], skip_special_tokens=True)
        logger.info(f"After training: {response}")

if __name__ == "__main__":
    test_simple_finetuning()
