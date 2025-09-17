#!/usr/bin/env python3
"""
GRPO Trainer for Dataset Classification

Implements GRPO (Group Relative Policy Optimization) training for dataset classification
using the Gemma model with Unsloth optimizations.
"""

from __future__ import annotations

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import asdict

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastModel

from .whisper_transcriber import WhisperTranscriber
from .dataset_loader import OnlineDatasetLoader, TrainingSample
from .reward_functions import create_reward_functions
from configs.training_config import TrainingConfig

logger = logging.getLogger(__name__)


class EarlyStoppingCallback:
    """Early stopping callback for GRPO training."""
    
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = -float('inf')
        self.patience_counter = 0
        self.should_stop = False
    
    def on_log(self, logs: Dict[str, float]) -> Dict[str, Any]:
        """Called when metrics are logged."""
        if 'reward' in logs:
            current_metric = logs['reward']
            
            # Check if we have improvement
            if current_metric > self.best_metric + self.min_delta:
                self.best_metric = current_metric
                self.patience_counter = 0
                logger.info(f"New best reward: {current_metric:.4f}")
            else:
                self.patience_counter += 1
                logger.info(f"No improvement for {self.patience_counter} steps (best: {self.best_metric:.4f})")
                
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered after {self.patience_counter} steps without improvement")
                    self.should_stop = True
                    return {"should_stop": True}
        
        return {}
    
    def on_train_begin(self, logs: Dict[str, Any]):
        """Called at the beginning of training."""
        logger.info("Starting training with early stopping callback")
        self.should_stop = False
    
    def on_train_end(self, logs: Dict[str, Any]):
        """Called at the end of training."""
        if self.should_stop:
            logger.info("Training stopped early due to no improvement")
        else:
            logger.info("Training completed normally")


class DatasetClassificationTrainer:
    """Trainer for dataset classification using GRPO."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the GRPO trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Training state
        self.step = 0
        self.best_score = 0.0
        self.training_history = []
        
        # Initialize components
        self._setup_model()
        self._setup_transcriber()
        self._setup_dataset_loader()
    
    def _setup_model(self):
        """Setup Unsloth model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            
            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                load_in_4bit=False,
                load_in_8bit=False,
                full_finetuning=False,
            )
            
            # Add LoRA adapters
            self.model = FastModel.get_peft_model(
                self.model,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias=self.config.lora_bias,
                random_state=self.config.random_seed,
            )
            
            logger.info("✓ Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup model: {e}")
            raise
    
    def _setup_transcriber(self):
        """Setup Whisper transcriber."""
        self.transcriber = WhisperTranscriber(self.config.whisper_model_path)
        logger.info("✓ Whisper transcriber setup complete")
    
    def _setup_dataset_loader(self):
        """Setup online dataset loader."""
        self.dataset_loader = OnlineDatasetLoader(self.config, self.transcriber)
        logger.info("✓ Dataset loader setup complete")
    
    def _prepare_training_dataset(self) -> Dataset:
        """Prepare the training dataset from online samples."""
        logger.info("Preparing training dataset...")
        
        samples = list(self.dataset_loader.generate_training_samples())
        logger.info(f"Generated {len(samples)} training samples")
        
        if len(samples) == 0:
            raise RuntimeError("No training samples generated")
        
        # Convert to HuggingFace dataset format
        dataset_dict = {
            "prompt": [sample.prompt for sample in samples],
            "answer": [sample.answer for sample in samples]
        }
        
        # Add metadata for analysis
        for key in ["dataset_name", "transcript", "transcript_length"]:
            dataset_dict[key] = [sample.metadata[key] for sample in samples]
        
        dataset = Dataset.from_dict(dataset_dict)
        logger.info(f"Created dataset with {len(dataset)} samples")
        
        # Log dataset distribution
        from collections import Counter
        distribution = Counter(dataset["dataset_name"])
        logger.info(f"Dataset distribution: {dict(distribution)}")
        
        return dataset
    
    def _create_grpo_config(self) -> GRPOConfig:
        """Create GRPO configuration."""
        return GRPOConfig(
            learning_rate=self.config.learning_rate,
            adam_beta1=self.config.adam_beta1,
            adam_beta2=self.config.adam_beta2,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            optim=self.config.optim,
            logging_steps=self.config.logging_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_generations=self.config.num_generations,
            max_prompt_length=self.config.max_prompt_length,
            max_completion_length=self.config.max_completion_length,
            max_steps=self.config.max_steps,
            save_steps=self.config.save_steps,
            max_grad_norm=self.config.max_grad_norm,
            report_to="none",
            output_dir=self.config.output_dir,
            remove_unused_columns=False,
            dataloader_drop_last=False,
        )
    
    def _create_reward_functions(self) -> List[Callable]:
        """Create reward functions for GRPO training."""
        return create_reward_functions(self.config.reward_weights)
    
    def train(self):
        """Main training loop with early stopping."""
        try:
            # Prepare dataset
            train_dataset = self._prepare_training_dataset()
            
            # Create GRPO configuration
            grpo_config = self._create_grpo_config()
            
            # Create reward functions
            reward_funcs = self._create_reward_functions()
            
            # Create trainer
            self.trainer = GRPOTrainer(
                model=self.model,
                processing_class=self.tokenizer,
                reward_funcs=reward_funcs,
                args=grpo_config,
                train_dataset=train_dataset,
            )
            
            # Add early stopping callback
            early_stopping = EarlyStoppingCallback(
                patience=self.config.patience,
                min_delta=self.config.min_delta
            )
            self.trainer.add_callback(early_stopping)
            
            # Save training configuration
            self._save_training_config()
            
            logger.info("Starting GRPO training...")
            start_time = time.time()
            
            # Start training
            self.trainer.train()
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Save final model
            self.save_model()
            
            # Save training history
            self._save_training_history()
            
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def save_model(self):
        """Save the trained model."""
        try:
            save_path = Path(self.config.model_dir) / "final_model"
            save_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving model to {save_path}")
            self.model.save_pretrained(str(save_path))
            self.tokenizer.save_pretrained(str(save_path))
            
            # Save training config
            config_path = save_path / "training_config.json"
            with open(config_path, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
            
            logger.info("✓ Model saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def _save_training_config(self):
        """Save training configuration."""
        config_path = Path(self.config.output_dir) / "training_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        logger.info(f"Training configuration saved to {config_path}")
    
    def _save_training_history(self):
        """Save training history."""
        history_path = Path(self.config.output_dir) / "training_history.json"
        
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Training history saved to {history_path}")
    
    def evaluate_model(self, test_samples: List[TrainingSample]) -> Dict[str, float]:
        """
        Evaluate the model on test samples.
        
        Args:
            test_samples: List of test samples
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        correct = 0
        total = len(test_samples)
        
        for sample in test_samples:
            # Format input
            messages = sample.prompt
            input_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Generate response
            inputs = self.tokenizer(input_text, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_completion_length,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            response = response.strip().lower()
            
            # Check if correct
            if response == sample.answer.lower():
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }


def create_trainer(config: TrainingConfig) -> DatasetClassificationTrainer:
    """
    Create a DatasetClassificationTrainer instance.
    
    Args:
        config: Training configuration
        
    Returns:
        DatasetClassificationTrainer instance
    """
    return DatasetClassificationTrainer(config)


# Test function
def test_trainer():
    """Test the trainer with a small configuration."""
    from configs.training_config import get_small_config
    
    config = get_small_config()
    config.samples_per_dataset = 2
    config.max_steps = 2
    
    print("Testing GRPO trainer...")
    trainer = create_trainer(config)
    print("Trainer created successfully")
    print(f"Model loaded: {trainer.model is not None}")
    print(f"Tokenizer loaded: {trainer.tokenizer is not None}")


if __name__ == "__main__":
    test_trainer()
