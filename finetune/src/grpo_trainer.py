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
from .simple_grpo_trainer import SimpleGRPOConfig as GRPOConfig, SimpleGRPOTrainer as GRPOTrainer
# from unsloth import FastModel  # Temporarily disabled due to compatibility issues

from .whisper_transcriber import WhisperTranscriber
from .dataset_loader import OnlineDatasetLoader, TrainingSample
from .reward_functions import create_reward_functions
from .wandb_integration import WandbLogger
from .accelerate_compatibility import apply_compatibility_patches
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
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        logger.info(f"Early stopping callback initialized with patience={self.patience}")
        return control
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch."""
        return control
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        return control
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each step."""
        return control
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each step."""
        return control
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when metrics are logged."""
        if logs and 'reward' in logs:
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
                    control.should_training_stop = True
        
        return control
    
    def on_save(self, args, state, control, **kwargs):
        """Called when model is saved."""
        return control
    
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """Called before the optimizer step."""
        return control
    
    def on_optimizer_step(self, args, state, control, **kwargs):
        """Called after the optimizer step."""
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        if self.should_stop:
            logger.info("Training stopped early due to no improvement")
        else:
            logger.info("Training completed normally")
        return control


class DatasetClassificationTrainer:
    """Trainer for dataset classification using GRPO."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the GRPO trainer.
        
        Args:
            config: Training configuration
        """
        # Apply compatibility patches first
        apply_compatibility_patches()
        
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Training state
        self.step = 0
        self.best_score = 0.0
        self.training_history = []
        
        # Setup wandb logging
        self.wandb_logger = WandbLogger(config)
        
        # Initialize components
        self._setup_model()
        self._setup_transcriber()
        self._setup_dataset_loader()
    
    def _setup_model(self):
        """Setup model and tokenizer using standard transformers."""
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float32,  # Use float32 for DialoGPT compatibility
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager"  # Use eager attention to prevent NaN
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
        
        # Convert to HuggingFace dataset format with proper tokenization
        prompts = [str(sample.prompt) for sample in samples]
        answers = [str(sample.answer) for sample in samples]
        
        # Tokenize prompts and answers
        prompt_encodings = self.tokenizer(
            prompts,
            truncation=True,
            padding=True,
            max_length=self.config.max_prompt_length,
            return_tensors="pt"
        )
        
        answer_encodings = self.tokenizer(
            answers,
            truncation=True,
            padding=True,
            max_length=self.config.max_completion_length,
            return_tensors="pt"
        )
        
        # Create dataset with tokenized inputs (only training fields)
        # For language modeling, we need to concatenate prompt and answer
        # and create proper labels with -100 for prompt tokens
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        for i in range(len(samples)):
            # Concatenate prompt and answer
            prompt_ids = prompt_encodings["input_ids"][i]
            answer_ids = answer_encodings["input_ids"][i]
            
            # Remove padding tokens
            prompt_ids = prompt_ids[prompt_ids != self.tokenizer.pad_token_id]
            answer_ids = answer_ids[answer_ids != self.tokenizer.pad_token_id]
            
            # Concatenate prompt and answer
            full_input_ids = torch.cat([prompt_ids, answer_ids])
            
            # Create labels: -100 for prompt tokens, actual tokens for answer
            labels = torch.full_like(full_input_ids, -100)
            labels[len(prompt_ids):] = answer_ids
            
            # Create attention mask
            attention_mask = torch.ones_like(full_input_ids)
            
            input_ids_list.append(full_input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
        
        # Pad sequences to same length
        max_length = max(len(ids) for ids in input_ids_list)
        
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for i in range(len(input_ids_list)):
            # Pad input_ids
            pad_length = max_length - len(input_ids_list[i])
            padded_input = torch.cat([
                input_ids_list[i], 
                torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=input_ids_list[i].dtype)
            ])
            padded_input_ids.append(padded_input)
            
            # Pad attention_mask
            padded_attn = torch.cat([
                attention_mask_list[i],
                torch.zeros(pad_length, dtype=attention_mask_list[i].dtype)
            ])
            padded_attention_mask.append(padded_attn)
            
            # Pad labels
            padded_label = torch.cat([
                labels_list[i],
                torch.full((pad_length,), -100, dtype=labels_list[i].dtype)
            ])
            padded_labels.append(padded_label)
        
        dataset_dict = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels)
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Store metadata separately for analysis
        metadata = {
            "dataset_name": [sample.metadata["dataset_name"] for sample in samples],
            "transcript": [sample.metadata["transcript"] for sample in samples],
            "transcript_length": [sample.metadata["transcript_length"] for sample in samples]
        }
        logger.info(f"Created dataset with {len(dataset)} samples")
        
        # Log dataset distribution
        from collections import Counter
        distribution = Counter(metadata["dataset_name"])
        logger.info(f"Dataset distribution: {dict(distribution)}")
        
        # Log audio samples to wandb
        audio_samples = []
        for i, sample in enumerate(samples[:self.config.wandb_log_samples]):
            if hasattr(sample, 'metadata') and 'audio' in sample.metadata:
                audio_samples.append({
                    'audio': sample.metadata['audio'],
                    'transcript': sample.metadata.get('transcript', ''),
                    'dataset_name': sample.metadata.get('dataset_name', 'unknown'),
                })
        
        if audio_samples:
            self.wandb_logger.log_audio_samples(audio_samples)
        
        # Log sample predictions
        sample_predictions = []
        for i, sample in enumerate(samples[:self.config.wandb_log_samples]):
            sample_predictions.append({
                'dataset_name': sample.metadata.get('dataset_name', 'unknown'),
                'prompt': sample.prompt[:200],
                'prediction': sample.answer[:200],
                'ground_truth': sample.metadata.get('transcript', '')[:200],
                'reward': 0.0,  # Will be calculated during training
                'accuracy': 0.0,  # Will be calculated during training
            })
        
        if sample_predictions:
            self.wandb_logger.log_predictions(sample_predictions)
        
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
            eval_steps=self.config.save_steps,
            eval_strategy="steps",
            load_best_model_at_end=False,
            max_grad_norm=self.config.max_grad_norm,
            report_to=["wandb"] if self.config.use_wandb else "none",
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
            
            # Disable evaluation and wandb for now to focus on training
            grpo_config.eval_strategy = "no"
            grpo_config.eval_steps = None
            grpo_config.report_to = []  # Disable all reporting
            
            # Create reward functions
            reward_funcs = self._create_reward_functions()
            
            # Create trainer
            self.trainer = GRPOTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                args=grpo_config,
                train_dataset=train_dataset,
                optimizers=(None, None),  # Provide default optimizers
            )
            
            # Add early stopping callback
            early_stopping = EarlyStoppingCallback(
                patience=self.config.patience,
                min_delta=self.config.min_delta
            )
            self.trainer.add_callback(early_stopping)
            
            # Log model architecture to wandb
            self.wandb_logger.log_model_architecture(self.model)
            
            # Log dataset distribution
            distribution = {}
            for sample in train_dataset:
                dataset_name = sample.get("dataset_name", "unknown")
                distribution[dataset_name] = distribution.get(dataset_name, 0) + 1
            self.wandb_logger.log_dataset_distribution(distribution)
            
            # Save training configuration
            self._save_training_config()
            
            logger.info("Starting GRPO training...")
            start_time = time.time()
            
            # Start training with wandb logging
            self._train_with_wandb_logging()
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Log final metrics
            self.wandb_logger.log_training_metrics({
                "final_training_time": training_time,
                "total_steps": self.step,
                "best_score": self.best_score,
            })
            
            # Save final model
            self.save_model()
            
            # Save training history
            self._save_training_history()
            
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Finish wandb run
            self.wandb_logger.finish()
    
    def _train_with_wandb_logging(self):
        """Custom training loop with comprehensive wandb logging."""
        try:
            # Start training
            self.trainer.train()
            
            # Log training history
            if hasattr(self.trainer, 'state') and hasattr(self.trainer.state, 'log_history'):
                for log_entry in self.trainer.state.log_history:
                    if 'train_loss' in log_entry:
                        self.wandb_logger.log_training_metrics({
                            'loss': log_entry['train_loss'],
                            'learning_rate': log_entry.get('learning_rate', 0.0),
                            'epoch': log_entry.get('epoch', 0.0),
                        }, step=log_entry.get('step', 0))
                    
                    if 'eval_loss' in log_entry:
                        self.wandb_logger.log_evaluation_metrics({
                            'loss': log_entry['eval_loss'],
                        }, step=log_entry.get('step', 0))
            
        except Exception as e:
            logger.error(f"Training with wandb logging failed: {e}")
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
