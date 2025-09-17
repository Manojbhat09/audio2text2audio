#!/usr/bin/env python3
"""
Wandb Integration for GRPO Training

Comprehensive Weights & Biases integration for tracking GRPO training metrics,
model performance, audio samples, and experiment management.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Dict, List, Any, Optional, Union
import numpy as np
import torch
import wandb
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

from configs.training_config import TrainingConfig

logger = logging.getLogger(__name__)


class WandbLogger:
    """Comprehensive Wandb logging for GRPO training."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize Wandb logger."""
        self.config = config
        self.run = None
        self.step = 0
        self.sample_logs = []
        
        if config.use_wandb:
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize wandb run."""
        try:
            # Generate run name if not provided
            if not self.config.wandb_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.config.wandb_name = f"grpo-{self.config.model_name.split('/')[-1]}-{timestamp}"
            
            # Initialize wandb
            self.run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.wandb_name,
                tags=self.config.wandb_tags or ["grpo", "dataset-classification", "reinforcement-learning"],
                config=self._get_wandb_config(),
                reinit=True
            )
            
            logger.info(f"✓ Wandb initialized: {self.run.url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}")
            self.run = None
    
    def _get_wandb_config(self) -> Dict[str, Any]:
        """Convert training config to wandb config."""
        config_dict = {}
        
        # Model settings
        config_dict.update({
            "model_name": self.config.model_name,
            "max_seq_length": self.config.max_seq_length,
            "max_prompt_length": self.config.max_prompt_length,
            "max_completion_length": self.config.max_completion_length,
        })
        
        # Training settings
        config_dict.update({
            "learning_rate": self.config.learning_rate,
            "num_generations": self.config.num_generations,
            "max_steps": self.config.max_steps,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "per_device_train_batch_size": self.config.per_device_train_batch_size,
            "save_steps": self.config.save_steps,
            "logging_steps": self.config.logging_steps,
        })
        
        # Optimizer settings
        config_dict.update({
            "adam_beta1": self.config.adam_beta1,
            "adam_beta2": self.config.adam_beta2,
            "weight_decay": self.config.weight_decay,
            "warmup_ratio": self.config.warmup_ratio,
            "lr_scheduler_type": self.config.lr_scheduler_type,
            "optim": self.config.optim,
            "max_grad_norm": self.config.max_grad_norm,
        })
        
        # Dataset settings
        config_dict.update({
            "target_datasets": self.config.target_datasets,
            "samples_per_dataset": self.config.samples_per_dataset,
            "sampling_method": self.config.sampling_method,
            "max_audio_duration": self.config.max_audio_duration,
        })
        
        # LoRA settings
        config_dict.update({
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "lora_bias": self.config.lora_bias,
        })
        
        return config_dict
    
    def log_training_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log training metrics to wandb."""
        if not self.run:
            return
        
        if step is None:
            step = self.step
        
        # Log basic metrics
        wandb.log({
            "train/loss": metrics.get("loss", 0.0),
            "train/learning_rate": metrics.get("learning_rate", 0.0),
            "train/grad_norm": metrics.get("grad_norm", 0.0),
            "train/epoch": metrics.get("epoch", 0.0),
            "train/step": step,
        }, step=step)
        
        # Log GRPO-specific metrics
        if "grpo_loss" in metrics:
            wandb.log({
                "grpo/grpo_loss": metrics["grpo_loss"],
                "grpo/policy_loss": metrics.get("policy_loss", 0.0),
                "grpo/kl_penalty": metrics.get("kl_penalty", 0.0),
                "grpo/reward": metrics.get("reward", 0.0),
                "grpo/advantage": metrics.get("advantage", 0.0),
            }, step=step)
        
        # Log dataset-specific metrics
        for dataset in self.config.target_datasets:
            if f"{dataset}_accuracy" in metrics:
                wandb.log({
                    f"dataset/{dataset}_accuracy": metrics[f"{dataset}_accuracy"],
                    f"dataset/{dataset}_loss": metrics.get(f"{dataset}_loss", 0.0),
                }, step=step)
    
    def log_evaluation_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log evaluation metrics to wandb."""
        if not self.run:
            return
        
        if step is None:
            step = self.step
        
        # Log evaluation metrics
        eval_metrics = {}
        for key, value in metrics.items():
            if key.startswith("eval_"):
                eval_metrics[f"eval/{key[5:]}"] = value
            else:
                eval_metrics[f"eval/{key}"] = value
        
        wandb.log(eval_metrics, step=step)
    
    def log_dataset_distribution(self, distribution: Dict[str, int], step: Optional[int] = None):
        """Log dataset distribution as a bar chart."""
        if not self.run:
            return
        
        if step is None:
            step = self.step
        
        # Create bar chart
        data = [[dataset, count] for dataset, count in distribution.items()]
        table = wandb.Table(data=data, columns=["Dataset", "Count"])
        
        wandb.log({
            "dataset/distribution": wandb.plot.bar(table, "Dataset", "Count", title="Dataset Distribution")
        }, step=step)
    
    def log_audio_samples(self, samples: List[Dict[str, Any]], step: Optional[int] = None):
        """Log audio samples to wandb."""
        if not self.run or not self.config.wandb_log_audio:
            return
        
        if step is None:
            step = self.step
        
        # Limit number of samples to log
        samples_to_log = samples[:self.config.wandb_log_samples]
        
        for i, sample in enumerate(samples_to_log):
            if "audio" in sample and "transcript" in sample:
                # Create audio wandb object
                audio_data = sample["audio"]["array"]
                sample_rate = sample["audio"]["sampling_rate"]
                
                # Convert to numpy if needed
                if isinstance(audio_data, list):
                    audio_data = np.array(audio_data, dtype=np.float32)
                
                # Log audio
                wandb.log({
                    f"audio/sample_{i}": wandb.Audio(
                        audio_data, 
                        sample_rate=sample_rate,
                        caption=f"Dataset: {sample.get('dataset_name', 'unknown')} - Transcript: {sample['transcript'][:100]}..."
                    )
                }, step=step)
    
    def log_predictions(self, predictions: List[Dict[str, Any]], step: Optional[int] = None):
        """Log model predictions as a table."""
        if not self.run or not self.config.wandb_log_predictions:
            return
        
        if step is None:
            step = self.step
        
        # Limit number of predictions to log
        predictions_to_log = predictions[:self.config.wandb_log_samples]
        
        # Create predictions table
        data = []
        for i, pred in enumerate(predictions_to_log):
            data.append([
                i,
                pred.get("dataset_name", "unknown"),
                str(pred.get("prompt", ""))[:100] + "...",
                str(pred.get("prediction", ""))[:100] + "...",
                str(pred.get("ground_truth", ""))[:100] + "...",
                pred.get("reward", 0.0),
                pred.get("accuracy", 0.0),
            ])
        
        table = wandb.Table(
            data=data,
            columns=["Index", "Dataset", "Prompt", "Prediction", "Ground Truth", "Reward", "Accuracy"]
        )
        
        wandb.log({
            "predictions/table": table,
            "predictions/count": len(predictions_to_log)
        }, step=step)
    
    def log_model_architecture(self, model: torch.nn.Module):
        """Log model architecture to wandb."""
        if not self.run or not self.config.wandb_log_model:
            return
        
        try:
            # Log model summary
            wandb.watch(model, log="all", log_freq=100)
            
            # Log model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            wandb.log({
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params,
                "model/parameter_ratio": trainable_params / total_params if total_params > 0 else 0,
            })
            
        except Exception as e:
            logger.warning(f"Failed to log model architecture: {e}")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Log hyperparameters to wandb."""
        if not self.run:
            return
        
        wandb.log({"hyperparameters": hyperparams})
    
    def log_system_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log system metrics (GPU, memory, etc.)."""
        if not self.run:
            return
        
        if step is None:
            step = self.step
        
        wandb.log({
            "system/gpu_memory_used": metrics.get("gpu_memory_used", 0),
            "system/gpu_memory_total": metrics.get("gpu_memory_total", 0),
            "system/cpu_usage": metrics.get("cpu_usage", 0),
            "system/ram_usage": metrics.get("ram_usage", 0),
        }, step=step)
    
    def log_learning_curves(self, train_losses: List[float], eval_losses: List[float] = None):
        """Log learning curves."""
        if not self.run:
            return
        
        # Create learning curve data
        steps = list(range(len(train_losses)))
        data = [[step, loss, "train"] for step, loss in zip(steps, train_losses)]
        
        if eval_losses:
            data.extend([[step, loss, "eval"] for step, loss in zip(steps, eval_losses)])
        
        table = wandb.Table(data=data, columns=["Step", "Loss", "Type"])
        
        wandb.log({
            "learning_curves/loss": wandb.plot.line(
                table, "Step", "Loss", title="Learning Curves"
            )
        })
    
    def log_confusion_matrix(self, y_true: List[str], y_pred: List[str], step: Optional[int] = None):
        """Log confusion matrix for dataset classification."""
        if not self.run:
            return
        
        if step is None:
            step = self.step
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Add labels
        classes = list(set(y_true + y_pred))
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes,
               yticklabels=classes,
               title='Confusion Matrix',
               ylabel='True Label',
               xlabel='Predicted Label')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        
        # Log to wandb
        wandb.log({"confusion_matrix": wandb.Image(fig)}, step=step)
        plt.close(fig)
    
    def log_sample_predictions(self, samples: List[Dict[str, Any]], step: Optional[int] = None):
        """Log detailed sample predictions."""
        if not self.run:
            return
        
        if step is None:
            step = self.step
        
        # Create detailed samples table
        data = []
        for sample in samples[:5]:  # Log first 5 samples
            data.append([
                sample.get("dataset_name", "unknown"),
                str(sample.get("prompt", ""))[:200],
                str(sample.get("prediction", ""))[:200],
                str(sample.get("ground_truth", ""))[:200],
                sample.get("reward", 0.0),
                sample.get("confidence", 0.0),
            ])
        
        table = wandb.Table(
            data=data,
            columns=["Dataset", "Prompt", "Prediction", "Ground Truth", "Reward", "Confidence"]
        )
        
        wandb.log({
            "samples/detailed": table
        }, step=step)
    
    def finish(self):
        """Finish wandb run."""
        if self.run:
            wandb.finish()
            logger.info("✓ Wandb run finished")
    
    def update_step(self, step: int):
        """Update current step."""
        self.step = step
