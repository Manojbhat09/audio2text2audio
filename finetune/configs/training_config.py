#!/usr/bin/env python3
"""
Training Configuration for Dataset Classification

Configuration settings for training a Gemma model to classify VoiceBench datasets
using GRPO (Group Relative Policy Optimization) reinforcement learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for dataset classification training using GRPO."""
    
    # Dataset settings
    target_datasets: List[str] = None
    samples_per_dataset: int = 100
    sampling_method: str = "random"  # random, first, last
    max_audio_duration: float = 30.0  # seconds
    
    # Model settings
    model_name: str = "google/gemma-2-2b-it"
    max_seq_length: int = 1024
    max_prompt_length: int = 256
    max_completion_length: int = 64
    
    # Training settings
    learning_rate: float = 5e-6
    num_generations: int = 4
    max_steps: int = 100
    gradient_accumulation_steps: int = 1
    per_device_train_batch_size: int = 1
    save_steps: int = 25
    logging_steps: int = 1
    
    # Optimizer settings
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch_fused"
    max_grad_norm: float = 0.1
    
    # LoRA settings
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    
    # Early stopping
    patience: int = 10
    min_delta: float = 0.01
    
    # Wandb settings
    use_wandb: bool = True
    wandb_project: str = "grpo-dataset-classification"
    wandb_entity: str = None
    wandb_name: str = None
    wandb_tags: List[str] = None
    wandb_log_model: bool = True
    wandb_log_predictions: bool = True
    wandb_log_audio: bool = True
    wandb_log_samples: int = 10
    
    # Paths
    whisper_model_path: str = "/home/mbhat/alien-invasion-r3-05/models/wpt/wpt.pt"
    output_dir: str = "/home/mbhat/omegalabs-anytoany-bittensor/finetune/outputs"
    data_dir: str = "/home/mbhat/omegalabs-anytoany-bittensor/finetune/data"
    model_dir: str = "/home/mbhat/omegalabs-anytoany-bittensor/finetune/models"
    
    # Reward function weights
    reward_weights: Dict[str, float] = None
    
    # Random seed
    random_seed: int = 42
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.target_datasets is None:
            self.target_datasets = ["ifeval", "commoneval", "wildvoice"]
        
        if self.reward_weights is None:
            self.reward_weights = {
                "exact_match": 3.0,
                "format_compliance": 2.0,
                "confidence": 1.0,
                "dataset_balance": 0.5
            }
        
        # Create output directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetConfig:
    """Configuration for individual datasets."""
    
    name: str
    hf_dataset_name: str = "hlt-lab/voicebench"
    split: str = "test"
    sampling_rate: int = 16000
    max_samples: int = 1000
    description: str = ""
    
    def __post_init__(self):
        """Set default description if not provided."""
        if not self.description:
            descriptions = {
                "ifeval": "Instruction-following tasks with specific formatting requirements",
                "commoneval": "Common sense reasoning and knowledge questions",
                "wildvoice": "Conversational scenarios and open-ended discussions"
            }
            self.description = descriptions.get(self.name, f"Dataset: {self.name}")


# Predefined dataset configurations
DATASET_CONFIGS = {
    "ifeval": DatasetConfig(
        name="ifeval",
        description="Instruction-following tasks with specific formatting requirements"
    ),
    "commoneval": DatasetConfig(
        name="commoneval", 
        description="Common sense reasoning and knowledge questions"
    ),
    "wildvoice": DatasetConfig(
        name="wildvoice",
        description="Conversational scenarios and open-ended discussions"
    )
}


def get_default_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()


def get_small_config() -> TrainingConfig:
    """Get configuration for small-scale testing."""
    return TrainingConfig(
        samples_per_dataset=20,
        max_steps=20,
        save_steps=5,
        patience=3
    )


def get_large_config() -> TrainingConfig:
    """Get configuration for large-scale training."""
    return TrainingConfig(
        samples_per_dataset=500,
        max_steps=500,
        save_steps=50,
        patience=20,
        learning_rate=2e-6
    )
