"""
Dataset Classification Training Package

This package provides tools for training a Gemma model to classify VoiceBench datasets
using GRPO (Group Relative Policy Optimization) reinforcement learning.
"""

from .whisper_transcriber import WhisperTranscriber, create_transcriber
from .dataset_loader import OnlineDatasetLoader, TrainingSample, create_dataset_loader
from .reward_functions import DatasetClassificationRewards, create_reward_functions
from .grpo_trainer import DatasetClassificationTrainer, create_trainer

__version__ = "1.0.0"
__author__ = "OmegaLabs"

__all__ = [
    "WhisperTranscriber",
    "create_transcriber",
    "OnlineDatasetLoader", 
    "TrainingSample",
    "create_dataset_loader",
    "DatasetClassificationRewards",
    "create_reward_functions",
    "DatasetClassificationTrainer",
    "create_trainer"
]





