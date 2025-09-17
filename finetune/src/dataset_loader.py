#!/usr/bin/env python3
"""
Online Dataset Loader for VoiceBench Datasets

Handles online loading and processing of VoiceBench datasets for training.
Supports ifeval, commoneval, and wildvoice datasets with on-demand transcription.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Tuple
from dataclasses import dataclass
from collections import Counter
import json

from datasets import load_dataset, Audio, Dataset
import numpy as np

from .whisper_transcriber import WhisperTranscriber
from configs.training_config import TrainingConfig, DatasetConfig, DATASET_CONFIGS

logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """A single training sample for dataset classification."""
    
    prompt: List[Dict[str, str]]
    answer: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "prompt": self.prompt,
            "answer": self.answer,
            "metadata": self.metadata
        }


class OnlineDatasetLoader:
    """
    Online dataset loader that fetches samples and generates transcripts on-demand.
    This avoids downloading all samples at once and enables memory-efficient training.
    """
    
    def __init__(self, config: TrainingConfig, transcriber: WhisperTranscriber):
        """
        Initialize the online dataset loader.
        
        Args:
            config: Training configuration
            transcriber: Whisper transcriber instance
        """
        self.config = config
        self.transcriber = transcriber
        self.datasets = {}
        self.sample_indices = {}
        self.dataset_configs = {}
        
        # Set random seed for reproducibility
        random.seed(config.random_seed)
        
        self._prepare_datasets()
    
    def _prepare_datasets(self):
        """Prepare datasets and determine sampling indices."""
        logger.info("Preparing datasets for online loading...")
        
        for dataset_name in self.config.target_datasets:
            try:
                logger.info(f"Loading dataset info: {dataset_name}")
                
                # Get dataset configuration
                dataset_config = DATASET_CONFIGS.get(dataset_name, DatasetConfig(name=dataset_name))
                self.dataset_configs[dataset_name] = dataset_config
                
                # Load dataset
                dataset = load_dataset(
                    dataset_config.hf_dataset_name, 
                    dataset_name, 
                    split=dataset_config.split
                )
                
                # Cast audio column to proper format
                dataset = dataset.cast_column("audio", Audio(sampling_rate=dataset_config.sampling_rate))
                
                total_samples = len(dataset)
                logger.info(f"{dataset_name}: {total_samples} total samples")
                
                # Determine sampling indices
                indices = self._get_sampling_indices(total_samples, dataset_name)
                
                self.datasets[dataset_name] = dataset
                self.sample_indices[dataset_name] = indices
                
                logger.info(f"{dataset_name}: Selected {len(indices)} samples")
                
            except Exception as e:
                logger.error(f"Failed to prepare dataset {dataset_name}: {e}")
                raise
    
    def _get_sampling_indices(self, total_samples: int, dataset_name: str) -> List[int]:
        """Get sampling indices based on the sampling method."""
        max_samples = min(self.config.samples_per_dataset, total_samples)
        
        if self.config.sampling_method == "random":
            indices = random.sample(range(total_samples), max_samples)
            indices.sort()  # Keep sorted for easier debugging
        elif self.config.sampling_method == "first":
            indices = list(range(max_samples))
        elif self.config.sampling_method == "last":
            start_idx = max(0, total_samples - max_samples)
            indices = list(range(start_idx, total_samples))
        else:
            raise ValueError(f"Unknown sampling method: {self.config.sampling_method}")
        
        return indices
    
    def generate_training_samples(self) -> Iterator[TrainingSample]:
        """
        Generate training samples on-demand with transcription.
        
        Yields:
            TrainingSample containing prompt, answer, and metadata
        """
        total_samples = sum(len(indices) for indices in self.sample_indices.values())
        sample_count = 0
        
        logger.info(f"Starting to generate {total_samples} training samples...")
        
        # Create shuffled list of (dataset_name, index) pairs
        all_samples = []
        for dataset_name, indices in self.sample_indices.items():
            for idx in indices:
                all_samples.append((dataset_name, idx))
        
        # Shuffle for varied training
        random.shuffle(all_samples)
        
        for dataset_name, idx in all_samples:
            sample_count += 1
            
            try:
                logger.info(f"Processing sample {sample_count}/{total_samples}: {dataset_name}[{idx}]")
                
                # Get the sample
                sample = self.datasets[dataset_name][idx]
                
                # Extract audio
                audio_array = sample['audio']['array']
                sample_rate = sample['audio']['sampling_rate']
                
                # Check audio duration
                duration = len(audio_array) / sample_rate
                if duration > self.config.max_audio_duration:
                    logger.warning(f"Audio too long ({duration:.1f}s), skipping {dataset_name}[{idx}]")
                    continue
                
                # Transcribe audio
                logger.debug(f"Transcribing audio (shape: {audio_array.shape}, sr: {sample_rate})")
                transcript = self.transcriber.transcribe(audio_array, sample_rate)
                
                if transcript.startswith("Error"):
                    logger.warning(f"Transcription failed for {dataset_name}[{idx}]: {transcript}")
                    continue
                
                # Create training sample
                training_sample = self._create_training_sample(
                    transcript, dataset_name, idx, sample
                )
                
                logger.debug(f"Generated training sample: {dataset_name} -> {transcript[:100]}...")
                yield training_sample
                
            except Exception as e:
                logger.error(f"Error processing sample {dataset_name}[{idx}]: {e}")
                continue
    
    def _create_training_sample(
        self, 
        transcript: str, 
        dataset_name: str, 
        idx: int, 
        original_sample: Dict[str, Any]
    ) -> TrainingSample:
        """Create a training sample from transcript and metadata."""
        
        # Get dataset configuration
        dataset_config = self.dataset_configs[dataset_name]
        
        # Create system prompt
        system_prompt = self._get_system_prompt()
        
        # Create user prompt
        user_prompt = self._format_user_prompt(transcript)
        
        # Create prompt structure
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # Create metadata
        metadata = {
            "dataset_name": dataset_name,
            "sample_index": idx,
            "transcript": transcript,
            "transcript_length": len(transcript),
            "original_prompt": original_sample.get("prompt", ""),
            "audio_duration": len(original_sample['audio']['array']) / original_sample['audio']['sampling_rate'],
            "dataset_description": dataset_config.description
        }
        
        return TrainingSample(
            prompt=prompt,
            answer=dataset_name,
            metadata=metadata
        )
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for dataset classification."""
        return f"""You are an expert at analyzing speech transcripts and identifying which type of evaluation dataset they come from.

Your task is to classify transcripts into one of these three categories:
- ifeval: {DATASET_CONFIGS['ifeval'].description}
- commoneval: {DATASET_CONFIGS['commoneval'].description}  
- wildvoice: {DATASET_CONFIGS['wildvoice'].description}

Analyze the content, style, and structure of the transcript to determine the most likely source dataset.
Respond with only the dataset name: ifeval, commoneval, or wildvoice."""
    
    def _format_user_prompt(self, transcript: str) -> str:
        """Format the user prompt with the transcript."""
        return f"""Please classify this transcript into one of the three categories: ifeval, commoneval, or wildvoice.

Transcript: "{transcript}"

Which dataset category does this transcript most likely come from? Answer with only the dataset name: ifeval, commoneval, or wildvoice."""
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded datasets."""
        stats = {}
        
        for dataset_name, indices in self.sample_indices.items():
            dataset = self.datasets[dataset_name]
            stats[dataset_name] = {
                "total_samples": len(dataset),
                "selected_samples": len(indices),
                "sampling_rate": dataset_configs[dataset_name].sampling_rate,
                "description": self.dataset_configs[dataset_name].description
            }
        
        return stats
    
    def save_sample_metadata(self, output_path: str):
        """Save metadata about the selected samples."""
        metadata = {
            "config": {
                "target_datasets": self.config.target_datasets,
                "samples_per_dataset": self.config.samples_per_dataset,
                "sampling_method": self.config.sampling_method,
                "random_seed": self.config.random_seed
            },
            "dataset_stats": self.get_dataset_stats(),
            "sample_indices": self.sample_indices
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Sample metadata saved to {output_path}")


def create_dataset_loader(config: TrainingConfig, transcriber: WhisperTranscriber) -> OnlineDatasetLoader:
    """
    Create an OnlineDatasetLoader instance.
    
    Args:
        config: Training configuration
        transcriber: Whisper transcriber instance
        
    Returns:
        OnlineDatasetLoader instance
    """
    return OnlineDatasetLoader(config, transcriber)


# Test function
def test_dataset_loader():
    """Test the dataset loader with a small sample."""
    from configs.training_config import get_small_config
    from .whisper_transcriber import create_transcriber
    
    # Create test configuration
    config = get_small_config()
    config.samples_per_dataset = 2  # Very small for testing
    
    # Create transcriber
    transcriber = create_transcriber(config.whisper_model_path)
    
    # Create dataset loader
    loader = create_dataset_loader(config, transcriber)
    
    # Test sample generation
    print("Testing dataset loader...")
    samples = list(loader.generate_training_samples())
    print(f"Generated {len(samples)} samples")
    
    for i, sample in enumerate(samples):
        print(f"Sample {i+1}: {sample.answer} - {sample.metadata['transcript'][:50]}...")
    
    # Print stats
    stats = loader.get_dataset_stats()
    print(f"Dataset stats: {stats}")


if __name__ == "__main__":
    test_dataset_loader()
