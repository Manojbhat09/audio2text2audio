"""
Improved dataset loader with better prompts and more diverse samples.
"""

import logging
import random
from typing import List, Iterator, Dict, Any
import numpy as np
from datasets import Dataset
from dataclasses import dataclass

from .training_sample import TrainingSample
from .dataset_config import DatasetConfig

logger = logging.getLogger(__name__)

class ImprovedDatasetLoader:
    """Improved dataset loader with better prompts and diverse samples."""
    
    def __init__(self, config):
        """Initialize the improved dataset loader."""
        self.config = config
        self.datasets = {}
        self.sample_indices = {}
        
        # Better prompt templates
        self.prompt_templates = [
            "Classify this audio transcript as one of these datasets: ifeval, commoneval, wildvoice. Transcript: {transcript}",
            "Which dataset does this audio belong to? Options: ifeval, commoneval, wildvoice. Audio: {transcript}",
            "Dataset classification task. Choose from: ifeval, commoneval, wildvoice. Input: {transcript}",
            "Identify the dataset for this audio transcript. Categories: ifeval, commoneval, wildvoice. Text: {transcript}",
            "Classify the following audio as ifeval, commoneval, or wildvoice: {transcript}"
        ]
        
        # Better response templates
        self.response_templates = {
            "ifeval": [
                "ifeval",
                "This belongs to the ifeval dataset.",
                "Dataset: ifeval",
                "Classification: ifeval",
                "ifeval dataset"
            ],
            "commoneval": [
                "commoneval", 
                "This belongs to the commoneval dataset.",
                "Dataset: commoneval",
                "Classification: commoneval",
                "commoneval dataset"
            ],
            "wildvoice": [
                "wildvoice",
                "This belongs to the wildvoice dataset.", 
                "Dataset: wildvoice",
                "Classification: wildvoice",
                "wildvoice dataset"
            ]
        }
        
        # More diverse transcript templates
        self.transcript_templates = {
            "ifeval": [
                "Follow these instructions step by step to complete the task.",
                "Given the following instructions, provide a detailed response.",
                "Execute the following command and explain your reasoning.",
                "Please follow the instructions carefully and show your work.",
                "Complete this instruction following task with detailed steps.",
                "Given these instructions, what would you do?",
                "Follow the step-by-step instructions provided.",
                "Execute the given instructions and explain your approach.",
                "Please complete this instruction following evaluation.",
                "Given the instructions, provide a comprehensive answer.",
                "Instruction following task with complex reasoning requirements.",
                "Follow the detailed instructions and provide step-by-step solution.",
                "Execute the given instructions with proper explanation.",
                "Complete this instruction following evaluation task.",
                "Given the instructions, show your work and reasoning."
            ],
            "commoneval": [
                "This is a common evaluation benchmark for language understanding.",
                "Standard evaluation task for natural language processing models.",
                "Common benchmark dataset for evaluating language models.",
                "This is a typical evaluation example from CommonEval.",
                "Standard NLP evaluation task with multiple choice questions.",
                "Common evaluation benchmark for assessing model performance.",
                "Typical evaluation example from the CommonEval dataset.",
                "Standard benchmark task for language model evaluation.",
                "Common evaluation dataset for testing model capabilities.",
                "This is a standard evaluation example from CommonEval.",
                "Standard benchmark evaluation with multiple choice format.",
                "Common evaluation task for language model assessment.",
                "Typical CommonEval benchmark example for testing.",
                "Standard evaluation dataset with multiple choice questions.",
                "Common benchmark evaluation for language understanding."
            ],
            "wildvoice": [
                "This is a natural conversation with diverse speaking styles.",
                "Wild voice data with various accents and speaking patterns.",
                "Natural speech sample with conversational characteristics.",
                "Diverse voice data with different speaking styles and accents.",
                "Wild voice sample with natural conversational patterns.",
                "This audio contains natural speech with various characteristics.",
                "Wild voice data with diverse speaking patterns and accents.",
                "Natural conversation sample with varied speaking styles.",
                "This is a wild voice sample with conversational characteristics.",
                "Diverse voice data with natural speaking patterns and accents.",
                "Natural conversational speech with diverse characteristics.",
                "Wild voice sample with varied speaking patterns and styles.",
                "Diverse conversational data with natural speech patterns.",
                "Wild voice audio with various accents and speaking styles.",
                "Natural speech sample with conversational diversity."
            ]
        }
    
    def prepare_datasets(self, dataset_names: List[str]) -> None:
        """Prepare improved datasets with better diversity."""
        logger.info(f"Preparing improved datasets: {dataset_names}")
        
        for dataset_name in dataset_names:
            logger.info(f"Preparing {dataset_name}...")
            
            # Create dataset config
            dataset_config = DatasetConfig(
                name=dataset_name,
                max_samples=self.config.samples_per_dataset,
                sampling_method=self.config.sampling_method
            )
            
            # Create improved mock dataset
            dataset = self._create_improved_mock_dataset(dataset_name, dataset_config)
            
            total_samples = len(dataset)
            logger.info(f"{dataset_name}: {total_samples} total samples")
            
            # Determine sampling indices
            indices = self._get_sampling_indices(total_samples, dataset_name)
            
            self.datasets[dataset_name] = dataset
            self.sample_indices[dataset_name] = indices
            
            logger.info(f"{dataset_name}: Selected {len(indices)} samples")
    
    def _create_improved_mock_dataset(self, dataset_name: str, dataset_config: DatasetConfig) -> Dataset:
        """Create improved mock dataset with better diversity."""
        mock_samples = []
        
        # Generate more samples for better training
        num_samples = max(50, self.config.samples_per_dataset * 2)  # Generate 2x more samples
        
        for i in range(num_samples):
            # Generate mock audio (random array)
            audio_length = np.random.randint(1000, 5000)
            audio = np.random.randn(audio_length).astype(np.float32)
            
            # Select random transcript template
            transcript_template = random.choice(self.transcript_templates[dataset_name])
            transcript = f"{transcript_template} Sample {i+1}."
            
            # Select random prompt template
            prompt_template = random.choice(self.prompt_templates)
            prompt = prompt_template.format(transcript=transcript)
            
            # Select random response template
            answer_template = random.choice(self.response_templates[dataset_name])
            answer = answer_template
            
            sample = {
                "prompt": prompt,
                "output": answer,
                "audio": {
                    "array": audio.tolist(),
                    "sampling_rate": 16000
                }
            }
            mock_samples.append(sample)
        
        # Create Dataset from mock samples
        return Dataset.from_list(mock_samples)
    
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
        """Generate improved training samples on-demand."""
        total_samples = sum(len(indices) for indices in self.sample_indices.values())
        sample_count = 0
        
        logger.info(f"Starting to generate {total_samples} improved training samples...")
        
        # Create shuffled list of (dataset_name, index) pairs
        all_samples = []
        for dataset_name, indices in self.sample_indices.items():
            for idx in indices:
                all_samples.append((dataset_name, idx))
        
        random.shuffle(all_samples)
        
        for dataset_name, idx in all_samples:
            try:
                # Get the sample from the dataset
                sample_data = self.datasets[dataset_name][idx]
                
                # Convert audio array to numpy
                audio_array = np.array(sample_data["audio"]["array"], dtype=np.float32)
                
                # Create training sample
                training_sample = TrainingSample(
                    prompt=sample_data["prompt"],
                    answer=sample_data["output"],
                    audio=audio_array,
                    metadata={
                        "dataset_name": dataset_name,
                        "transcript": sample_data["prompt"].split("Transcript: ")[-1] if "Transcript: " in sample_data["prompt"] else "",
                        "transcript_length": len(sample_data["prompt"]),
                        "sample_id": idx
                    }
                )
                
                sample_count += 1
                if sample_count % 10 == 0:
                    logger.info(f"Generated {sample_count}/{total_samples} samples...")
                
                yield training_sample
                
            except Exception as e:
                logger.error(f"Error generating sample {sample_count}: {e}")
                continue
        
        logger.info(f"Completed generating {sample_count} training samples")
