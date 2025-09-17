#!/usr/bin/env python3
"""
Extract VoiceBench dataset data without audio decoding

This script extracts the text data from VoiceBench datasets and saves it as JSON,
avoiding the torchcodec dependency issues.
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
from typing import Dict, Any, List

def extract_dataset_data(dataset_name: str, split: str = 'test', max_samples: int = None) -> List[Dict[str, Any]]:
    """Extract dataset data without audio decoding."""
    print(f"Extracting data from {dataset_name} dataset...")
    
    try:
        # Load dataset without audio decoding
        dataset = load_dataset('hlt-lab/voicebench', dataset_name, split=split)
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Features: {list(dataset.features.keys())}")
        
        # Extract text data only
        samples = []
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
                
            sample = {
                'prompt': item.get('prompt', ''),
                'output': item.get('output', ''),
                'index': i
            }
            
            # Add any other text fields
            for key, value in item.items():
                if key not in ['audio', 'prompt', 'output'] and isinstance(value, (str, int, float, bool)):
                    sample[key] = value
            
            samples.append(sample)
            
            if i % 50 == 0:
                print(f"  Processed {i+1} samples...")
        
        print(f"Extracted {len(samples)} samples from {dataset_name}")
        return samples
        
    except Exception as e:
        print(f"Error extracting {dataset_name}: {e}")
        return []

def save_dataset_data(samples: List[Dict[str, Any]], dataset_name: str, split: str = 'test'):
    """Save extracted data to JSON file."""
    output_file = f"voicebench_{dataset_name}_{split}.json"
    
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"Saved {len(samples)} samples to {output_file}")
    return output_file

def main():
    """Extract data from all VoiceBench datasets."""
    datasets = ['commoneval', 'wildvoice', 'ifeval', 'advbench']
    
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"Processing {dataset_name}")
        print('='*50)
        
        # Extract data
        samples = extract_dataset_data(dataset_name, split='test', max_samples=50)
        
        if samples:
            # Save to file
            output_file = save_dataset_data(samples, dataset_name)
            print(f"✅ {dataset_name}: {len(samples)} samples saved to {output_file}")
        else:
            print(f"❌ {dataset_name}: Failed to extract data")

if __name__ == "__main__":
    main()





