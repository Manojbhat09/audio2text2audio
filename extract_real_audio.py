#!/usr/bin/env python3
"""
Extract real audio data from VoiceBench parquet file

This script loads the parquet file directly and extracts the real audio data
without using torchcodec or the datasets library.
"""

import pandas as pd
import numpy as np
import json
import io
from typing import Dict, Any, List
import soundfile as sf

def extract_audio_from_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Extract audio array from bytes using soundfile."""
    try:
        # Use soundfile to read the audio bytes
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        return audio_data, sample_rate
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None, None

def load_voicebench_from_parquet(parquet_file: str, max_samples: int = 10) -> List[Dict[str, Any]]:
    """Load VoiceBench dataset from parquet file with real audio data."""
    print(f"Loading VoiceBench dataset from {parquet_file}...")
    
    try:
        # Load the parquet file
        df = pd.read_parquet(parquet_file)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Extract samples with real audio
        samples = []
        for i in range(min(max_samples, len(df))):
            row = df.iloc[i]
            
            # Get the prompt
            prompt = row.get('prompt', '')
            
            # Get the audio data
            audio_data = row.get('audio', {})
            
            if isinstance(audio_data, dict) and 'bytes' in audio_data:
                # Extract real audio from bytes
                audio_bytes = audio_data['bytes']
                audio_array, sample_rate = extract_audio_from_bytes(audio_bytes)
                
                if audio_array is not None:
                    samples.append({
                        'prompt': prompt,
                        'output': '',  # No output in this dataset
                        'audio': {
                            'array': audio_array,
                            'sampling_rate': sample_rate
                        },
                        'index': i
                    })
                    
                    if i % 5 == 0:
                        print(f"  Processed {i+1}/{min(max_samples, len(df))} samples")
                else:
                    print(f"  Failed to extract audio for sample {i}")
            else:
                print(f"  No audio bytes found for sample {i}")
        
        print(f"Successfully loaded {len(samples)} samples with real audio")
        return samples
        
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        return []

def save_samples_to_json(samples: List[Dict[str, Any]], filename: str = "real_voicebench_samples.json"):
    """Save samples to JSON file (without audio data)."""
    # Convert numpy arrays to lists for JSON serialization
    json_samples = []
    for sample in samples:
        json_sample = {
            'prompt': sample['prompt'],
            'output': sample['output'],
            'index': sample['index'],
            'audio_info': {
                'sampling_rate': sample['audio']['sampling_rate'],
                'length': len(sample['audio']['array']),
                'duration': len(sample['audio']['array']) / sample['audio']['sampling_rate']
            }
        }
        json_samples.append(json_sample)
    
    with open(filename, 'w') as f:
        json.dump(json_samples, f, indent=2)
    
    print(f"Saved {len(json_samples)} samples to {filename}")

def main():
    """Extract real audio data from VoiceBench parquet file."""
    parquet_file = "test-00000-of-00001.parquet"
    
    # Load real audio data
    samples = load_voicebench_from_parquet(parquet_file, max_samples=10)
    
    if samples:
        print(f"Successfully loaded {len(samples)} samples with real audio")
        
        # Save to JSON for inspection
        save_samples_to_json(samples)
        
        # Print sample information
        print("\nSample information:")
        for i, sample in enumerate(samples[:3]):
            print(f"Sample {i+1}:")
            print(f"  Prompt: {sample['prompt'][:100]}...")
            print(f"  Audio duration: {len(sample['audio']['array']) / sample['audio']['sampling_rate']:.2f}s")
            print(f"  Sample rate: {sample['audio']['sampling_rate']} Hz")
            print(f"  Audio shape: {sample['audio']['array'].shape}")
            print()
    else:
        print("Failed to load any samples")

if __name__ == "__main__":
    main()

