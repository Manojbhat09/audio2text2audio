#!/usr/bin/env python3
"""
Download Real VoiceBench Dataset Samples
This script downloads actual samples from all 4 VoiceBench datasets.
"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, Any, List

def download_real_voicebench_samples(num_samples_per_dataset: int = 10) -> Dict[str, List[Dict[str, Any]]]:
    """Download real VoiceBench samples from all datasets."""
    print("🔍 Downloading Real VoiceBench Dataset Samples")
    print("=" * 60)

    # Set cache directory with proper permissions
    cache_dir = "/tmp/huggingface_cache"
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    datasets_to_download = ['commoneval', 'wildvoice', 'ifeval', 'advbench']
    all_samples = {}

    try:
        from datasets import load_dataset
        print("✅ datasets library imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import datasets: {e}")
        return {}

    for dataset_name in datasets_to_download:
        print(f"\n📊 Downloading {dataset_name} dataset ({num_samples_per_dataset} samples)...")

        try:
            # Load dataset with limited samples
            dataset = load_dataset('hlt-lab/voicebench', dataset_name, split=f'test[:{num_samples_per_dataset}]')
            print(f"✅ {dataset_name}: Loaded {len(dataset)} real samples")

            samples = []
            for i, item in enumerate(dataset):
                # Extract text data only (skip audio to avoid torchcodec issues)
                sample = {
                    'prompt': item.get('prompt', ''),
                    'output': item.get('output', ''),
                    'dataset': dataset_name,
                    'sample_index': i,
                    'hf_index': item.get('index', i),
                    'audio_duration': 0  # Skip audio data
                }
                samples.append(sample)

                if i < 3:  # Show first 3 samples as examples
                    print(f"   Sample {i+1}: {sample['prompt'][:80]}...")

            all_samples[dataset_name] = samples

        except Exception as e:
            print(f"❌ Error downloading {dataset_name}: {e}")
            all_samples[dataset_name] = []

    # Save the downloaded data
    if all_samples:
        save_downloaded_samples(all_samples)

    return all_samples

def save_downloaded_samples(all_samples: Dict[str, List[Dict[str, Any]]]):
    """Save downloaded samples to files."""
    print("
💾 Saving downloaded samples..."    )

    # Save combined file
    all_samples_flat = []
    for dataset_name, samples in all_samples.items():
        all_samples_flat.extend(samples)

    with open('downloaded_real_voicebench_samples.json', 'w') as f:
        json.dump(all_samples_flat, f, indent=2)

    print(f"✅ Saved {len(all_samples_flat)} total samples to downloaded_real_voicebench_samples.json")

    # Save per-dataset files
    for dataset_name, samples in all_samples.items():
        filename = f'{dataset_name}_real_samples.json'
        with open(filename, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f"✅ Saved {len(samples)} {dataset_name} samples to {filename}")

def load_existing_samples() -> Dict[str, List[Dict[str, Any]]]:
    """Load existing downloaded samples if available."""
    print("🔍 Checking for existing downloaded samples...")

    try:
        with open('downloaded_real_voicebench_samples.json', 'r') as f:
            all_samples_flat = json.load(f)

        # Group by dataset
        datasets = {}
        for sample in all_samples_flat:
            dataset_name = sample['dataset']
            if dataset_name not in datasets:
                datasets[dataset_name] = []
            datasets[dataset_name].append(sample)

        total_samples = sum(len(samples) for samples in datasets.values())
        print(f"✅ Found {total_samples} existing real samples")

        for dataset_name, samples in datasets.items():
            print(f"   {dataset_name}: {len(samples)} samples")

        return datasets

    except FileNotFoundError:
        print("❌ No existing downloaded samples found")
        return {}

def main():
    """Main function to download real VoiceBench samples."""
    print("🚀 Download Real VoiceBench Dataset Samples")
    print("=" * 60)

    # Check if we already have downloaded samples
    existing_samples = load_existing_samples()

    if existing_samples and all(len(samples) >= 10 for samples in existing_samples.values()):
        print("✅ Already have sufficient real samples!")
        return existing_samples
    else:
        print("📥 Downloading real VoiceBench samples...")
        downloaded_samples = download_real_voicebench_samples(10)

        if downloaded_samples:
            total_downloaded = sum(len(samples) for samples in downloaded_samples.values())
            print(f"\n🎯 Successfully downloaded {total_downloaded} real VoiceBench samples!")
            print("Ready for experiment!")
            return downloaded_samples
        else:
            print("\n❌ Failed to download real samples")
            return {}

if __name__ == "__main__":
    main()
