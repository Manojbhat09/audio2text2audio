#!/usr/bin/env python3
"""
Get Real VoiceBench Data - Critical Fix
This script downloads actual VoiceBench samples from all datasets.
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List

def download_real_voicebench_critical() -> Dict[str, List[Dict[str, Any]]]:
    """Download real VoiceBench samples with critical error handling."""
    print("🔥 CRITICAL: Downloading Real VoiceBench Dataset")
    print("=" * 60)

    # Create temporary cache directory with proper permissions
    temp_cache = tempfile.mkdtemp()
    print(f"📁 Using cache directory: {temp_cache}")

    # Set environment variables for HuggingFace
    os.environ["HF_HOME"] = temp_cache
    os.environ["HF_DATASETS_CACHE"] = temp_cache
    os.environ["HF_HUB_CACHE"] = temp_cache

    datasets_to_download = ['commoneval', 'wildvoice', 'ifeval', 'advbench']
    all_samples = {}

    try:
        from datasets import load_dataset
        print("✅ datasets library imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import datasets: {e}")
        return {}

    for dataset_name in datasets_to_download:
        print(f"\n📊 CRITICAL: Downloading {dataset_name} dataset...")

        try:
            # Use streaming to avoid downloading everything
            print(f"   Loading {dataset_name} with streaming...")
            dataset = load_dataset(
                'hlt-lab/voicebench',
                dataset_name,
                split='test[:10]',  # Only first 10 samples
                streaming=True
            )

            samples = []
            count = 0

            for item in dataset:
                if count >= 10:
                    break

                # Extract only text data (avoid audio)
                sample = {
                    'prompt': item.get('prompt', ''),
                    'output': item.get('output', ''),  # Usually empty in VoiceBench
                    'dataset': dataset_name,
                    'sample_index': count,
                    'hf_index': item.get('index', count)
                }

                if sample['prompt']:  # Only add if we have a prompt
                    samples.append(sample)
                    count += 1

                    if count <= 3:  # Show first 3 samples
                        print(f"   Sample {count}: {sample['prompt'][:80]}...")

            all_samples[dataset_name] = samples
            print(f"✅ {dataset_name}: Downloaded {len(samples)} real samples")

        except Exception as e:
            print(f"❌ CRITICAL ERROR downloading {dataset_name}: {e}")
            print(f"   Skipping {dataset_name} dataset")
            all_samples[dataset_name] = []

    # Save the real data
    if all_samples:
        save_real_voicebench_data(all_samples)

    return all_samples

def save_real_voicebench_data(all_samples: Dict[str, List[Dict[str, Any]]]):
    """Save the real VoiceBench data we downloaded."""
    print("\n💾 Saving Real VoiceBench Data...")

    # Save combined file
    all_samples_flat = []
    for dataset_name, samples in all_samples.items():
        all_samples_flat.extend(samples)

    with open('real_voicebench_all_datasets.json', 'w') as f:
        json.dump(all_samples_flat, f, indent=2)

    print(f"✅ Saved {len(all_samples_flat)} total real samples")

    # Save per-dataset files
    total_real = 0
    for dataset_name, samples in all_samples.items():
        if samples:
            filename = f'real_{dataset_name}_samples.json'
            with open(filename, 'w') as f:
                json.dump(samples, f, indent=2)
            print(f"✅ Saved {len(samples)} {dataset_name} samples")
            total_real += len(samples)

    print(f"\n🎯 TOTAL REAL VOICEBENCH SAMPLES DOWNLOADED: {total_real}")

def verify_real_data(all_samples: Dict[str, List[Dict[str, Any]]]):
    """Verify the real data we downloaded."""
    print("\n🔍 Verifying Real VoiceBench Data...")
    print("=" * 60)

    total_samples = sum(len(samples) for samples in all_samples.values())
    print(f"Total samples downloaded: {total_samples}")

    for dataset_name, samples in all_samples.items():
        print(f"\n📊 {dataset_name.upper()}:")
        print(f"   Samples: {len(samples)}")

        if samples:
            # Show sample diversity
            print("   Sample prompts:")
            for i, sample in enumerate(samples[:3]):
                prompt = sample['prompt'][:100]
                print(f"     {i+1}. {prompt}...")
        else:
            print("   ❌ No samples downloaded")

    return total_samples > 0

def main():
    """Main function - get real VoiceBench data."""
    print("🚀 CRITICAL MISSION: Get Real VoiceBench Data")
    print("=" * 60)

    # Check if we already have real data
    if os.path.exists('real_voicebench_all_datasets.json'):
        print("✅ Found existing real VoiceBench data!")
        try:
            with open('real_voicebench_all_datasets.json', 'r') as f:
                existing_data = json.load(f)

            # Group by dataset
            datasets = {}
            for sample in existing_data:
                dataset_name = sample['dataset']
                if dataset_name not in datasets:
                    datasets[dataset_name] = []
                datasets[dataset_name].append(sample)

            total_existing = sum(len(samples) for samples in datasets.values())
            print(f"✅ Existing data: {total_existing} real samples")

            if total_existing >= 40:  # 10 samples per dataset
                print("✅ Already have sufficient real data!")
                return datasets

        except Exception as e:
            print(f"❌ Error reading existing data: {e}")

    # Download real data
    print("\n🔥 DOWNLOADING REAL VOICEBENCH DATA...")
    real_samples = download_real_voicebench_critical()

    if real_samples:
        total_downloaded = sum(len(samples) for samples in real_samples.values())

        if total_downloaded > 0:
            print(f"\n🎯 SUCCESS! Downloaded {total_downloaded} real VoiceBench samples!")
            verify_real_data(real_samples)
            return real_samples
        else:
            print("\n❌ CRITICAL FAILURE: No real samples downloaded!")
            return {}
    else:
        print("\n❌ CRITICAL FAILURE: Download process failed!")
        return {}

if __name__ == "__main__":
    main()
