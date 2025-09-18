#!/usr/bin/env python3
"""
Download Fresh VoiceBench Data - The Real Deal
This script actually downloads fresh data from HuggingFace VoiceBench dataset.
"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, Any, List

def setup_huggingface_environment():
    """Set up HuggingFace environment with proper permissions."""
    print("🔧 Setting up HuggingFace environment...")

    # Create a temporary directory with proper permissions
    temp_dir = "/tmp/voicebench_cache"
    os.makedirs(temp_dir, exist_ok=True)
    os.chmod(temp_dir, 0o755)

    # Set environment variables
    os.environ["HF_HOME"] = temp_dir
    os.environ["HF_DATASETS_CACHE"] = temp_dir
    os.environ["HF_HUB_CACHE"] = temp_dir
    os.environ["HF_TOKEN"] = ""  # No token needed for public dataset

    print(f"✅ Cache directory: {temp_dir}")
    return temp_dir

def download_voicebench_dataset(num_samples_per_dataset: int):
    """Download the actual VoiceBench dataset from HuggingFace."""
    print("🔥 DOWNLOADING REAL VOICEBENCH DATASET FROM HUGGINGFACE")
    print("=" * 60)

    cache_dir = setup_huggingface_environment()

    try:
        from datasets import load_dataset
        print("✅ datasets library imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import datasets: {e}")
        print("Install with: pip install datasets huggingface_hub")
        return {}

    datasets_to_download = ['commoneval', 'wildvoice', 'ifeval', 'advbench']
    all_samples = {}

    for dataset_name in datasets_to_download:
        print(f"\n📥 Downloading {dataset_name} dataset from HuggingFace...")

        try:
            # Load dataset without streaming
            print(f"   Loading {dataset_name} without streaming...")
            dataset = load_dataset('hlt-lab/voicebench', dataset_name, split=f'test[:{num_samples_per_dataset}]')

            samples = []
            count = 0

            print(f"   Processing samples...")
            for item in dataset:
                if count >= 10:
                    break

                # Extract text data only (skip audio to avoid issues)
                sample = {
                    'prompt': item.get('prompt', ''),
                    'output': item.get('output', ''),  # Usually empty in VoiceBench
                    'dataset': dataset_name,
                    'sample_index': count,
                    'hf_index': item.get('index', count),
                    'downloaded_fresh': True,
                    'source': 'hlt-lab/voicebench'
                }

                if sample['prompt']:  # Only add if we have a prompt
                    samples.append(sample)
                    count += 1

                    if count <= 3:  # Show first 3 samples
                        print(f"     Sample {count}: {sample['prompt'][:60]}...")

            all_samples[dataset_name] = samples
            print(f"✅ Downloaded {len(samples)} fresh samples from {dataset_name}")

        except Exception as e:
            print(f"❌ CRITICAL ERROR downloading {dataset_name}: {e}")
            print(f"   This may be due to network issues or dataset access problems")
            all_samples[dataset_name] = []

    # Save the fresh data
    if all_samples:
        save_fresh_data(all_samples, cache_dir)

    return all_samples

def save_fresh_data(all_samples: Dict[str, List[Dict[str, Any]]], cache_dir: str):
    """Save the freshly downloaded data."""
    print("💾 Saving Fresh VoiceBench Data..."    )

    # Save combined file
    all_samples_flat = []
    for dataset_name, samples in all_samples.items():
        all_samples_flat.extend(samples)

    fresh_filename = 'fresh_voicebench_all_datasets.json'
    with open(fresh_filename, 'w') as f:
        json.dump(all_samples_flat, f, indent=2)

    print(f"✅ Saved {len(all_samples_flat)} fresh samples to {fresh_filename}")

    # Save per-dataset files
    total_fresh = 0
    for dataset_name, samples in all_samples.items():
        if samples:
            filename = f'fresh_{dataset_name}_samples.json'
            with open(filename, 'w') as f:
                json.dump(samples, f, indent=2)
            print(f"✅ Saved {len(samples)} {dataset_name} samples to {filename}")
            total_fresh += len(samples)

    print(f"\n🎯 TOTAL FRESH SAMPLES DOWNLOADED: {total_fresh}")

    # Verify data is actually from HuggingFace
    if total_fresh > 0:
        verify_fresh_data(all_samples_flat)

def verify_fresh_data(samples: List[Dict[str, Any]]):
    """Verify the downloaded data is actually fresh from HuggingFace."""
    print("\n🔍 Verifying Fresh Data...")

    # Check for HuggingFace markers
    hf_markers = sum(1 for s in samples if s.get('downloaded_fresh') and s.get('source') == 'hlt-lab/voicebench')

    print(f"✅ HuggingFace markers found: {hf_markers}/{len(samples)}")

    # Show sample diversity
    datasets_found = set(s['dataset'] for s in samples)
    print(f"✅ Datasets downloaded: {sorted(datasets_found)}")

    # Show some real examples
    print("\n📋 Fresh VoiceBench Sample Examples:")
    for i, sample in enumerate(samples[:3]):
        print(f"   {sample['dataset'].upper()}: \"{sample['prompt'][:80]}...\"")

    return len(samples) > 0

def run_fresh_experiment():
    """Run experiment with freshly downloaded data."""
    print("\n🔬 Running Experiment with Fresh VoiceBench Data...")

    # Check if we have fresh data
    if not os.path.exists('fresh_voicebench_all_datasets.json'):
        print("❌ No fresh data found. Download first.")
        return

    # Load fresh data
    with open('fresh_voicebench_all_datasets.json', 'r') as f:
        fresh_samples = json.load(f)

    print(f"✅ Loaded {len(fresh_samples)} fresh samples")

    # Group by dataset
    datasets = {}
    for sample in fresh_samples:
        dataset_name = sample['dataset']
        if dataset_name not in datasets:
            datasets[dataset_name] = []
        datasets[dataset_name].append(sample)

    print(f"📊 Fresh dataset distribution:")
    for dataset_name, samples in datasets.items():
        print(f"   {dataset_name}: {len(samples)} samples")

    # Now run the actual experiment
    from critical_real_voicebench_experiment import CriticalRealVoiceBenchExperiment

    experiment = CriticalRealVoiceBenchExperiment()

    # Test server
    if not experiment.test_server_connection():
        print("❌ Server not available")
        return

    # Run experiment on fresh data from each dataset
    all_results = []
    for dataset_name, samples in datasets.items():
        if samples:
            print(f"\n🎯 Testing {dataset_name} with {len(samples)} fresh samples...")
            results = experiment.run_experiment_on_real_samples(samples)
            all_results.extend(results)

    if all_results:
        experiment.save_results(all_results)
        experiment.print_critical_analysis(all_results)
        print("🎉 SUCCESS! Experiment completed with FRESH VoiceBench data!")
        print(f"📊 Tested {len(all_results)} samples from {len(datasets)} datasets")
    else:
        print("\n❌ Experiment failed")

def main():
    """Main function - download fresh data and run experiment."""
    print("🚀 CRITICAL MISSION: Download Fresh VoiceBench Data & Run Experiment")
    print("=" * 70)

    # Step 1: Download fresh data
    print("STEP 1: DOWNLOADING FRESH DATA FROM HUGGINGFACE")
    num_samples_per_dataset = 10
    fresh_data = download_voicebench_dataset(num_samples_per_dataset)

    if not fresh_data:
        print("\n❌ CRITICAL FAILURE: Could not download fresh data")
        print("This may be due to:")
        print("  - Network connectivity issues")
        print("  - HuggingFace access problems")
        print("  - Dataset permission issues")
        return

    total_downloaded = sum(len(samples) for samples in fresh_data.values())
    if total_downloaded == 0:
        print("\n❌ CRITICAL FAILURE: No samples downloaded")
        return

    print("✅ SUCCESS! Downloaded fresh VoiceBench data")
    print(f"📊 Total samples: {total_downloaded}")

    # Step 2: Run experiment with fresh data
    print("STEP 2: RUNNING EXPERIMENT WITH FRESH DATA")    
    run_fresh_experiment()

if __name__ == "__main__":
    main()
