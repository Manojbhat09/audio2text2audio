#!/usr/bin/env python3
"""
Test script to verify dataset loading is working correctly.
"""

import json
from critical_real_voicebench_experiment import load_actual_real_voicebench_samples

def test_dataset_loading():
    """Test the dataset loading functionality."""
    print("🧪 Testing Dataset Loading")
    print("=" * 50)
    
    # Load datasets
    datasets = load_actual_real_voicebench_samples()
    
    if not datasets:
        print("❌ No datasets loaded")
        return
    
    # Show dataset distribution
    print(f"\n📊 Dataset Distribution:")
    total_samples = 0
    for dataset_name, samples in datasets.items():
        print(f"   {dataset_name}: {len(samples)} samples")
        total_samples += len(samples)
    
    print(f"\n✅ Total samples: {total_samples}")
    
    # Show sample examples with dataset info
    print(f"\n📋 Sample Examples:")
    for dataset_name, samples in datasets.items():
        if samples:
            sample = samples[0]
            print(f"   [{dataset_name}] {sample['prompt'][:80]}...")
    
    # Test the dataset assignment logic
    print(f"\n🔍 Testing Dataset Assignment:")
    all_samples = []
    for dataset_samples in datasets.values():
        all_samples.extend(dataset_samples)
    
    dataset_counts = {}
    for sample in all_samples:
        dataset = sample.get('dataset', 'unknown')
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
    
    print(f"   Dataset counts in loaded samples:")
    for dataset, count in dataset_counts.items():
        print(f"     {dataset}: {count}")

if __name__ == "__main__":
    test_dataset_loading()
