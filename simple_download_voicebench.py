#!/usr/bin/env python3
"""
Simple VoiceBench Data Downloader
Downloads VoiceBench data using HuggingFace Hub directly.
"""

import os
import json
import requests
from typing import Dict, Any, List

def download_voicebench_with_hub():
    """Download VoiceBench data using HuggingFace Hub API."""
    print("🚀 Downloading VoiceBench Data with HuggingFace Hub")
    print("=" * 60)
    
    # Set up environment
    os.environ["HF_HOME"] = "/tmp/huggingface_cache"
    os.makedirs("/tmp/huggingface_cache", exist_ok=True)
    
    datasets_to_download = ['commoneval', 'wildvoice', 'ifeval', 'advbench']
    all_samples = {}
    
    for dataset_name in datasets_to_download:
        print(f"\n📥 Downloading {dataset_name} dataset...")
        
        try:
            # Use HuggingFace Hub to get dataset info
            from huggingface_hub import HfApi, hf_hub_download
            api = HfApi()
            
            # Get dataset files
            repo_files = api.list_repo_files("hlt-lab/voicebench", repo_type="dataset")
            parquet_files = [f for f in repo_files if f.endswith('.parquet') and dataset_name in f]
            
            print(f"   Found {len(parquet_files)} parquet files for {dataset_name}")
            
            if parquet_files:
                # Download the first parquet file
                parquet_file = parquet_files[0]
                print(f"   Downloading {parquet_file}...")
                
                local_file = hf_hub_download(
                    repo_id="hlt-lab/voicebench",
                    filename=parquet_file,
                    repo_type="dataset"
                )
                
                # Read parquet file
                import pandas as pd
                df = pd.read_parquet(local_file)
                
                # Convert to samples
                samples = []
                for i, row in df.head(5).iterrows():  # Take first 5 samples
                    sample = {
                        'prompt': row.get('prompt', ''),
                        'output': row.get('output', ''),
                        'dataset': dataset_name,
                        'sample_index': i,
                        'hf_index': row.get('index', i),
                        'downloaded_fresh': True,
                        'source': 'hlt-lab/voicebench'
                    }
                    samples.append(sample)
                    
                    if i < 3:  # Show first 3 samples
                        print(f"     Sample {i+1}: {sample['prompt'][:60]}...")
                
                all_samples[dataset_name] = samples
                print(f"✅ Downloaded {len(samples)} samples from {dataset_name}")
            else:
                print(f"❌ No parquet files found for {dataset_name}")
                all_samples[dataset_name] = []
                
        except Exception as e:
            print(f"❌ Error downloading {dataset_name}: {e}")
            all_samples[dataset_name] = []
    
    # Save the data
    if all_samples:
        save_voicebench_data(all_samples)
    
    return all_samples

def save_voicebench_data(all_samples: Dict[str, List[Dict[str, Any]]]):
    """Save the downloaded VoiceBench data."""
    print("\n💾 Saving VoiceBench Data...")
    
    # Save combined file
    all_samples_flat = []
    for dataset_name, samples in all_samples.items():
        all_samples_flat.extend(samples)
    
    with open('fresh_voicebench_all_datasets.json', 'w') as f:
        json.dump(all_samples_flat, f, indent=2)
    
    print(f"✅ Saved {len(all_samples_flat)} total samples")
    
    # Save per-dataset files
    total_samples = 0
    for dataset_name, samples in all_samples.items():
        if samples:
            filename = f'fresh_{dataset_name}_samples.json'
            with open(filename, 'w') as f:
                json.dump(samples, f, indent=2)
            print(f"✅ Saved {len(samples)} {dataset_name} samples")
            total_samples += len(samples)
    
    print(f"\n🎯 TOTAL SAMPLES DOWNLOADED: {total_samples}")
    
    # Show summary
    print("\n📊 Dataset Summary:")
    for dataset_name, samples in all_samples.items():
        print(f"   {dataset_name}: {len(samples)} samples")

def main():
    """Main function."""
    print("🔥 SIMPLE VOICEBENCH DOWNLOADER")
    print("=" * 40)
    
    # Download data
    samples = download_voicebench_with_hub()
    
    if samples:
        total = sum(len(samples) for samples in samples.values())
        if total > 0:
            print(f"\n🎉 SUCCESS! Downloaded {total} samples from {len(samples)} datasets")
        else:
            print("\n❌ No samples downloaded")
    else:
        print("\n❌ Download failed")

if __name__ == "__main__":
    main()
