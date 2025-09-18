#!/usr/bin/env python3
"""
Quick VoiceBench Data Downloader
Downloads just a few samples from each dataset quickly.
"""

import os
import json
import requests
from typing import Dict, Any, List

def download_quick_voicebench():
    """Download VoiceBench data quickly with minimal samples."""
    print("🚀 Quick VoiceBench Data Download")
    print("=" * 50)
    
    # Set up environment
    os.environ["HF_HOME"] = "/tmp/huggingface_cache"
    os.makedirs("/tmp/huggingface_cache", exist_ok=True)
    
    datasets_to_download = ['commoneval', 'wildvoice', 'ifeval', 'advbench']
    all_samples = {}
    
    for dataset_name in datasets_to_download:
        print(f"\n📥 Downloading {dataset_name} dataset...")
        
        try:
            from huggingface_hub import hf_hub_download
            import pandas as pd
            
            # Try to download the first parquet file
            parquet_file = f"{dataset_name}/test-00000-of-00001.parquet"
            print(f"   Downloading {parquet_file}...")
            
            local_file = hf_hub_download(
                repo_id="hlt-lab/voicebench",
                filename=parquet_file,
                repo_type="dataset"
            )
            
            # Read parquet file
            df = pd.read_parquet(local_file)
            print(f"   Loaded {len(df)} rows from parquet file")
            
            # Convert to samples (take first 3 samples)
            samples = []
            for i, row in df.head(3).iterrows():
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
                
                print(f"     Sample {i+1}: {sample['prompt'][:60]}...")
            
            all_samples[dataset_name] = samples
            print(f"✅ Downloaded {len(samples)} samples from {dataset_name}")
                
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
    
    with open('quick_voicebench_all_datasets.json', 'w') as f:
        json.dump(all_samples_flat, f, indent=2)
    
    print(f"✅ Saved {len(all_samples_flat)} total samples")
    
    # Save per-dataset files
    total_samples = 0
    for dataset_name, samples in all_samples.items():
        if samples:
            filename = f'quick_{dataset_name}_samples.json'
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
    print("🔥 QUICK VOICEBENCH DOWNLOADER")
    print("=" * 40)
    
    # Download data
    samples = download_quick_voicebench()
    
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
