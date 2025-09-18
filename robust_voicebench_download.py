#!/usr/bin/env python3
"""
Robust VoiceBench Data Downloader
Downloads samples from all datasets by first checking what files are available.
"""

import os
import json
import requests
from typing import Dict, Any, List

def get_available_files():
    """Get list of available parquet files for each dataset."""
    print("🔍 Checking available files...")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Get all files in the dataset
        repo_files = api.list_repo_files("hlt-lab/voicebench", repo_type="dataset")
        parquet_files = [f for f in repo_files if f.endswith('.parquet')]
        
        # Group by dataset
        dataset_files = {}
        for file in parquet_files:
            dataset_name = file.split('/')[0]
            if dataset_name not in dataset_files:
                dataset_files[dataset_name] = []
            dataset_files[dataset_name].append(file)
        
        print("📁 Available files:")
        for dataset_name, files in dataset_files.items():
            print(f"   {dataset_name}: {len(files)} files")
            for file in files[:3]:  # Show first 3 files
                print(f"     - {file}")
        
        return dataset_files
        
    except Exception as e:
        print(f"❌ Error getting file list: {e}")
        return {}

def download_robust_voicebench():
    """Download VoiceBench data robustly."""
    print("🚀 Robust VoiceBench Data Download")
    print("=" * 50)
    
    # Set up environment
    os.environ["HF_HOME"] = "/tmp/huggingface_cache"
    os.makedirs("/tmp/huggingface_cache", exist_ok=True)
    
    # Get available files
    dataset_files = get_available_files()
    
    if not dataset_files:
        print("❌ Could not get file list")
        return {}
    
    all_samples = {}
    
    for dataset_name, files in dataset_files.items():
        print(f"\n📥 Downloading {dataset_name} dataset...")
        
        if not files:
            print(f"   No files found for {dataset_name}")
            all_samples[dataset_name] = []
            continue
        
        try:
            from huggingface_hub import hf_hub_download
            import pandas as pd
            
            # Try the first file
            parquet_file = files[0]
            print(f"   Downloading {parquet_file}...")
            
            local_file = hf_hub_download(
                repo_id="hlt-lab/voicebench",
                filename=parquet_file,
                repo_type="dataset"
            )
            
            # Read parquet file
            df = pd.read_parquet(local_file)
            print(f"   Loaded {len(df)} rows from parquet file")
            
            # Convert to samples (take first 5 samples)
            samples = []
            for i, row in df.head(5).iterrows():
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
    
    with open('robust_voicebench_all_datasets.json', 'w') as f:
        json.dump(all_samples_flat, f, indent=2)
    
    print(f"✅ Saved {len(all_samples_flat)} total samples")
    
    # Save per-dataset files
    total_samples = 0
    for dataset_name, samples in all_samples.items():
        if samples:
            filename = f'robust_{dataset_name}_samples.json'
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
    print("🔥 ROBUST VOICEBENCH DOWNLOADER")
    print("=" * 40)
    
    # Download data
    samples = download_robust_voicebench()
    
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
