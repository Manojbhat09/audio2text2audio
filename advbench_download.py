#!/usr/bin/env python3
"""
Advbench Dataset Downloader
Downloads the advbench dataset.
"""

import os
import json
from typing import Dict, Any, List

def download_advbench():
    """Download the advbench dataset."""
    print("🚀 Downloading Advbench Dataset")
    print("=" * 40)
    
    # Set up environment
    os.environ["HF_HOME"] = "/tmp/huggingface_cache"
    os.makedirs("/tmp/huggingface_cache", exist_ok=True)
    
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        
        print("📥 Downloading advbench dataset...")
        parquet_file = "advbench/test-00000-of-00001.parquet"
        
        local_file = hf_hub_download(
            repo_id="hlt-lab/voicebench",
            filename=parquet_file,
            repo_type="dataset"
        )
        
        # Read parquet file
        df = pd.read_parquet(local_file)
        print(f"✅ Loaded {len(df)} rows from advbench")
        
        # Convert to samples (take first 20 samples)
        samples = []
        for i, row in df.head(20).iterrows():
            sample = {
                'prompt': row.get('prompt', ''),
                'output': row.get('output', ''),
                'dataset': 'advbench',
                'sample_index': i,
                'hf_index': row.get('index', i),
                'downloaded_fresh': True,
                'source': 'hlt-lab/voicebench'
            }
            samples.append(sample)
            
            if i < 5:  # Show first 5 samples
                print(f"   Sample {i+1}: {sample['prompt'][:60]}...")
        
        print(f"✅ Converted {len(samples)} samples")
        
        # Save the data
        with open('advbench_samples.json', 'w') as f:
            json.dump(samples, f, indent=2)
        
        print(f"💾 Saved {len(samples)} advbench samples to advbench_samples.json")
        
        return samples
        
    except Exception as e:
        print(f"❌ Error downloading advbench: {e}")
        return []

def main():
    """Main function."""
    print("🔥 ADVBENCH DOWNLOADER")
    print("=" * 30)
    
    samples = download_advbench()
    
    if samples:
        print(f"\n🎉 SUCCESS! Downloaded {len(samples)} advbench samples")
    else:
        print("\n❌ Download failed")

if __name__ == "__main__":
    main()
