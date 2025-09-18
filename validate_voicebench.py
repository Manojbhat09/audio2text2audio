#!/usr/bin/env python3
"""
Validate VoiceBench dataset loading and check real data structure.
"""

import sys
from pathlib import Path

def test_dataset_loading():
    """Test loading VoiceBench dataset with real data."""
    print("🔍 Validating VoiceBench Dataset Loading")
    print("=" * 50)
    
    try:
        from datasets import load_dataset
        print("✅ datasets library imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import datasets: {e}")
        return False
    
    # Test each dataset configuration
    datasets_to_test = ['commoneval', 'wildvoice', 'ifeval', 'advbench']
    
    for dataset_name in datasets_to_test:
        print(f"\n📊 Testing {dataset_name} dataset...")
        
        try:
            # Load with streaming to avoid downloading everything
            dataset = load_dataset('hlt-lab/voicebench', dataset_name, split='test', streaming=True)
            print(f"✅ {dataset_name} dataset loaded successfully")
            
            # Get first item to check structure
            item = next(iter(dataset))
            print(f"   Sample keys: {list(item.keys())}")
            
            # Check for required fields
            if 'prompt' in item:
                prompt = item['prompt']
                print(f"   Prompt: {prompt[:100]}...")
            else:
                print("   ❌ No 'prompt' field found")
            
            if 'output' in item:
                output = item['output']
                print(f"   Output: {output[:100]}...")
            else:
                print("   ❌ No 'output' field found")
            
            # Check audio field
            if 'audio' in item:
                audio_info = item['audio']
                print(f"   Audio type: {type(audio_info)}")
                if isinstance(audio_info, dict):
                    print(f"   Audio keys: {list(audio_info.keys())}")
                    if 'array' in audio_info:
                        audio_array = audio_info['array']
                        print(f"   Audio array type: {type(audio_array)}")
                        if hasattr(audio_array, 'shape'):
                            print(f"   Audio shape: {audio_array.shape}")
                        if hasattr(audio_array, 'dtype'):
                            print(f"   Audio dtype: {audio_array.dtype}")
                    if 'sampling_rate' in audio_info:
                        print(f"   Sampling rate: {audio_info['sampling_rate']}")
            else:
                print("   ❌ No 'audio' field found")
            
            print(f"✅ {dataset_name} validation completed")
            
        except Exception as e:
            print(f"❌ Error loading {dataset_name}: {e}")
            continue
    
    return True

def test_audio_processing():
    """Test audio processing capabilities."""
    print(f"\n🎵 Testing Audio Processing")
    print("=" * 30)
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import numpy: {e}")
        return False
    
    try:
        import librosa
        print("✅ librosa imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import librosa: {e}")
        return False
    
    try:
        import whisper
        print("✅ whisper imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import whisper: {e}")
        return False
    
    return True

def test_whisper_model():
    """Test if Whisper model is available."""
    print(f"\n🎤 Testing Whisper Model")
    print("=" * 25)
    
    whisper_model_path = Path("models/wpt/wpt.pt")
    if whisper_model_path.exists():
        print(f"✅ Whisper model found at: {whisper_model_path}")
        return True
    else:
        print(f"❌ Whisper model not found at: {whisper_model_path}")
        print("   Available files in models/wpt/:")
        wpt_dir = Path("models/wpt")
        if wpt_dir.exists():
            for file in wpt_dir.iterdir():
                print(f"     {file.name}")
        return False

def main():
    """Main validation function."""
    print("🚀 VoiceBench Dataset Validation")
    print("=" * 40)
    
    # Test dataset loading
    dataset_ok = test_dataset_loading()
    
    # Test audio processing
    audio_ok = test_audio_processing()
    
    # Test whisper model
    whisper_ok = test_whisper_model()
    
    print(f"\n📋 Validation Summary:")
    print(f"   Dataset Loading: {'✅' if dataset_ok else '❌'}")
    print(f"   Audio Processing: {'✅' if audio_ok else '❌'}")
    print(f"   Whisper Model: {'✅' if whisper_ok else '❌'}")
    
    if dataset_ok and audio_ok and whisper_ok:
        print(f"\n🎉 All validations passed! Ready to use real VoiceBench data.")
        return True
    else:
        print(f"\n⚠️ Some validations failed. Check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
