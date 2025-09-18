#!/usr/bin/env python3
"""Test environment and imports."""

print("Testing environment...")
print("Python is working!")

import sys
print(f"Python version: {sys.version}")

try:
    from datasets import load_dataset
    print("✅ datasets imported successfully")
    
    print("Testing dataset loading...")
    dataset = load_dataset('hlt-lab/voicebench', 'commoneval', split='test', streaming=True)
    print("✅ Dataset loaded")
    
    item = next(iter(dataset))
    print(f"Sample keys: {list(item.keys())}")
    print(f"Prompt: {item.get('prompt', 'No prompt')[:100]}...")
    print("✅ Dataset sample loaded successfully")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Environment test completed!")
