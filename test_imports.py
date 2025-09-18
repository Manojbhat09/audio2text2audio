#!/usr/bin/env python3
"""Test imports for VoiceBench testing."""

print("Testing imports...")

try:
    import requests
    print("✅ requests imported")
except ImportError as e:
    print(f"❌ requests failed: {e}")

try:
    import json
    print("✅ json imported")
except ImportError as e:
    print(f"❌ json failed: {e}")

try:
    import numpy as np
    print("✅ numpy imported")
except ImportError as e:
    print(f"❌ numpy failed: {e}")

try:
    from datasets import load_dataset
    print("✅ datasets imported")
except ImportError as e:
    print(f"❌ datasets failed: {e}")

try:
    import librosa
    print("✅ librosa imported")
except ImportError as e:
    print(f"❌ librosa failed: {e}")

try:
    import whisper
    print("✅ whisper imported")
except ImportError as e:
    print(f"❌ whisper failed: {e}")

print("Import test completed!")
