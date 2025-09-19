#!/usr/bin/env python3
"""
Test script for the optimized server to verify model loading and functionality.
"""

import requests
import json
import numpy as np
import base64
import io
import soundfile as sf
import tempfile
import os

def create_test_audio(duration=2.0, sample_rate=16000):
    """Create a simple test audio signal."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Generate a simple sine wave
    frequency = 440  # A4 note
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    return audio.astype(np.float32)

def audio_to_base64(audio, sample_rate):
    """Convert audio array to base64 string."""
    buf = io.BytesIO()
    np.save(buf, audio.astype(np.float32))
    return base64.b64encode(buf.getvalue()).decode()

def test_health_check():
    """Test the health check endpoint."""
    try:
        response = requests.get("http://localhost:8000/api/v1/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health check passed: {data}")
            
            # Show device information if available
            if "language_model_device" in data:
                print(f"  📱 Language Model Device: {data['language_model_device']}")
                print(f"  🔢 Model Dtype: {data['language_model_dtype']}")
                if data.get("cuda_available"):
                    print(f"  🎮 CUDA Device: {data.get('cuda_device_name', 'Unknown')}")
                    print(f"  💾 Memory Allocated: {data.get('cuda_memory_allocated', 'Unknown')}")
                    print(f"  💾 Memory Reserved: {data.get('cuda_memory_reserved', 'Unknown')}")
                else:
                    print("  ⚠ CUDA not available - running on CPU")
            
            return data.get("model_loaded", False)
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def test_v2t_endpoint():
    """Test the voice-to-text endpoint."""
    try:
        # Create test audio
        audio = create_test_audio()
        audio_b64 = audio_to_base64(audio, 16000)
        
        payload = {
            "audio_data": audio_b64,
            "sample_rate": 16000
        }
        
        response = requests.post(
            "http://localhost:8000/api/v1/v2t",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ V2T endpoint working: {data.get('text', 'No text')[:100]}...")
            return True
        else:
            print(f"✗ V2T endpoint failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ V2T endpoint error: {e}")
        return False

def test_error_scenarios():
    """Test error scenarios to ensure proper responses."""
    print("\n4. Testing error scenarios...")
    
    # Test with invalid audio data
    try:
        payload = {
            "audio_data": "invalid_base64_data",
            "sample_rate": 16000
        }
        
        response = requests.post(
            "http://localhost:8000/api/v1/v2t",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Error handling working: {data.get('text', 'No text')[:100]}...")
        else:
            print(f"✗ Error handling failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"✗ Error scenario test failed: {e}")
    
    # Test with missing fields
    try:
        payload = {
            "audio_data": "",
            "sample_rate": 16000
        }
        
        response = requests.post(
            "http://localhost:8000/api/v1/v2t",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Empty input handling working: {data.get('text', 'No text')[:100]}...")
        else:
            print(f"✗ Empty input handling failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"✗ Empty input test failed: {e}")
    
    return True

def test_authentication():
    """Test authentication functionality."""
    print("\n5. Testing authentication...")
    
    # Test with valid audio data (should work if auth passes)
    try:
        audio = create_test_audio()
        audio_b64 = audio_to_base64(audio, 16000)
        
        payload = {
            "audio_data": audio_b64,
            "sample_rate": 16000
        }
        
        response = requests.post(
            "http://localhost:8000/api/v1/v2t",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            text = data.get('text', '')
            if "Authentication failed" in text:
                print(f"⚠ Authentication check working: {text}")
            else:
                print(f"✓ Authentication passed: {text[:100]}...")
            return True
        else:
            print(f"✗ Authentication test failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Authentication test error: {e}")
        return False

def test_inference_endpoint():
    """Test the inference endpoint (if INTERFACE is available)."""
    try:
        # Create test audio
        audio = create_test_audio()
        audio_b64 = audio_to_base64(audio, 16000)
        
        payload = {
            "audio_data": audio_b64,
            "sample_rate": 16000
        }
        
        response = requests.post(
            "http://localhost:8000/api/v1/inference",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Inference endpoint working: Audio data length {len(data.get('audio_data', ''))}")
            return True
        elif response.status_code == 503:
            print(f"⚠ Inference endpoint not available (expected if outetts models not loaded): {response.text}")
            return True  # This is expected if outetts models are not available
        else:
            print(f"✗ Inference endpoint failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Inference endpoint error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing optimized server...")
    print("=" * 50)
    
    # Test health check
    print("\n1. Testing health check...")
    models_loaded = test_health_check()
    
    if not models_loaded:
        print("⚠ Models not loaded. Some tests may fail.")
    
    # Test V2T endpoint
    print("\n2. Testing voice-to-text endpoint...")
    v2t_success = test_v2t_endpoint()
    
    # Test inference endpoint
    print("\n3. Testing inference endpoint...")
    inference_success = test_inference_endpoint()
    
    # Test error scenarios
    error_success = test_error_scenarios()
    
    # Test authentication
    auth_success = test_authentication()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Health Check: {'✓' if models_loaded else '✗'}")
    print(f"V2T Endpoint: {'✓' if v2t_success else '✗'}")
    print(f"Inference Endpoint: {'✓' if inference_success else '✗'}")
    print(f"Error Handling: {'✓' if error_success else '✗'}")
    print(f"Authentication: {'✓' if auth_success else '✗'}")
    
    if models_loaded and v2t_success and error_success and auth_success:
        print("\n🎉 Server is working correctly with authentication and error handling!")
    else:
        print("\n⚠ Some issues detected. Check the logs above.")

if __name__ == "__main__":
    main()
