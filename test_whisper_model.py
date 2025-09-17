#!/usr/bin/env python3
"""
Test Whisper Model Directly

This script tests the Whisper model directly without going through the server.
Based on the patterns seen in the markdown files, this tests:
1. Model loading
2. Direct transcription
3. Audio processing
"""

import os
import sys
import time
import logging
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WhisperModelTester:
    """Test Whisper model directly."""
    
    def __init__(self, model_path: str = "models/wpt/wpt.pt"):
        self.model_path = model_path
        self.model = None
        
    def load_model(self) -> bool:
        """Load the Whisper model."""
        try:
            logger.info(f"Loading Whisper model from {self.model_path}...")
            
            # Import whisper
            import whisper
            
            # Load model
            self.model = whisper.load_model(self.model_path)
            logger.info("✅ Whisper model loaded successfully")
            return True
            
        except ImportError:
            logger.error("❌ Whisper not installed. Install with: pip install openai-whisper")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model: {e}")
            return False
    
    def create_test_audio(self, duration: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
        """Create test audio with speech-like characteristics."""
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Create a more complex audio signal that mimics speech
        # Multiple frequencies to simulate formants
        audio = (
            0.3 * np.sin(2 * np.pi * 200 * t) +  # Fundamental frequency
            0.2 * np.sin(2 * np.pi * 400 * t) +  # First harmonic
            0.1 * np.sin(2 * np.pi * 600 * t) +  # Second harmonic
            0.05 * np.random.randn(len(t))        # Add some noise
        )
        
        # Apply envelope to make it more speech-like
        envelope = np.exp(-t / duration * 2)  # Decay envelope
        audio = audio * envelope
        
        return audio.astype(np.float32)
    
    def load_audio_file(self, audio_path: str) -> Optional[tuple[np.ndarray, int]]:
        """Load audio file."""
        try:
            audio, sample_rate = sf.read(audio_path)
            logger.info(f"✅ Loaded audio: {audio_path}")
            logger.info(f"   Duration: {len(audio) / sample_rate:.2f}s, Sample rate: {sample_rate}Hz")
            return audio, sample_rate
        except Exception as e:
            logger.error(f"❌ Failed to load audio file: {e}")
            return None
    
    def transcribe_audio(self, audio: np.ndarray, sample_rate: int) -> Optional[str]:
        """Transcribe audio using Whisper."""
        if self.model is None:
            logger.error("❌ Model not loaded")
            return None
        
        try:
            logger.info("🎤 Transcribing audio...")
            start_time = time.time()
            
            # Transcribe using Whisper
            result = self.model.transcribe(audio, fp16=False, language=None)
            
            transcription_time = time.time() - start_time
            logger.info(f"⏱️  Transcription took: {transcription_time:.3f}s")
            
            # Extract text from result
            if isinstance(result, dict):
                text = result.get('text', '')
            else:
                text = str(result)
            
            logger.info(f"✅ Transcription: {text}")
            return text
            
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            return None
    
    def test_model_loading(self) -> bool:
        """Test model loading."""
        logger.info("🔍 Testing model loading...")
        return self.load_model()
    
    def test_transcription(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Test transcription."""
        logger.info("🔍 Testing transcription...")
        result = self.transcribe_audio(audio, sample_rate)
        return result is not None
    
    def run_comprehensive_test(self, audio_path: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive test."""
        logger.info("🚀 Testing Whisper Model Directly")
        logger.info("=" * 50)
        
        results = {
            "model_loading": False,
            "transcription": False,
            "audio_loading": False,
            "transcription_text": None,
            "errors": []
        }
        
        # Test model loading
        if not self.test_model_loading():
            results["errors"].append("Model loading failed")
            return results
        
        results["model_loading"] = True
        
        # Load or create audio
        if audio_path and os.path.exists(audio_path):
            audio_data = self.load_audio_file(audio_path)
            if audio_data is None:
                results["errors"].append("Audio loading failed")
                return results
            audio, sample_rate = audio_data
            results["audio_loading"] = True
        else:
            logger.info("🎵 Creating test audio...")
            audio = self.create_test_audio()
            sample_rate = 16000
            results["audio_loading"] = True
        
        # Test transcription
        if self.test_transcription(audio, sample_rate):
            results["transcription"] = True
            results["transcription_text"] = self.transcribe_audio(audio, sample_rate)
        
        return results

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Whisper Model Directly")
    parser.add_argument("--model-path", default="models/wpt/wpt.pt",
                       help="Path to Whisper model (default: models/wpt/wpt.pt)")
    parser.add_argument("--audio-file", help="Path to audio file to test with")
    parser.add_argument("--create-test-audio", action="store_true",
                       help="Create test audio instead of using file")
    
    args = parser.parse_args()
    
    tester = WhisperModelTester(args.model_path)
    
    if args.create_test_audio:
        results = tester.run_comprehensive_test()
    else:
        results = tester.run_comprehensive_test(args.audio_file)
    
    # Print results
    logger.info("\n📊 Test Results:")
    logger.info(f"Model Loading: {'✅' if results['model_loading'] else '❌'}")
    logger.info(f"Audio Loading: {'✅' if results['audio_loading'] else '❌'}")
    logger.info(f"Transcription: {'✅' if results['transcription'] else '❌'}")
    
    if results['transcription_text']:
        logger.info(f"Transcription Text: {results['transcription_text']}")
    
    if results['errors']:
        logger.error("Errors:")
        for error in results['errors']:
            logger.error(f"  - {error}")
    
    success = results['model_loading'] and results['transcription']
    logger.info(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed!'}")
    
    exit(0 if success else 1)

if __name__ == "__main__":
    main()





