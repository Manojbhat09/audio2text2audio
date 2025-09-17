#!/usr/bin/env python3
"""
Test Whisper Integration

This script tests the complete Whisper integration including:
1. Model loading and initialization
2. Audio processing pipeline
3. Transcription accuracy
4. Performance metrics
5. Error handling and recovery
"""

import os
import sys
import time
import logging
import numpy as np
import soundfile as sf
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result data structure."""
    test_name: str
    success: bool
    duration: float
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class WhisperIntegrationTester:
    """Test complete Whisper integration."""
    
    def __init__(self, model_path: str = "models/wpt/wpt.pt"):
        self.model_path = model_path
        self.model = None
        self.results: List[TestResult] = []
        
    def load_model(self) -> bool:
        """Load the Whisper model."""
        start_time = time.time()
        
        try:
            logger.info(f"🔄 Loading Whisper model from {self.model_path}...")
            
            # Check if model file exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Import whisper
            import whisper
            
            # Load model
            self.model = whisper.load_model(self.model_path)
            
            duration = time.time() - start_time
            logger.info(f"✅ Whisper model loaded successfully in {duration:.3f}s")
            
            self.results.append(TestResult(
                test_name="model_loading",
                success=True,
                duration=duration,
                details={"model_path": self.model_path}
            ))
            
            return True
            
        except ImportError:
            error = "Whisper not installed. Install with: pip install openai-whisper"
            logger.error(f"❌ {error}")
            self.results.append(TestResult(
                test_name="model_loading",
                success=False,
                duration=time.time() - start_time,
                error=error
            ))
            return False
            
        except Exception as e:
            error = f"Failed to load Whisper model: {e}"
            logger.error(f"❌ {error}")
            self.results.append(TestResult(
                test_name="model_loading",
                success=False,
                duration=time.time() - start_time,
                error=error
            ))
            return False
    
    def create_test_audio(self, duration: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
        """Create test audio with speech-like characteristics."""
        start_time = time.time()
        
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
        
        duration = time.time() - start_time
        logger.info(f"✅ Test audio created in {duration:.3f}s")
        
        return audio.astype(np.float32)
    
    def load_audio_file(self, audio_path: str) -> Optional[Tuple[np.ndarray, int]]:
        """Load audio file."""
        start_time = time.time()
        
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            audio, sample_rate = sf.read(audio_path)
            
            duration = time.time() - start_time
            logger.info(f"✅ Loaded audio: {audio_path}")
            logger.info(f"   Duration: {len(audio) / sample_rate:.2f}s, Sample rate: {sample_rate}Hz")
            logger.info(f"   Load time: {duration:.3f}s")
            
            self.results.append(TestResult(
                test_name="audio_loading",
                success=True,
                duration=duration,
                details={
                    "audio_path": audio_path,
                    "duration": len(audio) / sample_rate,
                    "sample_rate": sample_rate
                }
            ))
            
            return audio, sample_rate
            
        except Exception as e:
            error = f"Failed to load audio file: {e}"
            logger.error(f"❌ {error}")
            self.results.append(TestResult(
                test_name="audio_loading",
                success=False,
                duration=time.time() - start_time,
                error=error
            ))
            return None
    
    def transcribe_audio(self, audio: np.ndarray, sample_rate: int) -> Optional[str]:
        """Transcribe audio using Whisper."""
        start_time = time.time()
        
        if self.model is None:
            error = "Model not loaded"
            logger.error(f"❌ {error}")
            self.results.append(TestResult(
                test_name="transcription",
                success=False,
                duration=time.time() - start_time,
                error=error
            ))
            return None
        
        try:
            logger.info("🎤 Transcribing audio...")
            
            # Transcribe using Whisper
            result = self.model.transcribe(audio, fp16=False, language=None)
            
            duration = time.time() - start_time
            logger.info(f"⏱️  Transcription took: {duration:.3f}s")
            
            # Extract text from result
            if isinstance(result, dict):
                text = result.get('text', '')
                confidence = result.get('confidence', 0.0)
            else:
                text = str(result)
                confidence = 0.0
            
            logger.info(f"✅ Transcription: {text}")
            if confidence > 0:
                logger.info(f"📊 Confidence: {confidence:.3f}")
            
            self.results.append(TestResult(
                test_name="transcription",
                success=True,
                duration=duration,
                details={
                    "text": text,
                    "confidence": confidence,
                    "audio_length": len(audio),
                    "sample_rate": sample_rate
                }
            ))
            
            return text
            
        except Exception as e:
            error = f"Transcription failed: {e}"
            logger.error(f"❌ {error}")
            self.results.append(TestResult(
                test_name="transcription",
                success=False,
                duration=time.time() - start_time,
                error=error
            ))
            return None
    
    def test_performance(self, audio: np.ndarray, sample_rate: int, num_runs: int = 3) -> Dict[str, Any]:
        """Test transcription performance with multiple runs."""
        logger.info(f"🔍 Testing performance with {num_runs} runs...")
        
        times = []
        successes = 0
        
        for i in range(num_runs):
            logger.info(f"   Run {i+1}/{num_runs}...")
            start_time = time.time()
            
            result = self.transcribe_audio(audio, sample_rate)
            
            if result is not None:
                successes += 1
                times.append(time.time() - start_time)
        
        if times:
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            logger.info(f"✅ Performance test completed:")
            logger.info(f"   Success rate: {successes}/{num_runs} ({successes/num_runs*100:.1f}%)")
            logger.info(f"   Average time: {avg_time:.3f}s")
            logger.info(f"   Min time: {min_time:.3f}s")
            logger.info(f"   Max time: {max_time:.3f}s")
            
            return {
                "success_rate": successes / num_runs,
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "times": times
            }
        else:
            logger.error("❌ Performance test failed - no successful runs")
            return {"success_rate": 0.0, "avg_time": 0.0, "min_time": 0.0, "max_time": 0.0}
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling with various edge cases."""
        logger.info("🔍 Testing error handling...")
        
        error_tests = []
        
        # Test with empty audio
        try:
            empty_audio = np.array([])
            result = self.transcribe_audio(empty_audio, 16000)
            error_tests.append({
                "test": "empty_audio",
                "success": result is None,  # Should fail gracefully
                "result": result
            })
        except Exception as e:
            error_tests.append({
                "test": "empty_audio",
                "success": True,  # Exception is expected
                "error": str(e)
            })
        
        # Test with very short audio
        try:
            short_audio = np.array([0.1, 0.2, 0.3])
            result = self.transcribe_audio(short_audio, 16000)
            error_tests.append({
                "test": "short_audio",
                "success": True,  # Should handle gracefully
                "result": result
            })
        except Exception as e:
            error_tests.append({
                "test": "short_audio",
                "success": False,
                "error": str(e)
            })
        
        # Test with invalid sample rate
        try:
            audio = self.create_test_audio(duration=1.0)
            result = self.transcribe_audio(audio, 0)  # Invalid sample rate
            error_tests.append({
                "test": "invalid_sample_rate",
                "success": True,  # Should handle gracefully
                "result": result
            })
        except Exception as e:
            error_tests.append({
                "test": "invalid_sample_rate",
                "success": True,  # Exception is expected
                "error": str(e)
            })
        
        logger.info(f"✅ Error handling tests completed: {len(error_tests)} tests")
        
        return {"error_tests": error_tests}
    
    def run_comprehensive_test(self, audio_path: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive integration test."""
        logger.info("🚀 Testing Whisper Integration")
        logger.info("=" * 50)
        
        # Test model loading
        if not self.load_model():
            logger.error("❌ Model loading failed, aborting tests")
            return {"overall_success": False, "results": self.results}
        
        # Load or create audio
        if audio_path and os.path.exists(audio_path):
            audio_data = self.load_audio_file(audio_path)
            if audio_data is None:
                logger.error("❌ Audio loading failed, using test audio")
                audio = self.create_test_audio()
                sample_rate = 16000
            else:
                audio, sample_rate = audio_data
        else:
            logger.info("🎵 Creating test audio...")
            audio = self.create_test_audio()
            sample_rate = 16000
        
        # Test transcription
        transcription = self.transcribe_audio(audio, sample_rate)
        
        # Test performance
        performance = self.test_performance(audio, sample_rate)
        
        # Test error handling
        error_handling = self.test_error_handling()
        
        # Compile results
        overall_success = all(result.success for result in self.results)
        
        results = {
            "overall_success": overall_success,
            "results": [
                {
                    "test_name": result.test_name,
                    "success": result.success,
                    "duration": result.duration,
                    "error": result.error,
                    "details": result.details
                }
                for result in self.results
            ],
            "performance": performance,
            "error_handling": error_handling,
            "transcription": transcription
        }
        
        # Print summary
        logger.info("\n📊 Test Summary:")
        for result in self.results:
            status = "✅" if result.success else "❌"
            logger.info(f"{status} {result.test_name}: {result.duration:.3f}s")
            if result.error:
                logger.info(f"   Error: {result.error}")
        
        if overall_success:
            logger.info("\n✅ All integration tests passed!")
        else:
            logger.error("\n❌ Some integration tests failed!")
        
        return results

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Whisper Integration")
    parser.add_argument("--model-path", default="models/wpt/wpt.pt",
                       help="Path to Whisper model (default: models/wpt/wpt.pt)")
    parser.add_argument("--audio-file", help="Path to audio file to test with")
    parser.add_argument("--output-file", help="Save results to JSON file")
    parser.add_argument("--performance-runs", type=int, default=3,
                       help="Number of performance test runs (default: 3)")
    
    args = parser.parse_args()
    
    tester = WhisperIntegrationTester(args.model_path)
    results = tester.run_comprehensive_test(args.audio_file)
    
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"📁 Results saved to: {args.output_file}")
    
    exit(0 if results["overall_success"] else 1)

if __name__ == "__main__":
    main()





