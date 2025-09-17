#!/usr/bin/env python3
"""
Whisper Server Analysis

This script analyzes why the Whisper server is giving low scores and provides
insights into the transcription quality issues.
"""

import requests
import json
import numpy as np
import base64
import io
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import soundfile as sf

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WhisperAnalysis:
    """Analyze Whisper server transcription quality."""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.session = requests.Session()
        self.session.timeout = 60
        
    def create_test_audio(self, text: str, method: str = "synthetic") -> np.ndarray:
        """Create test audio using different methods."""
        if method == "synthetic":
            return self.create_synthetic_audio(text)
        elif method == "silence":
            return self.create_silence_audio(text)
        elif method == "noise":
            return self.create_noise_audio(text)
        else:
            return self.create_synthetic_audio(text)
    
    def create_synthetic_audio(self, text: str) -> np.ndarray:
        """Create synthetic audio (current approach)."""
        words = len(text.split())
        duration = max(1.0, min(10.0, words * 0.3))
        sample_rate = 16000
        num_samples = int(duration * sample_rate)
        
        t = np.linspace(0, duration, num_samples)
        audio = np.zeros(num_samples)
        
        # Add fundamental frequency
        fundamental_freq = 150 + (hash(text) % 100)
        audio += 0.3 * np.sin(2 * np.pi * fundamental_freq * t)
        
        # Add harmonics
        for harmonic in [2, 3, 4, 5]:
            audio += 0.1 * np.sin(2 * np.pi * fundamental_freq * harmonic * t)
        
        # Add noise
        audio += 0.05 * np.random.randn(num_samples)
        
        # Add envelope
        envelope = np.exp(-t / duration) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
        audio *= envelope
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio.astype(np.float32)
    
    def create_silence_audio(self, text: str) -> np.ndarray:
        """Create silence audio to test fallback behavior."""
        words = len(text.split())
        duration = max(1.0, min(10.0, words * 0.3))
        sample_rate = 16000
        num_samples = int(duration * sample_rate)
        
        # Pure silence
        return np.zeros(num_samples, dtype=np.float32)
    
    def create_noise_audio(self, text: str) -> np.ndarray:
        """Create noise-only audio."""
        words = len(text.split())
        duration = max(1.0, min(10.0, words * 0.3))
        sample_rate = 16000
        num_samples = int(duration * sample_rate)
        
        # Pure noise
        return np.random.randn(num_samples).astype(np.float32) * 0.1
    
    def audio_to_base64(self, audio: np.ndarray) -> str:
        """Convert audio array to base64 string."""
        buffer = io.BytesIO()
        np.save(buffer, audio, allow_pickle=False)
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def test_transcription(self, text: str, audio_method: str = "synthetic") -> Dict[str, Any]:
        """Test transcription with different audio generation methods."""
        print(f"🎤 Testing: '{text[:50]}...' with {audio_method} audio")
        
        # Create audio
        audio = self.create_test_audio(text, audio_method)
        audio_b64 = self.audio_to_base64(audio)
        
        # Prepare request
        request_data = {
            "audio_data": audio_b64,
            "sample_rate": 16000
        }
        
        # Send request
        start_time = time.time()
        try:
            response = self.session.post(
                f"{self.server_url}/api/v1/v2t",
                json=request_data,
                timeout=60
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                transcription = result.get('text', '')
                
                # Analyze the response
                analysis = self.analyze_transcription(text, transcription, audio_method)
                
                return {
                    'success': True,
                    'original_text': text,
                    'transcription': transcription,
                    'response_time': response_time,
                    'audio_method': audio_method,
                    'analysis': analysis
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}",
                    'original_text': text,
                    'audio_method': audio_method
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'original_text': text,
                'audio_method': audio_method
            }
    
    def analyze_transcription(self, original: str, transcription: str, audio_method: str) -> Dict[str, Any]:
        """Analyze the quality of the transcription."""
        analysis = {
            'is_fallback_response': False,
            'is_accurate': False,
            'response_type': 'unknown',
            'confidence': 'low'
        }
        
        # Check for common fallback responses
        fallback_phrases = [
            "i didn't hear anything clearly",
            "could you please repeat",
            "it seems like your message is quite short",
            "could you please rephrase",
            "i'll do my best to help",
            "as an answer 5 points",
            "the response below gives detailed information"
        ]
        
        transcription_lower = transcription.lower()
        for phrase in fallback_phrases:
            if phrase in transcription_lower:
                analysis['is_fallback_response'] = True
                analysis['response_type'] = 'fallback'
                break
        
        # Check if it's a generic response
        if not analysis['is_fallback_response']:
            if len(transcription) < 10:
                analysis['response_type'] = 'too_short'
            elif any(word in transcription_lower for word in ['question', 'help', 'assist']):
                analysis['response_type'] = 'generic_help'
            else:
                analysis['response_type'] = 'actual_transcription'
                analysis['confidence'] = 'medium'
        
        # Check accuracy (simple word overlap)
        if analysis['response_type'] == 'actual_transcription':
            original_words = set(original.lower().split())
            transcription_words = set(transcription.lower().split())
            overlap = len(original_words.intersection(transcription_words))
            if overlap > 0:
                analysis['is_accurate'] = True
                analysis['confidence'] = 'high'
                analysis['word_overlap'] = overlap
        
        return analysis
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis of the Whisper server."""
        print("🔍 WHISPER SERVER ANALYSIS")
        print("=" * 50)
        
        # Test cases
        test_cases = [
            "What is the capital of France?",
            "Hello, how are you today?",
            "The quick brown fox jumps over the lazy dog.",
            "Can you help me with this problem?",
            "Testing one two three four five."
        ]
        
        audio_methods = ["synthetic", "silence", "noise"]
        
        results = []
        
        for method in audio_methods:
            print(f"\n📊 Testing with {method} audio:")
            print("-" * 30)
            
            for text in test_cases:
                result = self.test_transcription(text, method)
                results.append(result)
                
                if result['success']:
                    analysis = result['analysis']
                    print(f"✅ {text[:30]}...")
                    print(f"   Transcription: {result['transcription'][:60]}...")
                    print(f"   Type: {analysis['response_type']}")
                    print(f"   Fallback: {analysis['is_fallback_response']}")
                    print(f"   Accurate: {analysis['is_accurate']}")
                    print(f"   Time: {result['response_time']:.3f}s")
                else:
                    print(f"❌ {text[:30]}... - {result['error']}")
                print()
        
        # Overall analysis
        self.analyze_overall_results(results)
        
        return results
    
    def analyze_overall_results(self, results: List[Dict[str, Any]]):
        """Analyze overall results and provide recommendations."""
        print("\n📈 OVERALL ANALYSIS")
        print("=" * 50)
        
        # Group by audio method
        by_method = {}
        for result in results:
            method = result['audio_method']
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(result)
        
        for method, method_results in by_method.items():
            successful = [r for r in method_results if r['success']]
            fallback_responses = [r for r in successful if r['analysis']['is_fallback_response']]
            accurate_responses = [r for r in successful if r['analysis']['is_accurate']]
            
            print(f"\n🎵 {method.upper()} AUDIO:")
            print(f"   Total tests: {len(method_results)}")
            print(f"   Successful: {len(successful)}")
            print(f"   Fallback responses: {len(fallback_responses)}")
            print(f"   Accurate transcriptions: {len(accurate_responses)}")
            
            if successful:
                avg_time = np.mean([r['response_time'] for r in successful])
                print(f"   Average response time: {avg_time:.3f}s")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("=" * 50)
        
        all_successful = [r for r in results if r['success']]
        all_fallback = [r for r in all_successful if r['analysis']['is_fallback_response']]
        
        if len(all_fallback) > len(all_successful) * 0.8:
            print("❌ MAJOR ISSUE: Most responses are fallback responses")
            print("   This indicates the audio is not being recognized as speech")
            print("   Solutions:")
            print("   1. Use real audio files instead of synthetic audio")
            print("   2. Improve synthetic audio generation to be more speech-like")
            print("   3. Check if the server is properly processing the audio data")
        elif len(all_fallback) > len(all_successful) * 0.5:
            print("⚠️  MODERATE ISSUE: Many responses are fallback responses")
            print("   The audio quality needs improvement")
        else:
            print("✅ GOOD: Most responses are actual transcriptions")
        
        # Check for accuracy
        accurate_responses = [r for r in all_successful if r['analysis']['is_accurate']]
        if len(accurate_responses) < len(all_successful) * 0.3:
            print("❌ ACCURACY ISSUE: Low transcription accuracy")
            print("   The model is transcribing but not accurately")
            print("   This could be due to:")
            print("   1. Poor audio quality")
            print("   2. Model not trained for the audio characteristics")
            print("   3. Audio preprocessing issues")
        
        print(f"\n🎯 SUMMARY:")
        print(f"   Total tests: {len(results)}")
        print(f"   Successful: {len(all_successful)}")
        print(f"   Fallback responses: {len(all_fallback)}")
        print(f"   Accurate transcriptions: {len(accurate_responses)}")
        
        if len(all_successful) > 0:
            success_rate = len(all_successful) / len(results) * 100
            fallback_rate = len(all_fallback) / len(all_successful) * 100
            accuracy_rate = len(accurate_responses) / len(all_successful) * 100
            
            print(f"   Success rate: {success_rate:.1f}%")
            print(f"   Fallback rate: {fallback_rate:.1f}%")
            print(f"   Accuracy rate: {accuracy_rate:.1f}%")


def main():
    """Main function to run the analysis."""
    analyzer = WhisperAnalysis()
    results = analyzer.run_comprehensive_analysis()
    
    # Exit with appropriate code
    if results:
        all_successful = [r for r in results if r['success']]
        all_fallback = [r for r in all_successful if r['analysis']['is_fallback_response']]
        
        if len(all_fallback) > len(all_successful) * 0.8:
            print("\n❌ Analysis complete - Major issues detected")
            exit(1)
        else:
            print("\n✅ Analysis complete")
            exit(0)
    else:
        print("\n❌ Analysis failed")
        exit(1)


if __name__ == "__main__":
    main()





