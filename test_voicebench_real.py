#!/usr/bin/env python3
"""
Real VoiceBench test using actual dataset with proper audio transcription.
"""

import requests
import json
import time
import numpy as np
from typing import Dict, Any, List
from datasets import load_dataset
import librosa
import whisper
from pathlib import Path

class RealVoiceBenchTester:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.results = []
        self.whisper_model = None
        
    def load_whisper_model(self):
        """Load Whisper model for transcription."""
        try:
            whisper_path = Path("models/wpt/wpt.pt")
            if whisper_path.exists():
                print(f"Loading Whisper model from {whisper_path}")
                self.whisper_model = whisper.load_model(str(whisper_path))
                print("✅ Whisper model loaded successfully")
                return True
            else:
                print(f"❌ Whisper model not found at {whisper_path}")
                return False
        except Exception as e:
            print(f"❌ Error loading Whisper model: {e}")
            return False
    
    def transcribe_audio(self, audio_array: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio using Whisper."""
        if self.whisper_model is None:
            return ""
        
        try:
            # Ensure audio is in correct format
            if sample_rate != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            
            # Transcribe
            result = self.whisper_model.transcribe(audio_array, fp16=False, language=None)
            return result["text"].strip()
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return ""
    
    def load_real_voicebench_dataset(self, dataset_name: str, max_samples: int = 5) -> List[Dict[str, Any]]:
        """Load real VoiceBench dataset with actual audio data."""
        print(f"📊 Loading real {dataset_name} dataset...")
        
        try:
            # Load with streaming to avoid downloading everything
            dataset = load_dataset('hlt-lab/voicebench', dataset_name, split='test', streaming=True)
            print(f"✅ Dataset loaded successfully")
            
            samples = []
            count = 0
            
            for item in dataset:
                if count >= max_samples:
                    break
                
                print(f"Processing sample {count + 1}/{max_samples}...")
                
                # Extract text data
                prompt = item.get('prompt', '')
                output = item.get('output', '')
                
                # Get audio data
                audio_data = None
                sample_rate = 16000
                
                if 'audio' in item:
                    audio_info = item['audio']
                    if isinstance(audio_info, dict) and 'array' in audio_info:
                        audio_array = audio_info['array']
                        sample_rate = audio_info.get('sampling_rate', 16000)
                        
                        # Convert to numpy array if needed
                        if not isinstance(audio_array, np.ndarray):
                            audio_array = np.array(audio_array, dtype=np.float32)
                        
                        # Normalize audio
                        if audio_array.max() > 1.0 or audio_array.min() < -1.0:
                            audio_array = audio_array / np.max(np.abs(audio_array))
                        
                        audio_data = audio_array
                        print(f"   Audio shape: {audio_data.shape}, duration: {len(audio_data)/sample_rate:.2f}s")
                    else:
                        print(f"   No audio array found")
                else:
                    print(f"   No audio field found")
                
                # Transcribe audio if available
                transcribed_text = ""
                if audio_data is not None and self.whisper_model is not None:
                    print(f"   Transcribing audio...")
                    transcribed_text = self.transcribe_audio(audio_data, sample_rate)
                    print(f"   Transcribed: {transcribed_text[:100]}...")
                else:
                    # Use prompt as fallback
                    transcribed_text = prompt
                    print(f"   Using prompt as fallback: {prompt[:100]}...")
                
                sample = {
                    'prompt': prompt,
                    'output': output,
                    'transcribed_text': transcribed_text,
                    'audio_available': audio_data is not None,
                    'dataset': dataset_name,
                    'sample_index': count
                }
                
                samples.append(sample)
                count += 1
            
            print(f"✅ Successfully loaded {len(samples)} real samples from {dataset_name}")
            return samples
            
        except Exception as e:
            print(f"❌ Error loading dataset {dataset_name}: {e}")
            return []
    
    def test_single_turn_endpoint(self, text: str) -> Dict[str, Any]:
        """Test single-turn endpoint."""
        try:
            response = requests.post(
                f"{self.api_url}/api/v1/t2t",
                json={"text_data": text},
                timeout=20
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}", "text": response.text}
        except requests.exceptions.Timeout:
            return {"error": "Timeout", "text": "Request timed out"}
        except Exception as e:
            return {"error": str(e), "text": "Request failed"}
    
    def run_real_voicebench_test(self, dataset_name: str, max_samples: int = 5):
        """Run test on real VoiceBench dataset."""
        print(f"\n🚀 Testing Real VoiceBench Dataset: {dataset_name}")
        print("=" * 60)
        
        # Load real dataset
        samples = self.load_real_voicebench_dataset(dataset_name, max_samples)
        
        if not samples:
            print(f"❌ No samples loaded for {dataset_name}")
            return []
        
        results = []
        
        for i, sample in enumerate(samples):
            print(f"\n--- Sample {i+1}/{len(samples)} ---")
            print(f"Original prompt: {sample['prompt']}")
            print(f"Transcribed text: {sample['transcribed_text']}")
            print(f"Expected output: {sample['output']}")
            print(f"Audio available: {sample['audio_available']}")
            
            # Test with transcribed text
            response_data = self.test_single_turn_endpoint(sample['transcribed_text'])
            
            if 'error' in response_data:
                print(f"❌ Error: {response_data['error']}")
                response_text = response_data.get('text', 'Error')
            else:
                response_text = response_data.get('text', 'No response')
                print(f"✅ Response received")
            
            print(f"Response: {response_text[:200]}...")
            
            # Calculate metrics
            response_length = len(response_text)
            expected_length = len(sample['output'])
            length_ratio = response_length / expected_length if expected_length > 0 else 0
            
            result = {
                'dataset': dataset_name,
                'sample_index': i,
                'original_prompt': sample['prompt'],
                'transcribed_text': sample['transcribed_text'],
                'expected_output': sample['output'],
                'response': response_text,
                'response_length': response_length,
                'expected_length': expected_length,
                'length_ratio': length_ratio,
                'audio_available': sample['audio_available'],
                'experiment_type': 'single_turn_real',
                'error': response_data.get('error', None)
            }
            
            results.append(result)
            
            # Small delay between requests
            time.sleep(1)
        
        return results
    
    def run_all_real_tests(self, max_samples_per_dataset: int = 3):
        """Run tests on all VoiceBench datasets with real data."""
        print("🚀 Real VoiceBench Dataset Test")
        print("=" * 50)
        
        # Load Whisper model
        if not self.load_whisper_model():
            print("⚠️ Continuing without Whisper transcription...")
        
        # Test each dataset
        datasets = ['commoneval', 'wildvoice', 'ifeval', 'advbench']
        all_results = []
        
        for dataset_name in datasets:
            results = self.run_real_voicebench_test(dataset_name, max_samples_per_dataset)
            all_results.extend(results)
        
        # Save results
        self.save_results(all_results)
        
        # Print summary
        self.print_summary(all_results)
        
        return all_results
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save results to JSON file."""
        timestamp = int(time.time())
        filename = f"voicebench_real_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print summary statistics."""
        print("\n" + "=" * 60)
        print("📊 REAL VOICEBENCH TEST SUMMARY")
        print("=" * 60)
        
        # Group by dataset
        datasets = {}
        for result in results:
            dataset = result['dataset']
            if dataset not in datasets:
                datasets[dataset] = []
            datasets[dataset].append(result)
        
        for dataset_name, dataset_results in datasets.items():
            print(f"\n📈 {dataset_name.upper()} Dataset:")
            print(f"  Samples: {len(dataset_results)}")
            
            # Calculate statistics
            successful_results = [r for r in dataset_results if r.get('error') is None]
            error_results = [r for r in dataset_results if r.get('error') is not None]
            audio_available = [r for r in dataset_results if r.get('audio_available', False)]
            
            print(f"  Successful: {len(successful_results)}")
            print(f"  Errors: {len(error_results)}")
            print(f"  Audio available: {len(audio_available)}")
            
            if successful_results:
                response_lengths = [r['response_length'] for r in successful_results]
                expected_lengths = [r['expected_length'] for r in successful_results]
                length_ratios = [r['length_ratio'] for r in successful_results]
                
                print(f"  Avg Response Length: {sum(response_lengths) / len(response_lengths):.1f}")
                print(f"  Avg Expected Length: {sum(expected_lengths) / len(expected_lengths):.1f}")
                print(f"  Avg Length Ratio: {sum(length_ratios) / len(length_ratios):.2f}")
        
        # Overall statistics
        all_successful = [r for r in results if r.get('error') is None]
        all_errors = [r for r in results if r.get('error') is not None]
        all_audio = [r for r in results if r.get('audio_available', False)]
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"  Total Samples: {len(results)}")
        print(f"  Successful: {len(all_successful)}")
        print(f"  Errors: {len(all_errors)}")
        print(f"  Audio Available: {len(all_audio)}")
        print(f"  Success Rate: {len(all_successful) / len(results) * 100:.1f}%")

def main():
    """Main function."""
    tester = RealVoiceBenchTester()
    results = tester.run_all_real_tests(max_samples_per_dataset=3)
    
    if results:
        successful = len([r for r in results if r.get('error') is None])
        total = len(results)
        print(f"\n✅ Real VoiceBench test completed! {successful}/{total} samples successful")
    else:
        print("\n❌ Real VoiceBench test failed!")

if __name__ == "__main__":
    main()
