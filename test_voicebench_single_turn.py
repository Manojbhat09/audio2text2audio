#!/usr/bin/env python3
"""
Test VoiceBench datasets with single-turn conversation.
Tests 10 samples from each dataset: commoneval, wildvoice, ifeval, advbench
"""

import requests
import json
import time
import numpy as np
from typing import Dict, Any, List
from load_voicebench_direct import load_voicebench_direct, create_manual_voicebench_samples

class VoiceBenchSingleTurnTester:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.results = []
        
    def test_server_connection(self) -> bool:
        """Test if server is running and accessible."""
        try:
            response = requests.get(f"{self.api_url}/api/v1/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server is running and accessible")
                return True
            else:
                print(f"❌ Server returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return False
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio using the server's Whisper endpoint."""
        try:
            # Convert audio to base64 for transmission
            import base64
            import io
            import soundfile as sf
            
            # Save audio to bytes
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, audio_data, sample_rate, format='WAV')
            audio_bytes.seek(0)
            
            # Encode to base64
            audio_b64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
            
            # Send to server
            response = requests.post(
                f"{self.api_url}/api/v1/audio2text",
                json={"audio_data": audio_b64},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('text', '')
            else:
                print(f"❌ Transcription failed: {response.status_code}")
                return ""
                
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""
    
    def get_single_turn_response(self, text: str) -> Dict[str, Any]:
        """Get single-turn response from the server."""
        try:
            response = requests.post(
                f"{self.api_url}/api/v1/t2t",
                json={"text_data": text},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Single-turn request failed: {response.status_code}")
                return {"text": "Error", "error": response.text}
                
        except Exception as e:
            print(f"❌ Single-turn error: {e}")
            return {"text": "Error", "error": str(e)}
    
    def run_single_turn_test(self, dataset_name: str, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run single-turn test on a dataset."""
        print(f"\n🔍 Testing {dataset_name} dataset ({len(samples)} samples)")
        print("=" * 60)
        
        results = []
        
        for i, sample in enumerate(samples):
            print(f"\n--- Sample {i+1}/{len(samples)} ---")
            print(f"Prompt: {sample['prompt']}")
            print(f"Expected: {sample['output']}")
            
            # Get single-turn response
            response_data = self.get_single_turn_response(sample['prompt'])
            response_text = response_data.get('text', '')
            
            print(f"Response: {response_text}")
            
            # Calculate metrics
            response_length = len(response_text)
            expected_length = len(sample['output'])
            length_ratio = response_length / expected_length if expected_length > 0 else 0
            
            result = {
                'dataset': dataset_name,
                'sample_index': i,
                'prompt': sample['prompt'],
                'expected': sample['output'],
                'response': response_text,
                'response_length': response_length,
                'expected_length': expected_length,
                'length_ratio': length_ratio,
                'experiment_type': 'single_turn'
            }
            
            results.append(result)
            
            # Small delay between requests
            time.sleep(0.5)
        
        return results
    
    def run_all_tests(self, max_samples: int = 10):
        """Run single-turn tests on all VoiceBench datasets."""
        print("🚀 VoiceBench Single-Turn Test")
        print("=" * 50)
        
        # Test server connection
        if not self.test_server_connection():
            print("❌ Server not accessible. Please start the server first.")
            return
        
        # Load VoiceBench data
        print(f"\n📊 Loading VoiceBench data ({max_samples} samples per dataset)...")
        try:
            voicebench_data = load_voicebench_direct(max_samples)
        except Exception as e:
            print(f"❌ Error loading VoiceBench data: {e}")
            print("Using manual samples instead...")
            voicebench_data = create_manual_voicebench_samples()
        
        # Group samples by dataset
        datasets = {}
        for sample in voicebench_data:
            dataset = sample.get('dataset', 'unknown')
            if dataset not in datasets:
                datasets[dataset] = []
            datasets[dataset].append(sample)
        
        print(f"Found datasets: {list(datasets.keys())}")
        
        # Run tests for each dataset
        all_results = []
        for dataset_name, samples in datasets.items():
            if len(samples) > 0:
                results = self.run_single_turn_test(dataset_name, samples[:max_samples])
                all_results.extend(results)
        
        # Save results
        self.save_results(all_results)
        
        # Print summary
        self.print_summary(all_results)
        
        return all_results
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save results to JSON file."""
        timestamp = int(time.time())
        filename = f"voicebench_single_turn_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print summary statistics."""
        print("\n" + "=" * 60)
        print("📊 SUMMARY STATISTICS")
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
            response_lengths = [r['response_length'] for r in dataset_results]
            expected_lengths = [r['expected_length'] for r in dataset_results]
            length_ratios = [r['length_ratio'] for r in dataset_results]
            
            print(f"  Avg Response Length: {sum(response_lengths) / len(response_lengths):.1f}")
            print(f"  Avg Expected Length: {sum(expected_lengths) / len(expected_lengths):.1f}")
            print(f"  Avg Length Ratio: {sum(length_ratios) / len(length_ratios):.2f}")
            print(f"  Min Length Ratio: {min(length_ratios):.2f}")
            print(f"  Max Length Ratio: {max(length_ratios):.2f}")
        
        # Overall statistics
        all_response_lengths = [r['response_length'] for r in results]
        all_expected_lengths = [r['expected_length'] for r in results]
        all_length_ratios = [r['length_ratio'] for r in results]
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"  Total Samples: {len(results)}")
        print(f"  Avg Response Length: {sum(all_response_lengths) / len(all_response_lengths):.1f}")
        print(f"  Avg Expected Length: {sum(all_expected_lengths) / len(all_expected_lengths):.1f}")
        print(f"  Avg Length Ratio: {sum(all_length_ratios) / len(all_length_ratios):.2f}")

def main():
    """Main function."""
    tester = VoiceBenchSingleTurnTester()
    results = tester.run_all_tests(max_samples=10)
    
    if results:
        print(f"\n✅ Single-turn test completed successfully!")
        print(f"Tested {len(results)} samples across {len(set(r['dataset'] for r in results))} datasets")
    else:
        print("\n❌ Single-turn test failed!")

if __name__ == "__main__":
    main()
