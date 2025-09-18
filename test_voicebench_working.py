#!/usr/bin/env python3
"""
Working VoiceBench test with real dataset and proper error handling.
"""

import requests
import json
import time
from typing import Dict, Any, List
from datasets import load_dataset

class WorkingVoiceBenchTester:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.results = []
        
    def test_server_connection(self) -> bool:
        """Test if server is running and healthy."""
        try:
            response = requests.get(f"{self.api_url}/api/v1/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Server is healthy (uptime: {health_data.get('uptime_seconds', 0):.1f}s)")
                return True
            else:
                print(f"❌ Server returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return False
    
    def load_real_dataset_sample(self, dataset_name: str) -> Dict[str, Any]:
        """Load a single sample from real VoiceBench dataset."""
        print(f"📊 Loading sample from {dataset_name} dataset...")
        
        try:
            # Load with streaming
            dataset = load_dataset('hlt-lab/voicebench', dataset_name, split='test', streaming=True)
            print(f"✅ Dataset loaded successfully")
            
            # Get first item
            item = next(iter(dataset))
            print(f"✅ Sample loaded")
            
            # Extract data
            prompt = item.get('prompt', '')
            output = item.get('output', '')
            
            print(f"   Prompt: {prompt[:100]}...")
            print(f"   Output: {output[:100]}...")
            
            # Check for audio
            audio_available = False
            if 'audio' in item:
                audio_info = item['audio']
                if isinstance(audio_info, dict) and 'array' in audio_info:
                    audio_available = True
                    print(f"   Audio available: {len(audio_info['array'])} samples")
                else:
                    print(f"   Audio field present but no array")
            else:
                print(f"   No audio field")
            
            return {
                'prompt': prompt,
                'output': output,
                'audio_available': audio_available,
                'dataset': dataset_name
            }
            
        except Exception as e:
            print(f"❌ Error loading dataset {dataset_name}: {e}")
            return None
    
    def test_single_turn_response(self, text: str) -> Dict[str, Any]:
        """Test single-turn response."""
        try:
            print(f"   Sending request: {text[:50]}...")
            response = requests.post(
                f"{self.api_url}/api/v1/t2t",
                json={"text_data": text},
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('text', '')
                print(f"   ✅ Response received ({len(response_text)} chars)")
                return {
                    'success': True,
                    'response': response_text,
                    'error': None
                }
            else:
                print(f"   ❌ HTTP {response.status_code}: {response.text}")
                return {
                    'success': False,
                    'response': '',
                    'error': f"HTTP {response.status_code}"
                }
                
        except requests.exceptions.Timeout:
            print(f"   ❌ Request timed out")
            return {
                'success': False,
                'response': '',
                'error': 'Timeout'
            }
        except Exception as e:
            print(f"   ❌ Request error: {e}")
            return {
                'success': False,
                'response': '',
                'error': str(e)
            }
    
    def run_dataset_test(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Run test on a single dataset."""
        print(f"\n🚀 Testing {dataset_name.upper()} Dataset")
        print("=" * 50)
        
        # Load sample
        sample = self.load_real_dataset_sample(dataset_name)
        if not sample:
            return []
        
        # Test response
        result = self.test_single_turn_response(sample['prompt'])
        
        # Calculate metrics
        response_length = len(result['response'])
        expected_length = len(sample['output'])
        length_ratio = response_length / expected_length if expected_length > 0 else 0
        
        test_result = {
            'dataset': dataset_name,
            'prompt': sample['prompt'],
            'expected_output': sample['output'],
            'response': result['response'],
            'response_length': response_length,
            'expected_length': expected_length,
            'length_ratio': length_ratio,
            'audio_available': sample['audio_available'],
            'success': result['success'],
            'error': result['error'],
            'experiment_type': 'single_turn_real'
        }
        
        return [test_result]
    
    def run_all_tests(self):
        """Run tests on all VoiceBench datasets."""
        print("🚀 Working VoiceBench Test with Real Dataset")
        print("=" * 60)
        
        # Test server connection
        if not self.test_server_connection():
            print("❌ Server not available. Please start the server first.")
            return []
        
        # Test each dataset
        datasets = ['commoneval', 'wildvoice', 'ifeval', 'advbench']
        all_results = []
        
        for dataset_name in datasets:
            try:
                results = self.run_dataset_test(dataset_name)
                all_results.extend(results)
                time.sleep(1)  # Small delay between datasets
            except Exception as e:
                print(f"❌ Error testing {dataset_name}: {e}")
                continue
        
        # Save results
        if all_results:
            self.save_results(all_results)
            self.print_summary(all_results)
        
        return all_results
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save results to JSON file."""
        timestamp = int(time.time())
        filename = f"voicebench_working_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print summary statistics."""
        print("\n" + "=" * 60)
        print("📊 VOICEBENCH TEST SUMMARY")
        print("=" * 60)
        
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]
        
        print(f"Total Tests: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Success Rate: {len(successful) / len(results) * 100:.1f}%")
        
        if successful:
            response_lengths = [r['response_length'] for r in successful]
            expected_lengths = [r['expected_length'] for r in successful]
            length_ratios = [r['length_ratio'] for r in successful]
            
            print(f"\nResponse Statistics:")
            print(f"  Avg Response Length: {sum(response_lengths) / len(response_lengths):.1f}")
            print(f"  Avg Expected Length: {sum(expected_lengths) / len(expected_lengths):.1f}")
            print(f"  Avg Length Ratio: {sum(length_ratios) / len(length_ratios):.2f}")
        
        if failed:
            print(f"\nError Summary:")
            error_types = {}
            for r in failed:
                error = r.get('error', 'Unknown')
                error_types[error] = error_types.get(error, 0) + 1
            for error, count in error_types.items():
                print(f"  {error}: {count}")

def main():
    """Main function."""
    tester = WorkingVoiceBenchTester()
    results = tester.run_all_tests()
    
    if results:
        successful = len([r for r in results if r.get('success', False)])
        total = len(results)
        print(f"\n✅ VoiceBench test completed! {successful}/{total} tests successful")
    else:
        print("\n❌ VoiceBench test failed!")

if __name__ == "__main__":
    main()
