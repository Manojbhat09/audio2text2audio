#!/usr/bin/env python3
"""
Final Real VoiceBench Experiment - The Critical Truth
This script uses actual real VoiceBench samples and tests single-turn vs multi-turn.
"""

import requests
import json
import time
from typing import Dict, Any, List
from datetime import datetime

def load_real_voicebench_samples() -> List[Dict[str, Any]]:
    """Load the actual real VoiceBench samples we have."""
    print("🔍 Loading Real VoiceBench Samples...")

    try:
        with open('real_voicebench_samples.json', 'r') as f:
            samples = json.load(f)

        print(f"✅ Loaded {len(samples)} actual VoiceBench samples")
        print("\n📋 Real VoiceBench Sample Examples:")
        for i, sample in enumerate(samples[:3]):
            print(f"   {i+1}. \"{sample['prompt']}\"")

        return samples

    except FileNotFoundError:
        print("❌ real_voicebench_samples.json not found")
        return []

class FinalRealVoiceBenchExperiment:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.results = []

    def test_server(self) -> bool:
        """Test if server is running."""
        try:
            response = requests.post(
                f"{self.api_url}/api/v1/t2t",
                json={"text_data": "Hello"},
                timeout=30
            )
            if response.status_code == 200:
                print("✅ Server is responding")
                return True
            else:
                print(f"❌ Server returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return False

    def run_single_turn_test(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Test single-turn response."""
        try:
            response = requests.post(
                f"{self.api_url}/api/v1/t2t",
                json={"text_data": sample['prompt']},
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get('text', '')
                return {
                    'success': True,
                    'response': response_text,
                    'length': len(response_text),
                    'word_count': len(response_text.split())
                }
            else:
                return {
                    'success': False,
                    'response': '',
                    'length': 0,
                    'word_count': 0,
                    'error': f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                'success': False,
                'response': '',
                'length': 0,
                'word_count': 0,
                'error': str(e)
            }

    def run_multi_turn_test(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Test multi-turn response with context."""
        try:
            # Reset conversation
            requests.post(f"{self.api_url}/api/v1/reset_conversation", timeout=10)
            requests.post(f"{self.api_url}/api/v1/set_max_turns", json={"max_turns": 3}, timeout=10)

            # Send context message
            context_response = requests.post(
                f"{self.api_url}/api/v1/multiturn",
                json={"user_message": "I have some general knowledge questions for you."},
                timeout=60
            )

            # Send actual question
            question_response = requests.post(
                f"{self.api_url}/api/v1/multiturn",
                json={"user_message": sample['prompt']},
                timeout=60
            )

            if question_response.status_code == 200:
                result = question_response.json()
                response_text = result.get('text', '')
                return {
                    'success': True,
                    'response': response_text,
                    'length': len(response_text),
                    'word_count': len(response_text.split())
                }
            else:
                return {
                    'success': False,
                    'response': '',
                    'length': 0,
                    'word_count': 0,
                    'error': f"HTTP {question_response.status_code}"
                }
        except Exception as e:
            return {
                'success': False,
                'response': '',
                'length': 0,
                'word_count': 0,
                'error': str(e)
            }

    def run_experiment(self):
        """Run the final real VoiceBench experiment."""
        print("🚀 FINAL REAL VOICEBENCH EXPERIMENT")
        print("=" * 60)
        print("Testing single-turn vs multi-turn on ACTUAL VoiceBench samples")
        print("=" * 60)

        # Test server
        if not self.test_server():
            print("❌ Server not available. Please start the server first.")
            return

        # Load real samples
        samples = load_real_voicebench_samples()
        if not samples:
            print("❌ No real VoiceBench samples found.")
            return

        print(f"\n🔬 Running experiment on {len(samples)} real VoiceBench samples...")

        # Run tests on all samples
        for i, sample in enumerate(samples):
            print(f"\n--- Sample {i+1}/{len(samples)} ---")
            print(f"Prompt: {sample['prompt']}")

            # Single-turn test
            print("Testing single-turn...")
            single_result = self.run_single_turn_test(sample)
            if single_result['success']:
                print(f"✅ Single-turn: {single_result['response'][:100]}...")
            else:
                print(f"❌ Single-turn failed: {single_result.get('error', 'Unknown error')}")

            # Multi-turn test
            print("Testing multi-turn...")
            multi_result = self.run_multi_turn_test(sample)
            if multi_result['success']:
                print(f"✅ Multi-turn: {multi_result['response'][:100]}...")
            else:
                print(f"❌ Multi-turn failed: {multi_result.get('error', 'Unknown error')}")

            # Calculate comparison
            length_diff = multi_result['length'] - single_result['length']
            word_diff = multi_result['word_count'] - single_result['word_count']

            result = {
                'sample_index': i,
                'prompt': sample['prompt'],
                'audio_duration': sample.get('audio_info', {}).get('duration', 0),
                'single_turn': single_result,
                'multi_turn': multi_result,
                'comparison': {
                    'length_difference': length_diff,
                    'word_difference': word_diff,
                    'multi_turn_longer': multi_result['length'] > single_result['length'],
                    'multi_turn_more_words': multi_result['word_count'] > single_result['word_count']
                }
            }

            self.results.append(result)
            time.sleep(1)

        # Analyze results
        self.analyze_results()

    def analyze_results(self):
        """Analyze the experiment results."""
        if not self.results:
            print("❌ No results to analyze")
            return

        print("\n" + "=" * 80)
        print("📊 FINAL REAL VOICEBENCH EXPERIMENT RESULTS")
        print("=" * 80)

        total_samples = len(self.results)
        single_successful = len([r for r in self.results if r['single_turn']['success']])
        multi_successful = len([r for r in self.results if r['multi_turn']['success']])

        print(f"Total Real Samples: {total_samples}")
        print(f"Single-Turn Successful: {single_successful}/{total_samples} ({single_successful/total_samples*100:.1f}%)")
        print(f"Multi-Turn Successful: {multi_successful}/{total_samples} ({multi_successful/total_samples*100:.1f}%)")

        # Calculate averages
        single_lengths = [r['single_turn']['length'] for r in self.results if r['single_turn']['success']]
        multi_lengths = [r['multi_turn']['length'] for r in self.results if r['multi_turn']['success']]
        single_words = [r['single_turn']['word_count'] for r in self.results if r['single_turn']['success']]
        multi_words = [r['multi_turn']['word_count'] for r in self.results if r['multi_turn']['success']]

        if single_lengths and multi_lengths:
            avg_single_length = sum(single_lengths) / len(single_lengths)
            avg_multi_length = sum(multi_lengths) / len(multi_lengths)
            avg_single_words = sum(single_words) / len(single_words)
            avg_multi_words = sum(multi_words) / len(multi_words)

            print(f"\n📈 Quality Comparison:")
            print(f"  Average Single-Turn Length: {avg_single_length:.1f} chars")
            print(f"  Average Multi-Turn Length: {avg_multi_length:.1f} chars")
            print(f"  Length Difference: {avg_multi_length - avg_single_length:+.1f} chars")
            print(f"  Average Single-Turn Words: {avg_single_words:.1f}")
            print(f"  Average Multi-Turn Words: {avg_multi_words:.1f}")
            print(f"  Word Difference: {avg_multi_words - avg_single_words:+.1f}")

            # Count improvements
            multi_longer = len([r for r in self.results if r['comparison']['multi_turn_longer']])
            multi_more_words = len([r for r in self.results if r['comparison']['multi_turn_more_words']])

            print(f"\n🎯 Chain of Thought Effectiveness:")
            print(f"  Samples where Multi-Turn was longer: {multi_longer}/{total_samples}")
            print(f"  Samples where Multi-Turn had more words: {multi_more_words}/{total_samples}")
            print(f"  Multi-Turn advantage: {multi_longer/total_samples*100:.1f}%")

            print(f"\n🔬 CRITICAL CONCLUSION:")
            if avg_multi_length > avg_single_length:
                print(f"  ✅ Multi-turn conversation context DOES improve response quality!")
                print(f"  📈 Average improvement: {avg_multi_length - avg_single_length:.1f} characters")
                print(f"  📈 Word improvement: {avg_multi_words - avg_single_words:.1f} words")
            else:
                print(f"  ❌ Multi-turn conversation context does NOT improve response quality")
                print(f"  📉 Average difference: {avg_multi_length - avg_single_length:.1f} characters")

        # Save results
        self.save_results()

    def save_results(self):
        """Save experiment results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"final_real_voicebench_experiment_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n💾 Results saved to: {filename}")

def main():
    """Main function."""
    print("🔥 CRITICAL MISSION: Real VoiceBench Experiment")
    print("=" * 60)

    experiment = FinalRealVoiceBenchExperiment()
    experiment.run_experiment()

if __name__ == "__main__":
    main()
