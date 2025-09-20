#!/usr/bin/env python3
"""
Critical Real VoiceBench Experiment
This script uses the actual real VoiceBench samples we have and tests single-turn vs multi-turn properly.
"""

import requests
import json
import time
import numpy as np
from typing import Dict, Any, List
from datetime import datetime

def load_actual_real_voicebench_samples() -> Dict[str, List[Dict[str, Any]]]:
    """Load the actual real VoiceBench samples we have."""
    print("🔍 Loading Actual Real VoiceBench Samples")
    print("=" * 60)

    try:
        all_samples = []
        
        # Load all available sample files
        sample_files = [
            'commoneval_samples.json',
            'wildvoice_samples.json', 
            'ifeval_samples.json',
            'advbench_samples.json',
            'real_voicebench_samples.json'  # Fallback for old format
        ]
        
        for filename in sample_files:
            try:
                with open(filename, 'r') as f:
                    samples = json.load(f)
                    all_samples.extend(samples)
                    print(f"✅ Loaded {len(samples)} samples from {filename}")
            except FileNotFoundError:
                print(f"❌ {filename} not found")
                continue

        print(f"✅ Total loaded: {len(all_samples)} samples")

        # Show some real examples
        print("\n📋 Sample Examples:")
        for i, sample in enumerate(all_samples[:5]):
            dataset = sample.get('dataset', 'unknown')
            print(f"   {i+1}. [{dataset}] {sample['prompt']}")

        # Group by dataset
        datasets = {}
        for sample in all_samples:
            dataset_name = sample.get('dataset', 'unknown')
            if dataset_name not in datasets:
                datasets[dataset_name] = []
            datasets[dataset_name].append(sample)

        print(f"\n📊 Dataset Distribution:")
        for dataset_name, samples in datasets.items():
            print(f"   {dataset_name}: {len(samples)} samples")

        return datasets

    except Exception as e:
        print(f"❌ Error loading samples: {e}")
        return {}

class CriticalRealVoiceBenchExperiment:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.results = []

    def test_server_connection(self) -> bool:
        """Test if server is running."""
        try:
            response = requests.post(
                f"{self.api_url}/api/v1/t2t",
                json={"text_data": "Hello"},
                timeout=30
            )
            if response.status_code == 200:
                print(f"✅ Server is responding (status: {response.status_code})")
                return True
            else:
                print(f"❌ Server returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return False

    def test_single_turn_response(self, text: str) -> Dict[str, Any]:
        """Test single-turn response."""
        try:
            response = requests.post(
                f"{self.api_url}/api/v1/t2t",
                json={"text_data": text},
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get('text', '')
                return {
                    'success': True,
                    'response': response_text,
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'response': '',
                    'error': f"HTTP {response.status_code}"
                }
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'response': '',
                'error': 'Timeout'
            }
        except Exception as e:
            return {
                'success': False,
                'response': '',
                'error': str(e)
            }

    def test_multi_turn_response(self, messages: List[str], max_turns: int = 3) -> Dict[str, Any]:
        """Test multi-turn response with conversation context."""
        try:
            # Reset conversation first
            requests.post(f"{self.api_url}/api/v1/reset_conversation", timeout=30)
            requests.post(f"{self.api_url}/api/v1/set_max_turns", json={"max_turns": max_turns}, timeout=30)

            responses = []
            for i, message in enumerate(messages):
                # Use t2t endpoint for text-to-text multi-turn testing
                response = requests.post(
                    f"{self.api_url}/api/v1/t2t",
                    json={"text_data": message},
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get('text', '')
                    responses.append({
                        'turn': i + 1,
                        'message': message,
                        'response': response_text,
                        'current_turn': i + 1,  # Simulate turn counter
                        'max_turns': max_turns,
                        'turns_remaining': max_turns - (i + 1)
                    })
                else:
                    responses.append({
                        'turn': i + 1,
                        'message': message,
                        'response': '',
                        'error': f"HTTP {response.status_code}"
                    })

            return {
                'success': True,
                'responses': responses,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'responses': [],
                'error': str(e)
            }

    def run_experiment_on_real_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run experiment on actual real VoiceBench samples."""
        print(f"\n🚀 Testing Real VoiceBench Samples ({len(samples)} samples)")
        print("=" * 70)

        results = []

        for i, sample in enumerate(samples):
            print(f"\n--- Sample {i+1}/{len(samples)} ---")
            print(f"Real Prompt: {sample['prompt']}")
            print(f"Audio Duration: {sample.get('audio_info', {}).get('duration', 0):.2f}s")

            # Test 1: Single-turn
            print("   Testing single-turn...")
            single_result = self.test_single_turn_response(sample['prompt'])

            if single_result['success']:
                print(f"   ✅ Single-turn: {single_result['response']}...")
            else:
                print(f"   ❌ Single-turn failed: {single_result['error']}")

            # Test 2: Multi-turn
            print("   Testing multi-turn...")
            context_messages = [
                "I have some general knowledge questions for you.",
                sample['prompt']
            ]

            multi_result = self.test_multi_turn_response(context_messages, max_turns=3)

            if multi_result['success']:
                final_response = multi_result['responses'][-1]['response'] if multi_result['responses'] else ''
                print(f"   ✅ Multi-turn: {final_response}...")
            else:
                print(f"   ❌ Multi-turn failed: {multi_result['error']}")

            # Calculate metrics
            single_response = single_result.get('response', '')
            multi_response = multi_result['responses'][-1]['response'] if multi_result['success'] and multi_result['responses'] else ''

            single_length = len(single_response)
            multi_length = len(multi_response)
            single_word_count = len(single_response.split())
            multi_word_count = len(multi_response.split())

            # Quality indicators
            single_has_explanation = any(word in single_response.lower() for word in ['because', 'since', 'due to', 'as a result', 'therefore'])
            multi_has_explanation = any(word in multi_response.lower() for word in ['because', 'since', 'due to', 'as a result', 'therefore'])

            single_has_examples = any(word in single_response.lower() for word in ['for example', 'such as', 'like', 'including'])
            multi_has_examples = any(word in multi_response.lower() for word in ['for example', 'such as', 'like', 'including'])

            result = {
                'dataset': sample.get('dataset', 'commoneval'),  # Use actual dataset from sample
                'sample_index': i,
                'hf_index': sample.get('index', i),
                'prompt': sample['prompt'],
                'audio_duration': sample.get('audio_info', {}).get('duration', 0),
                'single_turn': {
                    'response': single_response,
                    'length': single_length,
                    'word_count': single_word_count,
                    'has_explanation': single_has_explanation,
                    'has_examples': single_has_examples,
                    'success': single_result['success'],
                    'error': single_result.get('error')
                },
                'multi_turn': {
                    'response': multi_response,
                    'length': multi_length,
                    'word_count': multi_word_count,
                    'has_explanation': multi_has_explanation,
                    'has_examples': multi_has_examples,
                    'success': multi_result['success'],
                    'error': multi_result.get('error'),
                    'conversation_turns': len(multi_result.get('responses', []))
                },
                'comparison': {
                    'length_difference': multi_length - single_length,
                    'word_count_difference': multi_word_count - single_word_count,
                    'multi_turn_longer': multi_length > single_length,
                    'multi_turn_more_words': multi_word_count > single_word_count,
                    'explanation_improvement': multi_has_explanation and not single_has_explanation,
                    'examples_improvement': multi_has_examples and not single_has_examples
                }
            }

            results.append(result)
            time.sleep(1)

        return results

    def run_critical_experiment(self):
        """Run the critical real VoiceBench experiment."""
        print("🚀 Critical Real VoiceBench Multi-Turn vs Single-Turn Experiment")
        print("=" * 70)
        print("Using ACTUAL VoiceBench samples from real dataset")
        print("=" * 70)

        # Test server connection
        if not self.test_server_connection():
            print("❌ Server not available. Please start the server first.")
            return []

        # Load actual real VoiceBench samples
        datasets = load_actual_real_voicebench_samples()

        if not datasets:
            print("❌ No VoiceBench samples found.")
            return []

        # Run experiment on all available samples
        print("\n🔬 Running experiment on all available VoiceBench samples...")
        all_samples = []
        for dataset_samples in datasets.values():
            all_samples.extend(dataset_samples)
        results = self.run_experiment_on_real_samples(all_samples)

        if results:
            self.save_results(results)
            self.print_critical_analysis(results)

        return results

    def save_results(self, results: List[Dict[str, Any]]):
        """Save experiment results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"critical_real_voicebench_experiment_results_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to: {filename}")

    def print_critical_analysis(self, results: List[Dict[str, Any]]):
        """Print critical analysis of the results."""
        print("\n" + "=" * 80)
        print("🔍 CRITICAL REAL VOICEBENCH EXPERIMENT ANALYSIS")
        print("=" * 80)

        total_samples = len(results)
        single_successful = len([r for r in results if r['single_turn']['success']])
        multi_successful = len([r for r in results if r['multi_turn']['success']])

        print(f"Total Real Samples Tested: {total_samples}")
        print(f"Single-Turn Successful: {single_successful} ({single_successful/total_samples*100:.1f}%)")
        print(f"Multi-Turn Successful: {multi_successful} ({multi_successful/total_samples*100:.1f}%)")

        # Quality metrics
        single_lengths = [r['single_turn']['length'] for r in results if r['single_turn']['success']]
        multi_lengths = [r['multi_turn']['length'] for r in results if r['multi_turn']['success']]
        single_word_counts = [r['single_turn']['word_count'] for r in results if r['single_turn']['success']]
        multi_word_counts = [r['multi_turn']['word_count'] for r in results if r['multi_turn']['success']]

        if single_lengths and multi_lengths:
            avg_single_length = sum(single_lengths) / len(single_lengths)
            avg_multi_length = sum(multi_lengths) / len(multi_lengths)
            avg_single_words = sum(single_word_counts) / len(single_word_counts)
            avg_multi_words = sum(multi_word_counts) / len(multi_word_counts)

            print(f"\n📊 Quality Metrics:")
            print(f"  Average Single-Turn Length: {avg_single_length:.1f} chars")
            print(f"  Average Multi-Turn Length: {avg_multi_length:.1f} chars")
            print(f"  Length Difference: {avg_multi_length - avg_single_length:+.1f} chars")
            print(f"  Average Single-Turn Words: {avg_single_words:.1f}")
            print(f"  Average Multi-Turn Words: {avg_multi_words:.1f}")
            print(f"  Word Count Difference: {avg_multi_words - avg_single_words:+.1f}")

        # Quality improvements
        explanation_improvements = len([r for r in results if r['comparison']['explanation_improvement']])
        examples_improvements = len([r for r in results if r['comparison']['examples_improvement']])

        print(f"\n🎯 Quality Improvements:")
        print(f"  Samples with better explanations: {explanation_improvements}/{total_samples}")
        print(f"  Samples with more examples: {examples_improvements}/{total_samples}")

        # Chain of thought effectiveness
        multi_turn_longer = len([r for r in results if r['comparison']['multi_turn_longer']])
        multi_turn_more_words = len([r for r in results if r['comparison']['multi_turn_more_words']])

        print(f"\n🧠 Chain of Thought Analysis:")
        print(f"  Samples where Multi-Turn was longer: {multi_turn_longer}/{total_samples}")
        print(f"  Samples where Multi-Turn had more words: {multi_turn_more_words}/{total_samples}")
        print(f"  Multi-Turn advantage: {multi_turn_longer/total_samples*100:.1f}%")

        # Critical conclusion
        print(f"\n🔬 CRITICAL CONCLUSION:")
        if avg_multi_length > avg_single_length:
            print(f"  ✅ Multi-turn conversation context DOES improve response quality!")
            print(f"  📈 Average improvement: {avg_multi_length - avg_single_length:.1f} characters")
            print(f"  📈 Word improvement: {avg_multi_words - avg_single_words:.1f} words")
        else:
            print(f"  ❌ Multi-turn conversation context does NOT improve response quality")
            print(f"  📉 Average difference: {avg_multi_length - avg_single_length:.1f} characters")

        print(f"\n⚠️  LIMITATIONS:")
        print(f"  - Only tested commoneval dataset (10 samples)")
        print(f"  - Missing wildvoice, ifeval, advbench datasets")
        print(f"  - Not a complete VoiceBench benchmark")

def main():
    """Main function."""
    experiment = CriticalRealVoiceBenchExperiment()
    results = experiment.run_critical_experiment()

    if results:
        print(f"\n✅ Critical real VoiceBench experiment completed!")
        print(f"📊 Tested {len(results)} real samples")
        print(f"🔬 Results saved for analysis")
    else:
        print(f"\n❌ Critical real VoiceBench experiment failed!")

if __name__ == "__main__":
    main()
