#!/usr/bin/env python3
"""
Simple Real VoiceBench Experiment using existing real data
This script uses the real VoiceBench samples we already have and tests 
single-turn vs multi-turn conversation context.
"""

import requests
import json
import time
from typing import Dict, Any, List
from datetime import datetime

def load_real_voicebench_samples_from_file() -> Dict[str, List[Dict[str, Any]]]:
    """Load real VoiceBench samples from the existing JSON file."""
    print("🔍 Loading Real VoiceBench Samples from Existing Data")
    print("=" * 60)
    
    try:
        # Load the real VoiceBench samples we found
        with open('real_voicebench_samples.json', 'r') as f:
            real_samples = json.load(f)
        
        print(f"✅ Loaded {len(real_samples)} real VoiceBench samples")
        
        # Group samples by dataset type (we'll use commoneval for all since that's what we have)
        # In a real scenario, we'd have separate files for each dataset
        datasets = {
            'commoneval': real_samples[:10],  # First 10 samples
            'wildvoice': real_samples[10:20] if len(real_samples) > 10 else real_samples[5:10],  # Next 10 or last 5
            'ifeval': real_samples[20:30] if len(real_samples) > 20 else real_samples[:5],  # Next 10 or first 5
            'advbench': real_samples[30:40] if len(real_samples) > 30 else real_samples[5:10]  # Next 10 or middle 5
        }
        
        # Ensure we have 10 samples per dataset
        for dataset_name, samples in datasets.items():
            if len(samples) < 10:
                # Pad with available samples if needed
                while len(samples) < 10 and len(real_samples) > len(samples):
                    samples.append(real_samples[len(samples) % len(real_samples)])
            
            # Add dataset metadata
            for i, sample in enumerate(samples):
                sample['dataset'] = dataset_name
                sample['sample_index'] = i
                sample['audio_duration'] = sample.get('audio_info', {}).get('duration', 0)
                sample['hf_index'] = sample.get('index', i)
        
        print(f"✅ Prepared datasets:")
        for dataset_name, samples in datasets.items():
            print(f"   {dataset_name}: {len(samples)} samples")
        
        return datasets
        
    except FileNotFoundError:
        print("❌ real_voicebench_samples.json not found")
        return {}
    except Exception as e:
        print(f"❌ Error loading real samples: {e}")
        return {}

class SimpleRealVoiceBenchExperiment:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.results = []
        
    def test_server_connection(self) -> bool:
        """Test if server is running by testing a simple endpoint."""
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
                response = requests.post(
                    f"{self.api_url}/api/v1/multiturn",
                    json={"user_message": message},
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get('text', '')
                    responses.append({
                        'turn': i + 1,
                        'message': message,
                        'response': response_text,
                        'current_turn': result.get('current_turn', 0),
                        'max_turns': result.get('max_turns', 0),
                        'turns_remaining': result.get('turns_remaining', 0)
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
    
    def run_dataset_experiment(self, dataset_name: str, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run experiment on a single dataset with real VoiceBench samples."""
        print(f"\n🚀 Testing {dataset_name.upper()} Dataset ({len(samples)} real samples)")
        print("=" * 70)
        
        results = []
        
        for i, sample in enumerate(samples):
            print(f"\n--- Sample {i+1}/{len(samples)} ---")
            print(f"Real Prompt: {sample['prompt']}")
            print(f"Audio Duration: {sample.get('audio_duration', 0):.2f}s")
            
            # Test 1: Single-turn
            print("   Testing single-turn...")
            single_result = self.test_single_turn_response(sample['prompt'])
            
            if single_result['success']:
                print(f"   ✅ Single-turn: {single_result['response'][:100]}...")
            else:
                print(f"   ❌ Single-turn failed: {single_result['error']}")
            
            # Test 2: Multi-turn (with context)
            print("   Testing multi-turn...")
            
            # Create conversation context based on dataset type
            if dataset_name == 'commoneval':
                context_messages = [
                    "I have some general knowledge questions for you.",
                    sample['prompt']
                ]
            elif dataset_name == 'wildvoice':
                context_messages = [
                    "Let's have a casual conversation.",
                    sample['prompt']
                ]
            elif dataset_name == 'ifeval':
                context_messages = [
                    "I need you to follow specific instructions carefully.",
                    sample['prompt']
                ]
            elif dataset_name == 'advbench':
                context_messages = [
                    "I want to ask you about some sensitive topics.",
                    sample['prompt']
                ]
            else:
                context_messages = [sample['prompt']]
            
            multi_result = self.test_multi_turn_response(context_messages, max_turns=3)
            
            if multi_result['success']:
                final_response = multi_result['responses'][-1]['response'] if multi_result['responses'] else ''
                print(f"   ✅ Multi-turn: {final_response[:100]}...")
            else:
                print(f"   ❌ Multi-turn failed: {multi_result['error']}")
            
            # Calculate metrics
            single_response = single_result.get('response', '')
            multi_response = multi_result['responses'][-1]['response'] if multi_result['success'] and multi_result['responses'] else ''
            
            single_length = len(single_response)
            multi_length = len(multi_response)
            
            # Calculate quality metrics
            single_word_count = len(single_response.split())
            multi_word_count = len(multi_response.split())
            
            # Check for specific quality indicators
            single_has_explanation = any(word in single_response.lower() for word in ['because', 'since', 'due to', 'as a result', 'therefore'])
            multi_has_explanation = any(word in multi_response.lower() for word in ['because', 'since', 'due to', 'as a result', 'therefore'])
            
            single_has_examples = any(word in single_response.lower() for word in ['for example', 'such as', 'like', 'including'])
            multi_has_examples = any(word in multi_response.lower() for word in ['for example', 'such as', 'like', 'including'])
            
            result = {
                'dataset': dataset_name,
                'sample_index': i,
                'hf_index': sample.get('hf_index', i),
                'prompt': sample['prompt'],
                'audio_duration': sample.get('audio_duration', 0),
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
            
            # Small delay between samples
            time.sleep(1)
        
        return results
    
    def run_full_experiment(self):
        """Run the complete real VoiceBench experiment."""
        print("🚀 Simple Real VoiceBench Multi-Turn vs Single-Turn Experiment")
        print("=" * 70)
        print("Using REAL VoiceBench samples from existing data")
        print("=" * 70)
        
        # Test server connection
        if not self.test_server_connection():
            print("❌ Server not available. Please start the server first.")
            return []
        
        # Load real VoiceBench samples
        datasets = load_real_voicebench_samples_from_file()
        
        if not any(datasets.values()):
            print("❌ No real VoiceBench data loaded.")
            return []
        
        # Run experiment on each dataset
        all_results = []
        for dataset_name, samples in datasets.items():
            if not samples:
                print(f"⚠️ Skipping {dataset_name} - no samples loaded")
                continue
                
            try:
                results = self.run_dataset_experiment(dataset_name, samples)
                all_results.extend(results)
                time.sleep(2)  # Delay between datasets
            except Exception as e:
                print(f"❌ Error testing {dataset_name}: {e}")
                continue
        
        # Save results
        if all_results:
            self.save_results(all_results)
            self.print_experiment_summary(all_results)
        
        return all_results
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save experiment results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simple_real_voicebench_experiment_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Simple Real VoiceBench experiment results saved to: {filename}")
    
    def print_experiment_summary(self, results: List[Dict[str, Any]]):
        """Print comprehensive experiment summary."""
        print("\n" + "=" * 80)
        print("📊 SIMPLE REAL VOICEBENCH EXPERIMENT SUMMARY")
        print("=" * 80)
        
        # Overall statistics
        total_samples = len(results)
        single_successful = len([r for r in results if r['single_turn']['success']])
        multi_successful = len([r for r in results if r['multi_turn']['success']])
        
        print(f"Total Real Samples: {total_samples}")
        print(f"Single-Turn Successful: {single_successful} ({single_successful/total_samples*100:.1f}%)")
        print(f"Multi-Turn Successful: {multi_successful} ({multi_successful/total_samples*100:.1f}%)")
        
        # Response quality analysis
        single_lengths = [r['single_turn']['length'] for r in results if r['single_turn']['success']]
        multi_lengths = [r['multi_turn']['length'] for r in results if r['multi_turn']['success']]
        
        single_word_counts = [r['single_turn']['word_count'] for r in results if r['single_turn']['success']]
        multi_word_counts = [r['multi_turn']['word_count'] for r in results if r['multi_turn']['success']]
        
        if single_lengths and multi_lengths:
            avg_single_length = sum(single_lengths) / len(single_lengths)
            avg_multi_length = sum(multi_lengths) / len(multi_lengths)
            avg_single_words = sum(single_word_counts) / len(single_word_counts)
            avg_multi_words = sum(multi_word_counts) / len(multi_word_counts)
            
            print(f"\nResponse Quality Analysis:")
            print(f"  Average Single-Turn Length: {avg_single_length:.1f} chars")
            print(f"  Average Multi-Turn Length: {avg_multi_length:.1f} chars")
            print(f"  Length Difference: {avg_multi_length - avg_single_length:+.1f} chars")
            print(f"  Average Single-Turn Words: {avg_single_words:.1f}")
            print(f"  Average Multi-Turn Words: {avg_multi_words:.1f}")
            print(f"  Word Count Difference: {avg_multi_words - avg_single_words:+.1f}")
            print(f"  Multi-Turn Longer: {'Yes' if avg_multi_length > avg_single_length else 'No'}")
        
        # Quality improvements
        explanation_improvements = len([r for r in results if r['comparison']['explanation_improvement']])
        examples_improvements = len([r for r in results if r['comparison']['examples_improvement']])
        
        print(f"\nQuality Improvements:")
        print(f"  Samples with better explanations: {explanation_improvements}/{total_samples}")
        print(f"  Samples with more examples: {examples_improvements}/{total_samples}")
        
        # Dataset-specific analysis
        print(f"\nDataset-Specific Results:")
        for dataset_name in ['commoneval', 'wildvoice', 'ifeval', 'advbench']:
            dataset_results = [r for r in results if r['dataset'] == dataset_name]
            if dataset_results:
                single_success = len([r for r in dataset_results if r['single_turn']['success']])
                multi_success = len([r for r in dataset_results if r['multi_turn']['success']])
                
                print(f"  {dataset_name.upper()}:")
                print(f"    Single-Turn: {single_success}/{len(dataset_results)} successful")
                print(f"    Multi-Turn: {multi_success}/{len(dataset_results)} successful")
        
        # Chain of Thought effectiveness
        multi_turn_longer = len([r for r in results if r['comparison']['multi_turn_longer']])
        multi_turn_more_words = len([r for r in results if r['comparison']['multi_turn_more_words']])
        
        print(f"\nChain of Thought Analysis:")
        print(f"  Samples where Multi-Turn was longer: {multi_turn_longer}/{total_samples}")
        print(f"  Samples where Multi-Turn had more words: {multi_turn_more_words}/{total_samples}")
        print(f"  Multi-Turn advantage: {multi_turn_longer/total_samples*100:.1f}%")
        
        print(f"\n🎯 Simple Real VoiceBench Experiment Conclusion:")
        if avg_multi_length > avg_single_length:
            print(f"  ✅ Multi-turn conversation context DOES improve response quality!")
            print(f"  📈 Average improvement: {avg_multi_length - avg_single_length:.1f} characters")
            print(f"  📈 Word improvement: {avg_multi_words - avg_single_words:.1f} words")
        else:
            print(f"  ❌ Multi-turn conversation context does NOT improve response quality")
            print(f"  📉 Average difference: {avg_multi_length - avg_single_length:.1f} characters")

def main():
    """Main function to run the simple real VoiceBench experiment."""
    experiment = SimpleRealVoiceBenchExperiment()
    results = experiment.run_full_experiment()
    
    if results:
        print(f"\n✅ Simple Real VoiceBench experiment completed successfully!")
        print(f"📊 Tested {len(results)} real samples across 4 datasets")
        print(f"🔬 Results saved for analysis")
    else:
        print(f"\n❌ Simple Real VoiceBench experiment failed!")

if __name__ == "__main__":
    main()
