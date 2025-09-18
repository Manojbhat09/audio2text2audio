#!/usr/bin/env python3
"""
Analyze VoiceBench Experiment Results
"""

import json
import os
from typing import Dict, Any, List

def analyze_results():
    """Analyze the experiment results."""
    print("📊 Analyzing VoiceBench Experiment Results")
    print("=" * 60)
    
    # Find the most recent results file
    result_files = [f for f in os.listdir('.') if f.startswith('critical_real_voicebench_experiment_results_') and f.endswith('.json')]
    
    if not result_files:
        print("❌ No results files found")
        return
    
    # Use the most recent file
    latest_file = sorted(result_files)[-1]
    print(f"📁 Analyzing {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    print(f"📊 Total samples: {len(data)}")
    
    # Calculate statistics
    single_turn_lengths = []
    multi_turn_lengths = []
    
    for result in data:
        single_turn = result.get('single_turn', {}).get('response', '')
        multi_turn = result.get('multi_turn', {}).get('response', '')
        
        single_turn_lengths.append(len(single_turn))
        multi_turn_lengths.append(len(multi_turn))
    
    # Calculate averages
    avg_single_turn = sum(single_turn_lengths) / len(single_turn_lengths)
    avg_multi_turn = sum(multi_turn_lengths) / len(multi_turn_lengths)
    
    print(f"\n📈 Response Length Analysis:")
    print(f"   Single-turn average length: {avg_single_turn:.1f} characters")
    print(f"   Multi-turn average length: {avg_multi_turn:.1f} characters")
    print(f"   Length difference: {avg_multi_turn - avg_single_turn:.1f} characters")
    
    # Show some examples
    print(f"\n📋 Sample Results:")
    for i, result in enumerate(data[:3]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Prompt: {result.get('prompt', '')[:80]}...")
        print(f"Single-turn: {result.get('single_turn', {}).get('response', '')[:100]}...")
        print(f"Multi-turn: {result.get('multi_turn', {}).get('response', '')[:100]}...")
    
    # Dataset breakdown
    datasets = {}
    for result in data:
        dataset = result.get('dataset', 'unknown')
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(result)
    
    print(f"\n📊 Dataset Breakdown:")
    for dataset_name, samples in datasets.items():
        print(f"   {dataset_name}: {len(samples)} samples")

def main():
    """Main function."""
    analyze_results()

if __name__ == "__main__":
    main()
