#!/usr/bin/env python3
"""
Simple script to run the multi-turn conversation experiment.

This script:
1. Starts the server (if not running)
2. Runs the multi-turn vs single-turn experiment
3. Compares results and shows improvements
"""

import subprocess
import time
import requests
import json
import sys
from pathlib import Path

def check_server_running(api_url: str) -> bool:
    """Check if the server is running."""
    try:
        response = requests.get(api_url.replace('/v2t', '/api/v1/health'), timeout=5)
        return response.status_code == 200
    except:
        return False

def start_server():
    """Start the server in the background."""
    print("Starting server...")
    server_script = Path(__file__).parent / "getsetgo-ele-01" / "server.py"
    
    if not server_script.exists():
        print(f"Server script not found at {server_script}")
        return False
    
    try:
        # Start server in background
        process = subprocess.Popen([
            sys.executable, str(server_script)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for server to start
        time.sleep(10)
        
        # Check if server is running
        if check_server_running("http://localhost:8000/api/v1/v2t"):
            print("✅ Server started successfully")
            return True
        else:
            print("❌ Server failed to start")
            return False
            
    except Exception as e:
        print(f"Error starting server: {e}")
        return False

def run_experiment():
    """Run the multi-turn experiment."""
    print("\n" + "="*60)
    print("RUNNING MULTI-TURN CONVERSATION EXPERIMENT")
    print("="*60)
    
    # Import and run the experiment
    sys.path.append(str(Path(__file__).parent / "tests"))
    
    from test_multiturn_experiment import MultiTurnExperiment
    
    # Create experiment instance
    experiment = MultiTurnExperiment("http://localhost:8000/api/v1/v2t")
    
    # Run experiment on a subset of datasets for testing
    datasets = ['commoneval', 'ifeval']  # Start with 2 datasets
    results = experiment.run_experiment(datasets)
    
    # Save results
    output_file = "multiturn_experiment_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Experiment completed! Results saved to {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    for dataset_name, dataset_results in results.items():
        single_eval = dataset_results['single_turn']['evaluation']
        multi_eval = dataset_results['multi_turn']['evaluation']
        
        print(f"\n{dataset_name.upper()}:")
        print(f"  Single-turn success rate: {single_eval['success_rate']:.3f}")
        print(f"  Multi-turn success rate:  {multi_eval['success_rate']:.3f}")
        improvement = multi_eval['success_rate'] - single_eval['success_rate']
        print(f"  Improvement: {improvement:+.3f} ({improvement*100:+.1f}%)")
        
        if improvement > 0:
            print(f"  🎉 Multi-turn is BETTER by {improvement*100:.1f}%")
        elif improvement < 0:
            print(f"  📉 Single-turn is BETTER by {abs(improvement)*100:.1f}%")
        else:
            print(f"  ➖ No difference")
    
    return results

def main():
    """Main function."""
    print("Multi-Turn Conversation Experiment for VoiceBench")
    print("=" * 60)
    
    api_url = "http://localhost:8000/api/v1/v2t"
    
    # Check if server is running
    if not check_server_running(api_url):
        print("Server not running. Starting server...")
        if not start_server():
            print("Failed to start server. Please start it manually and try again.")
            return
    else:
        print("✅ Server is already running")
    
    # Run the experiment
    try:
        results = run_experiment()
        
        # Calculate overall improvement
        total_single = 0
        total_multi = 0
        count = 0
        
        for dataset_name, dataset_results in results.items():
            single_eval = dataset_results['single_turn']['evaluation']
            multi_eval = dataset_results['multi_turn']['evaluation']
            total_single += single_eval['success_rate']
            total_multi += multi_eval['success_rate']
            count += 1
        
        if count > 0:
            avg_single = total_single / count
            avg_multi = total_multi / count
            overall_improvement = avg_multi - avg_single
            
            print(f"\n" + "="*60)
            print("OVERALL RESULTS")
            print("="*60)
            print(f"Average Single-turn success rate: {avg_single:.3f}")
            print(f"Average Multi-turn success rate:  {avg_multi:.3f}")
            print(f"Overall improvement: {overall_improvement:+.3f} ({overall_improvement*100:+.1f}%)")
            
            if overall_improvement > 0.05:  # 5% improvement threshold
                print(f"\n🎉 CONCLUSION: Multi-turn conversation shows SIGNIFICANT improvement!")
                print(f"   This suggests that conversation context helps the model perform better.")
            elif overall_improvement > 0:
                print(f"\n✅ CONCLUSION: Multi-turn conversation shows modest improvement.")
                print(f"   Conversation context may provide some benefit.")
            elif overall_improvement < -0.05:
                print(f"\n📉 CONCLUSION: Single-turn performs better.")
                print(f"   Conversation context may be adding noise or confusion.")
            else:
                print(f"\n➖ CONCLUSION: No significant difference between approaches.")
                print(f"   Both single-turn and multi-turn perform similarly.")
        
    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
