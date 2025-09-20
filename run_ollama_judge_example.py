#!/usr/bin/env python3
"""
Example script showing how to use the comprehensive VoiceBench scoring with Ollama judge.
"""

import subprocess
import sys
import os

def check_ollama_running():
    """Check if Ollama is running."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_qwen3_model():
    """Check if qwen3 model is available."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            return any('qwen3' in name for name in model_names)
        return False
    except:
        return False

def main():
    """Main function to run the example."""
    print("🚀 VoiceBench Scoring with Ollama Judge Example")
    print("=" * 60)
    
    # Check if Ollama is running
    if not check_ollama_running():
        print("❌ Ollama is not running!")
        print("Please start Ollama first:")
        print("  1. Install Ollama: https://ollama.ai/")
        print("  2. Start Ollama: ollama serve")
        print("  3. Pull qwen3 model: ollama pull qwen3:14b")
        return
    
    print("✅ Ollama is running")
    
    # Check if qwen3 model is available
    if not check_qwen3_model():
        print("⚠️  qwen3 model not found. Attempting to pull...")
        try:
            subprocess.run(["ollama", "pull", "qwen3:14b"], check=True)
            print("✅ qwen3:14b model pulled successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to pull qwen3:14b model")
            return
    else:
        print("✅ qwen3 model is available")
    
    # Check if we have experiment results
    result_files = [f for f in os.listdir('.') if f.startswith('critical_real_voicebench_experiment_results_') and f.endswith('.json')]
    
    if not result_files:
        print("❌ No experiment results found!")
        print("Please run the experiment first:")
        print("  python critical_real_voicebench_experiment.py")
        return
    
    latest_file = sorted(result_files)[-1]
    print(f"✅ Found experiment results: {latest_file}")
    
    # Run comprehensive scoring with Ollama judge
    print("\n🤖 Running comprehensive scoring with Ollama judge...")
    print("=" * 60)
    
    cmd = [
        "python", "comprehensive_voicebench_scoring.py",
        "--ollama-judge",
        "--ollama-model", "qwen3:14b",
        "--ollama-url", "http://localhost:11434",
        "--results-file", latest_file,
        "--samples", "0-4"  # Test with first 5 samples
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Comprehensive scoring completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Comprehensive scoring failed: {e}")
    except KeyboardInterrupt:
        print("\n⏹️  Scoring interrupted by user")

if __name__ == "__main__":
    main()


