#!/usr/bin/env python3
"""
Analyze what the trained model actually learned.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import json

# Enable synchronous CUDA execution for better debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_trained_model():
    """Analyze what the trained model actually learned."""
    
    # Load model
    model_path = "/home/mbhat/omegalabs-anytoany-bittensor/finetune/models/final_model"
    logger.info(f"Loading model from {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Test various prompts to see what the model learned
    test_cases = [
        # Basic math
        ("What is 2+2?", "4"),
        ("What is 3+3?", "6"),
        ("What is 4+4?", "8"),
        
        # Dataset classification
        ("Classify this transcript: 'This is an instruction following task'", "ifeval"),
        ("Classify this transcript: 'This is a common evaluation task'", "commoneval"),
        ("Classify this transcript: 'This is a wild voice sample'", "wildvoice"),
        
        # Simple completion
        ("The answer is:", "42"),
        ("The dataset is:", "ifeval"),
        ("The category is:", "commoneval"),
        
        # Pattern completion
        ("ifeval, commoneval,", "wildvoice"),
        ("1, 2, 3,", "4"),
        ("A, B, C,", "D"),
    ]
    
    results = []
    
    for prompt, expected in test_cases:
        logger.info(f"\n=== Testing: {prompt} ===")
        logger.info(f"Expected: {expected}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate with different methods
        generation_methods = [
            {"name": "Greedy", "do_sample": False, "temperature": 1.0, "max_new_tokens": 10},
            {"name": "Sampling", "do_sample": True, "temperature": 0.7, "max_new_tokens": 10},
            {"name": "High Temp", "do_sample": True, "temperature": 1.5, "max_new_tokens": 10},
        ]
        
        for method in generation_methods:
            try:
                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=method["max_new_tokens"],
                        do_sample=method["do_sample"],
                        temperature=method["temperature"],
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                    )
                    
                    response = tokenizer.decode(generated[0], skip_special_tokens=True)
                    assistant_response = response[len(prompt):].strip()
                    
                    # Check if response contains expected answer
                    contains_expected = expected.lower() in assistant_response.lower()
                    
                    logger.info(f"{method['name']}: '{assistant_response}' (contains expected: {contains_expected})")
                    
                    results.append({
                        "prompt": prompt,
                        "expected": expected,
                        "method": method["name"],
                        "response": assistant_response,
                        "contains_expected": contains_expected
                    })
                    
            except Exception as e:
                logger.error(f"{method['name']} failed: {e}")
                results.append({
                    "prompt": prompt,
                    "expected": expected,
                    "method": method["name"],
                    "response": f"ERROR: {e}",
                    "contains_expected": False
                })
    
    # Save results
    with open("model_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    logger.info("\n=== SUMMARY ===")
    total_tests = len(test_cases) * len(generation_methods)
    correct_tests = sum(1 for r in results if r["contains_expected"])
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Correct tests: {correct_tests}")
    logger.info(f"Accuracy: {correct_tests/total_tests:.2%}")
    
    # Check if model learned dataset names
    dataset_tests = [r for r in results if "dataset" in r["prompt"].lower() or "classify" in r["prompt"].lower()]
    dataset_correct = sum(1 for r in dataset_tests if r["contains_expected"])
    logger.info(f"Dataset classification tests: {len(dataset_tests)}")
    logger.info(f"Dataset classification correct: {dataset_correct}")
    logger.info(f"Dataset accuracy: {dataset_correct/len(dataset_tests):.2%}" if dataset_tests else "No dataset tests")

if __name__ == "__main__":
    analyze_trained_model()
