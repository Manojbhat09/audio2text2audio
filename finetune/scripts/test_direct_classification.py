#!/usr/bin/env python3
"""
Test the direct classification trained model on comprehensive classification performance.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
from typing import Dict, List, Any
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DirectClassificationTester:
    """Test the direct classification trained model."""
    
    def __init__(self, base_model_path: str, lora_model_path: str, max_length: int = 128):
        self.max_length = max_length
        
        # Load the base model
        logger.info(f"Loading base model from {base_model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load LoRA weights
        logger.info(f"Loading LoRA weights from {lora_model_path}")
        self.model = PeftModel.from_pretrained(self.model, lora_model_path)
        
        logger.info("✓ Model and LoRA weights loaded successfully")
        
        # Test cases for each dataset type
        self.test_cases = {
            'ifeval': [
                "Follow these instructions exactly: Write 3 sentences about cats.",
                "Your task: Create a list with exactly 5 items about weather.",
                "Instruction: Write a paragraph that contains the word 'example' three times.",
                "Complete this: Write exactly 2 sentences, each starting with 'The'.",
                "Follow this format: Write 4 sentences about food, numbered 1-4.",
                "Your assignment: Write a response that ends with 'Task completed.'",
                "Instruction: Create a response with exactly 6 words about nature.",
                "Follow these steps: 1) Write about trees 2) Write about flowers 3) Write about animals",
                "Your task: Write a sentence that contains both 'blue' and 'sky'.",
                "Complete this instruction: Write a paragraph about cars in exactly 3 sentences."
            ],
            'commoneval': [
                "What is the capital of France?",
                "How many days are in a week?",
                "What color is the sun?",
                "What is 2 plus 2?",
                "What is the largest planet?",
                "What do plants need to grow?",
                "What is the opposite of hot?",
                "What do we use to write?",
                "What is the name of our planet?",
                "What do we call water when it's frozen?"
            ],
            'wildvoice': [
                "Hey, how are you doing today?",
                "So what do you think about this idea?",
                "You know, I was just thinking about that.",
                "Wow, that's really interesting! Tell me more.",
                "Hmm, I'm not sure about that approach.",
                "That sounds like a great plan to me.",
                "I'm really excited about this project!",
                "Oh, I see what you mean now.",
                "You're absolutely right about that point.",
                "Well, that's one way to look at it."
            ]
        }
    
    def create_test_prompt(self, text: str) -> str:
        """Create test prompt for classification."""
        return f"Text: {text}\nLabel:"
    
    def test_single_classification(self, text: str, expected_label: str) -> Dict[str, Any]:
        """Test classification of a single text sample."""
        prompt = self.create_test_prompt(text)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=True,
                temperature=0.1,  # Low temperature for more deterministic output
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        assistant_response = response[len(prompt):].strip()
        
        # Extract predicted label
        predicted_label = assistant_response.lower().strip()
        
        # Clean up the prediction - look for the exact labels
        if 'ifeval' in predicted_label:
            predicted_label = 'ifeval'
        elif 'commoneval' in predicted_label:
            predicted_label = 'commoneval'
        elif 'wildvoice' in predicted_label:
            predicted_label = 'wildvoice'
        else:
            predicted_label = 'unknown'
        
        is_correct = predicted_label == expected_label
        
        return {
            'text': text,
            'expected': expected_label,
            'predicted': predicted_label,
            'correct': is_correct,
            'raw_response': assistant_response
        }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive classification test."""
        logger.info("Starting comprehensive direct classification test...")
        
        results = {
            'overall_accuracy': 0.0,
            'per_dataset_accuracy': {},
            'detailed_results': [],
            'confusion_matrix': {},
            'total_samples': 0,
            'correct_predictions': 0
        }
        
        # Test each dataset type
        for dataset_name, test_texts in self.test_cases.items():
            logger.info(f"\nTesting {dataset_name} samples...")
            
            dataset_results = []
            correct_count = 0
            
            for i, text in enumerate(test_texts):
                result = self.test_single_classification(text, dataset_name)
                dataset_results.append(result)
                
                if result['correct']:
                    correct_count += 1
                    results['correct_predictions'] += 1
                
                results['total_samples'] += 1
                
                logger.info(f"  Sample {i+1}: {'✓' if result['correct'] else '✗'} "
                          f"Expected: {result['expected']:12} Predicted: {result['predicted']:12}")
                if not result['correct']:
                    logger.info(f"    Text: {text[:60]}...")
                    logger.info(f"    Raw response: {result['raw_response']}")
            
            # Calculate accuracy for this dataset
            dataset_accuracy = correct_count / len(test_texts)
            results['per_dataset_accuracy'][dataset_name] = dataset_accuracy
            
            logger.info(f"  {dataset_name} accuracy: {dataset_accuracy:.2%} ({correct_count}/{len(test_texts)})")
            
            results['detailed_results'].extend(dataset_results)
        
        # Calculate overall accuracy
        results['overall_accuracy'] = results['correct_predictions'] / results['total_samples']
        
        # Build confusion matrix
        confusion_matrix = {}
        for result in results['detailed_results']:
            expected = result['expected']
            predicted = result['predicted']
            
            if expected not in confusion_matrix:
                confusion_matrix[expected] = {}
            if predicted not in confusion_matrix[expected]:
                confusion_matrix[expected][predicted] = 0
            
            confusion_matrix[expected][predicted] += 1
        
        results['confusion_matrix'] = confusion_matrix
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print test results in a formatted way."""
        logger.info("\n" + "="*60)
        logger.info("DIRECT CLASSIFICATION TEST RESULTS")
        logger.info("="*60)
        
        logger.info(f"Overall Accuracy: {results['overall_accuracy']:.2%} "
                   f"({results['correct_predictions']}/{results['total_samples']})")
        
        logger.info("\nPer-Dataset Accuracy:")
        for dataset, accuracy in results['per_dataset_accuracy'].items():
            logger.info(f"  {dataset}: {accuracy:.2%}")
        
        logger.info("\nConfusion Matrix:")
        for expected, predictions in results['confusion_matrix'].items():
            logger.info(f"  {expected}:")
            for predicted, count in predictions.items():
                logger.info(f"    -> {predicted}: {count}")
        
        logger.info("\nDetailed Results:")
        for i, result in enumerate(results['detailed_results']):
            status = "✓" if result['correct'] else "✗"
            logger.info(f"  {i+1:2d}. {status} Expected: {result['expected']:12} "
                       f"Predicted: {result['predicted']:12} "
                       f"Text: {result['text'][:50]}...")

def main():
    parser = argparse.ArgumentParser(description='Test direct classification trained model')
    parser.add_argument('--base-model-path', type=str, 
                       default='Qwen/Qwen2.5-0.5B-Instruct',
                       help='Path to base model')
    parser.add_argument('--lora-model-path', type=str, 
                       default='./direct_classification_models/final_direct_classification_model',
                       help='Path to LoRA model')
    parser.add_argument('--max-length', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--output-file', type=str, default='direct_classification_test_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Check if LoRA model exists
    if not Path(args.lora_model_path).exists():
        logger.error(f"LoRA model path {args.lora_model_path} does not exist!")
        return
    
    # Initialize tester
    tester = DirectClassificationTester(args.base_model_path, args.lora_model_path, args.max_length)
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    # Print results
    tester.print_results(results)
    
    # Save results to file
    output_path = Path(args.output_file)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Overall Performance: {results['overall_accuracy']:.2%}")
    
    if results['overall_accuracy'] >= 0.8:
        logger.info("🎉 Excellent performance! The direct classification model is learning very well.")
    elif results['overall_accuracy'] >= 0.6:
        logger.info("👍 Good performance! The direct classification model is learning but could be improved.")
    elif results['overall_accuracy'] >= 0.4:
        logger.info("⚠️  Moderate performance. The direct classification model needs more training.")
    else:
        logger.info("❌ Poor performance. The direct classification model is not learning the task effectively.")

if __name__ == "__main__":
    main()
