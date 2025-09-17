#!/usr/bin/env python3
"""
Test the trained Qwen model on dataset classification performance.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from typing import Dict, List, Any
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClassificationTester:
    """Test the trained model on classification tasks."""
    
    def __init__(self, model_path: str, max_length: int = 256):
        self.max_length = max_length
        
        # Load the trained model
        logger.info(f"Loading trained model from {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("✓ Model and tokenizer loaded successfully")
        
        # Test cases for each dataset type
        self.test_cases = {
            'ifeval': [
                "Please follow these instructions step by step: First, identify the main topic. Then, provide three supporting arguments.",
                "Your task is to analyze the following text and extract the key information. Make sure to follow the format exactly as specified.",
                "Complete this instruction: Write a response that contains exactly 5 sentences, each starting with a different letter.",
                "Follow this instruction precisely: Create a list of 10 items, but do not use any commas in your response.",
                "Your assignment is to write a paragraph that contains at least 3 placeholders in square brackets, like [this].",
                "Please respond to this question: What are the main components of a successful project? Use exactly 50 words.",
                "Follow these steps: 1) Read the question carefully 2) Think about the answer 3) Write your response",
                "Your task is to create a response that follows this exact format: SECTION 1: Introduction, SECTION 2: Main points",
                "Please answer this question in all lowercase letters only, without any capital letters.",
                "Complete this instruction: Write a response that ends with the exact phrase 'This completes the task.'"
            ],
            'commoneval': [
                "What is the capital city of Australia?",
                "Explain the concept of photosynthesis in simple terms.",
                "List three benefits of regular exercise.",
                "What are the main causes of climate change?",
                "Describe the process of making bread from flour.",
                "What is the difference between a virus and bacteria?",
                "Explain why the sky appears blue during the day.",
                "What are the primary colors and how do they mix?",
                "Describe the water cycle in nature.",
                "What is the purpose of the United Nations?"
            ],
            'wildvoice': [
                "Hey there! How's it going today? I'm doing pretty well, thanks for asking.",
                "So I was thinking about what we discussed earlier, and I have some ideas to share.",
                "You know what? That's actually a really interesting point you made there.",
                "I'm not sure about that, but I think we should consider all the options first.",
                "Wow, that's amazing! I never thought about it that way before.",
                "Hmm, let me think about this for a moment. What do you think we should do?",
                "That sounds like a great plan! When do you want to get started?",
                "I'm really excited about this project. It's going to be so much fun!",
                "Oh, I see what you mean now. That makes a lot more sense.",
                "You're absolutely right about that. I completely agree with your assessment."
            ]
        }
    
    def create_test_prompt(self, text: str) -> str:
        """Create test prompt for classification."""
        return f"""You are an expert at analyzing text and identifying which type of evaluation dataset it comes from.

Your task is to classify text into one of these categories:
- ifeval: Instruction following evaluation with complex reasoning tasks
- commoneval: Common evaluation benchmark for natural language processing  
- wildvoice: Wild voice data with diverse speaking styles and accents

Analyze the content, style, and structure of the text to determine the most likely source dataset.

Text: "{text}"

Which dataset category does this text most likely come from? Answer with only the dataset name: ifeval, commoneval, or wildvoice.

Answer:"""
    
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
                max_new_tokens=20,
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
        
        # Clean up the prediction
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
        logger.info("Starting comprehensive classification test...")
        
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
                          f"Expected: {result['expected']}, Predicted: {result['predicted']}")
                if not result['correct']:
                    logger.info(f"    Text: {text[:100]}...")
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
        logger.info("CLASSIFICATION TEST RESULTS")
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
    parser = argparse.ArgumentParser(description='Test trained Qwen model on classification')
    parser.add_argument('--model-path', type=str, 
                       default='./qwen_voicebench_models/final_qwen_voicebench_model',
                       help='Path to trained model')
    parser.add_argument('--max-length', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--output-file', type=str, default='classification_test_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model_path).exists():
        logger.error(f"Model path {args.model_path} does not exist!")
        return
    
    # Initialize tester
    tester = ClassificationTester(args.model_path, args.max_length)
    
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
        logger.info("🎉 Excellent performance! The model is learning well.")
    elif results['overall_accuracy'] >= 0.6:
        logger.info("👍 Good performance! The model is learning but could be improved.")
    elif results['overall_accuracy'] >= 0.4:
        logger.info("⚠️  Moderate performance. The model needs more training.")
    else:
        logger.info("❌ Poor performance. The model is not learning the task effectively.")

if __name__ == "__main__":
    main()
