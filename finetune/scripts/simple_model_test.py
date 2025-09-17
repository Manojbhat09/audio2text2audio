#!/usr/bin/env python3
"""
Simple test script for the trained GRPO dataset classifier model.
This script tests the model without complex imports.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from collections import Counter

# Enable synchronous CUDA execution for better debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleModelTester:
    """Simple test for the trained GRPO model."""
    
    def __init__(self, model_path: str):
        """Initialize the model tester."""
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        # Test results
        self.test_results = {
            "predictions": [],
            "ground_truth": [],
            "confidence_scores": [],
            "test_prompts": []
        }
    
    def load_model(self):
        """Load the trained model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            logger.info("✓ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def test_with_synthetic_data(self, num_tests: int = 10):
        """Test the model with synthetic prompts."""
        logger.info(f"Testing with {num_tests} synthetic prompts")
        
        # Create synthetic test prompts
        test_cases = [
            ("ifeval", "This is a test audio for ifeval dataset with instruction following tasks."),
            ("commoneval", "This is a common evaluation task for language understanding."),
            ("wildvoice", "This is a wild voice sample with natural speech patterns."),
            ("ifeval", "Instruction following evaluation with complex reasoning tasks."),
            ("commoneval", "Common evaluation benchmark for natural language processing."),
            ("wildvoice", "Wild voice data with diverse speaking styles and accents."),
            ("ifeval", "IfEval dataset containing instruction following examples."),
            ("commoneval", "CommonEval benchmark for evaluating language models."),
            ("wildvoice", "WildVoice dataset with natural conversational speech."),
            ("ifeval", "Instruction following evaluation with step-by-step reasoning.")
        ]
        
        all_predictions = []
        all_ground_truth = []
        all_confidence_scores = []
        all_prompts = []
        
        for i, (expected_dataset, transcript) in enumerate(test_cases[:num_tests]):
            logger.info(f"Processing test {i+1}/{num_tests}")
            
            # Create prompt for classification
            prompt = f"Classify this audio transcript as one of these datasets: ifeval, commoneval, wildvoice. Transcript: {transcript}"
            
            try:
                # Use safe generation method
                full_response = self._safe_generate(prompt, max_new_tokens=30, temperature=0.1)
                
                if full_response.startswith("ERROR"):
                    logger.error(f"Generation failed: {full_response}")
                    continue
                
                # Extract prediction (remove the prompt)
                prediction = full_response[len(prompt):].strip()
                
                # Extract dataset name from prediction
                predicted_dataset = self._extract_dataset_name(prediction)
                
                # Calculate confidence (simple heuristic)
                confidence = self._calculate_confidence(prediction, expected_dataset)
                
                # Store results
                all_predictions.append(predicted_dataset)
                all_ground_truth.append(expected_dataset)
                all_confidence_scores.append(confidence)
                all_prompts.append(prompt)
                
                logger.info(f"Expected: {expected_dataset} | Predicted: {predicted_dataset} | Confidence: {confidence:.3f}")
                logger.info(f"Response: {prediction[:100]}...")
                
            except Exception as e:
                logger.error(f"Error processing test {i+1}: {e}")
                continue
        
        # Store results
        self.test_results = {
            "predictions": all_predictions,
            "ground_truth": all_ground_truth,
            "confidence_scores": all_confidence_scores,
            "test_prompts": all_prompts
        }
        
        return self.test_results
    
    def _extract_dataset_name(self, prediction: str) -> str:
        """Extract dataset name from model prediction."""
        prediction_lower = prediction.lower()
        
        if "ifeval" in prediction_lower:
            return "ifeval"
        elif "commoneval" in prediction_lower:
            return "commoneval"
        elif "wildvoice" in prediction_lower:
            return "wildvoice"
        else:
            return "unknown"
    
    def _calculate_confidence(self, prediction: str, ground_truth: str) -> float:
        """Calculate confidence score for prediction."""
        prediction_lower = prediction.lower()
        ground_truth_lower = ground_truth.lower()
        
        # Simple confidence based on exact match
        if ground_truth_lower in prediction_lower:
            return 1.0
        else:
            return 0.0
    
    def _safe_generate(self, prompt: str, max_new_tokens: int = 30, temperature: float = 0.7) -> str:
        """
        Safely generate text with comprehensive NaN handling and fallback mechanisms.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text or error message
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Method 1: Try deterministic generation first
            try:
                with torch.no_grad():
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,  # Deterministic generation
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=2,
                    )
                    
                    response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                    
                    # Check if response is valid
                    if response and len(response.strip()) > 0:
                        return response
                        
            except Exception as e:
                logger.warning(f"Deterministic generation failed: {e}")
            
            # Method 2: Try with custom logits processing
            try:
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    # Comprehensive NaN/Inf handling
                    if torch.isnan(logits).any():
                        logger.warning("NaN detected in logits, applying fixes...")
                        logits = torch.where(torch.isnan(logits), torch.tensor(-10.0, device=logits.device), logits)
                    
                    if torch.isinf(logits).any():
                        logger.warning("Inf detected in logits, applying fixes...")
                        logits = torch.where(torch.isinf(logits), torch.tensor(10.0, device=logits.device), logits)
                    
                    # Clamp extreme values
                    logits = torch.clamp(logits, min=-20.0, max=20.0)
                    
                    # Use greedy decoding with processed logits
                    generated_ids = []
                    current_input = inputs["input_ids"]
                    
                    for _ in range(max_new_tokens):
                        # Get next token logits
                        next_logits = logits[0, -1, :]  # Last token logits
                        
                        # Apply temperature if needed
                        if temperature > 0:
                            next_logits = next_logits / max(temperature, 0.1)
                        
                        # Get the most likely token
                        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                        
                        # Check for EOS token
                        if next_token.item() == self.tokenizer.eos_token_id:
                            break
                        
                        generated_ids.append(next_token.item())
                        
                        # Update input for next iteration
                        current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1)
                        
                        # Get new logits
                        with torch.no_grad():
                            new_outputs = self.model(current_input)
                            logits = new_outputs.logits
                            
                            # Apply same fixes to new logits
                            if torch.isnan(logits).any():
                                logits = torch.where(torch.isnan(logits), torch.tensor(-10.0, device=logits.device), logits)
                            if torch.isinf(logits).any():
                                logits = torch.where(torch.isinf(logits), torch.tensor(10.0, device=logits.device), logits)
                            logits = torch.clamp(logits, min=-20.0, max=20.0)
                    
                    # Decode generated tokens
                    if generated_ids:
                        full_sequence = torch.cat([inputs["input_ids"], torch.tensor(generated_ids, device=device).unsqueeze(0)], dim=1)
                        response = self.tokenizer.decode(full_sequence[0], skip_special_tokens=True)
                        return response
                    else:
                        return prompt  # Return original prompt if no generation
                        
            except Exception as e:
                logger.warning(f"Custom logits processing failed: {e}")
            
            # Method 3: Fallback to simple response
            logger.warning("All generation methods failed, using fallback response")
            return "I apologize, but I'm having trouble generating a response right now."
                
        except Exception as e:
            logger.error(f"All generation methods failed: {e}")
            return f"ERROR: {str(e)}"
    
    def calculate_metrics(self) -> dict:
        """Calculate performance metrics."""
        predictions = self.test_results["predictions"]
        ground_truth = self.test_results["ground_truth"]
        confidence_scores = self.test_results["confidence_scores"]
        
        # Overall accuracy
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0.0
        
        # Per-dataset accuracy
        dataset_accuracy = {}
        for dataset in set(ground_truth):
            dataset_predictions = [p for p, g in zip(predictions, ground_truth) if g == dataset]
            dataset_correct = sum(1 for p in dataset_predictions if p == dataset)
            dataset_total = len(dataset_predictions)
            dataset_accuracy[dataset] = dataset_correct / dataset_total if dataset_total > 0 else 0.0
        
        # Average confidence
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # Confusion matrix
        confusion_matrix = self._create_confusion_matrix(predictions, ground_truth)
        
        metrics = {
            "overall_accuracy": accuracy,
            "dataset_accuracy": dataset_accuracy,
            "average_confidence": avg_confidence,
            "total_samples": total,
            "correct_predictions": correct,
            "confusion_matrix": confusion_matrix
        }
        
        return metrics
    
    def _create_confusion_matrix(self, predictions: list, ground_truth: list) -> dict:
        """Create confusion matrix."""
        datasets = sorted(set(predictions + ground_truth))
        matrix = {}
        
        for true_dataset in datasets:
            matrix[true_dataset] = {}
            for pred_dataset in datasets:
                count = sum(1 for p, g in zip(predictions, ground_truth) 
                           if g == true_dataset and p == pred_dataset)
                matrix[true_dataset][pred_dataset] = count
        
        return matrix
    
    def print_summary(self, metrics: dict):
        """Print test summary."""
        print("\n" + "="*60)
        print("🎯 MODEL TESTING SUMMARY")
        print("="*60)
        
        print(f"📊 Overall Accuracy: {metrics['overall_accuracy']:.3f} ({metrics['correct_predictions']}/{metrics['total_samples']})")
        print(f"🎯 Average Confidence: {metrics['average_confidence']:.3f}")
        
        print("\n📈 Per-Dataset Accuracy:")
        for dataset, acc in metrics['dataset_accuracy'].items():
            print(f"  {dataset}: {acc:.3f}")
        
        print("\n🔍 Confusion Matrix:")
        datasets = sorted(metrics['confusion_matrix'].keys())
        print("     " + " ".join(f"{d:>10}" for d in datasets))
        for true_dataset in datasets:
            row = f"{true_dataset:>10}"
            for pred_dataset in datasets:
                count = metrics['confusion_matrix'][true_dataset][pred_dataset]
                row += f" {count:>10}"
            print(row)
        
        print("\n✅ Test completed successfully!")
        print("="*60)
    
    def save_results(self, output_file: str = "test_results.json"):
        """Save test results to file."""
        metrics = self.calculate_metrics()
        
        results = {
            "test_results": self.test_results,
            "metrics": metrics
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test the trained GRPO model")
    parser.add_argument("--model-path", type=str, 
                       default="/home/mbhat/omegalabs-anytoany-bittensor/finetune/models/final_model",
                       help="Path to the trained model")
    parser.add_argument("--num-tests", type=int, default=10,
                       help="Number of test cases")
    parser.add_argument("--output-file", type=str, default="test_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    try:
        # Initialize tester
        tester = SimpleModelTester(args.model_path)
        
        # Load model
        tester.load_model()
        
        # Run tests
        test_results = tester.test_with_synthetic_data(args.num_tests)
        
        # Calculate and print metrics
        metrics = tester.calculate_metrics()
        tester.print_summary(metrics)
        
        # Save results
        tester.save_results(args.output_file)
        
        logger.info("🎉 Testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
