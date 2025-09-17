#!/usr/bin/env python3
"""
Comprehensive model evaluation script with better metrics and analysis.
"""

import argparse
import logging
import sys
from pathlib import Path
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluator with better metrics."""
    
    def __init__(self, model_path: str):
        """Initialize the model evaluator."""
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        # Test results
        self.test_results = {
            "predictions": [],
            "ground_truth": [],
            "confidence_scores": [],
            "test_prompts": [],
            "response_lengths": [],
            "generation_times": []
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
    
    def create_comprehensive_test_cases(self, num_tests: int = 50):
        """Create comprehensive test cases with better diversity."""
        
        # More diverse and realistic test cases
        test_cases = []
        
        # ifeval test cases
        ifeval_cases = [
            ("ifeval", "Follow these instructions step by step to complete the task."),
            ("ifeval", "Given the following instructions, provide a detailed response."),
            ("ifeval", "Execute the following command and explain your reasoning."),
            ("ifeval", "Please follow the instructions carefully and show your work."),
            ("ifeval", "Complete this instruction following task with detailed steps."),
            ("ifeval", "Given these instructions, what would you do?"),
            ("ifeval", "Follow the step-by-step instructions provided."),
            ("ifeval", "Execute the given instructions and explain your approach."),
            ("ifeval", "Please complete this instruction following evaluation."),
            ("ifeval", "Given the instructions, provide a comprehensive answer."),
            ("ifeval", "Instruction following task with complex reasoning requirements."),
            ("ifeval", "Follow the detailed instructions and provide step-by-step solution."),
            ("ifeval", "Execute the given instructions with proper explanation."),
            ("ifeval", "Complete this instruction following evaluation task."),
            ("ifeval", "Given the instructions, show your work and reasoning.")
        ]
        
        # commoneval test cases
        commoneval_cases = [
            ("commoneval", "This is a common evaluation benchmark for language understanding."),
            ("commoneval", "Standard evaluation task for natural language processing models."),
            ("commoneval", "Common benchmark dataset for evaluating language models."),
            ("commoneval", "This is a typical evaluation example from CommonEval."),
            ("commoneval", "Standard NLP evaluation task with multiple choice questions."),
            ("commoneval", "Common evaluation benchmark for assessing model performance."),
            ("commoneval", "Typical evaluation example from the CommonEval dataset."),
            ("commoneval", "Standard benchmark task for language model evaluation."),
            ("commoneval", "Common evaluation dataset for testing model capabilities."),
            ("commoneval", "This is a standard evaluation example from CommonEval."),
            ("commoneval", "Standard benchmark evaluation with multiple choice format."),
            ("commoneval", "Common evaluation task for language model assessment."),
            ("commoneval", "Typical CommonEval benchmark example for testing."),
            ("commoneval", "Standard evaluation dataset with multiple choice questions."),
            ("commoneval", "Common benchmark evaluation for language understanding.")
        ]
        
        # wildvoice test cases
        wildvoice_cases = [
            ("wildvoice", "This is a natural conversation with diverse speaking styles."),
            ("wildvoice", "Wild voice data with various accents and speaking patterns."),
            ("wildvoice", "Natural speech sample with conversational characteristics."),
            ("wildvoice", "Diverse voice data with different speaking styles and accents."),
            ("wildvoice", "Wild voice sample with natural conversational patterns."),
            ("wildvoice", "This audio contains natural speech with various characteristics."),
            ("wildvoice", "Wild voice data with diverse speaking patterns and accents."),
            ("wildvoice", "Natural conversation sample with varied speaking styles."),
            ("wildvoice", "This is a wild voice sample with conversational characteristics."),
            ("wildvoice", "Diverse voice data with natural speaking patterns and accents."),
            ("wildvoice", "Natural conversational speech with diverse characteristics."),
            ("wildvoice", "Wild voice sample with varied speaking patterns and styles."),
            ("wildvoice", "Diverse conversational data with natural speech patterns."),
            ("wildvoice", "Wild voice audio with various accents and speaking styles."),
            ("wildvoice", "Natural speech sample with conversational diversity.")
        ]
        
        # Combine all test cases
        all_cases = ifeval_cases + commoneval_cases + wildvoice_cases
        
        # Select random subset
        import random
        selected_cases = random.sample(all_cases, min(num_tests, len(all_cases)))
        
        return selected_cases
    
    def evaluate_model(self, num_tests: int = 50):
        """Evaluate the model with comprehensive test cases."""
        logger.info(f"Evaluating model with {num_tests} test cases")
        
        # Create test cases
        test_cases = self.create_comprehensive_test_cases(num_tests)
        
        all_predictions = []
        all_ground_truth = []
        all_confidence_scores = []
        all_prompts = []
        all_response_lengths = []
        all_generation_times = []
        
        for i, (expected_dataset, transcript) in enumerate(test_cases):
            logger.info(f"Processing test {i+1}/{len(test_cases)}")
            
            # Create prompt for classification
            prompt = f"Classify this audio transcript as one of these datasets: ifeval, commoneval, wildvoice. Transcript: {transcript}"
            
            try:
                import time
                start_time = time.time()
                
                # Tokenize and generate
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.model.device)
                
                # Generate prediction with better parameters
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=20,  # Shorter responses
                        temperature=0.3,   # Lower temperature
                        top_p=0.9,        # Nucleus sampling
                        top_k=50,         # Top-k sampling
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1,  # Reduce repetition
                        early_stopping=True     # Stop at EOS
                    )
                
                generation_time = time.time() - start_time
                
                # Decode prediction
                prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                prediction = prediction[len(prompt):].strip()
                
                # Extract dataset name from prediction
                predicted_dataset = self._extract_dataset_name(prediction)
                
                # Calculate confidence (improved heuristic)
                confidence = self._calculate_confidence(prediction, expected_dataset)
                
                # Store results
                all_predictions.append(predicted_dataset)
                all_ground_truth.append(expected_dataset)
                all_confidence_scores.append(confidence)
                all_prompts.append(prompt)
                all_response_lengths.append(len(prediction))
                all_generation_times.append(generation_time)
                
                logger.info(f"Expected: {expected_dataset} | Predicted: {predicted_dataset} | Confidence: {confidence:.3f}")
                logger.info(f"Response: {prediction[:50]}...")
                
            except Exception as e:
                logger.error(f"Error processing test {i+1}: {e}")
                continue
        
        # Store results
        self.test_results = {
            "predictions": all_predictions,
            "ground_truth": all_ground_truth,
            "confidence_scores": all_confidence_scores,
            "test_prompts": all_prompts,
            "response_lengths": all_response_lengths,
            "generation_times": all_generation_times
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
        """Calculate improved confidence score for prediction."""
        prediction_lower = prediction.lower()
        ground_truth_lower = ground_truth.lower()
        
        # Check for exact match
        if ground_truth_lower in prediction_lower:
            return 1.0
        
        # Check for partial match
        if any(word in prediction_lower for word in ground_truth_lower.split()):
            return 0.5
        
        # Check for similar words
        similar_words = {
            "ifeval": ["instruction", "follow", "task", "step"],
            "commoneval": ["common", "evaluation", "benchmark", "standard"],
            "wildvoice": ["wild", "voice", "natural", "conversation", "speech"]
        }
        
        if any(word in prediction_lower for word in similar_words.get(ground_truth_lower, [])):
            return 0.3
        
        return 0.0
    
    def calculate_comprehensive_metrics(self) -> dict:
        """Calculate comprehensive performance metrics."""
        predictions = self.test_results["predictions"]
        ground_truth = self.test_results["ground_truth"]
        confidence_scores = self.test_results["confidence_scores"]
        response_lengths = self.test_results["response_lengths"]
        generation_times = self.test_results["generation_times"]
        
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
        
        # Response length statistics
        avg_response_length = np.mean(response_lengths) if response_lengths else 0.0
        
        # Generation time statistics
        avg_generation_time = np.mean(generation_times) if generation_times else 0.0
        
        # Confusion matrix
        confusion_matrix = self._create_confusion_matrix(predictions, ground_truth)
        
        # Precision, Recall, F1 for each class
        precision_recall_f1 = self._calculate_precision_recall_f1(predictions, ground_truth)
        
        metrics = {
            "overall_accuracy": accuracy,
            "dataset_accuracy": dataset_accuracy,
            "average_confidence": avg_confidence,
            "total_samples": total,
            "correct_predictions": correct,
            "confusion_matrix": confusion_matrix,
            "precision_recall_f1": precision_recall_f1,
            "avg_response_length": avg_response_length,
            "avg_generation_time": avg_generation_time
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
    
    def _calculate_precision_recall_f1(self, predictions: list, ground_truth: list) -> dict:
        """Calculate precision, recall, and F1 for each class."""
        datasets = set(predictions + ground_truth)
        metrics = {}
        
        for dataset in datasets:
            # True positives: correctly predicted as this dataset
            tp = sum(1 for p, g in zip(predictions, ground_truth) if p == dataset and g == dataset)
            
            # False positives: predicted as this dataset but actually different
            fp = sum(1 for p, g in zip(predictions, ground_truth) if p == dataset and g != dataset)
            
            # False negatives: actually this dataset but predicted as different
            fn = sum(1 for p, g in zip(predictions, ground_truth) if p != dataset and g == dataset)
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[dataset] = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        
        return metrics
    
    def print_comprehensive_summary(self, metrics: dict):
        """Print comprehensive evaluation summary."""
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE MODEL EVALUATION SUMMARY")
        print("="*80)
        
        print(f"📊 Overall Accuracy: {metrics['overall_accuracy']:.3f} ({metrics['correct_predictions']}/{metrics['total_samples']})")
        print(f"🎯 Average Confidence: {metrics['average_confidence']:.3f}")
        print(f"📏 Average Response Length: {metrics['avg_response_length']:.1f} characters")
        print(f"⏱️  Average Generation Time: {metrics['avg_generation_time']:.3f} seconds")
        
        print("\n📈 Per-Dataset Accuracy:")
        for dataset, acc in metrics['dataset_accuracy'].items():
            print(f"  {dataset}: {acc:.3f}")
        
        print("\n🎯 Precision, Recall, F1:")
        for dataset, prf in metrics['precision_recall_f1'].items():
            print(f"  {dataset}:")
            print(f"    Precision: {prf['precision']:.3f}")
            print(f"    Recall: {prf['recall']:.3f}")
            print(f"    F1: {prf['f1']:.3f}")
        
        print("\n🔍 Confusion Matrix:")
        datasets = sorted(metrics['confusion_matrix'].keys())
        print("     " + " ".join(f"{d:>10}" for d in datasets))
        for true_dataset in datasets:
            row = f"{true_dataset:>10}"
            for pred_dataset in datasets:
                count = metrics['confusion_matrix'][true_dataset][pred_dataset]
                row += f" {count:>10}"
            print(row)
        
        print("\n✅ Comprehensive evaluation completed!")
        print("="*80)
    
    def save_results(self, output_file: str = "comprehensive_evaluation_results.json"):
        """Save comprehensive evaluation results to file."""
        metrics = self.calculate_comprehensive_metrics()
        
        results = {
            "test_results": self.test_results,
            "metrics": metrics
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Comprehensive results saved to {output_file}")

def main():
    """Main function for comprehensive evaluation."""
    parser = argparse.ArgumentParser(description="Comprehensive model evaluation")
    parser.add_argument("--model-path", type=str, 
                       default="/home/mbhat/omegalabs-anytoany-bittensor/finetune/models/final_model",
                       help="Path to the trained model")
    parser.add_argument("--num-tests", type=int, default=50,
                       help="Number of test cases")
    parser.add_argument("--output-file", type=str, default="comprehensive_evaluation_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(args.model_path)
        
        # Load model
        evaluator.load_model()
        
        # Run evaluation
        test_results = evaluator.evaluate_model(args.num_tests)
        
        # Calculate and print metrics
        metrics = evaluator.calculate_comprehensive_metrics()
        evaluator.print_comprehensive_summary(metrics)
        
        # Save results
        evaluator.save_results(args.output_file)
        
        logger.info("🎉 Comprehensive evaluation completed!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
