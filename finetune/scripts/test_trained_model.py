#!/usr/bin/env python3
"""
Test script for the trained GRPO dataset classifier model.
This script loads the trained model and tests it on real datasets to evaluate performance.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataset_loader import DatasetLoader
from whisper_transcriber import WhisperTranscriber
from reward_functions import DatasetClassificationReward

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_results.log')
    ]
)
logger = logging.getLogger(__name__)

class ModelTester:
    """Test the trained GRPO model on real datasets."""
    
    def __init__(self, model_path: str, whisper_model_path: str = "openai/whisper-tiny"):
        """
        Initialize the model tester.
        
        Args:
            model_path: Path to the trained model
            whisper_model_path: Path to Whisper model for transcription
        """
        self.model_path = model_path
        self.whisper_model_path = whisper_model_path
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.whisper_transcriber = None
        self.reward_function = None
        self.dataset_loader = None
        
        # Test results
        self.test_results = {
            "predictions": [],
            "ground_truth": [],
            "confidence_scores": [],
            "transcripts": [],
            "dataset_names": []
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
    
    def setup_components(self):
        """Setup Whisper transcriber and reward function."""
        logger.info("Setting up components...")
        
        try:
            # Initialize Whisper transcriber
            self.whisper_transcriber = WhisperTranscriber(self.whisper_model_path)
            logger.info("✓ Whisper transcriber initialized")
            
            # Initialize reward function
            self.reward_function = DatasetClassificationReward()
            logger.info("✓ Reward function initialized")
            
            # Initialize dataset loader
            self.dataset_loader = DatasetLoader()
            logger.info("✓ Dataset loader initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup components: {e}")
            raise
    
    def test_on_datasets(self, datasets: List[str], samples_per_dataset: int = 10) -> Dict:
        """
        Test the model on specified datasets.
        
        Args:
            datasets: List of dataset names to test
            samples_per_dataset: Number of samples per dataset
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing on datasets: {datasets}")
        logger.info(f"Samples per dataset: {samples_per_dataset}")
        
        all_predictions = []
        all_ground_truth = []
        all_confidence_scores = []
        all_transcripts = []
        all_dataset_names = []
        
        for dataset_name in datasets:
            logger.info(f"\n--- Testing {dataset_name} ---")
            
            try:
                # Generate test samples for this dataset
                samples = list(self.dataset_loader.generate_training_samples(
                    datasets=[dataset_name],
                    max_samples=samples_per_dataset
                ))
                
                logger.info(f"Generated {len(samples)} test samples for {dataset_name}")
                
                for i, sample in enumerate(samples):
                    logger.info(f"Processing sample {i+1}/{len(samples)}")
                    
                    # Transcribe audio
                    transcript = self.whisper_transcriber.transcribe(sample.audio)
                    logger.info(f"Transcript: {transcript[:100]}...")
                    
                    # Create prompt for classification
                    prompt = f"Classify this audio transcript as one of these datasets: ifeval, commoneval, wildvoice. Transcript: {transcript}"
                    
                    # Tokenize and generate
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    ).to(self.model.device)
                    
                    # Generate prediction
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=50,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    
                    # Decode prediction
                    prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    prediction = prediction[len(prompt):].strip()
                    
                    # Extract dataset name from prediction
                    predicted_dataset = self._extract_dataset_name(prediction)
                    
                    # Calculate confidence (simple heuristic)
                    confidence = self._calculate_confidence(prediction, dataset_name)
                    
                    # Store results
                    all_predictions.append(predicted_dataset)
                    all_ground_truth.append(dataset_name)
                    all_confidence_scores.append(confidence)
                    all_transcripts.append(transcript)
                    all_dataset_names.append(dataset_name)
                    
                    logger.info(f"Predicted: {predicted_dataset} | Ground Truth: {dataset_name} | Confidence: {confidence:.3f}")
                    
            except Exception as e:
                logger.error(f"Error testing {dataset_name}: {e}")
                continue
        
        # Store results
        self.test_results = {
            "predictions": all_predictions,
            "ground_truth": all_ground_truth,
            "confidence_scores": all_confidence_scores,
            "transcripts": all_transcripts,
            "dataset_names": all_dataset_names
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
    
    def calculate_metrics(self) -> Dict:
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
    
    def _create_confusion_matrix(self, predictions: List[str], ground_truth: List[str]) -> Dict:
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
    
    def generate_report(self, output_dir: str = "test_outputs"):
        """Generate comprehensive test report."""
        logger.info("Generating test report...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Save detailed results
        results_file = output_path / "detailed_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "test_results": self.test_results,
                "metrics": metrics
            }, f, indent=2)
        
        # Generate confusion matrix plot
        self._plot_confusion_matrix(metrics["confusion_matrix"], output_path)
        
        # Generate accuracy plot
        self._plot_accuracy(metrics["dataset_accuracy"], output_path)
        
        # Print summary
        self._print_summary(metrics)
        
        logger.info(f"Test report saved to {output_path}")
    
    def _plot_confusion_matrix(self, confusion_matrix: Dict, output_path: Path):
        """Plot confusion matrix."""
        datasets = sorted(confusion_matrix.keys())
        matrix = np.array([[confusion_matrix[true][pred] for pred in datasets] 
                          for true in datasets])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=datasets, yticklabels=datasets)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_accuracy(self, dataset_accuracy: Dict, output_path: Path):
        """Plot per-dataset accuracy."""
        datasets = list(dataset_accuracy.keys())
        accuracies = list(dataset_accuracy.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(datasets, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
        plt.title('Accuracy by Dataset')
        plt.xlabel('Dataset')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'dataset_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _print_summary(self, metrics: Dict):
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

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test the trained GRPO model")
    parser.add_argument("--model-path", type=str, 
                       default="/home/mbhat/omegalabs-anytoany-bittensor/finetune/models/final_model",
                       help="Path to the trained model")
    parser.add_argument("--whisper-model", type=str, default="openai/whisper-tiny",
                       help="Whisper model for transcription")
    parser.add_argument("--datasets", nargs="+", 
                       default=["ifeval", "commoneval", "wildvoice"],
                       help="Datasets to test on")
    parser.add_argument("--samples-per-dataset", type=int, default=10,
                       help="Number of samples per dataset")
    parser.add_argument("--output-dir", type=str, default="test_outputs",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    try:
        # Initialize tester
        tester = ModelTester(args.model_path, args.whisper_model)
        
        # Load model
        tester.load_model()
        
        # Setup components
        tester.setup_components()
        
        # Run tests
        test_results = tester.test_on_datasets(args.datasets, args.samples_per_dataset)
        
        # Generate report
        tester.generate_report(args.output_dir)
        
        logger.info("🎉 Testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
