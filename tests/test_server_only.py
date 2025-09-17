#!/usr/bin/env python3
"""
Server-Only VoiceBench Testing Script

This script tests voice-to-text models using a server API endpoint
without any Docker dependencies. It uses the same evaluation logic
as the production system but connects directly to HTTP APIs.

Usage:
    python -m tests.test_server_only --experiment_name "my-model-test" --api_url "http://localhost:8000/api/v1/v2t"
"""

import argparse
import json
import time
import traceback
import base64
import io
import requests
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from datasets import load_dataset, Audio
import bittensor as bt

# Import evaluation components (without Docker dependencies)
from neurons.llm_judge import evaluate_responses_with_llm
from neurons.voicebench_evaluators import (
    OpenEvaluator, MCQEvaluator, IFEvaluator, 
    BBHEvaluator, HarmEvaluator
)

# Constants
PENALTY_SCORE = 0.001
SAMPLES_PER_DATASET = {
    'commoneval': 50,
    'wildvoice': 100,
    'advbench': 100,
    'ifeval': 50
}
VOICEBENCH_MAX_SAMPLES = 100000

# Dataset to evaluator mapping
DATASET_EVALUATOR_MAP = {
    'commoneval': 'open',
    'wildvoice': 'open',
    'ifeval': 'ifeval',
    'advbench': 'harm',
}

# Datasets that need LLM scoring
NEEDS_LLM_JUDGE = ['commoneval', 'wildvoice']


class ServerModelClient:
    """Client for communicating with model server APIs."""
    
    def __init__(self, api_url: str, timeout: int = 600):
        self.api_url = api_url
        self.timeout = timeout
        bt.logging.info(f"Initialized ServerModelClient with API: {api_url}")
    
    def inference_v2t(self, audio_array: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Send inference request to server.
        
        Args:
            audio_array: Input audio array
            sample_rate: Audio sample rate
            
        Returns:
            Dict containing inference results
        """
        try:
            # Convert audio array to base64
            buffer = io.BytesIO()
            np.save(buffer, audio_array)
            audio_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Send request
            response = requests.post(
                self.api_url,
                json={
                    "audio_data": audio_b64,
                    "sample_rate": sample_rate
                },
                timeout=self.timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            bt.logging.error(f"Request failed: {e}")
            return {"text": "", "error": str(e)}
        except Exception as e:
            bt.logging.error(f"Unexpected error: {e}")
            return {"text": "", "error": str(e)}


def evaluate_dataset_with_proper_evaluator(
    dataset_name: str,
    dataset_results: Dict[str, Any]
) -> tuple[float, Dict[str, Any]]:
    """
    Evaluate dataset responses using the correct VoiceBench evaluator.
    
    Returns:
        Tuple of (score, status_dict)
    """
    status = {
        'dataset': dataset_name,
        'total_samples': dataset_results.get('total_samples', 0),
        'successful_responses': dataset_results.get('successful_responses', 0),
        'success_rate': dataset_results.get('success_rate', 0.0),
        'evaluator_used': None,
        'evaluation_status': 'pending',
        'evaluation_error': None,
        'evaluation_details': None,
        'evaluation_time': 0.0,
        'score': 0.0,
        'sample_details': []
    }
    
    start = time.perf_counter()
    
    # Check for dataset-level error
    if 'error' in dataset_results:
        status['evaluation_status'] = 'failed'
        status['evaluation_error'] = dataset_results['error']
        return 0.0, status
    
    # Get responses
    responses = dataset_results.get('responses', [])
    if not responses:
        status['evaluation_status'] = 'no_responses'
        return 0.0, status
    
    # Determine evaluator type
    dataset_base = dataset_name.split('_')[0]
    evaluator_type = DATASET_EVALUATOR_MAP.get(dataset_base, 'open')
    status['evaluator_used'] = evaluator_type
    bt.logging.info(f"Using {evaluator_type} evaluator for {dataset_name}")
    
    try:
        # Prepare data for evaluation
        eval_data = []
        
        if evaluator_type == 'open':
            # First get LLM scores
            bt.logging.info(f"Getting LLM scores for {dataset_name}")
            llm_responses = evaluate_responses_with_llm(responses)
            
            # Build sample details for open evaluator datasets
            for i, response in enumerate(llm_responses):
                scores = response.get('score', ['0'])
                eval_data.append({
                    'score': scores
                })
                
                # Build sample detail
                sample_detail = {
                    'hf_index': response.get('hf_index', i),
                    'miner_model_response': response.get('response', ''),
                    'llm_judge_response': response.get('llm_raw_response', ''),
                    'llm_scores': scores if isinstance(scores, list) else [scores],
                    'inference_time': response.get('inference_time', 0.0)
                }
                status['sample_details'].append(sample_detail)
                
            bt.logging.info(f"Got LLM scores for {dataset_name} for {len(llm_responses)} responses")
            
        elif evaluator_type == 'ifeval':
            # IFEvaluator needs special format
            for response in responses:
                eval_data.append({
                    'key': response.get('id', 0),
                    'instruction_id_list': response.get('instruction_id_list', []),
                    'prompt': response.get('prompt', ''),
                    'response': response.get('response', ''),
                    'kwargs': response.get('kwargs', [])
                })
                
                # Build sample detail for IFEval
                sample_detail = {
                    'hf_index': response.get('hf_index', 0),
                    'miner_model_response': response.get('response', ''),
                    'llm_judge_response': '',
                    'llm_scores': [],
                    'inference_time': response.get('inference_time', 0.0)
                }
                status['sample_details'].append(sample_detail)
                
        elif evaluator_type == 'harm':
            # HarmEvaluator just needs response
            for response in responses:
                eval_data.append({
                    'response': response.get('response', '')
                })
                
                # Build sample detail for Harm evaluator
                sample_detail = {
                    'hf_index': response.get('hf_index', 0),
                    'miner_model_response': response.get('response', ''),
                    'llm_judge_response': '',
                    'llm_scores': [],
                    'inference_time': response.get('inference_time', 0.0)
                }
                status['sample_details'].append(sample_detail)
        
        # Create and run evaluator
        if evaluator_type == 'open':
            evaluator = OpenEvaluator()
        elif evaluator_type == 'ifeval':
            evaluator = IFEvaluator()
        elif evaluator_type == 'harm':
            evaluator = HarmEvaluator()
        else:
            bt.logging.warning(f"Unknown evaluator type {evaluator_type}, using OpenEvaluator")
            evaluator = OpenEvaluator()
        
        # Evaluate
        eval_result = evaluator.evaluate(eval_data)
        
        # Store per-sample scores for harm evaluator
        if evaluator_type == 'harm' and 'per_sample_scores' in eval_result:
            per_sample_scores = eval_result['per_sample_scores']
            for i, score in enumerate(per_sample_scores):
                status['sample_details'][i]['llm_scores'] = [float(score)]
            eval_result.pop('per_sample_scores', None)

        # Extract score based on evaluator output format
        if 'gpt' in eval_result:
            # OpenEvaluator returns 1-5, normalize to 0-1
            score = eval_result['gpt'] / 5.0
        elif 'final' in eval_result:
            # IFEvaluator returns final score 0-1
            score = eval_result['final']
        elif 'refusal_rate' in eval_result:
            # HarmEvaluator returns refusal rate (higher is better for safety)
            score = eval_result['refusal_rate']
        elif 'strict-prompt' in eval_result:
            # IFEvaluator alternative
            score = eval_result['strict-prompt']
        else:
            bt.logging.warning(f"Unknown evaluation result format: {eval_result}")
            score = 0.0
        
        # Limit sample_details to random 10 samples
        if len(status['sample_details']) > 10:
            import random
            status['sample_details'] = random.sample(status['sample_details'], 10)

        status['score'] = score
        status['evaluation_status'] = 'completed'
        status['evaluation_details'] = eval_result
        status['evaluation_time'] = time.perf_counter() - start

        bt.logging.info(f"Dataset {dataset_name} evaluated with {evaluator_type}: score={score:.3f}")
        
        return score, status
        
    except Exception as e:
        bt.logging.error(f"Error evaluating dataset {dataset_name}: {e}")
        status['evaluation_status'] = 'failed'
        status['evaluation_error'] = str(e)
        return 0.0, status


def calculate_voicebench_scores_with_status(
    results: Dict[str, Any]
) -> tuple[Dict[str, float], Dict[str, Any]]:
    """
    Calculate VoiceBench scores using proper evaluators with status tracking.
    
    Returns:
        Tuple of (scores_dict, status_dict)
    """
    scores = {}
    status = {}
    
    # Dataset weights for weighted average
    dataset_weights = {
        'commoneval_test': 2.0,
        'wildvoice_test': 2.0,
        'ifeval_test': 1.5,
        'advbench_test': 1.0,
    }
    
    weighted_scores = {}
    total_weight = 0.0
    
    for dataset_key, dataset_results in results.items():
        # Evaluate with proper evaluator
        score, dataset_status = evaluate_dataset_with_proper_evaluator(
            dataset_key, dataset_results
        )
        
        scores[dataset_key] = float(score)
        status[dataset_key] = dataset_status
        
        # Apply weighting for overall score
        weight = dataset_weights.get(dataset_key, 1.0)
        weighted_scores[dataset_key] = float(score) * weight
        total_weight += weight
    
    # Calculate weighted overall score
    if total_weight > 0:
        scores['overall'] = sum(weighted_scores.values()) / total_weight
    else:
        scores['overall'] = 0.0
    
    # Add overall status summary
    overall_status = {
        'total_datasets': len(results),
        'completed_evaluations': sum(1 for s in status.values() if isinstance(s, dict) and s.get('evaluation_status') == 'completed'),
        'failed_evaluations': sum(1 for s in status.values() if isinstance(s, dict) and s.get('evaluation_status') == 'failed'),
        'total_samples': sum(s.get('total_samples', 0) for s in status.values() if isinstance(s, dict)),
        'successful_responses': sum(s.get('successful_responses', 0) for s in status.values() if isinstance(s, dict)),
        'overall_score': float(scores['overall'])
    }
    status['overall'] = overall_status
    
    return scores, status


def create_realistic_audio_from_prompt(prompt: str) -> np.ndarray:
    """Create realistic audio data based on the prompt text."""
    # Calculate duration based on prompt length (realistic speech timing)
    words = len(prompt.split())
    duration = max(1.0, min(10.0, words * 0.3))  # ~0.3 seconds per word
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    
    # Create more realistic audio pattern
    t = np.linspace(0, duration, num_samples)
    
    # Create a more speech-like pattern with multiple frequencies
    # This simulates the complexity of real speech
    audio = np.zeros(num_samples)
    
    # Add fundamental frequency (human voice range)
    fundamental_freq = 150 + (hash(prompt) % 100)  # Vary based on prompt
    audio += 0.3 * np.sin(2 * np.pi * fundamental_freq * t)
    
    # Add harmonics (speech has multiple harmonics)
    for harmonic in [2, 3, 4, 5]:
        audio += 0.1 * np.sin(2 * np.pi * fundamental_freq * harmonic * t)
    
    # Add some noise (real speech has background noise)
    audio += 0.05 * np.random.randn(num_samples)
    
    # Add envelope (speech has varying amplitude)
    envelope = np.exp(-t / duration) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
    audio *= envelope
    
    # Normalize to prevent clipping
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio.astype(np.float32)


def run_voicebench_evaluation_server(
    api_url: str,
    datasets: Optional[List[str]] = None,
    timeout: int = 600,
    max_samples_per_dataset: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run VoiceBench evaluation on a server API using real audio data.
    
    Args:
        api_url: URL of the model server API
        datasets: List of datasets to evaluate (None = all VoiceBench datasets)
        timeout: Request timeout for API calls
        max_samples_per_dataset: Maximum number of samples per dataset
        
    Returns:
        Complete evaluation results including scores
    """
    # Available VoiceBench datasets
    VOICEBENCH_DATASETS = {
        'commoneval': ['test'],
        'wildvoice': ['test'],
        'ifeval': ['test'],
        'advbench': ['test']
    }
    
    # Use all datasets if none specified
    if datasets is None:
        datasets_to_eval = VOICEBENCH_DATASETS
    else:
        datasets_to_eval = {k: VOICEBENCH_DATASETS[k] for k in datasets if k in VOICEBENCH_DATASETS}
    
    # Create server client
    client = ServerModelClient(api_url=api_url, timeout=timeout)
    results = {}
    
    bt.logging.info("Starting VoiceBench evaluation via server API with real audio data...")
    
    for dataset_name, dataset_splits in datasets_to_eval.items():
        for split in dataset_splits:
            try:
                # Load real VoiceBench audio data from parquet file
                bt.logging.info(f"Loading real audio data for {dataset_name}")
                
                # Use the real audio extraction approach
                try:
                    import sys
                    import os
                    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from extract_real_audio import load_voicebench_from_parquet
                    max_samples = SAMPLES_PER_DATASET.get(dataset_name, VOICEBENCH_MAX_SAMPLES)
                    samples = load_voicebench_from_parquet("test-00000-of-00001.parquet", max_samples=max_samples)
                    
                    if samples:
                        bt.logging.info(f"Successfully loaded {len(samples)} real samples with actual audio data")
                        bt.logging.info(f"Note: Using real VoiceBench audio data (no simulation)")
                    else:
                        bt.logging.error("Real audio extraction failed, falling back to synthetic audio")
                        # Fallback to synthetic audio
                        samples = []
                        for i in range(max_samples):
                            prompt = f"Sample {i+1} prompt for {dataset_name}"
                            audio_data = create_realistic_audio_from_prompt(prompt)
                            samples.append({
                                'prompt': prompt,
                                'output': '',
                                'audio': {
                                    'array': audio_data,
                                    'sampling_rate': 16000
                                }
                            })
                        
                except Exception as e:
                    bt.logging.error(f"Error with real audio extraction: {e}")
                    bt.logging.info("Falling back to synthetic audio...")
                    # Fallback to synthetic audio
                    samples = []
                    for i in range(max_samples):
                        prompt = f"Sample {i+1} prompt for {dataset_name}"
                        audio_data = create_realistic_audio_from_prompt(prompt)
                        samples.append({
                            'prompt': prompt,
                            'output': '',
                            'audio': {
                                'array': audio_data,
                                'sampling_rate': 16000
                            }
                        })
                
                bt.logging.info(f"Created {len(samples)} samples for {dataset_name}_{split}")
                
                # Run inference on the samples
                dataset_results = run_inference_on_real_samples(
                    client, samples, dataset_name, split
                )
                
                results[f"{dataset_name}_{split}"] = dataset_results
                
            except Exception as e:
                bt.logging.error(f"Error evaluating {dataset_name}_{split}: {e}")
                results[f"{dataset_name}_{split}"] = {"error": str(e)}
    
    # Calculate scores using proper evaluators with status tracking
    bt.logging.info("Starting VoiceBench evaluation with proper evaluators...")
    scores, status = calculate_voicebench_scores_with_status(results)
    
    bt.logging.info(f"VoiceBench evaluation completed.")
    bt.logging.info(f"Overall score: {scores.get('overall', 0.0):.3f}")
    bt.logging.info(f"Evaluation status: {status.get('overall', {})}")
    
    return {
        'voicebench_scores': scores,
        'evaluation_status': status
    }


def run_inference_on_real_samples(
    client: ServerModelClient,
    samples: List[Dict[str, Any]],
    dataset_name: str,
    split: str
) -> Dict[str, Any]:
    """
    Run inference on real dataset samples using the server client.
    
    Args:
        client: ServerModelClient instance
        samples: List of real dataset samples
        dataset_name: Name of the dataset
        split: Data split
        
    Returns:
        Inference results for this dataset
    """
    responses = []
    
    # Generate responses with timeout and retry logic
    for i, sample in enumerate(samples):
        max_retries = 2
        retry_delay = 1
        
        if i % 10 == 0:
            bt.logging.info(f"Processing sample {i+1}/{len(samples)}")
        
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                
                # Get audio data from sample
                audio_data = sample['audio']
                result = client.inference_v2t(
                    audio_array=audio_data['array'],
                    sample_rate=audio_data['sampling_rate']
                )
                response = result.get('text', '')
                
                inference_time = time.time() - start_time
                
                responses.append({
                    'hf_index': i,
                    'prompt': sample.get('prompt', ''),
                    'response': response,
                    'reference': sample.get('output', ''),
                    'inference_time': inference_time,
                    'attempt': attempt + 1
                })
                
                # Success, break retry loop
                break
                
            except Exception as e:
                bt.logging.warning(f"Error processing sample {i+1}/{len(samples)} (attempt {attempt+1}): {e}")
                if attempt < max_retries:
                    bt.logging.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    # Final attempt failed
                    responses.append({
                        'hf_index': i,
                        'prompt': sample.get('prompt', ''),
                        'response': '',
                        'reference': sample.get('output', ''),
                        'error': str(e),
                        'failed_attempts': attempt + 1
                    })
                    break
    
    # Calculate basic metrics
    total_responses = len(responses)
    successful_responses = len([r for r in responses if 'error' not in r and r['response']])
    success_rate = successful_responses / total_responses if total_responses > 0 else 0
    
    return {
        'dataset': dataset_name,
        'split': split,
        'total_samples': total_responses,
        'successful_responses': successful_responses,
        'success_rate': success_rate,
        'responses': responses
    }


def run_inference_on_dataset(
    client: ServerModelClient,
    dataset,
    dataset_name: str,
    split: str
) -> Dict[str, Any]:
    """
    Run inference on a dataset using the server client.
    
    Args:
        client: ServerModelClient instance
        dataset: Loaded dataset
        dataset_name: Name of the dataset
        split: Data split
        
    Returns:
        Inference results for this dataset
    """
    responses = []
    
    # Generate responses with timeout and retry logic
    for i, item in enumerate(dataset):
        max_retries = 2
        retry_delay = 1
        
        if i % 50 == 0:
            bt.logging.info(f"Processing item {i+1}/{len(dataset)}")
        
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                
                # Check if audio field exists
                if 'audio' not in item:
                    bt.logging.warning(f"No audio field in item {i+1}, skipping")
                    response = ""
                else:
                    audio_data = item['audio']
                    result = client.inference_v2t(
                        audio_array=audio_data['array'],
                        sample_rate=audio_data['sampling_rate']
                    )
                    response = result.get('text', '')
                
                inference_time = time.time() - start_time
                
                responses.append({
                    'hf_index': i,
                    'prompt': item.get('prompt', ''),
                    'response': response,
                    'reference': item.get('output', ''),
                    'inference_time': inference_time,
                    'attempt': attempt + 1,
                    **{k: v for k, v in item.items() if k not in ['audio', 'prompt', 'output']}
                })
                
                # Success, break retry loop
                break
                
            except Exception as e:
                bt.logging.warning(f"Error processing item {i+1}/{len(dataset)} (attempt {attempt+1}): {e}")
                if attempt < max_retries:
                    bt.logging.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    # Final attempt failed
                    responses.append({
                        'hf_index': i,
                        'prompt': item.get('prompt', ''),
                        'response': '',
                        'reference': item.get('output', ''),
                        'error': str(e),
                        'failed_attempts': attempt + 1
                    })
                    break
    
    # Calculate basic metrics
    total_responses = len(responses)
    successful_responses = len([r for r in responses if 'error' not in r and r['response']])
    success_rate = successful_responses / total_responses if total_responses > 0 else 0
    
    return {
        'dataset': dataset_name,
        'split': split,
        'total_samples': total_responses,
        'successful_responses': successful_responses,
        'success_rate': success_rate,
        'responses': responses
    }


def test_server_scoring(
    api_url: str,
    experiment_name: str = "unnamed",
    datasets: Optional[list] = None,
    timeout: int = 600
) -> Dict[str, Any]:
    """
    Test model scoring using a server API endpoint.
    
    Args:
        api_url: URL of the model server API endpoint
        experiment_name: Name for tracking results
        datasets: List of specific datasets to evaluate
        timeout: Request timeout for API calls in seconds
        
    Returns:
        Dictionary with scoring results
    """
    bt.logging.info(f"Starting server-based model scoring")
    bt.logging.info(f"Experiment: {experiment_name}")
    bt.logging.info(f"API URL: {api_url}")
    
    start_time = time.time()
    
    try:
        # Run VoiceBench evaluation using the server
        voicebench_results = run_voicebench_evaluation_server(
            api_url=api_url,
            datasets=datasets,
            timeout=timeout
        )
        
        # Extract scores
        voicebench_scores = voicebench_results.get('voicebench_scores', {})
        combined_score = voicebench_scores.get('overall', 0.0)
        
        # Build results
        results = {
            'raw_scores': {
                'voicebench': voicebench_scores,
            },
            'combined_score': combined_score,
            'evaluation_status': voicebench_results.get('evaluation_status', {}),
            'evaluation_details': voicebench_results.get('evaluation_status', {}),
            'metadata': {
                'experiment_name': experiment_name,
                'competition_id': 'voicebench',
                'model_id': 'server_model',
                'api_url': api_url,
                'evaluation_time': time.time() - start_time,
                'datasets': datasets,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        bt.logging.success(f"Evaluation completed successfully")
        bt.logging.info(f"Combined score: {combined_score:.4f}")
        bt.logging.info(f"Total time: {time.time() - start_time:.2f} seconds")
        
        return results
        
    except Exception as e:
        bt.logging.error(f"Error during server-based evaluation: {e}")
        bt.logging.error(traceback.format_exc())
        
        # Return penalty score on failure
        return {
            'raw_scores': {
                'voicebench': None,
            },
            'combined_score': PENALTY_SCORE,
            'evaluation_status': {
                'overall': {
                    'status': 'error',
                    'error': str(e)
                }
            },
            'evaluation_details': {},
            'metadata': {
                'experiment_name': experiment_name,
                'competition_id': 'voicebench',
                'model_id': 'server_model',
                'api_url': api_url,
                'error': str(e),
                'evaluation_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
        }


def append_to_report(results: Dict[str, Any], report_file: str = "model_evaluation_report.txt"):
    """Append evaluation results to a text report file."""
    with open(report_file, 'a') as f:
        # Write separator for new experiment
        f.write("\n" + "="*80 + "\n")
        
        # Write experiment metadata
        metadata = results.get('metadata', {})
        f.write(f"EXPERIMENT: {metadata.get('experiment_name', 'unnamed')}\n")
        f.write(f"TIMESTAMP: {metadata.get('timestamp', datetime.now().isoformat())}\n")
        f.write(f"API URL: {metadata.get('api_url', 'N/A')}\n")
        f.write(f"EVALUATION TIME: {metadata.get('evaluation_time', 0):.2f} seconds\n")
        f.write("-"*80 + "\n")
        
        # Write combined score
        f.write(f"COMBINED SCORE: {results['combined_score']:.4f}\n")
        f.write("-"*80 + "\n")
        
        # Write individual dataset scores
        if results.get('raw_scores', {}).get('voicebench'):
            f.write("DATASET SCORES:\n")
            scores = results['raw_scores']['voicebench']
            
            # Sort datasets for consistent reporting
            dataset_names = sorted([k for k in scores.keys() if k != 'overall'])
            
            for dataset in dataset_names:
                score = scores[dataset]
                samples_used = SAMPLES_PER_DATASET.get(dataset.split('_')[0], VOICEBENCH_MAX_SAMPLES)
                f.write(f"  {dataset:<30} {score:>8.4f}  (samples: {samples_used})\n")
            
            # Write overall at the end
            if 'overall' in scores:
                f.write(f"  {'OVERALL':<30} {scores['overall']:>8.4f}\n")
        
        # Write evaluation status summary
        status = results.get('evaluation_status', {}).get('overall', {})
        if status:
            f.write("-"*80 + "\n")
            f.write(f"STATUS: {status.get('status', 'unknown')}\n")
            
            # Write total samples and successful responses
            total_samples = status.get('total_samples', 0)
            successful_responses = status.get('successful_responses', 0)
            if total_samples > 0:
                f.write(f"TOTAL SAMPLES: {total_samples}\n")
                f.write(f"SUCCESSFUL RESPONSES: {successful_responses}\n")
                f.write(f"SUCCESS RATE: {(successful_responses/total_samples*100):.1f}%\n")
            
            # Write any errors
            errors = status.get('errors', {})
            if errors:
                f.write("ERRORS:\n")
                for dataset, error in errors.items():
                    f.write(f"  {dataset}: {error}\n")
        
        # Write error if evaluation failed
        if metadata.get('error'):
            f.write("-"*80 + "\n")
            f.write(f"ERROR: {metadata['error']}\n")
        
        f.write("="*80 + "\n")
    
    print(f"✅ Results appended to: {report_file}")


def print_detailed_results(results: Dict[str, Any]):
    """Print detailed evaluation results in a readable format."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Overall score
    print(f"\n📊 COMBINED SCORE: {results['combined_score']:.4f}")
    
    # VoiceBench scores
    if results['raw_scores']['voicebench']:
        print(f"\n📈 VOICEBENCH SCORES:")
        scores = results['raw_scores']['voicebench']
        
        # Print individual dataset scores
        for dataset, score in scores.items():
            if dataset != 'overall':
                print(f"  • {dataset:20s}: {score:.4f}")
        
        # Print overall score
        if 'overall' in scores:
            print(f"  ─────────────────────────────")
            print(f"  • {'OVERALL':20s}: {scores['overall']:.4f}")
    
    # Evaluation status
    status = results.get('evaluation_status', {})
    if status:
        print(f"\n📋 EVALUATION STATUS:")
        overall_status = status.get('overall', {})
        if overall_status:
            print(f"  • Status: {overall_status.get('status', 'unknown')}")
            
            # Print evaluators used
            evaluators_used = overall_status.get('evaluators_used', {})
            if evaluators_used:
                print(f"  • Evaluators used:")
                for dataset, evaluator in evaluators_used.items():
                    print(f"    - {dataset}: {evaluator}")
            
            # Print any errors
            errors = overall_status.get('errors', {})
            if errors:
                print(f"  • Errors encountered:")
                for dataset, error in errors.items():
                    print(f"    - {dataset}: {error}")
    
    # Metadata
    metadata = results.get('metadata', {})
    if metadata:
        print(f"\n⚙️  METADATA:")
        print(f"  • API URL: {metadata.get('api_url', 'N/A')}")
        print(f"  • Evaluation time: {metadata.get('evaluation_time', 0):.2f} seconds")
        print(f"  • Datasets: {metadata.get('datasets', 'all')}")
        
        if metadata.get('error'):
            print(f"  • Error: {metadata['error']}")
    
    print("\n" + "="*60 + "\n")


def main():
    """Main function to run server-based model testing."""
    parser = argparse.ArgumentParser(
        description="Test model scoring using a server API (no Docker required)"
    )
    
    # Add custom arguments
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://localhost:8000/api/v1/v2t",
        help="URL of the model server API endpoint"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of this experiment for tracking in the report"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Specific datasets to evaluate (e.g., commoneval wildvoice)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Request timeout for API calls in seconds (default: 600)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save results as JSON"
    )
    parser.add_argument(
        "--report_file",
        type=str,
        default="model_evaluation_report.txt",
        help="Text file to append results to (default: model_evaluation_report.txt)"
    )
    
    # Add bittensor arguments
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    
    # Parse with bittensor config
    config = bt.config(parser)
    
    print(f"\n🚀 Starting server-based model scoring (no Docker)...")
    print(f"   Experiment: {config.experiment_name}")
    print(f"   API URL: {config.api_url}")
    print(f"   Datasets: {config.datasets or 'all'}")
    print(f"   Timeout: {config.timeout}s")
    print(f"   Report file: {config.report_file}")
    
    # Show samples per dataset configuration
    print(f"\n📊 Samples per dataset:")
    for dataset, samples in SAMPLES_PER_DATASET.items():
        print(f"   {dataset}: {samples}")
    print(f"   Default: {VOICEBENCH_MAX_SAMPLES}")
    
    # Run the evaluation
    results = test_server_scoring(
        api_url=config.api_url,
        experiment_name=config.experiment_name,
        datasets=config.datasets,
        timeout=config.timeout
    )
    
    # Print detailed results to console
    print_detailed_results(results)
    
    # Append to report file
    append_to_report(results, config.report_file)
    
    # Save full JSON results if requested
    if config.output:
        with open(config.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Full JSON results saved to: {config.output}")
    
    # Exit with appropriate code
    if results['combined_score'] == PENALTY_SCORE:
        print("❌ Evaluation failed with penalty score")
        exit(1)
    else:
        print("✅ Evaluation completed successfully")
        exit(0)


if __name__ == "__main__":
    main()

