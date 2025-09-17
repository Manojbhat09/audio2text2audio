#!/usr/bin/env python3
"""
Reward Functions for GRPO Training

Implements reward functions for training the dataset classifier using GRPO.
These functions evaluate the quality of model responses for dataset classification.
"""

from __future__ import annotations

import logging
import re
from typing import List, Dict, Any, Callable
from collections import Counter

logger = logging.getLogger(__name__)


class DatasetClassificationRewards:
    """Collection of reward functions for dataset classification training."""
    
    # Valid dataset names
    VALID_DATASETS = {"ifeval", "commoneval", "wildvoice"}
    
    def __init__(self, reward_weights: Dict[str, float] = None):
        """
        Initialize reward functions with weights.
        
        Args:
            reward_weights: Dictionary mapping reward function names to weights
        """
        self.reward_weights = reward_weights or {
            "exact_match": 3.0,
            "format_compliance": 2.0,
            "confidence": 1.0,
            "dataset_balance": 0.5
        }
    
    def exact_match_reward(self, completions: List[List[Dict[str, str]]], answer: List[str], **kwargs) -> List[float]:
        """
        Reward for exact dataset name match.
        
        Args:
            completions: List of completions from the model
            answer: Ground truth answers
            
        Returns:
            List of reward scores
        """
        scores = []
        
        for completion in completions:
            if not completion or not completion[0].get("content"):
                scores.append(0.0)
                continue
                
            response = completion[0]["content"].strip().lower()
            target = answer[0].lower() if isinstance(answer, list) else answer.lower()
            
            # Exact match gets full reward
            if response == target:
                scores.append(3.0)
            # Partial match (contains target) gets partial reward
            elif target in response:
                scores.append(1.5)
            else:
                scores.append(0.0)
        
        return scores
    
    def format_compliance_reward(self, completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """
        Reward for proper response format (single dataset name).
        
        Args:
            completions: List of completions from the model
            
        Returns:
            List of reward scores
        """
        scores = []
        
        for completion in completions:
            if not completion or not completion[0].get("content"):
                scores.append(0.0)
                continue
                
            response = completion[0]["content"].strip().lower()
            words = response.split()
            
            # Single word response that's a valid dataset name
            if len(words) == 1 and words[0] in self.VALID_DATASETS:
                scores.append(2.0)
            # Contains valid dataset name but has extra words
            elif any(dataset in response for dataset in self.VALID_DATASETS):
                scores.append(1.0)
            # Long response (penalize verbosity)
            elif len(words) > 10:
                scores.append(-0.5)
            else:
                scores.append(0.0)
        
        return scores
    
    def confidence_reward(self, completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """
        Reward for confident, decisive responses.
        
        Args:
            completions: List of completions from the model
            
        Returns:
            List of reward scores
        """
        uncertain_phrases = [
            "not sure", "maybe", "possibly", "might be", "could be",
            "i think", "probably", "perhaps", "unclear", "uncertain"
        ]
        
        scores = []
        
        for completion in completions:
            if not completion or not completion[0].get("content"):
                scores.append(0.0)
                continue
                
            response = completion[0]["content"].lower()
            
            # Penalize uncertain language
            uncertainty_penalty = sum(0.5 for phrase in uncertain_phrases if phrase in response)
            score = max(0.0, 1.0 - uncertainty_penalty)
            scores.append(score)
        
        return scores
    
    def dataset_balance_reward(self, completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """
        Reward for balanced predictions across datasets.
        
        Args:
            completions: List of completions from the model
            
        Returns:
            List of reward scores
        """
        # Extract predictions
        predictions = []
        for completion in completions:
            if completion and completion[0].get("content"):
                response = completion[0]["content"].strip().lower()
                if response in self.VALID_DATASETS:
                    predictions.append(response)
        
        if not predictions:
            return [0.0] * len(completions)
        
        # Calculate distribution
        distribution = Counter(predictions)
        total = len(predictions)
        
        # Ideal distribution (equal for all datasets)
        ideal_count = total / len(self.VALID_DATASETS)
        
        # Calculate balance score (lower is better)
        balance_score = sum(abs(count - ideal_count) for count in distribution.values()) / total
        
        # Convert to reward (higher is better)
        balance_reward = max(0.0, 1.0 - balance_score)
        
        # Return same score for all completions
        return [balance_reward] * len(completions)
    
    def length_penalty_reward(self, completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """
        Penalty for responses that are too long or too short.
        
        Args:
            completions: List of completions from the model
            
        Returns:
            List of reward scores
        """
        scores = []
        
        for completion in completions:
            if not completion or not completion[0].get("content"):
                scores.append(0.0)
                continue
                
            response = completion[0]["content"].strip()
            word_count = len(response.split())
            
            # Optimal length is 1-3 words
            if word_count == 1:
                scores.append(1.0)  # Perfect
            elif word_count <= 3:
                scores.append(0.8)  # Good
            elif word_count <= 10:
                scores.append(0.5)  # Acceptable
            else:
                scores.append(0.0)  # Too verbose
        
        return scores
    
    def get_all_reward_functions(self) -> List[Callable]:
        """Get all reward functions as a list."""
        return [
            self.exact_match_reward,
            self.format_compliance_reward,
            self.confidence_reward,
            self.dataset_balance_reward,
            self.length_penalty_reward
        ]
    
    def get_weighted_reward_functions(self) -> List[Callable]:
        """Get reward functions with weights applied."""
        functions = self.get_all_reward_functions()
        weights = [
            self.reward_weights.get("exact_match", 1.0),
            self.reward_weights.get("format_compliance", 1.0),
            self.reward_weights.get("confidence", 1.0),
            self.reward_weights.get("dataset_balance", 1.0),
            self.reward_weights.get("length_penalty", 0.5)
        ]
        
        # Create weighted functions
        weighted_functions = []
        for func, weight in zip(functions, weights):
            def make_weighted_func(f, w):
                def weighted_func(*args, **kwargs):
                    scores = f(*args, **kwargs)
                    return [s * w for s in scores]
                return weighted_func
            
            weighted_functions.append(make_weighted_func(func, weight))
        
        return weighted_functions


def create_reward_functions(reward_weights: Dict[str, float] = None) -> List[Callable]:
    """
    Create reward functions for GRPO training.
    
    Args:
        reward_weights: Dictionary mapping reward function names to weights
        
    Returns:
        List of reward functions
    """
    rewards = DatasetClassificationRewards(reward_weights)
    return rewards.get_weighted_reward_functions()


def test_reward_functions():
    """Test the reward functions with sample data."""
    # Sample completions
    completions = [
        [{"role": "assistant", "content": "ifeval"}],
        [{"role": "assistant", "content": "commoneval"}],
        [{"role": "assistant", "content": "I think it might be wildvoice"}],
        [{"role": "assistant", "content": "This is definitely ifeval dataset"}],
        [{"role": "assistant", "content": "unknown"}]
    ]
    
    answers = ["ifeval", "commoneval", "wildvoice", "ifeval", "ifeval"]
    
    # Test reward functions
    rewards = DatasetClassificationRewards()
    
    print("Testing reward functions...")
    print(f"Exact match: {rewards.exact_match_reward(completions, answers)}")
    print(f"Format compliance: {rewards.format_compliance_reward(completions)}")
    print(f"Confidence: {rewards.confidence_reward(completions)}")
    print(f"Length penalty: {rewards.length_penalty_reward(completions)}")
    
    # Test weighted functions
    weighted_funcs = create_reward_functions()
    print(f"Number of weighted functions: {len(weighted_funcs)}")


if __name__ == "__main__":
    test_reward_functions()





