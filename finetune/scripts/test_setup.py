#!/usr/bin/env python3
"""
Test script to verify the finetune setup

This script tests all components of the training pipeline to ensure
everything is working correctly before starting actual training.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.whisper_transcriber import create_transcriber
from src.dataset_loader import create_dataset_loader
from src.reward_functions import create_reward_functions
from src.grpo_trainer import create_trainer
from configs.training_config import get_small_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all imports work correctly."""
    logger.info("Testing imports...")
    
    try:
        import torch
        logger.info(f"✓ PyTorch: {torch.__version__}")
    except ImportError as e:
        logger.error(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        logger.info(f"✓ Transformers: {transformers.__version__}")
    except ImportError as e:
        logger.error(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import datasets
        logger.info(f"✓ Datasets: {datasets.__version__}")
    except ImportError as e:
        logger.error(f"✗ Datasets import failed: {e}")
        return False
    
    try:
        import trl
        logger.info(f"✓ TRL: {trl.__version__}")
    except ImportError as e:
        logger.error(f"✗ TRL import failed: {e}")
        return False
    
    try:
        import unsloth
        logger.info(f"✓ Unsloth: {unsloth.__version__}")
    except ImportError as e:
        logger.error(f"✗ Unsloth import failed: {e}")
        return False
    
    try:
        import whisper
        logger.info(f"✓ Whisper: {whisper.__version__}")
    except ImportError as e:
        logger.error(f"✗ Whisper import failed: {e}")
        return False
    
    try:
        import librosa
        logger.info(f"✓ Librosa: {librosa.__version__}")
    except ImportError as e:
        logger.error(f"✗ Librosa import failed: {e}")
        return False
    
    return True


def test_whisper_transcriber():
    """Test Whisper transcriber."""
    logger.info("Testing Whisper transcriber...")
    
    try:
        config = get_small_config()
        transcriber = create_transcriber(config.whisper_model_path)
        
        if transcriber.is_loaded():
            logger.info("✓ Whisper transcriber loaded successfully")
            logger.info(f"Model info: {transcriber.get_model_info()}")
            return True
        else:
            logger.error("✗ Whisper transcriber failed to load")
            return False
            
    except Exception as e:
        logger.error(f"✗ Whisper transcriber test failed: {e}")
        return False


def test_reward_functions():
    """Test reward functions."""
    logger.info("Testing reward functions...")
    
    try:
        reward_funcs = create_reward_functions()
        logger.info(f"✓ Created {len(reward_funcs)} reward functions")
        
        # Test with dummy data
        completions = [
            [{"role": "assistant", "content": "ifeval"}],
            [{"role": "assistant", "content": "commoneval"}]
        ]
        answers = ["ifeval", "commoneval"]
        
        for i, func in enumerate(reward_funcs):
            scores = func(completions, answers)
            logger.info(f"✓ Reward function {i+1} returned {len(scores)} scores")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Reward functions test failed: {e}")
        return False


def test_configuration():
    """Test configuration loading."""
    logger.info("Testing configuration...")
    
    try:
        config = get_small_config()
        logger.info(f"✓ Configuration loaded: {config.model_name}")
        logger.info(f"Target datasets: {config.target_datasets}")
        logger.info(f"Samples per dataset: {config.samples_per_dataset}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False


def test_model_loading():
    """Test model loading (without full training)."""
    logger.info("Testing model loading...")
    
    try:
        config = get_small_config()
        config.samples_per_dataset = 1  # Minimal for testing
        
        # This will test model loading but not training
        trainer = create_trainer(config)
        
        if trainer.model is not None and trainer.tokenizer is not None:
            logger.info("✓ Model and tokenizer loaded successfully")
            return True
        else:
            logger.error("✗ Model or tokenizer failed to load")
            return False
            
    except Exception as e:
        logger.error(f"✗ Model loading test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("🧪 Starting finetune setup tests...")
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Reward Functions", test_reward_functions),
        ("Whisper Transcriber", test_whisper_transcriber),
        ("Model Loading", test_model_loading),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} test passed")
            else:
                logger.error(f"❌ {test_name} test failed")
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
    
    logger.info(f"\n🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Setup is ready for training.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please fix issues before training.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
