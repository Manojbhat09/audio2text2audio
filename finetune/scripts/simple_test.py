#!/usr/bin/env python3
"""
Simple test script to verify basic functionality

This script tests basic imports and functionality without the problematic unsloth imports.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_basic_imports():
    """Test basic imports."""
    logger.info("Testing basic imports...")
    
    try:
        import torch
        logger.info(f"✓ PyTorch: {torch.__version__}")
    except ImportError as e:
        logger.error(f"✗ PyTorch import failed: {e}")
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
    
    try:
        import numpy as np
        logger.info(f"✓ NumPy: {np.__version__}")
    except ImportError as e:
        logger.error(f"✗ NumPy import failed: {e}")
        return False
    
    return True


def test_whisper_transcriber():
    """Test Whisper transcriber without unsloth dependencies."""
    logger.info("Testing Whisper transcriber...")
    
    try:
        from src.whisper_transcriber import WhisperTranscriber
        
        # Test with actual model path
        model_path = "/home/mbhat/omegalabs-anytoany-bittensor/elephant-04/models/wpt/wpt.pt"
        transcriber = WhisperTranscriber(model_path)
        logger.info("✓ WhisperTranscriber loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ WhisperTranscriber test failed: {e}")
        return False


def test_reward_functions():
    """Test reward functions."""
    logger.info("Testing reward functions...")
    
    try:
        from src.standalone_reward_functions import DatasetClassificationRewards
        
        rewards = DatasetClassificationRewards()
        logger.info("✓ Reward functions imported successfully")
        
        # Test with dummy data
        completions = [
            [{"role": "assistant", "content": "ifeval"}],
            [{"role": "assistant", "content": "commoneval"}]
        ]
        answers = ["ifeval", "commoneval"]
        
        scores = rewards.exact_match_reward(completions, answers)
        logger.info(f"✓ Reward function test passed: {scores}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Reward functions test failed: {e}")
        return False


def test_configuration():
    """Test configuration loading."""
    logger.info("Testing configuration...")
    
    try:
        from configs.training_config import get_small_config
        
        config = get_small_config()
        logger.info(f"✓ Configuration loaded: {config.model_name}")
        logger.info(f"Target datasets: {config.target_datasets}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False


def test_whisper_model_exists():
    """Test if Whisper model file exists."""
    logger.info("Testing Whisper model file...")
    
    model_path = "/home/mbhat/omegalabs-anytoany-bittensor/elephant-04/models/wpt/wpt.pt"
    
    if Path(model_path).exists():
        logger.info(f"✓ Whisper model found at {model_path}")
        return True
    else:
        logger.warning(f"⚠ Whisper model not found at {model_path}")
        return False


def main():
    """Run all tests."""
    logger.info("🧪 Starting simple finetune tests...")
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Configuration", test_configuration),
        ("Reward Functions", test_reward_functions),
        ("Whisper Transcriber", test_whisper_transcriber),
        ("Whisper Model File", test_whisper_model_exists),
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
        logger.info("🎉 All basic tests passed! Core functionality is working.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please fix issues before training.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
