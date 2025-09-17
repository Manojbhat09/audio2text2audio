#!/usr/bin/env python3
"""
Debug script to see what the trained model is actually generating.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Enable synchronous CUDA execution for better debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_generation():
    """Test what the model actually generates."""
    
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
    
    # Test prompts
    test_prompts = [
        "Classify this audio transcript as one of these datasets: ifeval, commoneval, wildvoice. Transcript: This is a test audio for ifeval dataset with instruction following tasks.",
        "Classify this audio transcript as one of these datasets: ifeval, commoneval, wildvoice. Transcript: This is a common evaluation task for language understanding.",
        "Classify this audio transcript as one of these datasets: ifeval, commoneval, wildvoice. Transcript: This is a wild voice sample with natural speech patterns."
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        logger.info(f"\n=== Test {i} ===")
        logger.info(f"Prompt: {prompt}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate with different parameters
        generation_configs = [
            {"max_new_tokens": 10, "do_sample": False, "temperature": 1.0, "name": "Greedy"},
            {"max_new_tokens": 20, "do_sample": True, "temperature": 0.7, "top_p": 0.9, "name": "Sampling"},
            {"max_new_tokens": 30, "do_sample": True, "temperature": 1.0, "top_p": 0.95, "name": "High Temp"},
        ]
        
        for config in generation_configs:
            try:
                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=config["max_new_tokens"],
                        do_sample=config["do_sample"],
                        temperature=config.get("temperature", 1.0),
                        top_p=config.get("top_p", 1.0),
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=2,
                    )
                    
                    response = tokenizer.decode(generated[0], skip_special_tokens=True)
                    logger.info(f"{config['name']}: {response}")
                    
            except Exception as e:
                logger.error(f"{config['name']} failed: {e}")
        
        # Also try manual generation to see logits
        try:
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                logger.info(f"Logits shape: {logits.shape}")
                logger.info(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
                
                # Get next token probabilities
                next_logits = logits[0, -1, :]
                probs = torch.softmax(next_logits, dim=-1)
                top_tokens = torch.topk(probs, 10)
                
                logger.info("Top 10 next tokens:")
                for j, (prob, idx) in enumerate(zip(top_tokens.values, top_tokens.indices)):
                    token = tokenizer.decode([idx.item()])
                    logger.info(f"  {j+1}. {token} (prob: {prob:.4f})")
                    
        except Exception as e:
            logger.error(f"Manual generation failed: {e}")

if __name__ == "__main__":
    test_model_generation()
