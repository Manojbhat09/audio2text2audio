#!/usr/bin/env python3
"""
Test the model with the exact conversation format used during training.
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

def test_conversation_format():
    """Test the model with the exact conversation format used during training."""
    
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
    
    # Test with the exact conversation format from training
    test_conversations = [
        [
            {"role": "system", "content": "You are an expert at analyzing speech transcripts and identifying which type of evaluation dataset they come from.\n\nYour task is to classify transcripts into one of these three categories:\n- ifeval: Instruction following evaluation with complex reasoning tasks\n- commoneval: Common evaluation benchmark for natural language processing\n- wildvoice: Wild voice data with diverse speaking styles and accents\n\nAnalyze the content, style, and structure of the transcript to determine the most likely source dataset.\nRespond with only the dataset name: ifeval, commoneval, or wildvoice."},
            {"role": "user", "content": "Please classify this transcript into one of the three categories: ifeval, commoneval, or wildvoice.\n\nTranscript: \"This is a test audio for ifeval dataset with instruction following tasks.\"\n\nWhich dataset category does this transcript most likely come from? Answer with only the dataset name: ifeval, commoneval, or wildvoice."}
        ],
        [
            {"role": "system", "content": "You are an expert at analyzing speech transcripts and identifying which type of evaluation dataset they come from.\n\nYour task is to classify transcripts into one of these three categories:\n- ifeval: Instruction following evaluation with complex reasoning tasks\n- commoneval: Common evaluation benchmark for natural language processing\n- wildvoice: Wild voice data with diverse speaking styles and accents\n\nAnalyze the content, style, and structure of the transcript to determine the most likely source dataset.\nRespond with only the dataset name: ifeval, commoneval, or wildvoice."},
            {"role": "user", "content": "Please classify this transcript into one of the three categories: ifeval, commoneval, or wildvoice.\n\nTranscript: \"This is a common evaluation task for language understanding.\"\n\nWhich dataset category does this transcript most likely come from? Answer with only the dataset name: ifeval, commoneval, or wildvoice."}
        ],
        [
            {"role": "system", "content": "You are an expert at analyzing speech transcripts and identifying which type of evaluation dataset they come from.\n\nYour task is to classify transcripts into one of these three categories:\n- ifeval: Instruction following evaluation with complex reasoning tasks\n- commoneval: Common evaluation benchmark for natural language processing\n- wildvoice: Wild voice data with diverse speaking styles and accents\n\nAnalyze the content, style, and structure of the transcript to determine the most likely source dataset.\nRespond with only the dataset name: ifeval, commoneval, or wildvoice."},
            {"role": "user", "content": "Please classify this transcript into one of the three categories: ifeval, commoneval, or wildvoice.\n\nTranscript: \"This is a wild voice sample with natural speech patterns.\"\n\nWhich dataset category does this transcript most likely come from? Answer with only the dataset name: ifeval, commoneval, or wildvoice."}
        ]
    ]
    
    for i, conversation in enumerate(test_conversations, 1):
        logger.info(f"\n=== Test {i} ===")
        
        # Convert conversation to text format
        text_prompt = ""
        for message in conversation:
            if message["role"] == "system":
                text_prompt += f"System: {message['content']}\n\n"
            elif message["role"] == "user":
                text_prompt += f"User: {message['content']}\n\n"
        
        text_prompt += "Assistant: "
        
        logger.info(f"Full prompt:\n{text_prompt}")
        
        # Tokenize
        inputs = tokenizer(text_prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        try:
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=2,
                )
                
                response = tokenizer.decode(generated[0], skip_special_tokens=True)
                logger.info(f"Full response: {response}")
                
                # Extract just the assistant's response
                assistant_response = response[len(text_prompt):].strip()
                logger.info(f"Assistant response: '{assistant_response}'")
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")

if __name__ == "__main__":
    test_conversation_format()
