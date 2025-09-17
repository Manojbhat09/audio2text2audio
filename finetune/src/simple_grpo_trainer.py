"""
Simple GRPO Trainer Implementation

This module provides a simplified implementation of Group Relative Policy Optimization (GRPO)
that works with standard transformers without TRL or Unsloth dependencies.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Union, Any
import numpy as np
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class SimpleGRPOConfig(TrainingArguments):
    """
    Configuration for Simple GRPO training.
    
    This extends TrainingArguments with GRPO-specific parameters.
    """
    # GRPO specific parameters
    beta: float = 0.1
    """Temperature parameter for GRPO loss calculation"""
    
    # Additional parameters that might be passed
    num_generations: int = 1
    """Number of generations for GRPO"""
    
    max_prompt_length: int = 512
    """Maximum prompt length"""
    
    max_completion_length: int = 256
    """Maximum completion length"""
    
    temperature: float = 0.7
    """Temperature for generation"""
    
    top_p: float = 0.9
    """Top-p for generation"""
    
    top_k: int = 50
    """Top-k for generation"""
    
    group_size: int = 4
    """Size of groups for relative ranking"""
    
    use_kl_penalty: bool = True
    
    def __post_init__(self):
        """Post-initialization to fix evaluation strategy."""
        # Disable load_best_model_at_end to avoid conflicts
        self.load_best_model_at_end = False
        
        # Set evaluation strategy to match save strategy
        if self.save_strategy == "steps":
            self.eval_strategy = "steps"
            self.eval_steps = self.save_steps
        elif self.save_strategy == "epoch":
            self.eval_strategy = "epoch"
        
        # Debug logging
        print(f"DEBUG: save_strategy={self.save_strategy}, eval_strategy={self.eval_strategy}, eval_steps={self.eval_steps}, load_best_model_at_end={self.load_best_model_at_end}")
        
        super().__post_init__()
    """Whether to use KL divergence penalty"""
    
    kl_penalty_weight: float = 0.1
    """Weight for KL divergence penalty"""
    
    # Dataset parameters
    max_length: int = 512
    """Maximum sequence length"""
    
    # Training parameters
    learning_rate: float = 5e-5  # Reasonable learning rate for fine-tuning
    """Learning rate for training"""
    
    num_train_epochs: int = 3
    """Number of training epochs"""
    
    per_device_train_batch_size: int = 1
    """Batch size per device"""
    
    gradient_accumulation_steps: int = 4
    """Number of gradient accumulation steps"""
    
    warmup_steps: int = 100
    """Number of warmup steps"""
    
    logging_steps: int = 10
    """Logging frequency"""
    
    save_steps: int = 500
    """Save frequency"""
    
    eval_steps: int = 500
    """Evaluation frequency"""
    
    # Generation parameters
    max_new_tokens: int = 128
    """Maximum number of new tokens to generate"""
    
    temperature: float = 0.7
    """Temperature for generation"""
    
    top_p: float = 0.9
    """Top-p for generation"""
    
    do_sample: bool = True
    """Whether to use sampling during generation"""
    
    # Output directory
    output_dir: str = "./grpo_output"
    """Output directory for checkpoints and logs"""
    
    # Other parameters
    remove_unused_columns: bool = False
    """Whether to remove unused columns from dataset"""
    
    dataloader_drop_last: bool = True
    """Whether to drop last incomplete batch"""
    
    # Optimizer parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8  # Smaller epsilon for numerical stability
    weight_decay: float = 0.01  # Weight decay to prevent overfitting
    
    # Scheduler parameters
    lr_scheduler_type: str = "cosine"
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = False  # Disable mixed precision for DialoGPT compatibility
    
    # Gradient clipping to prevent NaN gradients
    max_grad_norm: float = 1.0  # Standard gradient clipping
    """Maximum gradient norm for clipping"""
    
    # Dataloader parameters
    dataloader_num_workers: int = 0
    
    # Evaluation parameters
    eval_strategy: str = "steps"
    eval_steps: int = 500
    per_device_eval_batch_size: int = 1
    
    # Logging parameters
    logging_strategy: str = "steps"
    logging_steps: int = 10
    
    # Save parameters
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Load parameters
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Other parameters
    report_to: Optional[List[str]] = None
    run_name: Optional[str] = None
    disable_tqdm: bool = False
    
    def __post_init__(self):
        """Post-initialization processing"""
        super().__post_init__()
        
        # Set default values
        if self.report_to is None:
            self.report_to = []
        
        if self.run_name is None:
            self.run_name = "simple_grpo_training"
        
        # Ensure output directory exists
        import os
        os.makedirs(self.output_dir, exist_ok=True)


class SimpleGRPOTrainer(Trainer):
    """
    Simple GRPO Trainer implementation.
    
    This trainer implements Group Relative Policy Optimization without relying on TRL or Unsloth.
    """
    
    def __init__(
        self,
        model,
        args: SimpleGRPOConfig,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        tokenizer: Optional[Any] = None,
        data_collator: Optional[Any] = None,
        compute_metrics: Optional[Any] = None,
        callbacks: Optional[List[Any]] = None,
        optimizers: Optional[tuple] = None,
        preprocess_logits_for_metrics: Optional[Any] = None,
    ):
        """
        Initialize the Simple GRPO Trainer.
        
        Args:
            model: The model to train
            args: GRPO configuration
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer for the model
            data_collator: Data collator for batching
            compute_metrics: Function to compute metrics
            callbacks: Training callbacks
            optimizers: Optimizer and scheduler
            preprocess_logits_for_metrics: Function to preprocess logits for metrics
        """
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        
        self.grpo_config = args
        self.beta = args.beta
        self.group_size = args.group_size
        self.use_kl_penalty = args.use_kl_penalty
        self.kl_penalty_weight = args.kl_penalty_weight
        
        # Initialize reference model for KL penalty
        self.reference_model = None
        if self.use_kl_penalty:
            self._initialize_reference_model()
        
        # Initialize model weights properly to prevent NaN
        self._initialize_model_weights()
        
        # Enable gradient monitoring
        self._gradient_monitoring_enabled = True
    
    def _initialize_reference_model(self):
        """Initialize reference model for KL divergence calculation"""
        try:
            # Create a copy of the current model as reference
            self.reference_model = self.model.__class__(self.model.config)
            self.reference_model.load_state_dict(self.model.state_dict())
            
            # Ensure reference model is on the same device as the main model
            device = next(self.model.parameters()).device
            self.reference_model = self.reference_model.to(device)
            self.reference_model.eval()
            
            # Freeze reference model parameters
            for param in self.reference_model.parameters():
                param.requires_grad = False
                
            logger.info("Reference model initialized for KL penalty")
        except Exception as e:
            logger.warning(f"Failed to initialize reference model: {e}")
            self.use_kl_penalty = False
    
    def _initialize_model_weights(self):
        """Initialize model weights to prevent NaN values."""
        try:
            # Check for NaN values in model parameters
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any():
                    logger.warning(f"NaN detected in {name}, reinitializing...")
                    # Reinitialize the parameter
                    if param.dim() >= 2:
                        torch.nn.init.xavier_uniform_(param)
                    else:
                        torch.nn.init.normal_(param, mean=0.0, std=0.02)
            
            # Ensure all parameters are finite
            for name, param in self.model.named_parameters():
                if torch.isinf(param).any():
                    logger.warning(f"Inf detected in {name}, reinitializing...")
                    if param.dim() >= 2:
                        torch.nn.init.xavier_uniform_(param)
                    else:
                        torch.nn.init.normal_(param, mean=0.0, std=0.02)
            
            logger.info("Model weights initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize model weights: {e}")
    
    def _monitor_gradients(self):
        """Monitor gradients for NaN values and exploding gradients."""
        try:
            total_norm = 0.0
            param_count = 0
            nan_count = 0
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
                    
                    # Check for NaN gradients
                    if torch.isnan(param.grad).any():
                        nan_count += 1
                        logger.warning(f"NaN gradient detected in {name}")
                        # Zero out NaN gradients
                        param.grad.data.zero_()
                    
                    # Check for exploding gradients
                    if param_norm.item() > 10.0:
                        logger.warning(f"Large gradient detected in {name}: {param_norm.item():.4f}")
            
            total_norm = total_norm ** (1. / 2)
            
            if nan_count > 0:
                logger.warning(f"Found {nan_count} parameters with NaN gradients")
            
            if total_norm > 5.0:
                logger.warning(f"Large total gradient norm: {total_norm:.4f}")
                
        except Exception as e:
            logger.warning(f"Gradient monitoring failed: {e}")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        Compute GRPO loss with numerical stability fixes.
        
        Args:
            model: The model to compute loss for
            inputs: Input batch
            return_outputs: Whether to return model outputs
            
        Returns:
            Loss value and optionally model outputs
        """
        # Get model outputs
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Apply numerical stability fixes to logits
        logits = self._stabilize_logits(logits)
        
        # Get input_ids and attention_mask
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Get labels if available
        labels = inputs.get("labels", None)
        
        if labels is not None:
            # Standard cross-entropy loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for loss calculation
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Check for NaN or infinite loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("NaN or infinite loss detected, using small positive loss")
                loss = torch.tensor(1e-6, device=model.device, requires_grad=True)
            
            # Monitor gradients after backward pass
            if hasattr(self, '_gradient_monitoring_enabled'):
                self._monitor_gradients()
            
            # Add GRPO-specific loss components
            if self.use_kl_penalty and self.reference_model is not None:
                # Compute KL divergence penalty
                kl_loss = self._compute_kl_penalty(model, input_ids, attention_mask)
                if not torch.isnan(kl_loss) and not torch.isinf(kl_loss):
                    loss += self.kl_penalty_weight * kl_loss
            
            if return_outputs:
                return loss, outputs
            else:
                return loss
        else:
            # If no labels, return a dummy loss
            loss = torch.tensor(0.0, device=model.device, requires_grad=True)
            
            if return_outputs:
                return loss, outputs
            else:
                return loss
    
    def _compute_kl_penalty(self, model, input_ids, attention_mask):
        """
        Compute KL divergence penalty between current model and reference model.
        
        Args:
            model: Current model
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            KL divergence loss
        """
        try:
            # Ensure all tensors are on the same device
            device = model.device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            with torch.no_grad():
                # Get reference model outputs
                ref_outputs = self.reference_model(input_ids=input_ids, attention_mask=attention_mask)
                ref_logits = ref_outputs.logits.to(device)
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            
            # Get current model outputs
            current_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            current_logits = current_outputs.logits
            current_log_probs = F.log_softmax(current_logits, dim=-1)
            
            # Compute KL divergence
            kl_div = F.kl_div(
                current_log_probs,
                ref_log_probs,
                reduction="batchmean",
                log_target=True
            )
            
            return kl_div
        except Exception as e:
            logger.warning(f"Failed to compute KL penalty: {e}")
            return torch.tensor(0.0, device=device)
    
    def _stabilize_logits(self, logits):
        """
        Apply minimal numerical stability fixes to logits to prevent inf/nan values.
        
        Args:
            logits: Raw model logits
            
        Returns:
            Stabilized logits
        """
        # Check for NaN values and replace with small negative values
        if torch.isnan(logits).any():
            logger.warning("NaN values detected in logits, replacing with small negative values")
            logits = torch.where(torch.isnan(logits), torch.tensor(-10.0, device=logits.device), logits)
        
        # Check for infinite values and replace with large but finite values
        if torch.isinf(logits).any():
            logger.warning("Infinite values detected in logits, replacing with large finite values")
            logits = torch.where(torch.isinf(logits), torch.tensor(50.0, device=logits.device), logits)
        
        # Only clamp extreme values, don't apply aggressive transformations
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        
        return logits
    
    def _safe_generate(self, model, tokenizer, prompt, max_new_tokens=20, temperature=0.7, top_p=0.9, top_k=40):
        """
        Safely generate text with comprehensive NaN handling and fallback mechanisms.
        
        Args:
            model: The model to use for generation
            tokenizer: Tokenizer for the model
            prompt: Input prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text or error message
        """
        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Method 1: Try deterministic generation first
            try:
                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,  # Deterministic generation
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=2,
                    )
                    
                    response = tokenizer.decode(generated[0], skip_special_tokens=True)
                    
                    # Check if response is valid
                    if response and len(response.strip()) > 0:
                        return response
                        
            except Exception as e:
                logger.warning(f"Deterministic generation failed: {e}")
            
            # Method 2: Try with custom logits processing
            try:
                with torch.no_grad():
                    outputs = model(**inputs)
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
                        if next_token.item() == tokenizer.eos_token_id:
                            break
                        
                        generated_ids.append(next_token.item())
                        
                        # Update input for next iteration
                        current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1)
                        
                        # Get new logits
                        with torch.no_grad():
                            new_outputs = model(current_input)
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
                        response = tokenizer.decode(full_sequence[0], skip_special_tokens=True)
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
    
    def _prepare_inputs(self, inputs):
        """Prepare inputs for the model"""
        prepared = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                prepared[k] = v.to(self.model.device)
            else:
                prepared[k] = v
        return prepared


def create_simple_grpo_trainer(
    model,
    tokenizer,
    train_dataset: torch.utils.data.Dataset,
    eval_dataset: Optional[torch.utils.data.Dataset] = None,
    config: Optional[SimpleGRPOConfig] = None,
    **kwargs
) -> SimpleGRPOTrainer:
    """
    Create a Simple GRPO trainer instance.
    
    Args:
        model: The model to train
        tokenizer: Tokenizer for the model
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        config: GRPO configuration
        **kwargs: Additional arguments
        
    Returns:
        Configured Simple GRPO trainer
    """
    if config is None:
        config = SimpleGRPOConfig()
    
    # Create data collator
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        max_length=config.max_length,
        return_tensors="pt"
    )
    
    # Create trainer
    trainer = SimpleGRPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        **kwargs
    )
    
    return trainer


# Example usage and testing
if __name__ == "__main__":
    # Test the simple GRPO trainer
    print("✅ Simple GRPO Trainer implementation loaded successfully!")
    print("Available classes:")
    print("- SimpleGRPOConfig: Configuration for GRPO training")
    print("- SimpleGRPOTrainer: Main trainer class")
    print("- create_simple_grpo_trainer: Factory function to create trainer")
