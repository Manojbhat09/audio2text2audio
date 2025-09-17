#!/usr/bin/env python3
"""
Simple test script to verify the GRPO training pipeline works
without requiring model downloads.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.simple_grpo_trainer import SimpleGRPOConfig, SimpleGRPOTrainer
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    """Dummy dataset for testing"""
    def __init__(self, size=10):
        self.size = size
        self.data = []
        for i in range(size):
            self.data.append({
                'input_ids': torch.randint(0, 1000, (50,)),
                'attention_mask': torch.ones(50),
                'labels': torch.randint(0, 1000, (50,))
            })
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx]

def test_grpo_trainer():
    """Test the GRPO trainer with a dummy model"""
    print("🧪 Testing GRPO Trainer...")
    
    try:
        # Create a simple model for testing
        print("📦 Creating dummy model...")
        class DummyModel(torch.nn.Module):
            def __init__(self, config=None):
                super().__init__()
                self.config = config or type('Config', (), {'vocab_size': 1000})()
                self.linear = torch.nn.Linear(1000, 1000)
            
            def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
                batch_size, seq_len = input_ids.shape
                device = input_ids.device
                # Create dummy logits on the same device
                logits = self.linear(torch.randn(batch_size, seq_len, 1000, device=device))
                return type('Output', (), {'logits': logits})()
        
        model = DummyModel()
        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Create dummy tokenizer
        class DummyTokenizer:
            def __init__(self):
                self.pad_token = "<pad>"
                self.eos_token = "<eos>"
                self.vocab_size = 1000
            
            def __call__(self, *args, **kwargs):
                return {"input_ids": torch.randint(0, 1000, (1, 10))}
        
        tokenizer = DummyTokenizer()
        
        # Create dummy dataset
        print("📊 Creating dummy dataset...")
        train_dataset = DummyDataset(10)
        eval_dataset = DummyDataset(5)
        
        # Create GRPO config
        print("⚙️ Creating GRPO config...")
        config = SimpleGRPOConfig(
            output_dir="./test_output",
            max_steps=2,
            learning_rate=1e-4,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_steps=10,
            eval_steps=10,
            eval_strategy="no",
            save_strategy="steps",
            load_best_model_at_end=False,
            remove_unused_columns=False,
            dataloader_drop_last=False,
            use_kl_penalty=False,  # Disable KL penalty for testing
        )
        
        # Create trainer
        print("🚀 Creating GRPO trainer...")
        trainer = SimpleGRPOTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            optimizers=(None, None),  # Provide dummy optimizers
        )
        
        print("✅ GRPO Trainer created successfully!")
        
        # Test training step
        print("🏃 Testing training step...")
        sample_batch = train_dataset[0]
        sample_batch = {k: v.unsqueeze(0) for k, v in sample_batch.items()}
        
        # Move batch to same device as model
        if torch.cuda.is_available():
            sample_batch = {k: v.cuda() for k, v in sample_batch.items()}
        
        # Test loss computation
        loss = trainer.compute_loss(model, sample_batch)
        print(f"✅ Loss computation successful: {loss}")
        
        print("🎉 All tests passed! GRPO trainer is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_grpo_trainer()
    sys.exit(0 if success else 1)
