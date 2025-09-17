#!/bin/bash
"""
Setup script for the finetune environment

This script sets up the conda environment and installs all required dependencies
for training the dataset classifier with GRPO.
"""

set -e  # Exit on any error

echo "🚀 Setting up finetune environment..."

# Activate conda environment
echo "📦 Activating conda environment..."
source /home/mbhat/miniconda/bin/activate
conda activate omega

# Check if we're in the right environment
echo "🔍 Checking environment..."
python -c "import sys; print(f'Python: {sys.executable}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Install required packages
echo "📥 Installing required packages..."

# Core ML packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate
pip install trl  # For GRPO
pip install unsloth  # For efficient training
pip install whisper  # For audio transcription
pip install librosa  # For audio processing
pip install numpy scipy  # For numerical operations

# Additional utilities
pip install wandb  # For experiment tracking (optional)
pip install tensorboard  # For logging
pip install tqdm  # For progress bars
pip install psutil  # For system monitoring

# Verify installations
echo "✅ Verifying installations..."
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')"
python -c "import datasets; print(f'✓ Datasets: {datasets.__version__}')"
python -c "import trl; print(f'✓ TRL: {trl.__version__}')"
python -c "import unsloth; print(f'✓ Unsloth: {unsloth.__version__}')"
python -c "import whisper; print(f'✓ Whisper: {whisper.__version__}')"
python -c "import librosa; print(f'✓ Librosa: {librosa.__version__}')"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p /home/mbhat/omegalabs-anytoany-bittensor/finetune/{outputs,data,models,logs}

# Set permissions
echo "🔐 Setting permissions..."
chmod +x /home/mbhat/omegalabs-anytoany-bittensor/finetune/scripts/*.py

echo "🎉 Environment setup complete!"
echo ""
echo "To start training, run:"
echo "  cd /home/mbhat/omegalabs-anytoany-bittensor/finetune"
echo "  python scripts/train_dataset_classifier.py --config small"
echo ""
echo "For full training:"
echo "  python scripts/train_dataset_classifier.py --config default"





