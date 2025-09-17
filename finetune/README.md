# Dataset Classification Training Pipeline

This directory contains the complete pipeline for training a Gemma model to classify VoiceBench datasets using GRPO (Group Relative Policy Optimization) reinforcement learning.

## Overview

The pipeline trains a model to classify speech transcripts into one of three VoiceBench dataset categories:
- **ifeval**: Instruction-following tasks with specific formatting requirements
- **commoneval**: Common sense reasoning and knowledge questions  
- **wildvoice**: Conversational scenarios and open-ended discussions

## Architecture

```
finetune/
├── src/                    # Source code
│   ├── whisper_transcriber.py    # Audio transcription using Whisper
│   ├── dataset_loader.py         # Online dataset loading and processing
│   ├── reward_functions.py       # GRPO reward functions
│   └── grpo_trainer.py          # Main GRPO training implementation
├── configs/                # Configuration files
│   └── training_config.py       # Training configuration classes
├── scripts/                # Executable scripts
│   ├── train_dataset_classifier.py  # Main training script
│   └── setup_environment.sh        # Environment setup script
├── data/                   # Data storage
├── models/                 # Model storage
├── outputs/                # Training outputs and logs
└── requirements.txt        # Python dependencies
```

## Quick Start

### 1. Setup Environment

```bash
# Activate conda environment
source /home/mbhat/miniconda/bin/activate
conda activate omega

# Run setup script
cd /home/mbhat/omegalabs-anytoany-bittensor/finetune
bash scripts/setup_environment.sh
```

### 2. Test with Small Configuration

```bash
python scripts/train_dataset_classifier.py --config small --verbose
```

### 3. Full Training

```bash
python scripts/train_dataset_classifier.py --config default
```

## Configuration Options

### Predefined Configurations

- **small**: For testing (20 samples per dataset, 20 steps)
- **default**: Standard training (100 samples per dataset, 100 steps)  
- **large**: Full-scale training (500 samples per dataset, 500 steps)

### Command Line Arguments

```bash
python scripts/train_dataset_classifier.py \
    --model-name unsloth/gemma-3-270m-it \
    --samples-per-dataset 100 \
    --max-steps 100 \
    --learning-rate 5e-6 \
    --datasets ifeval commoneval wildvoice \
    --sampling-method random \
    --output-dir ./outputs \
    --verbose
```

## Training Process

1. **Data Loading**: Online loading of VoiceBench datasets
2. **Transcription**: Audio-to-text conversion using Whisper wpt.pt model
3. **Sample Generation**: Creation of training samples with prompts and labels
4. **GRPO Training**: Reinforcement learning with multiple reward functions
5. **Model Saving**: Saving of trained model and training artifacts

## Reward Functions

The training uses multiple reward functions to guide learning:

- **Exact Match**: Rewards correct dataset name predictions
- **Format Compliance**: Rewards proper response format (single dataset name)
- **Confidence**: Rewards decisive responses without uncertainty
- **Dataset Balance**: Rewards balanced predictions across datasets
- **Length Penalty**: Penalizes overly verbose responses

## Outputs

Training produces:

- **Trained Model**: Saved in `models/final_model/`
- **Training Logs**: Detailed logs in `outputs/`
- **Configuration**: Training config saved as JSON
- **Metrics**: Training history and evaluation metrics

## Monitoring

Training progress is logged with:
- Step-by-step training loss and rewards
- Dataset distribution statistics
- Early stopping monitoring
- Model performance metrics

## Requirements

- Python 3.8+
- CUDA-compatible GPU
- 16GB+ RAM recommended
- 50GB+ disk space for models and data

## Dependencies

See `requirements.txt` for complete list. Key packages:
- PyTorch 2.0+
- Transformers 4.35+
- TRL 0.7+ (for GRPO)
- Unsloth (for efficient training)
- Whisper (for audio transcription)
- Datasets (for data loading)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or max sequence length
2. **Whisper Model Not Found**: Check path to wpt.pt model
3. **Dataset Loading Errors**: Verify internet connection and dataset availability
4. **Training Divergence**: Adjust learning rate or reward weights

### Debug Mode

Run with `--verbose` flag for detailed logging:

```bash
python scripts/train_dataset_classifier.py --config small --verbose
```

## Performance Tips

1. **Start Small**: Use `--config small` for initial testing
2. **Monitor Resources**: Watch GPU memory and CPU usage
3. **Adjust Batch Size**: Increase if you have more GPU memory
4. **Early Stopping**: Configure patience based on your needs
5. **Reward Tuning**: Adjust reward weights for better performance

## File Structure Details

### Source Code (`src/`)

- **whisper_transcriber.py**: Handles audio transcription using Whisper
- **dataset_loader.py**: Online dataset loading with on-demand transcription
- **reward_functions.py**: GRPO reward function implementations
- **grpo_trainer.py**: Main training loop with GRPO integration

### Configuration (`configs/`)

- **training_config.py**: Configuration classes and predefined settings

### Scripts (`scripts/`)

- **train_dataset_classifier.py**: Main training script with CLI interface
- **setup_environment.sh**: Environment setup and dependency installation

## Example Usage

```python
from configs.training_config import get_default_config
from src.grpo_trainer import create_trainer

# Create configuration
config = get_default_config()
config.samples_per_dataset = 50
config.max_steps = 50

# Create and run trainer
trainer = create_trainer(config)
trainer.train()
```

## Contributing

When modifying the code:

1. Follow the existing code style
2. Add type hints and docstrings
3. Test with small configuration first
4. Update documentation as needed
5. Run linting before committing

## License

This code is part of the OmegaLabs AnyToAny Bittensor project.





