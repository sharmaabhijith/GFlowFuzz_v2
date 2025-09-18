# PPO Training for Auto User Agent

This module implements Proximal Policy Optimization (PPO) training for the auto user agent using Hugging Face's TRL library.

## Overview

The training system trains an RL-based user agent to have natural conversations with the booking agent. Key features:

- **PPO Algorithm**: Uses Hugging Face TRL for stable PPO training
- **Quantization & LoRA**: Efficient training with 4-bit quantization and LoRA adapters
- **Dual Buffer System**: Separate buffers for adversarial (hallucination) and normal data
- **Verifier Integration**: Rewards based on booking verification against database
- **Professional Pipeline**: Comprehensive logging, checkpointing, and evaluation

## Architecture

```
┌─────────────────────────────────────────┐
│          Auto User Agent (RL)           │
│         (Quantized + LoRA)              │
└────────────┬────────────────────────────┘
             │ Generate Response
             ▼
┌─────────────────────────────────────────┐
│        Booking Environment              │
│   ┌─────────────────────────────┐       │
│   │   Booking Agent (Fixed)      │       │
│   └─────────────────────────────┘       │
│   ┌─────────────────────────────┐       │
│   │   Verifier Agent (Fixed)     │       │
│   └─────────────────────────────┘       │
└────────────┬────────────────────────────┘
             │ Reward Signal
             ▼
┌─────────────────────────────────────────┐
│         PPO Training Loop               │
│   • Experience Buffer                   │
│   • Adversarial Buffer                  │
│   • Policy Updates                      │
└─────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Training

Edit `configs/ppo_training_config.yaml`:

```yaml
model:
  base_model: "microsoft/DialoGPT-small"  # Or any conversational model
  use_quantization: true                   # Enable 4-bit quantization
  use_lora: true                           # Enable LoRA adapters

ppo:
  max_epochs: 10
  learning_rate: 1.0e-5
  batch_size: 8
```

### 3. Run Training

```bash
# Using the shell script
./training/run_training.sh

# Or directly with Python
python training/train_ppo.py --config configs/ppo_training_config.yaml
```

## Training Process

### State Space
- Previous conversation history
- Current booking objective
- Assistant's last response

### Action Space
- Natural language response (sequence of tokens)
- Maximum length controlled by config

### Reward Structure

1. **Verifier Reward** (70% weight):
   - 1.0: All booking claims verified in database
   - 0.0: Any hallucinations detected

2. **Immediate Rewards** (30% weight):
   - Length penalty/bonus
   - Naturalness bonus
   - Polite ending bonus
   - Progress reward

### Buffer System

- **Adversarial Buffer**: Stores conversations with hallucinations (reward = 0)
- **Normal Buffer**: Stores verified conversations (reward = 1)
- **Sampling**: Configurable ratio for off-policy training

## Configuration Options

### Model Settings
- `base_model`: Pre-trained conversational model
- `use_quantization`: Enable 4-bit quantization for memory efficiency
- `use_lora`: Use LoRA adapters for parameter-efficient training
- `lora_r`: LoRA rank (default: 16)
- `lora_alpha`: LoRA scaling (default: 32)

### PPO Hyperparameters
- `learning_rate`: Learning rate for optimization
- `batch_size`: Batch size for training
- `ppo_epochs`: Number of PPO update epochs per rollout
- `gamma`: Discount factor
- `lam`: GAE lambda
- `cliprange`: PPO clipping range

### Buffer Settings
- `use_buffer`: Enable experience replay
- `buffer_size`: Maximum buffer size
- `adversarial_ratio`: Ratio of adversarial samples in batch

## Output Structure

```
ppo_training_output/
├── checkpoint-{epoch}/
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer_config.json
│   └── buffer.json
├── final_model/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── tokenizer_config.json
├── training_config.yaml
└── training_metrics.json
```

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir ./tensorboard_logs
```

### Training Logs
Check `logs/ppo_training_*.log` for detailed training progress.

### Metrics
- Average reward per epoch
- Verification rate
- Loss values
- Buffer statistics

## Using Trained Model

```python
from agents.auto_user.module import AutoUserAgent, AutoUserConfig

# Load trained model
config = AutoUserConfig(
    model_name="./ppo_training_output/final_model",
    max_length=30,
    temperature=0.7
)

agent = AutoUserAgent(config)
agent.initialize_model()

# Use for conversation
response = await agent.generate_response(
    assistant_message="How can I help you today?"
)
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Enable `gradient_checkpointing`
- Use smaller base model

### Slow Training
- Enable `use_quantization` for 4-bit training
- Reduce `ppo_epochs`
- Use smaller `rollout_episodes`

### Poor Performance
- Increase `max_epochs`
- Adjust reward weights
- Fine-tune PPO hyperparameters
- Increase buffer size

## Advanced Features

### Custom Objectives
Add booking objectives in `evaluation` section of config:

```yaml
evaluation:
  eval_objectives:
    - "Complex multi-city booking request"
    - "Specific date and class requirements"
```

### Mixed Precision Training
Enable for faster training on compatible GPUs:

```yaml
hardware:
  mixed_precision: true
```

### Distributed Training
Use accelerate for multi-GPU:

```bash
accelerate launch training/train_ppo.py --config configs/ppo_training_config.yaml
```

## License

Part of GFlowFuzz_v2 framework.