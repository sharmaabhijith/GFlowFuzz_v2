#!/bin/bash

# Script to run PPO training for Auto User Agent

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  PPO Training for Auto User Agent${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
fi

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python --version

# Install/upgrade required packages if needed
echo -e "${YELLOW}Checking required packages...${NC}"
pip install -q --upgrade torch transformers trl peft bitsandbytes accelerate

# Create necessary directories
echo -e "${YELLOW}Creating output directories...${NC}"
mkdir -p ppo_training_output
mkdir -p logs
mkdir -p tensorboard_logs

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Run training
echo ""
echo -e "${GREEN}Starting PPO training...${NC}"
echo ""

python training/train_ppo.py --config configs/ppo_training_config.yaml "$@"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Training completed successfully!${NC}"
    echo -e "${GREEN}Output saved to: ppo_training_output/${NC}"
else
    echo ""
    echo -e "${RED}✗ Training failed. Check logs for details.${NC}"
    exit 1
fi