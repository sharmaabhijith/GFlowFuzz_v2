#!/bin/bash
#SBATCH -J sft_train
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:mi210:8
#SBATCH --partition=faculty
#SBATCH --qos=gtqos
#SBATCH --mem=256G
#SBATCH -t 24:00:00
#SBATCH -o logs/gflownet_%j.out
#SBATCH -e logs/gflownet_%j.err

set -euo pipefail
mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# ---- Container + binds ----
IMG="$HOME/asharma/pytorch-mybase.sif"
PIP_CACHE="$HOME/asharma/.cache/pip"
mkdir -p "$PIP_CACHE"
# Bind project dir + pip cache (comma-separated)
export APPTAINER_BINDPATH="$(printf "%s,%s" "$PWD" "$PIP_CACHE")"
export PIP_CACHE_DIR="$PIP_CACHE"
# Helper to run inside the image, rooted at the project dir
RUN=(apptainer exec --rocm --pwd "$PWD" "$IMG")

# ---- Smart venv build/reuse (. myenv ) ----
VENV_DIR="myenv"

# ---- Train (SFT) ----
SFT_MODEL="${SFT_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
SFT_DATA="${SFT_DATA:-dataset/cleaned_data/final_data_os_3.jsonl}"
SFT_OUT="${SFT_OUT:-SFT/trained_models}"
SFT_SEQ="${SFT_SEQ:-2048}"
SFT_BATCH="${SFT_BATCH:-1}"
SFT_ACCUM="${SFT_ACCUM:-16}"
SFT_EPOCHS="${SFT_EPOCHS:-2.0}"
SFT_LR="${SFT_LR:-2e-4}"
SFT_LORA_R="${SFT_LORA_R:-32}"
SFT_LORA_ALPHA="${SFT_LORA_ALPHA:-32}"
SFT_LORA_DROPOUT="${SFT_LORA_DROPOUT:-0.05}"

CMD=(
  "$VENV_DIR/bin/python"
  "SFT/train.py"
  --data "$SFT_DATA"
  --model "$SFT_MODEL"
  --out "$SFT_OUT"
  --seq "$SFT_SEQ"
  --batch "$SFT_BATCH"
  --accum "$SFT_ACCUM"
  --epochs "$SFT_EPOCHS"
  --lr "$SFT_LR"
  --lora_r "$SFT_LORA_R"
  --lora_alpha "$SFT_LORA_ALPHA"
  --lora_dropout "$SFT_LORA_DROPOUT"
)

echo "Starting SFT training with model $SFT_MODEL"
echo "Dataset: $SFT_DATA"
echo "Output dir: $SFT_OUT"

srun "${RUN[@]}" "${CMD[@]}"
