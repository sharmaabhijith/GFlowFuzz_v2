#!/usr/bin/env bash
set -euo pipefail
set -a
source .env
set +a

# ---------- Config (edit as you like) ----------
TRAIN_PY="${TRAIN_PY:-SFT/train.py}"                            # your training script path
SFT_MODEL="${SFT_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"           # NOT an FP8 checkpoint
SFT_DATA="${SFT_DATA:-dataset/cleaned_data/final_data_os_1.jsonl}"
SFT_OUT="${SFT_OUT:-SFT/trained_models/qwen3-4b-sft-os1-epoch}"

# ---------- Environment / venv ----------
VENV_DIR="${VENV_DIR:-myenv}"

# ---------- Sanity checks ----------
[[ -f "$TRAIN_PY" ]] || { echo "Training script not found: $TRAIN_PY"; exit 1; }
[[ -f "$SFT_DATA"  ]] || { echo "Dataset not found: $SFT_DATA"; exit 1; }
mkdir -p "$SFT_OUT" "$WANDB_DIR"

echo "---------------------------------------------"
echo "Model:     $SFT_MODEL"
echo "Data:      $SFT_DATA"
echo "Output:    $SFT_OUT"
echo "---------------------------------------------"

# ---------- Run training ----------
exec "python3" "$TRAIN_PY" \
  --data "$SFT_DATA" \
  --model "$SFT_MODEL" \
  --out "$SFT_OUT" \
