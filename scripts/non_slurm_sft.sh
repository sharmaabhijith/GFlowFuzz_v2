#!/usr/bin/env bash
set -euo pipefail
set -a
source .env
set +a

# ---------- Config (edit as you like) ----------
TRAIN_PY="${TRAIN_PY:-SFT/train.py}"                            # your training script path
SFT_MODEL="${SFT_MODEL:-Qwen/Qwen3-8B}"           # NOT an FP8 checkpoint
SFT_DATA="${SFT_DATA:-dataset/cleaned_data/final_data_os_3.jsonl}"
SFT_OUT="${SFT_OUT:-SFT/trained_models/qwen3-8b-sft}"
SFT_SEQ="${SFT_SEQ:-4096}"
SFT_BATCH="${SFT_BATCH:-1}"
SFT_ACCUM="${SFT_ACCUM:-16}"
SFT_EPOCHS="${SFT_EPOCHS:-2.0}"
SFT_LR="${SFT_LR:-2e-4}"
SFT_LORA_R="${SFT_LORA_R:-8}"
SFT_LORA_ALPHA="${SFT_LORA_ALPHA:-16}"
SFT_LORA_DROPOUT="${SFT_LORA_DROPOUT:-0.05}"

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
echo "Seq/Batch: $SFT_SEQ / $SFT_BATCH  (accum=$SFT_ACCUM)"
echo "Epochs/LR: $SFT_EPOCHS / $SFT_LR"
echo "LoRA:      r=$SFT_LORA_R, alpha=$SFT_LORA_ALPHA, dropout=$SFT_LORA_DROPOUT"
echo "bnb flag:  ${BINARY_FLAGS[*]:-(none)}"
echo "---------------------------------------------"

# ---------- Run training ----------
exec "python3" "$TRAIN_PY" \
  --data "$SFT_DATA" \
  --model "$SFT_MODEL" \
  --out "$SFT_OUT" \
  --seq "$SFT_SEQ" \
  --batch "$SFT_BATCH" \
  --accum "$SFT_ACCUM" \
  --epochs "$SFT_EPOCHS" \
  --lr "$SFT_LR" \
  --lora_r "$SFT_LORA_R" \
  --lora_alpha "$SFT_LORA_ALPHA" \
  --lora_dropout "$SFT_LORA_DROPOUT" \
