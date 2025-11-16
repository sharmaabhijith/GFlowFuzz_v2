#!/usr/bin/env bash
set -euo pipefail
set -a
source .env
set +a

# ---------- Config (edit as you like) ----------
TRAIN_PY="${TRAIN_PY:-SFT/evaluate_auditor.py}"                            # your training script path
SFT_MODEL="${SFT_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"           # NOT an FP8 checkpoint

# ---------- Environment / venv ----------
VENV_DIR="${VENV_DIR:-myenv}"

echo "---------------------------------------------"
echo "Model:     $SFT_MODEL"
echo "---------------------------------------------"

# ---------- Run training ----------
exec "python3" "$TRAIN_PY"
