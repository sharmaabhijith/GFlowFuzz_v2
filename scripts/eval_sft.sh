#!/usr/bin/env bash
set -euo pipefail
set -a
source .env
set +a

# ---------- Config (edit as you like) ----------
TRAIN_PY="${TRAIN_PY:-SFT/evaluate_auditor.py}"                            # your training script path

# ---------- Environment / venv ----------
VENV_DIR="${VENV_DIR:-myenv}"


# ---------- Run training ----------
exec "python3" "$TRAIN_PY"
