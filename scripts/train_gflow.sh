#!/bin/bash
#SBATCH -J gflownet_train             # Job name
#SBATCH -N 8                          # Number of nodes
#SBATCH --ntasks-per-node=8           # 1 task per GPU
#SBATCH --gres=gpu:8                  # GPUs per node
#SBATCH -p gpu                        # Partition/queue name
#SBATCH -t 12:00:00                   # Time limit hh:mm:ss
#SBATCH -o logs/gflownet_%j.out       # Standard output (make sure logs/ exists)
#SBATCH -e logs/gflownet_%j.err       # Standard error
#SBATCH --cpus-per-task=8             # CPU cores per GPU task

set -euo pipefail

echo "=== GFlowNet training job bootstrap ==="
echo "Submit directory : ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Job ID           : ${SLURM_JOB_ID:-unknown}"
echo "Nodes allocated  : ${SLURM_JOB_NUM_NODES:-1}"
echo "Tasks requested  : ${SLURM_NTASKS:-1}"
echo "GPUs per node    : ${SLURM_GPUS_ON_NODE:-8}"

#-------------
# Environment
#-------------
module purge
module load rocm
module load mpi

# Activate your Python environment if available.
# Override VENV_ACTIVATE to point to a different virtual environment.
VENV_ACTIVATE=${VENV_ACTIVATE:-"$SLURM_SUBMIT_DIR/../myenv/bin/activate"}
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
  echo "Activated Python environment at $VENV_ACTIVATE"
else
  echo "Virtual environment not found at $VENV_ACTIVATE; continuing without activation."
fi

PROJECT_ROOT=$(realpath "${PROJECT_ROOT:-"$SLURM_SUBMIT_DIR/.."}")
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

mkdir -p "${SLURM_SUBMIT_DIR:-.}/logs"

echo "Running on nodes:"
srun hostname | sort -u

#-------------
# Configuration
#-------------
CONFIG_FILE=${CONFIG_FILE:-"training/configs/training_config.yaml"}
CONFIG_PATH="$PROJECT_ROOT/$CONFIG_FILE"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found at $CONFIG_PATH" >&2
  exit 1
fi

ADJUSTED_CONFIG=$(mktemp "${SLURM_JOB_ID:-gflownet}_config_XXXX.yaml")
export CONFIG_IN="$CONFIG_PATH"
export CONFIG_OUT="$ADJUSTED_CONFIG"
trap 'rm -f "$ADJUSTED_CONFIG"' EXIT

python3 <<'PY'
import os
import pathlib
import yaml

src = pathlib.Path(os.environ["CONFIG_IN"]).resolve()
dst = pathlib.Path(os.environ["CONFIG_OUT"]).resolve()

with src.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

if "gflownet" not in config:
    raise SystemExit("Configuration missing 'gflownet' section.")

config["algorithm"] = "gflownet"

with dst.open("w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False)

print(f"Wrote adjusted config to {dst}")
PY

# Optional: resume from checkpoint by overriding RESUME_PATH=/path/to/checkpoint
EXTRA_ARGS=()
if [[ -n "${RESUME_PATH:-}" ]]; then
  EXTRA_ARGS+=(--resume "$RESUME_PATH")
fi

echo "=== Launching distributed training ==="
srun python3 -m training.train \
  --config "$ADJUSTED_CONFIG" \
  "${EXTRA_ARGS[@]}"

echo "Training finished with exit code $?"
