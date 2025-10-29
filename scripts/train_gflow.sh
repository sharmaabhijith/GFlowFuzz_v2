#!/bin/bash
#SBATCH -J gflownet_train
#SBATCH -N 8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH -o logs/gflownet_%j.out
#SBATCH -e logs/gflownet_%j.err
#SBATCH --cpus-per-task=8

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

module load rocm
module load mpi

VENV_PATH=${VENV_PATH:-myenv/bin/activate}
if [[ -f "$VENV_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH"
fi

CONFIG_FILE=${CONFIG_FILE:-training/configs/training_config.yaml}

echo "Starting GFlowNet training with config $CONFIG_FILE"

EXTRA_ARGS=()
if [[ -n "${RESUME_PATH:-}" ]]; then
  EXTRA_ARGS+=(--resume "$RESUME_PATH")
fi

srun python3 -m training.train --config "$CONFIG_FILE" "${EXTRA_ARGS[@]}"
