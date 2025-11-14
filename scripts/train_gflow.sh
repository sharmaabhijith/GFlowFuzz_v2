#!/bin/bash
#SBATCH -J gflownet_train
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
REQ_FILE="requirements.txt"
STAMP="$VENV_DIR/.req.sha256"

# Compute hash of requirements (empty if file missing)
REQ_HASH="$("${RUN[@]}" bash -lc "test -f '$REQ_FILE' && sha256sum '$REQ_FILE' | awk '{print \$1}'" || true)"

needs_rebuild=false

# 1) Validate venv interpreter exists and is runnable (ELF or symlink)
if ! "${RUN[@]}" bash -lc '
  set -e
  test -x "'"$VENV_DIR"'/bin/python" || exit 1
  T=$(file -b "'"$VENV_DIR"'/bin/python" || true)
  case "$T" in
    *ELF*|*symbolic\ link* ) exit 0 ;;
    * ) exit 1 ;;
  esac
'; then
  needs_rebuild=true
fi

# 2) Rebuild if requirements changed
if [[ -n "$REQ_HASH" ]]; then
  OLD_HASH="$("${RUN[@]}" bash -lc "cat '$STAMP' 2>/dev/null || true")"
  [[ "$REQ_HASH" != "$OLD_HASH" ]] && needs_rebuild=true
fi

# 3) Build/rebuild when needed
if $needs_rebuild; then
  echo "[venv] (re)creating container-native venv at $VENV_DIR ..."
  "${RUN[@]}" bash -lc "
    rm -rf '$VENV_DIR'
    python3 -m venv '$VENV_DIR'
    '$VENV_DIR/bin/pip' install --upgrade pip wheel
    if [ -f '$REQ_FILE' ]; then
      '$VENV_DIR/bin/pip' install -r '$REQ_FILE'
      echo '$REQ_HASH' > '$STAMP'
    fi
    chmod -R u+rwX,go+rX '$VENV_DIR'
    '$VENV_DIR/bin/python' -V
  "
else
  # Harden perms in case they were lost
  "${RUN[@]}" bash -lc "find '$VENV_DIR/bin' -type f -exec chmod +x {} + >/dev/null 2>&1 || true"
fi

# ---- Train ----
CONFIG_FILE=${CONFIG_FILE:-training/configs/training_config.yaml}
echo "Starting GFlowNet training with config $CONFIG_FILE"
srun "${RUN[@]}" "$VENV_DIR/bin/python" -m training.train --config "$CONFIG_FILE"
