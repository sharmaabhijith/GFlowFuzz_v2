#!/usr/bin/env bash
# Generate datasets for every (user_model, temperature, chat_model) combination.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

EPISODES="${EPISODES:-9}"
MAX_DIALOGUES="${MAX_DIALOGUES:-30}"
LOG_EVERY="${LOG_EVERY:-1}"

user_models=(
  "deepseek-ai/DeepSeek-V3.2-Exp"
)

assistant_models=(
  "Qwen/Qwen3-235B-A22B-Instruct-2507"
)

user_temps=(0.5)

for u_model in "${user_models[@]}"; do
  for temp in "${user_temps[@]}"; do
    for a_model in "${assistant_models[@]}"; do
      echo "=== user_model=${u_model} temp=${temp} | chat_model=${a_model} ==="
      python dataset/make_dataset.py \
        --episodes "${EPISODES}" \
        --max_dialogues "${MAX_DIALOGUES}" \
        --user-model "${u_model}" \
        --user-temperature "${temp}" \
        --chat-model "${a_model}"
    done
  done
done
