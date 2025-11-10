#!/usr/bin/env bash
# Generate datasets for every (user_model, temperature, chat_model) combination.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

EPISODES="${EPISODES:-25}"
MAX_DIALOGUES="${MAX_DIALOGUES:-30}"
LOG_EVERY="${LOG_EVERY:-1}"

user_models=(
  "meta-llama/Llama-3.2-3B-Instruct"
  "Qwen/Qwen3-14B"
  "Qwen/Qwen3-Next-80B-A3B-Instruct"
  "Qwen/Qwen3-235B-A22B-Instruct-2507"
)

assistant_models=(
  "Qwen/Qwen3-14B"
  "Qwen/Qwen3-235B-A22B-Instruct-2507"
)

user_temps=(0.1 0.5 0.9)

for temp in "${user_temps[@]}"; do
  for u_model in "${user_models[@]}"; do
    for a_model in "${assistant_models[@]}"; do
      echo "=== user_model=${u_model} temp=${temp} | chat_model=${a_model} ==="
      python dataset/make_dataset.py \
        --episodes "${EPISODES}" \
        --max_dialogues "${MAX_DIALOGUES}" \
        --log_every "${LOG_EVERY}" \
        --user-model "${u_model}" \
        --user-temperature "${temp}" \
        --chat-model "${a_model}"
    done
  done
done
