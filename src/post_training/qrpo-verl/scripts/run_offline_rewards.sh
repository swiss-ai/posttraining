#!/usr/bin/env bash

set -euo pipefail

EXTRA_HYDRA_OVERRIDES=("$@")

PROJECT_ROOT_AT="${PROJECT_ROOT_AT:-$(pwd)}"
cd "${PROJECT_ROOT_AT}"

if [[ -z "${JUDGE_BASE_URL:-}" ]]; then
  echo "ERROR: JUDGE_BASE_URL must point to the active judge router, e.g. http://<router-ip>:30000/v1" >&2
  exit 1
fi

export JUDGE_BASE_URL
export JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
export JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.6-27B-smatrenok}"
export ACTIVE_UF_PROMPTS_PATH="${ACTIVE_UF_PROMPTS_PATH:-/iopsstor/scratch/cscs/smatreno/posttraining-data/response_annotation/prompts.py}"
export ACTIVE_UF_ASPECTS="${ACTIVE_UF_ASPECTS:-helpfulness}"
export JUDGE_MAX_CONCURRENCY_PER_WORKER="${JUDGE_MAX_CONCURRENCY_PER_WORKER:-64}"
export JUDGE_MAX_CONNECTIONS="${JUDGE_MAX_CONNECTIONS:-2048}"
export JUDGE_TIMEOUT_S="${JUDGE_TIMEOUT_S:-60}"
export PYTHONPATH="${PROJECT_ROOT_AT}/src/post_training/qrpo-verl${PYTHONPATH:+:${PYTHONPATH}}"

REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-32}"
OFFLINE_REWARD_CHUNK_SIZE_COMPLETIONS="${OFFLINE_REWARD_CHUNK_SIZE_COMPLETIONS:-1024}"
RAY_WAIT_TIMEOUT_S="${RAY_WAIT_TIMEOUT_S:-600}"

if (( REWARD_NUM_WORKERS <= 0 )); then
  echo "ERROR: REWARD_NUM_WORKERS must be positive, got ${REWARD_NUM_WORKERS}." >&2
  exit 1
fi
if (( OFFLINE_REWARD_CHUNK_SIZE_COMPLETIONS <= 0 )); then
  echo "ERROR: OFFLINE_REWARD_CHUNK_SIZE_COMPLETIONS must be positive, got ${OFFLINE_REWARD_CHUNK_SIZE_COMPLETIONS}." >&2
  exit 1
fi
if (( OFFLINE_REWARD_CHUNK_SIZE_COMPLETIONS % REWARD_NUM_WORKERS != 0 )); then
  echo "ERROR: OFFLINE_REWARD_CHUNK_SIZE_COMPLETIONS=${OFFLINE_REWARD_CHUNK_SIZE_COMPLETIONS} must be divisible by REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS}." >&2
  exit 1
fi

cat <<EOF
Offline reward recomputation:
  reward.num_workers = ${REWARD_NUM_WORKERS}
  offline_rewards.chunk_size_completions = ${OFFLINE_REWARD_CHUNK_SIZE_COMPLETIONS}
  judge base url = ${JUDGE_BASE_URL}
  judge model = ${JUDGE_MODEL}
EOF

python -m entrypoints.recompute_offline_rewards \
  reward.num_workers="${REWARD_NUM_WORKERS}" \
  offline_rewards.chunk_size_completions="${OFFLINE_REWARD_CHUNK_SIZE_COMPLETIONS}" \
  ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS_PER_NODE:-$(nproc)}" \
  "${EXTRA_HYDRA_OVERRIDES[@]}"
