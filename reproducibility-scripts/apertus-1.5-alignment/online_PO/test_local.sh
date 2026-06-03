#!/bin/bash
set -xeuo pipefail

# Stop any existing Ray cluster
ray stop --force 2>/dev/null || true
sleep 2

# Start Ray (single node, 4 GPUs)
ray start --head --num-gpus 4
sleep 5

# Required env vars
export JUDGE_BASE_URL="http://localhost:8080/v1"
export JUDGE_API_KEY="dummy"
export JUDGE_MODEL="dummy"
export MODEL_PATH="Qwen/Qwen3-1.7B"
export TRAIN_DATA="/iopsstor/scratch/cscs/dmelikidze/dmelikidze/projects/posttraining/run/reproducibility-scripts/apertus-1.5-alignment/online_PO/data/train_dolci_small.parquet"
export VAL_DATA="${TRAIN_DATA}"
export EXPERIMENT_NAME="test-weight-sync"
export OUTPUT_DIR="/iopsstor/scratch/cscs/dmelikidze/verl-training/test-weight-sync"

# Small config for 4 GPUs
export TRAIN_BATCH_SIZE=4
export ROLLOUT_N=2
export FSDP_SIZE=4
export TP_SIZE=1
export MAX_PROMPT_LENGTH=512
export MAX_RESPONSE_LENGTH=512
export SAVE_FREQ=9999
export TOTAL_EPOCHS=1
export ACTOR_MICRO_BS=1
export LOGPROB_MICRO_BS=1
export REF_LOGPROB_MICRO_BS=1
export GPU_MEM_UTIL=0.35
export ENFORCE_EAGER=true
export MAX_NUM_BATCHED_TOKENS=4096
export LEARNING_RATE=5e-6
export DPO_BETA=0.1
export GRAD_CLIP=20.0
export LENGTH_NORMALIZE=false
export LR_SCHEDULER_TYPE=linear
export LR_WARMUP_STEPS=-1
export LR_WARMUP_STEPS_RATIO=0.1
export MIN_LR_RATIO=0.0
export ASYNC_ROLLOUT=false
export OFFPOLICY_DATA=""
export OFFPOLICY_BATCH_SIZE=4

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

bash train_spin.sh trainer.nnodes=1 trainer.n_gpus_per_node=4
