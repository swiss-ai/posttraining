#!/bin/bash
# ── Main launcher: HP grid x (inference server + training) ──────────────
# For each HP combination, submits an orchestrator job (1 node) that:
#   1. Launches an inference server via submit_job.py
#   2. Waits for the server URL
#   3. Submits the training job with that URL
#   4. Exits (server stays running for training to use)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGS_DIR="/iopsstor/scratch/cscs/dmelikidze/online-dpo/logs"
mkdir -p "${LOGS_DIR}/orchestrator" "${LOGS_DIR}/training"

# ── SLURM / Account ────────────────────────────────────────────────────
ACCOUNT="infra01"
RESERVATION="SD-69241-apertus-1-5"
PARTITION="normal"
JOB_TIME="12:00:00"

# ── Inference server config ─────────────────────────────────────────────
SERVER_MODEL="/iopsstor/scratch/cscs/dmelikidze/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
SERVER_SERVED_NAME="Qwen/Qwen3.6-27B-dmelikidze"
SERVER_NODES=8
SERVER_WORKERS=8
SERVER_NODES_PER_WORKER=1
SERVER_TP_SIZE=4
SERVER_FRAMEWORK="vllm"

# ── Training config (fixed across grid) ─────────────────────────────────
TRAIN_NODES=8
MODEL_PATH="/capstor/store/cscs/swissai/infra01/models/apertus-8b-sft-1.5--lr8e-5"
OUTPUT_BASE_DIR="/iopsstor/scratch/cscs/dmelikidze/verl-training"
JUDGE_MODEL="Qwen/Qwen3.6-27B-dmelikidze"
JUDGE_API_KEY="sk-rc-MH1IEiFLN35rXSJq5pWECQ"

# Fixed training params (override per-run via grid arrays below)
GPU_MEM_UTIL=0.35
ENFORCE_EAGER=false
MAX_NUM_BATCHED_TOKENS=8192
TRAIN_BATCH_SIZE=256
ROLLOUT_N=8
ACTOR_MICRO_BS=2
LOGPROB_MICRO_BS=2
REF_LOGPROB_MICRO_BS=2
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=1024
LR_SCHEDULER_TYPE=linear
LR_WARMUP_STEPS=-1
LR_WARMUP_STEPS_RATIO=0.1
MIN_LR_RATIO=0.0
TOTAL_EPOCHS=1
SAVE_FREQ=500
TP_SIZE=4
FSDP_SIZE=16
GRAD_CLIP=20.0
LENGTH_NORMALIZE=false

# ── Hyperparameter grid ─────────────────────────────────────────────────
# Each array defines values to sweep. All combinations are launched.
LEARNING_RATES=(1e-7 5e-7 1e-6 5e-6 1e-5)
DPO_BETAS=(0.1) #(0.01 0.1)

# ── Submit jobs ─────────────────────────────────────────────────────────
for LR in "${LEARNING_RATES[@]}"; do
for BETA in "${DPO_BETAS[@]}"; do

    RUN_NAME="online-DPO-lr${LR}-beta${BETA}"
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${RUN_NAME}"

    echo "Submitting: ${RUN_NAME}"

    sbatch \
        --job-name="orch-${RUN_NAME}" \
        --account="${ACCOUNT}" \
        --reservation="${RESERVATION}" \
        --partition="${PARTITION}" \
        --time="${JOB_TIME}" \
        --nodes=1 \
        --ntasks-per-node=1 \
        --output="${LOGS_DIR}/orchestrator/orch-${RUN_NAME}_%j.out" \
        --error="${LOGS_DIR}/orchestrator/orch-${RUN_NAME}_%j.err" \
        --export=ALL,\
SERVER_MODEL="${SERVER_MODEL}",\
SERVER_SERVED_NAME="${SERVER_SERVED_NAME}",\
SERVER_NODES="${SERVER_NODES}",\
SERVER_WORKERS="${SERVER_WORKERS}",\
SERVER_NODES_PER_WORKER="${SERVER_NODES_PER_WORKER}",\
SERVER_TP_SIZE="${SERVER_TP_SIZE}",\
SERVER_FRAMEWORK="${SERVER_FRAMEWORK}",\
TRAIN_NODES="${TRAIN_NODES}",\
MODEL_PATH="${MODEL_PATH}",\
OUTPUT_DIR="${OUTPUT_DIR}",\
EXPERIMENT_NAME="${RUN_NAME}",\
JUDGE_MODEL="${JUDGE_MODEL}",\
JUDGE_API_KEY="${JUDGE_API_KEY}",\
LEARNING_RATE="${LR}",\
DPO_BETA="${BETA}",\
GPU_MEM_UTIL="${GPU_MEM_UTIL}",\
ENFORCE_EAGER="${ENFORCE_EAGER}",\
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}",\
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}",\
ROLLOUT_N="${ROLLOUT_N}",\
ACTOR_MICRO_BS="${ACTOR_MICRO_BS}",\
LOGPROB_MICRO_BS="${LOGPROB_MICRO_BS}",\
REF_LOGPROB_MICRO_BS="${REF_LOGPROB_MICRO_BS}",\
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}",\
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}",\
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE}",\
LR_WARMUP_STEPS="${LR_WARMUP_STEPS}",\
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO}",\
MIN_LR_RATIO="${MIN_LR_RATIO}",\
TOTAL_EPOCHS="${TOTAL_EPOCHS}",\
SAVE_FREQ="${SAVE_FREQ}",\
TP_SIZE="${TP_SIZE}",\
FSDP_SIZE="${FSDP_SIZE}",\
GRAD_CLIP="${GRAD_CLIP}",\
LENGTH_NORMALIZE="${LENGTH_NORMALIZE}",\
ACCOUNT="${ACCOUNT}",\
RESERVATION="${RESERVATION}",\
PARTITION="${PARTITION}",\
JOB_TIME="${JOB_TIME}" \
        "${SCRIPT_DIR}/orchestrator.sh"

done
done

echo "All orchestrator jobs submitted."
