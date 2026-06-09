#!/bin/bash
# ── Main launcher: HP grid x (inference server + training) ──────────────
# For each HP combination, submits an orchestrator job (1 node) that:
#   1. Launches an inference server via submit_job.py
#   2. Waits for the server URL
#   3. Submits the training job with that URL
#   4. Exits (server stays running for training to use)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGS_DIR="${SCRATCH}/online-dpo/logs"
mkdir -p "${LOGS_DIR}/orchestrator" "${LOGS_DIR}/training"

# ── SLURM / Account ────────────────────────────────────────────────────
ACCOUNT="infra01"
RESERVATION="SD-69241-apertus-1-5-0"
PARTITION="normal"
JOB_TIME="12:00:00"
EXCLUDE_NODES="nid007613"

# ── Inference server config ─────────────────────────────────────────────
SERVER_MODEL="${SCRATCH}/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
SERVER_SERVED_NAME="Qwen/Qwen3.6-27B-dmelikidze"
SERVER_NODES=8
SERVER_WORKERS=8
SERVER_NODES_PER_WORKER=1
SERVER_TP_SIZE=4
SERVER_FRAMEWORK="vllm"

# ── Training config (fixed across grid) ─────────────────────────────────
TRAIN_NODES=16
# ── Model paths to ablate over ──────────────────────────────────────────
# Add one path per line; the script submits the full grid for each model.
# MODEL_NAME is derived from the basename of each path.
MODEL_PATHS=(
    # "${SCRATCH}/ap_mo/ap_1p5_sft/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_3000"
    # "${SCRATCH}/ap_mo/ap_1p5_sft/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_3600"
    "${SCRATCH}/ap_mo/ap_1p5_sft/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200"
    # "${SCRATCH}/ap_mo/ap_1p5_sft/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_6500"

    # "${SCRATCH}/ap_mo/new_era3/Apertus-1p5-8B-sft-16k-lr6e-5-constant-it38036"
    # "/capstor/store/cscs/swissai/infra01/models/apertus-8b-sft-1.5--lr8e-5"
    # "${SCRATCH}/ap_mo/active_dpo_new4/sft-image"
    # "${SCRATCH}/infra01/models/SFT/latest-sft-notooluse"
    # "${SCRATCH}/infra01/models/Alignment/current_newest_models/MultiModal-OffPolicy-DPO"
)
REF_MODEL_PATH=""  # leave empty to use MODEL_PATH as reference
OUTPUT_BASE_DIR="${SCRATCH}/verl-training"
JUDGE_MODEL="Qwen/Qwen3.6-27B-dmelikidze"
JUDGE_API_KEY="<API_KEY>"

# Fixed training params (override per-run via grid arrays below)
GPU_MEM_UTIL=0.35
ENFORCE_EAGER=false
MAX_NUM_BATCHED_TOKENS=8192
TRAIN_BATCH_SIZE=256
ROLLOUT_N=8
ACTOR_MICRO_BS=1
LOGPROB_MICRO_BS=1
REF_LOGPROB_MICRO_BS=1
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=2048
LR_SCHEDULER_TYPE=linear
LR_WARMUP_STEPS=-1
LR_WARMUP_STEPS_RATIO=0.1
MIN_LR_RATIO=0.0
TOTAL_EPOCHS=1
SAVE_FREQ=500
TP_SIZE=4
FSDP_SIZE=64
GRAD_CLIP=20.0
LENGTH_NORMALIZE=false
ASYNC_ROLLOUT=false
# Reward-loop workers: >0 enables STREAMING annotation (judge scores each rollout
# as it is generated, overlapping the judge with generation); 0 = post-hoc.
REWARD_NUM_WORKERS=16
TRAIN_DATA="${SCRIPT_DIR}/data/train_dolci_final.parquet"
VAL_DATA="${SCRIPT_DIR}/data/train_dolci_final.parquet"
# OFFPOLICY_DATA="${SCRIPT_DIR}/data/train_dolci_offpolicy.parquet"
OFFPOLICY_DATA=""
OFFPOLICY_BATCH_SIZE=256

# ── Resume from checkpoint (set to the exact output dir to resume) ──────
# RESUME_OUTPUT_DIR="${SCRATCH}/verl-training/apertus1.5-sft1.5-online-DPO-lr5e-6-beta0.1-bs256-lenNormfalse-maxPL2048-rollout16-offpolicy-2093197-2093226"
RESUME_OUTPUT_DIR=""

# ── Hyperparameter grid ─────────────────────────────────────────────────
# Each array defines values to sweep. All combinations are launched.
LEARNING_RATES=(5e-6) #(1e-7 5e-7 1e-6 5e-6 1e-5)
DPO_BETAS=(0.1) #(0.01 0.1)

# ── Submit jobs ─────────────────────────────────────────────────────────
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
MODEL_NAME="$(basename "${MODEL_PATH}")"
for LR in "${LEARNING_RATES[@]}"; do
for BETA in "${DPO_BETAS[@]}"; do

    RUN_NAME="${MODEL_NAME}-online-lr${LR}-beta${BETA}-bs${TRAIN_BATCH_SIZE}-lenNorm${LENGTH_NORMALIZE}-maxPL${MAX_PROMPT_LENGTH}-rollout${ROLLOUT_N}-images"
    if [[ -n "${RESUME_OUTPUT_DIR:-}" ]]; then
        OUTPUT_DIR="${RESUME_OUTPUT_DIR}"
    else
        OUTPUT_DIR="${OUTPUT_BASE_DIR}/${RUN_NAME}"
    fi

    echo "Submitting: ${RUN_NAME}"

    sbatch \
        --job-name="orch-${RUN_NAME}" \
        --account="${ACCOUNT}" \
        --reservation="${RESERVATION}" \
        --partition="${PARTITION}" \
        --time="${JOB_TIME}" \
        --nodes=1 \
        --ntasks-per-node=1 \
        --exclude="${EXCLUDE_NODES}" \
        --output="${LOGS_DIR}/orchestrator/%j_orch-${RUN_NAME}.out" \
        --error="${LOGS_DIR}/orchestrator/%j_orch-${RUN_NAME}.err" \
        --export=ALL,\
SERVER_MODEL="${SERVER_MODEL}",\
SERVER_SERVED_NAME="${SERVER_SERVED_NAME}",\
SERVER_NODES="${SERVER_NODES}",\
SERVER_WORKERS="${SERVER_WORKERS}",\
SERVER_NODES_PER_WORKER="${SERVER_NODES_PER_WORKER}",\
SERVER_TP_SIZE="${SERVER_TP_SIZE}",\
SERVER_FRAMEWORK="${SERVER_FRAMEWORK}",\
TRAIN_NODES="${TRAIN_NODES}",\
TRAIN_DATA="${TRAIN_DATA}",\
VAL_DATA="${VAL_DATA}",\
MODEL_PATH="${MODEL_PATH}",\
REF_MODEL_PATH="${REF_MODEL_PATH}",\
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
ASYNC_ROLLOUT="${ASYNC_ROLLOUT}",\
REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS}",\
OFFPOLICY_DATA="${OFFPOLICY_DATA}",\
OFFPOLICY_BATCH_SIZE="${OFFPOLICY_BATCH_SIZE}",\
ACCOUNT="${ACCOUNT}",\
RESERVATION="${RESERVATION}",\
PARTITION="${PARTITION}",\
JOB_TIME="${JOB_TIME}",\
EXCLUDE_NODES="${EXCLUDE_NODES}",\
RESUME_OUTPUT_DIR="${RESUME_OUTPUT_DIR:-}" \
        "${SCRIPT_DIR}/orchestrator.sh"

done
done
done

echo "All orchestrator jobs submitted."
