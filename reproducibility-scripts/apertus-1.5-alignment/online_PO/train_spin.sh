#!/bin/bash
set -x

export WANDB_ENTITY="apertus"
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Run-specific parameters (set via env vars, with defaults) ───────────
JUDGE_BASE_URL="${JUDGE_BASE_URL:?JUDGE_BASE_URL must be set}"
JUDGE_API_KEY="${JUDGE_API_KEY:?JUDGE_API_KEY must be set}"
JUDGE_MODEL="${JUDGE_MODEL:?JUDGE_MODEL must be set}"
export JUDGE_BASE_URL JUDGE_API_KEY JUDGE_MODEL

MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?EXPERIMENT_NAME must be set}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set}"

REF_MODEL_PATH="${REF_MODEL_PATH:-}"
TRAIN_DATA="${TRAIN_DATA:-}"
VAL_DATA="${VAL_DATA:-}"
PROJECT_NAME="${PROJECT_NAME:-apertus-1.5-online-dpo}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:--1}"
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.1}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.0}"
GRAD_CLIP="${GRAD_CLIP:-20.0}"
DPO_BETA="${DPO_BETA:-0.1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
ROLLOUT_N="${ROLLOUT_N:-2}"
TP_SIZE="${TP_SIZE:-4}"
FSDP_SIZE="${FSDP_SIZE:-16}"
SAVE_FREQ="${SAVE_FREQ:-125}"
LENGTH_NORMALIZE="${LENGTH_NORMALIZE:-false}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.35}"
ACTOR_MICRO_BS="${ACTOR_MICRO_BS:-1}"
LOGPROB_MICRO_BS="${LOGPROB_MICRO_BS:-1}"
REF_LOGPROB_MICRO_BS="${REF_LOGPROB_MICRO_BS:-1}"
REF_LOGPROB_MAX_TOKEN_LEN="${REF_LOGPROB_MAX_TOKEN_LEN:-16384}"
# DPO actor-update packing (Fix B). DPO_DYNAMIC_BSZ=false reverts to the
# original fixed-micro_bs loop with no code change. ACTOR_MAX_TOKEN_LEN bounds
# the padded token-slots (rows*width) per packed micro-batch; raise to pack more
# pairs, lower if memory is tight.
DPO_DYNAMIC_BSZ="${DPO_DYNAMIC_BSZ:-true}"
ACTOR_MAX_TOKEN_LEN="${ACTOR_MAX_TOKEN_LEN:-8192}"
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
ASYNC_ROLLOUT="${ASYNC_ROLLOUT:-false}"
REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-16}"
OFFPOLICY_DATA="${OFFPOLICY_DATA:-}"
OFFPOLICY_BATCH_SIZE="${OFFPOLICY_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
LARGE_MODEL="${LARGE_MODEL:-false}"

# ── Script paths ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RECIPE_DIR="${SCRIPT_DIR}/recipe"

if [ ! -e "$RECIPE_DIR" ]; then
    ln -sf /iopsstor/scratch/cscs/dmelikidze/verl-recipe "$RECIPE_DIR"
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Default data paths if not overridden
DATA_DIR="${SCRIPT_DIR}/data"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/train_dolci.parquet}"
VAL_DATA="${VAL_DATA:-${DATA_DIR}/train_dolci.parquet}"
OFFPOLICY_DATA="${OFFPOLICY_DATA:-}"

# ── Large-model memory optimizations (CPU offload) ──────────────────────
# Only applied for big models (e.g. 70B), gated by LARGE_MODEL from
# launch.sh. Offloading params/optimizer/activations to CPU is what lets a
# 70B fit on 96GB GH200s; at 8B it is pure CPU<->GPU paging overhead, so we
# leave it off and these args expand to nothing.
OFFLOAD_ARGS=()
if [[ "${LARGE_MODEL}" == "true" ]]; then
    OFFLOAD_ARGS=(
        # actor_rollout_ref.actor.fsdp_config.param_offload=true
        # actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
        # actor_rollout_ref.ref.fsdp_config.param_offload=true
        # actor_rollout_ref.model.enable_activation_offload=true
    )
fi

# ── Ref-pass dynamic batching (text-only data) ──────────────────────────
# With the multimodal false-positive fixed in compute_log_prob, the ref
# log-prob pass packs micro-batches by token length instead of padding every
# sequence to max_len. The reorder is reverted (get_reverse_idx) before
# return, so the positional chosen/rejected split is unaffected. Gated on
# LARGE_MODEL alongside the other 70B settings; tune the budget via
# REF_LOGPROB_MAX_TOKEN_LEN (must be >= the longest sequence, 4096 here).
REF_PACKING_ARGS=()
if [[ "${LARGE_MODEL}" == "true" ]]; then
    REF_PACKING_ARGS=(
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${REF_LOGPROB_MAX_TOKEN_LEN}
    )
fi

python3 -m recipe.spin.main_spin \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.dataloader_num_workers=0 \
    data.filter_overlong_prompts=true \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    ${REF_MODEL_PATH:+actor_rollout_ref.model.ref_path=${REF_MODEL_PATH}} \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${LR_WARMUP_STEPS_RATIO} \
    actor_rollout_ref.actor.optim.lr_scheduler_type=${LR_SCHEDULER_TYPE} \
    actor_rollout_ref.actor.optim.min_lr_ratio=${MIN_LR_RATIO} \
    actor_rollout_ref.actor.grad_clip=${GRAD_CLIP} \
    actor_rollout_ref.actor.dpo_beta=${DPO_BETA} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_MICRO_BS} \
    +actor_rollout_ref.actor.dpo_use_dynamic_bsz=${DPO_DYNAMIC_BSZ} \
    +actor_rollout_ref.actor.dpo_max_token_len_per_gpu=${ACTOR_MAX_TOKEN_LEN} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGPROB_MICRO_BS} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOGPROB_MICRO_BS} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE} \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=true \
    actor_rollout_ref.ref.fsdp_config.fsdp_size=${FSDP_SIZE} \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=true \
    actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.load_format=auto \
    data.trust_remote_code=true \
    reward_model.reward_manager=naive \
    reward.num_workers=${REWARD_NUM_WORKERS} \
    algorithm.adv_estimator=null \
    +algorithm.length_normalize=${LENGTH_NORMALIZE} \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.ref_update_freq=-1 \
    trainer.balance_batch=false \
    trainer.val_before_train=false \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.logger='["console", "wandb"]' \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.test_freq=-1 \
    trainer.val_only=false \
    +trainer.async_rollout=${ASYNC_ROLLOUT} \
    +data.offpolicy_files="${OFFPOLICY_DATA}" \
    +data.offpolicy_batch_size=${OFFPOLICY_BATCH_SIZE} \
    "${OFFLOAD_ARGS[@]}" \
    "${REF_PACKING_ARGS[@]}" \
    "$@"
