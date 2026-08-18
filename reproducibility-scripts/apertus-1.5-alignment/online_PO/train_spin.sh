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
REF_LOGPROB_MAX_TOKEN_LEN="${REF_LOGPROB_MAX_TOKEN_LEN:-4096}"
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
        # Offload the actor's fp32 optimizer master + Adam states to CPU. They are only
        # needed at optimizer.step(), NOT during the forward/backward that OOMs, so this
        # frees ~2.2GB at step 1 (fp32 master) and ~6.6GB from step 2 on (master + m/v)
        # while keeping enforce_eager=false. Small per-step CPU<->GPU transfer cost.
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
        # Ref shards (~2.3GB/rank fp32 at FSDP=128) page CPU<->GPU around each ref pass
        # (recipe fsdp_workers.compute_ref_log_prob) instead of squatting on the GPU
        # during the actor update, which OOMs by <1GB without this.
        actor_rollout_ref.ref.fsdp_config.param_offload=true
        # actor_rollout_ref.actor.fsdp_config.param_offload=true
        # actor_rollout_ref.model.enable_activation_offload=true
    )
fi

# ── Ref-pass dynamic batching (text-only data) ──────────────────────────
# Packs ref micro-batches by token length (~16 forwards/rank -> ~3-4), cutting
# ref_log_prob from ~45s to ~12s. This is SAFE now that recipe/spin/dp_actor.py
# passes dp_group=WORLD to rearrange_micro_batches, which all-reduces the
# micro-batch COUNT to the max across ranks (same_micro_num_in_dp) so the
# per-forward FSDP all-gathers stay in lockstep. Without that dp_group the
# uneven per-rank counts desynced the collective -> NCCL watchdog killed a rank
# (the ~673s crash). rmpad is already on, and the reorder is reverted via
# get_reverse_idx, so values are unchanged. REF_LOGPROB_MAX_TOKEN_LEN must be
# >= the longest sequence (4096 here).
REF_PACKING_ARGS=()
if [[ "${LARGE_MODEL}" == "true" ]]; then
    REF_PACKING_ARGS=(
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${REF_LOGPROB_MAX_TOKEN_LEN}
    )
fi

# ── Large-model actor load dtype ────────────────────────────────────────
# DO NOT set actor model_dtype=bf16. model_dtype is the *master/optimizer*
# dtype (the FSDP flat param AdamW updates), NOT the compute dtype. verl warns
# about this (fsdp_workers.py:380). bf16 master + lr=2e-6 => each update
# (~1e-6) is below the bf16 mantissa step of the weights (~1e-4) and rounds
# away -> the policy never moves (grad_norm large but KL~0, frozen policy).
# Correct setup = fp32 master (model_dtype=fp32, the default) + bf16 COMPUTE
# (fsdp.yaml `dtype: bfloat16`, applied via MixedPrecision) -> fast & low GPU
# mem without losing updates. We keep fp32 explicit here as a guardrail.
# Host-RAM note: fp32 load materializes ~292GB on rank-0, but low_cpu_mem_usage
# (streamed load) + removing the redundant ref-worker actor build keep it under
# the 450GB node budget.
ACTOR_DTYPE_ARGS=()
if [[ "${LARGE_MODEL}" == "true" ]]; then
    ACTOR_DTYPE_ARGS=(
        actor_rollout_ref.actor.fsdp_config.model_dtype=fp32
    )
fi

# ── Skip the redundant actor build on the ref-only worker ───────────────
# The ref worker (role="ref") reuses the combined-worker init path, which
# otherwise builds a full SECOND copy of the model + Adam optimizer that the
# ref never uses. On rank-0 (sync_module_states) both full copies land in host
# RAM at load -> ~2x146GB for 70B -> Ray OOM-kill. Skipping it roughly halves
# the ref worker's host peak. Gated on LARGE_MODEL so small-model runs keep the
# original path and this is trivially reversible.
REF_SLIM_ARGS=()
if [[ "${LARGE_MODEL}" == "true" ]]; then
    REF_SLIM_ARGS=(
        +actor_rollout_ref.ref.skip_redundant_actor_build=true
    )
fi

# ── Checkpoint load: low_cpu_mem_usage is HARDCODED True in verl ────────
# DO NOT pass actor_rollout_ref.model.low_cpu_mem_usage as a hydra arg: HFModelConfig
# is a strict dataclass and rejects the non-schema key (ConfigKeyError, even with `+`).
# It is hardcoded True in verl/workers/fsdp_workers.py (streams weights, keeps rank-0
# host RAM ~1x). This empty array is kept only so the passthrough below stays valid.
LOW_CPU_MEM_ARGS=()

# ── Rollout weight load: skip SGLang's redundant disk read (hybrid only) ─
# In colocated/hybrid mode the trainer does an initial actor->rollout
# update_weights (spin_trainer ~L1138, "Initial weight sync to rollout
# replicas complete.") BEFORE the first generation, so SGLang does not need to
# read the checkpoint from disk itself -- it re-reads the same 70B the actor
# already loaded. load_format=dummy inits the engine with empty weights and
# lets that sync populate them over NCCL, which (a) skips the ~8-10min
# "Multi-thread loading shards" stage and (b) removes the slow-load straggler
# that trips SGLang's 480s init barrier (the transient crash we hit). Only
# valid in hybrid mode (verl forces dummy->auto otherwise). Gated on
# LARGE_MODEL; small runs keep auto.
ROLLOUT_LOAD_FORMAT="auto"
if [[ "${LARGE_MODEL}" == "true" ]]; then
    ROLLOUT_LOAD_FORMAT="dummy"
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
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.load_format=${ROLLOUT_LOAD_FORMAT} \
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
    "${ACTOR_DTYPE_ARGS[@]}" \
    "${REF_SLIM_ARGS[@]}" \
    "${LOW_CPU_MEM_ARGS[@]}" \
    "$@"
