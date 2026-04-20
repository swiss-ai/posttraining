#!/bin/bash
set -x

export WANDB_ENTITY="apertus"
export HYDRA_FULL_ERROR=1

# Judge server for active_ultrafeedback custom reward (recipe.spin.active_ultrafeedback_reward)
# export JUDGE_BASE_URL="https://api.swissai.svc.cscs.ch/v1"
export JUDGE_BASE_URL="http://172.28.44.28:30000/v1"
export JUDGE_API_KEY="sk-rc-MH1IEiFLN35rXSJq5pWECQ"
export JUDGE_MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507-dmelikidze"       # e.g. Llama-3.3-70B-Instruct

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
RECIPE_DIR="${SCRIPT_DIR}/recipe"

# Symlink to local fork for development (switch to git clone for production)
if [ ! -e "$RECIPE_DIR" ]; then
    ln -sf /iopsstor/scratch/cscs/dmelikidze/verl-recipe "$RECIPE_DIR"
fi

SPIN_DIR="${RECIPE_DIR}/spin"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

#/iopsstor/scratch/cscs/dmelikidze/huggingface/hub/models--swiss-ai--Apertus-8B-Instruct-2509-SFT/snapshots/d57e4f1a3baa6315c60707346b5498b48b40a364 \

python3 -m recipe.spin.main_spin \
    data.train_files="${DATA_DIR}/train_dolci.parquet" \
    data.val_files="${DATA_DIR}/train_dolci.parquet" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.dataloader_num_workers=0 \
    data.filter_overlong_prompts=true \
    actor_rollout_ref.model.path=/iopsstor/scratch/cscs/dmelikidze/huggingface/hub/models--swiss-ai--Apertus-8B-Instruct-2509-SFT/snapshots/d57e4f1a3baa6315c60707346b5498b48b40a364 \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=150 \
    actor_rollout_ref.actor.dpo_beta=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=true \
    actor_rollout_ref.ref.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=true \
    actor_rollout_ref.rollout.load_format=auto \
    data.trust_remote_code=true \
    reward_model.reward_manager=naive \
    algorithm.adv_estimator=null \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=1 \
    trainer.ref_update_freq=-1 \
    trainer.val_before_train=false \
    trainer.project_name=apertus-1.5-online-dpo \
    trainer.experiment_name=online-dpo-run2 \
    trainer.logger='["console", "wandb"]' \
    trainer.save_freq=250 \
    trainer.default_local_dir=/iopsstor/scratch/cscs/dmelikidze/verl-training/online-dpo-run2/ \
    trainer.test_freq=-1 \
    trainer.val_only=false \
    "$@"
