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
export WANDB_ENTITY="${WANDB_ENTITY:-matrs01}"
export PYTHONPATH="${PROJECT_ROOT_AT}/src/post_training/qrpo-verl${PYTHONPATH:+:${PYTHONPATH}}"

TRAINER_NNODES="${TRAINER_NNODES:-${SLURM_NNODES:-2}}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
TRAIN_MICRO_BATCH_SIZE_PER_GPU="${TRAIN_MICRO_BATCH_SIZE_PER_GPU:-16}"
TRAIN_BATCH_SIZE_PER_GPU="${TRAIN_BATCH_SIZE_PER_GPU:-16}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-${TRAIN_MICRO_BATCH_SIZE_PER_GPU}}"
TRAJECTORIES_PER_PROMPT="${TRAJECTORIES_PER_PROMPT:-1}"
TOTAL_TRAIN_GPUS="$((TRAINER_NNODES * N_GPUS_PER_NODE))"
if [[ -n "${GLOBAL_TRAIN_BATCH_SIZE:-}" ]]; then
  QRPO_TRAIN_BATCH_SIZE="${GLOBAL_TRAIN_BATCH_SIZE}"
  if (( QRPO_TRAIN_BATCH_SIZE % TOTAL_TRAIN_GPUS != 0 )); then
    echo "ERROR: GLOBAL_TRAIN_BATCH_SIZE=${QRPO_TRAIN_BATCH_SIZE} must be divisible by total GPUs ${TOTAL_TRAIN_GPUS}." >&2
    exit 1
  fi
  TRAIN_BATCH_SIZE_PER_GPU="$((QRPO_TRAIN_BATCH_SIZE / TOTAL_TRAIN_GPUS))"
else
  QRPO_TRAIN_BATCH_SIZE="$((TOTAL_TRAIN_GPUS * TRAIN_BATCH_SIZE_PER_GPU))"
fi

if (( TRAJECTORIES_PER_PROMPT <= 0 )); then
  echo "ERROR: TRAJECTORIES_PER_PROMPT must be positive, got ${TRAJECTORIES_PER_PROMPT}." >&2
  exit 1
fi
if (( QRPO_TRAIN_BATCH_SIZE % TRAJECTORIES_PER_PROMPT != 0 )); then
  echo "ERROR: global train batch ${QRPO_TRAIN_BATCH_SIZE} must be divisible by TRAJECTORIES_PER_PROMPT=${TRAJECTORIES_PER_PROMPT}." >&2
  exit 1
fi
QRPO_PROMPT_BATCH_SIZE="$((QRPO_TRAIN_BATCH_SIZE / TRAJECTORIES_PER_PROMPT))"

if (( QRPO_TRAIN_BATCH_SIZE % (TOTAL_TRAIN_GPUS * TRAIN_MICRO_BATCH_SIZE_PER_GPU) != 0 )); then
  echo "ERROR: global train batch ${QRPO_TRAIN_BATCH_SIZE} must be divisible by total_gpus * train_micro_batch_size_per_gpu = ${TOTAL_TRAIN_GPUS} * ${TRAIN_MICRO_BATCH_SIZE_PER_GPU}." >&2
  exit 1
fi
if (( QRPO_TRAIN_BATCH_SIZE % (TOTAL_TRAIN_GPUS * LOG_PROB_MICRO_BATCH_SIZE_PER_GPU) != 0 )); then
  echo "ERROR: global train batch ${QRPO_TRAIN_BATCH_SIZE} must be divisible by total_gpus * log_prob_micro_batch_size_per_gpu = ${TOTAL_TRAIN_GPUS} * ${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}." >&2
  exit 1
fi
GRAD_ACCUM_STEPS="$((TRAIN_BATCH_SIZE_PER_GPU / TRAIN_MICRO_BATCH_SIZE_PER_GPU))"
LOG_PROB_ACCUM_STEPS="$((TRAIN_BATCH_SIZE_PER_GPU / LOG_PROB_MICRO_BATCH_SIZE_PER_GPU))"
REF_REWARD_GENERATION_PROMPT_BATCH_SIZE="${REF_REWARD_GENERATION_PROMPT_BATCH_SIZE:-128}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_WAIT_TIMEOUT_S="${RAY_WAIT_TIMEOUT_S:-600}"
RAY_JOB_DIR="${RAY_JOB_DIR:-${SCRATCH:-/tmp}/qrpo-ray/${SLURM_JOB_ID:-manual}}"
RAY_HEAD_IP_FILE="${RAY_JOB_DIR}/head_ip"
RAY_DONE_FILE="${RAY_JOB_DIR}/done"

mkdir -p "${RAY_JOB_DIR}"
rm -f "${RAY_DONE_FILE}"

cleanup() {
  set +e
  ray stop --force >/dev/null 2>&1
  if [[ "${SLURM_NODEID:-0}" == "0" ]]; then
    touch "${RAY_DONE_FILE}"
  fi
}
trap cleanup EXIT

ray stop --force >/dev/null 2>&1 || true

node_ip_address() {
  hostname --ip-address | tr ' ' '\n' | awk '/^[0-9]+\./ { print; found = 1; exit } { if (candidate == "") candidate = $0 } END { if (!found && candidate != "") print candidate }'
}

if [[ "${SLURM_NODEID:-0}" == "0" ]]; then
  HEAD_IP="$(node_ip_address)"
  echo "${HEAD_IP}" > "${RAY_HEAD_IP_FILE}"
  echo "Starting Ray head at ${HEAD_IP}:${RAY_PORT}"
  ray start \
    --head \
    --node-ip-address="${HEAD_IP}" \
    --port="${RAY_PORT}" \
    --num-gpus="${N_GPUS_PER_NODE}" \
    --num-cpus="${RAY_NUM_CPUS_PER_NODE:-$(nproc)}" \
    --dashboard-host=0.0.0.0
else
  echo "Waiting for Ray head address in ${RAY_HEAD_IP_FILE}"
  while [[ ! -s "${RAY_HEAD_IP_FILE}" ]]; do
    sleep 2
  done

  HEAD_IP="$(cat "${RAY_HEAD_IP_FILE}")"
  echo "Starting Ray worker for ${HEAD_IP}:${RAY_PORT}"
  until ray start \
    --address="${HEAD_IP}:${RAY_PORT}" \
    --num-gpus="${N_GPUS_PER_NODE}" \
    --num-cpus="${RAY_NUM_CPUS_PER_NODE:-$(nproc)}"; do
    sleep 5
  done

  while [[ ! -e "${RAY_DONE_FILE}" ]]; do
    sleep 10
  done
  exit 0
fi

export RAY_ADDRESS="$(cat "${RAY_HEAD_IP_FILE}"):${RAY_PORT}"
export TRAINER_NNODES
export N_GPUS_PER_NODE
export TOTAL_TRAIN_GPUS
export TRAIN_MICRO_BATCH_SIZE_PER_GPU
export TRAIN_BATCH_SIZE_PER_GPU
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU
export TRAJECTORIES_PER_PROMPT
export QRPO_PROMPT_BATCH_SIZE
export QRPO_TRAIN_BATCH_SIZE
export GRAD_ACCUM_STEPS
export LOG_PROB_ACCUM_STEPS
export REF_REWARD_GENERATION_PROMPT_BATCH_SIZE
export RAY_WAIT_TIMEOUT_S

cat <<EOF
QRPO batch-size math:
  total_gpus = trainer.nnodes * trainer.n_gpus_per_node = ${TRAINER_NNODES} * ${N_GPUS_PER_NODE} = ${TOTAL_TRAIN_GPUS}
  global_train_batch_size = trajectory batch size per optimizer update = ${QRPO_TRAIN_BATCH_SIZE}
  trajectories_per_prompt = n_online + n_offline = ${TRAJECTORIES_PER_PROMPT}
  prompt_batch_size = global_train_batch_size / trajectories_per_prompt = ${QRPO_PROMPT_BATCH_SIZE}
  train_batch_size_per_gpu = global_train_batch_size / total_gpus = ${TRAIN_BATCH_SIZE_PER_GPU}
  train_micro_batch_size_per_gpu = ${TRAIN_MICRO_BATCH_SIZE_PER_GPU}
  actor gradient accumulation steps = train_batch_size_per_gpu / train_micro_batch_size_per_gpu = ${GRAD_ACCUM_STEPS}
  log_prob_micro_batch_size_per_gpu = ${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
  ref-logprob accumulation steps = train_batch_size_per_gpu / log_prob_micro_batch_size_per_gpu = ${LOG_PROB_ACCUM_STEPS}
EOF

python - <<'PY'
import os
import time

import ray

address = os.environ["RAY_ADDRESS"]
expected_gpus = int(os.environ["TRAINER_NNODES"]) * int(os.environ["N_GPUS_PER_NODE"])
deadline = time.time() + int(os.environ["RAY_WAIT_TIMEOUT_S"])

ray.init(address=address)
try:
    while True:
        resources = ray.cluster_resources()
        available_gpus = int(resources.get("GPU", 0))
        if available_gpus >= expected_gpus:
            print(f"Ray cluster ready: {available_gpus} GPUs visible.")
            break
        if time.time() > deadline:
            raise TimeoutError(
                f"Timed out waiting for {expected_gpus} Ray GPUs; "
                f"currently visible: {available_gpus}."
            )
        print(
            f"Waiting for Ray cluster: {available_gpus}/{expected_gpus} GPUs visible."
        )
        time.sleep(5)
finally:
    ray.shutdown()
PY

python -m entrypoints.main_qrpo \
  data.path=/users/smatreno/projects/posttraining/dev/artifacts/private/datasets/MaxMin-Filtered-Ref-Completions-30-Annotated-Combined-Final-OnlineQRPOformat/train_split \
  actor_rollout_ref.model.path=/users/smatreno/projects/posttraining/dev/artifacts/private/baseline-checkpoints/Apertus-8b-sft-1.5--lr8e-5 \
  tokenizer.use_eos_as_pad=false \
  data.train_batch_size="${QRPO_PROMPT_BATCH_SIZE}" \
  qrpo_runtime.train_mini_batch_size="${QRPO_TRAIN_BATCH_SIZE}" \
  qrpo_runtime.train_micro_batch_size_per_gpu="${TRAIN_MICRO_BATCH_SIZE_PER_GPU}" \
  qrpo_runtime.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  source_schedule.n_online=1 \
  source_schedule.n_offline=0 \
  actor_rollout_ref.rollout.n=1 \
  reward.num_workers=32 \
  reward.custom_reward_function.path=/users/smatreno/projects/posttraining/dev/src/post_training/qrpo-verl/rewards/active_ultrafeedback_reward.py \
  reward.custom_reward_function.name=compute_score \
  online_rollout.data_source=activeultrafeedback \
  trainer.save_freq=100 \
  trainer.test_freq=0 \
  trainer.default_local_dir=/users/smatreno/projects/posttraining/dev/artifacts/private/outputs/online-qrpo/online-qrpo-test/apertus_8b_online_qrpo_debug \
  trainer.val_before_train=false \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=online-qrpo-test \
  trainer.experiment_name=apertus_8b_online_qrpo_debug \
  offline_selector=random \
  online_rollout.completion_logging.enabled=false \
  online_rollout.completion_logging.outputs='["wandb"]' \
  online_rollout.completion_logging.selection=all \
  ref_rewards.initial_source=generate \
  ref_rewards.initial_version=ref_step_000000 \
  ref_rewards.refresh_interval_epochs=null \
  data.ref_rewards_key=null \
  ref_rewards.store_dir=/users/smatreno/projects/posttraining/dev/artifacts/private/outputs/ref_reward_stores/apertus-8b-sft-1.5--lr8e-5_initial_activeuf_helpfulness \
  ref_rewards.generation_num_chunks=null \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
  actor_rollout_ref.rollout.agent.num_workers=32 \
  actor_rollout_ref.rollout.max_num_seqs=4096 \
  actor_rollout_ref.rollout.max_num_batched_tokens=131072 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.disable_log_stats=false \
  ++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL=INFO \
  trainer.nnodes="${TRAINER_NNODES}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  ref_rewards.generation_prompt_batch_size="${REF_REWARD_GENERATION_PROMPT_BATCH_SIZE}" \
  ++ray_kwargs.ray_init.address="${RAY_ADDRESS}" \
  "${EXTRA_HYDRA_OVERRIDES[@]}"
