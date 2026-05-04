#!/bin/bash
#SBATCH --job-name=spin-multinode
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/dmelikidze/online-dpo/logs/training/%j-%x.out
#SBATCH --error=/iopsstor/scratch/cscs/dmelikidze/online-dpo/logs/training/%j-%x.err
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --partition=normal

# ── Run configuration ───────────────────────────────────────────────────
# Change these for each run. Everything else stays the same.

export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://172.28.38.28:30000/v1}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-sk-rc-MH1IEiFLN35rXSJq5pWECQ}"
export JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.6-27B-dmelikidze}"

export MODEL_PATH="${MODEL_PATH:-/iopsstor/scratch/cscs/dmelikidze/huggingface/hub/models--swiss-ai--Apertus-8B-Instruct-2509-SFT/snapshots/d57e4f1a3baa6315c60707346b5498b48b40a364}"
export OUTPUT_DIR="${OUTPUT_DIR:-/iopsstor/scratch/cscs/dmelikidze/verl-training/online-dpo-run-from-SFT/}"
export OUTPUT_DIR="${OUTPUT_DIR%/}-${SLURM_JOB_ID}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(basename "${OUTPUT_DIR}")}"

# Optional overrides (uncomment or pass as env vars)
# export TRAIN_DATA=...
# export VAL_DATA=...
# export PROJECT_NAME=apertus-1.5-online-dpo
# export LEARNING_RATE=1e-6
# export DPO_BETA=0.1
# export SAVE_FREQ=50
# export LENGTH_NORMALIZE=false

# ─────────────────────────────────────────────────────────────────────────

set -x

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"

# ── Step 1: Resolve head node IP ────────────────────────────────────────
# SLURM_JOB_NODELIST contains compressed node names like "nid00[123-124]".
# scontrol expands them into one-per-line hostnames.
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
# Resolve the hostname to an IP address for Ray to bind to.
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" --environment=verl hostname --ip-address)

# If multiple IPs are returned (IPv4 + IPv6), pick the shorter (IPv4) one.
if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

# Ray temp dir must be on a node-local filesystem for Unix domain sockets.
# /dev/shm is RAM-backed and shared between containers on the same node.
RAY_TMPDIR="/dev/shm/ray_tmp_${SLURM_JOB_ID}"
export RAY_TMPDIR

# Unset AMD ROCm variable that conflicts with CUDA_VISIBLE_DEVICES in verl workers
unset ROCR_VISIBLE_DEVICES

# ── Helper: start Ray cluster on all nodes ──────────────────────────────
start_ray_cluster() {
    echo "Starting Ray HEAD at $head_node ($head_node_ip)"
    srun --nodes=1 --ntasks=1 -w "$head_node" --environment=verl \
        bash -c "
            unset ROCR_VISIBLE_DEVICES && \
            export WANDB_ENTITY=apertus && \
            cd ${SCRIPT_DIR}/verl && pip install -e . --quiet && cd ${SCRIPT_DIR} && \
            ray start --head --node-ip-address=${head_node_ip} --port=${port} \
                --num-gpus ${SLURM_GPUS_PER_NODE} --temp-dir=${RAY_TMPDIR} --block
        " &

    sleep 12

    worker_num=$((SLURM_JOB_NUM_NODES - 1))
    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        echo "Starting Ray WORKER $i at $node_i"
        srun --nodes=1 --ntasks=1 -w "$node_i" --environment=verl \
            bash -c "
                unset ROCR_VISIBLE_DEVICES && \
                export WANDB_ENTITY=apertus && \
                cd ${SCRIPT_DIR}/verl && pip install -e . --quiet && cd ${SCRIPT_DIR} && \
                ray start --address ${ip_head} \
                    --num-gpus ${SLURM_GPUS_PER_NODE} --temp-dir=${RAY_TMPDIR} --block
            " &
        sleep 1
    done
}

# ── Helper: stop Ray on all nodes ───────────────────────────────────────
stop_ray_cluster() {
    echo "Stopping Ray cluster on all nodes..."
    for node in "${nodes_array[@]}"; do
        srun --nodes=1 --ntasks=1 -w "$node" --environment=verl \
            bash -c "ray stop --force 2>/dev/null; true" &
    done
    wait
    sleep 10
}

# ── Step 2+3: Start Ray cluster and launch training with auto-retry ─────
# On transient CUDA/NCCL errors the entire Ray cluster is corrupted, so we
# tear it down, restart, and resume from the latest checkpoint.
# trainer.resume_mode=auto tells verl to find the latest checkpoint.
# trainer.save_freq should be low (e.g. 50) to minimise lost work.

MAX_RETRIES=5

start_ray_cluster

for attempt in $(seq 1 $MAX_RETRIES); do
    echo ""
    echo "===== Training attempt ${attempt}/${MAX_RETRIES} ====="
    echo ""

    PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" --environment=verl \
        bash -c "
            unset ROCR_VISIBLE_DEVICES && \
            export JUDGE_BASE_URL='${JUDGE_BASE_URL}' && \
            export JUDGE_API_KEY='${JUDGE_API_KEY}' && \
            export JUDGE_MODEL='${JUDGE_MODEL}' && \
            export MODEL_PATH='${MODEL_PATH}' && \
            export EXPERIMENT_NAME='${EXPERIMENT_NAME}' && \
            export OUTPUT_DIR='${OUTPUT_DIR}' && \
            export PROJECT_NAME='${PROJECT_NAME:-}' && \
            export TRAIN_DATA='${TRAIN_DATA:-}' && \
            export VAL_DATA='${VAL_DATA:-}' && \
            export LEARNING_RATE='${LEARNING_RATE:-}' && \
            export LR_WARMUP_STEPS='${LR_WARMUP_STEPS:-}' && \
            export LR_WARMUP_STEPS_RATIO='${LR_WARMUP_STEPS_RATIO:-}' && \
            export LR_SCHEDULER_TYPE='${LR_SCHEDULER_TYPE:-}' && \
            export MIN_LR_RATIO='${MIN_LR_RATIO:-}' && \
            export DPO_BETA='${DPO_BETA:-}' && \
            export GRAD_CLIP='${GRAD_CLIP:-}' && \
            export TRAIN_BATCH_SIZE='${TRAIN_BATCH_SIZE:-}' && \
            export MAX_PROMPT_LENGTH='${MAX_PROMPT_LENGTH:-}' && \
            export MAX_RESPONSE_LENGTH='${MAX_RESPONSE_LENGTH:-}' && \
            export ROLLOUT_N='${ROLLOUT_N:-}' && \
            export TP_SIZE='${TP_SIZE:-}' && \
            export FSDP_SIZE='${FSDP_SIZE:-}' && \
            export SAVE_FREQ='${SAVE_FREQ:-}' && \
            export LENGTH_NORMALIZE='${LENGTH_NORMALIZE:-}' && \
            export TOTAL_EPOCHS='${TOTAL_EPOCHS:-}' && \
            export GPU_MEM_UTIL='${GPU_MEM_UTIL:-}' && \
            export ACTOR_MICRO_BS='${ACTOR_MICRO_BS:-}' && \
            export ROLLOUT_N='${ROLLOUT_N:-}' && \
            export LOGPROB_MICRO_BS='${LOGPROB_MICRO_BS:-}' && \
            export REF_LOGPROB_MICRO_BS='${REF_LOGPROB_MICRO_BS:-}' && \
            export ENFORCE_EAGER='${ENFORCE_EAGER:-}' && \
            export MAX_NUM_BATCHED_TOKENS='${MAX_NUM_BATCHED_TOKENS:-}' && \
            cd ${SCRIPT_DIR}/verl && pip install -e . --quiet && cd ${SCRIPT_DIR} && \
            export RAY_ADDRESS=${ip_head} && \
            export RAY_TMPDIR=${RAY_TMPDIR} && \
            bash ${SCRIPT_DIR}/train_spin.sh \
                trainer.nnodes=${SLURM_NNODES} \
                trainer.n_gpus_per_node=${SLURM_GPUS_PER_NODE} \
                trainer.resume_mode=auto
        "
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "Training completed successfully on attempt ${attempt}."
        break
    fi

    echo "Training failed with exit code ${exit_code} on attempt ${attempt}/${MAX_RETRIES}."

    if [ $attempt -lt $MAX_RETRIES ]; then
        echo "Restarting Ray cluster and resuming from latest checkpoint..."
        stop_ray_cluster
        start_ray_cluster
    else
        echo "All ${MAX_RETRIES} attempts exhausted. Exiting."
        exit $exit_code
    fi
done
