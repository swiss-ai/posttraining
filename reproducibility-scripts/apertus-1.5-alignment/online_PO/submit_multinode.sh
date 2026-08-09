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
if [[ -z "${RESUME_OUTPUT_DIR:-}" ]]; then
    export OUTPUT_DIR="${OUTPUT_DIR%/}-${SLURM_JOB_ID}"
fi
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(basename "${OUTPUT_DIR}")}"
# Append the training job id so each launch gets a distinct wandb run name
# (avoids many runs sharing one display name and looking merged).
export EXPERIMENT_NAME="${EXPERIMENT_NAME}-${SLURM_JOB_ID}"

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

# ── Fast load toggle (default OFF) ──────────────────────────────────────
# FAST_LOAD=true -> only GLOBAL rank 0 reads the checkpoint; every other rank builds
# the model on the meta device and FSDP (sync_module_states) broadcasts rank-0's
# weights. Avoids BOTH the per-node OOM (4x fp32 materialization at the shard barrier)
# and the 64-way Lustre read contention -> ~2-3min init, no staging needed. verl reads
# this via the VERL_RANK0_ONLY_LOAD env var, exported into the ray-start srun env below
# so the worker processes inherit it. Default off = load on all ranks (unchanged).
FAST_LOAD="${FAST_LOAD:-false}"
RANK0_LOAD_FLAG=0
[[ "${FAST_LOAD}" == "true" ]] && RANK0_LOAD_FLAG=1
echo "FAST_LOAD=${FAST_LOAD} -> VERL_RANK0_ONLY_LOAD=${RANK0_LOAD_FLAG}"

# ── Helper: start Ray cluster on all nodes ──────────────────────────────
start_ray_cluster() {
    echo "Starting Ray HEAD at $head_node ($head_node_ip)"
    srun --nodes=1 --ntasks=1 -w "$head_node" --environment=verl \
        bash -c "
            unset ROCR_VISIBLE_DEVICES && \
            export WANDB_ENTITY=apertus && \
            export VERL_PPO_LOGGING_LEVEL=DEBUG && export VERL_LOGGING_LEVEL=DEBUG && \
            export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 && \
            export VERL_RANK0_ONLY_LOAD=${RANK0_LOAD_FLAG} && \
            export PYTHONPATH=${SCRIPT_DIR}/verl:${SCRIPT_DIR}:\${PYTHONPATH:-} && \
            ray start --head --node-ip-address=${head_node_ip} --port=${port} \
                --num-gpus ${SLURM_GPUS_PER_NODE} --temp-dir=${RAY_TMPDIR} --block
        " &

    sleep 12

    # ── Start workers: ONE srun step per node (matches the known-good run) ───
    # The batched + throwaway-`bash -c true` mount barrier was removed: that
    # extra `--environment=verl` container per node set up and tore down the CXI
    # fabric right before `ray start`, which turned rare VNI_NOT_FOUND fabric
    # errors into CONSISTENT ones at the first cross-node NCCL collective (the
    # known-good run 2763918 used this plain per-node loop and had ZERO VNI
    # errors). So: one clean `srun ... ray start` per node, nothing else touching
    # the container/fabric first. WORKER_START_GAP staggers launches (the known-
    # good run used 10s; lower is fine now that bad nodes are excluded).
    worker_num=$((SLURM_JOB_NUM_NODES - 1))
    WORKER_START_GAP="${WORKER_START_GAP:-2}"   # seconds between per-node launches
    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        echo "Starting Ray WORKER $i at $node_i"
        srun --nodes=1 --ntasks=1 -w "$node_i" --environment=verl \
            bash -c "
                unset ROCR_VISIBLE_DEVICES && \
                export WANDB_ENTITY=apertus && \
                export VERL_PPO_LOGGING_LEVEL=DEBUG && export VERL_LOGGING_LEVEL=DEBUG && \
                export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 && \
                export VERL_RANK0_ONLY_LOAD=${RANK0_LOAD_FLAG} && \
                export PYTHONPATH=${SCRIPT_DIR}/verl:${SCRIPT_DIR}:\${PYTHONPATH:-} && \
                ray start --address ${ip_head} \
                    --num-gpus ${SLURM_GPUS_PER_NODE} --temp-dir=${RAY_TMPDIR} --block
            " &
        sleep "${WORKER_START_GAP}"
    done
}

# ── Helper: stop Ray on all nodes ───────────────────────────────────────
stop_ray_cluster() {
    echo "Stopping Ray cluster on all nodes..."
    for node in "${nodes_array[@]}"; do
        # --overlap is REQUIRED: the backgrounded `ray start --block` steps still
        # hold each node's step slot, so without it these cleanup sruns hang on
        # "Requested nodes are busy" and the auto-retry can never restart the
        # cluster (a recoverable transient init flake then becomes a dead job).
        srun --overlap --nodes=1 --ntasks=1 -w "$node" --environment=verl \
            bash -c "ray stop --force 2>/dev/null; true" &
    done
    wait
    sleep 6
}

# ── Step 2+3: Start Ray cluster and launch training with auto-retry ─────
# On transient CUDA/NCCL errors the entire Ray cluster is corrupted, so we
# tear it down, restart, and resume from the latest checkpoint.
# trainer.resume_mode=auto tells verl to find the latest checkpoint.
# trainer.save_freq should be low (e.g. 50) to minimise lost work.

MAX_RETRIES=1  # auto-retry DISABLED: run once, exit on failure (no cluster restart / resume). Set >1 to re-enable.

# ── Stage checkpoint to node-local /dev/shm (RAM) ───────────────────────
# Lustre concurrent reads (64 FSDP workers x 1123 tensors on the same files)
# crawl (~18min) even after striping -- it's metadata/serving contention, not
# bandwidth (a single bulk read is ~4GB/s). /dev/shm is RAM-backed and shared
# across containers on a node (see RAY_TMPDIR note above), so we bulk-copy the
# checkpoint ONCE per node (fast sequential read, ~64->16 readers) and point the
# run at the local copy -> the per-tensor loads then hit RAM. Node RAM is 856GB
# (train peak ~501GB + ckpt ~140GB fits). Auto-on for LARGE_MODEL; set
# STAGE_MODEL=false to disable. LOCAL_MODEL_PATH is what training actually loads.
# DISABLED (default false): staging made the load so fast that all ranks on a node
# materialize their full fp32 model in host RAM simultaneously (the slow Lustre load
# used to stagger this) + the 140GB /dev/shm copy -> per-node host OOM-kill at init,
# every launch (the gloo/TCPStore "connection closed by peer" cascade was the symptom
# of an OOM-killed rank). Reverting to the slow-but-working Lustre load. Re-enable with
# STAGE_MODEL=true ONLY after adding staggered / rank-0-broadcast loading so the 4
# ranks/node don't peak together. See project memory for the full analysis.
STAGE_MODEL="${STAGE_MODEL:-false}"
LOCAL_MODEL_PATH="${MODEL_PATH}"
if [[ "${STAGE_MODEL}" == "true" ]]; then
    MODEL_NAME="$(basename "${MODEL_PATH}")"
    LOCAL_MODEL_PATH="/dev/shm/${MODEL_NAME}"
    echo "Staging ${MODEL_PATH} -> ${LOCAL_MODEL_PATH} on all ${SLURM_JOB_NUM_NODES} nodes..."
    srun --nodes="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 --environment=verl \
        bash -c "
            set -euo pipefail
            if [[ -f '${LOCAL_MODEL_PATH}/.stage_complete' ]]; then
                echo \"[stage] \$(hostname): already staged, reusing\"
            else
                rm -rf '${LOCAL_MODEL_PATH}'
                mkdir -p '${LOCAL_MODEL_PATH}'
                cp -rL '${MODEL_PATH}/.' '${LOCAL_MODEL_PATH}/'
                touch '${LOCAL_MODEL_PATH}/.stage_complete'
                echo \"[stage] \$(hostname): staged \$(du -sh '${LOCAL_MODEL_PATH}' | cut -f1)\"
            fi
        " || { echo "ERROR: model staging failed on one or more nodes; aborting."; exit 1; }
    echo "Staging complete; training will load from ${LOCAL_MODEL_PATH}"
fi

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
            export MODEL_PATH='${LOCAL_MODEL_PATH}' && \
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
            export LARGE_MODEL='${LARGE_MODEL:-}' && \
            export ACTOR_MICRO_BS='${ACTOR_MICRO_BS:-}' && \
            export ROLLOUT_N='${ROLLOUT_N:-}' && \
            export LOGPROB_MICRO_BS='${LOGPROB_MICRO_BS:-}' && \
            export REF_LOGPROB_MICRO_BS='${REF_LOGPROB_MICRO_BS:-}' && \
            export ENFORCE_EAGER='${ENFORCE_EAGER:-}' && \
            export MAX_NUM_BATCHED_TOKENS='${MAX_NUM_BATCHED_TOKENS:-}' && \
            export ASYNC_ROLLOUT='${ASYNC_ROLLOUT:-}' && \
            export REWARD_NUM_WORKERS='${REWARD_NUM_WORKERS:-}' && \
            export OFFPOLICY_DATA='${OFFPOLICY_DATA:-}' && \
            export OFFPOLICY_BATCH_SIZE='${OFFPOLICY_BATCH_SIZE:-}' && \
            export PYTHONPATH=${SCRIPT_DIR}/verl:${SCRIPT_DIR}:\${PYTHONPATH:-} && \
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
