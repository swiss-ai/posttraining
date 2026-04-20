#!/bin/bash
#SBATCH --job-name=spin-multinode
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --error=./logs/slurm-%j.err
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --partition=normal

# ── Explanation ──────────────────────────────────────────────────────────
#
# SLURM header:
#   --nodes=N              : allocate N nodes
#   --ntasks-per-node=1    : one srun task per node (Ray manages GPU workers internally)
#   --gpus-per-node=4      : 4 GPUs on each node
#
# How it works:
#   1. We resolve the IP of the first node (head node)
#   2. Start a Ray head process on node 0 — this is the cluster coordinator
#   3. Start Ray worker processes on all other nodes, connecting to the head
#   4. Run the training script on the head node — verl discovers all GPUs via Ray
#   5. Ray handles cross-node communication (NCCL for GPU tensors, gRPC for control)
#
# The "block" flag keeps Ray processes alive for the duration of the job.
# The "overlap" flag on the final srun lets the training script coexist
# with the Ray head process on the same node.
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

# ── Step 2: pip install verl + start Ray on all nodes ───────────────────
# Each srun --environment=verl spawns a fresh container. We must pip install
# the local verl fork AND start Ray in the same bash session so that Ray's
# worker processes (which it forks internally) inherit the installed package.
#
# --temp-dir: tells Ray to store session files (sockets, logs) on the shared
# filesystem instead of /tmp. This is critical because --overlap creates a
# separate container that doesn't share /tmp with the Ray head container.
#
# Head node: install verl, then start Ray as the cluster coordinator.
echo "Starting Ray HEAD at $head_node ($head_node_ip)"
srun --nodes=1 --ntasks=1 -w "$head_node" --environment=verl \
    bash -c "
        unset ROCR_VISIBLE_DEVICES && \
        export WANDB_ENTITY=apertus && \
        cd ${SCRIPT_DIR}/verl && pip install -e . --quiet && cd ${SCRIPT_DIR} && \
        ray start --head --node-ip-address=${head_node_ip} --port=${port} \
            --num-gpus ${SLURM_GPUS_PER_NODE} --temp-dir=${RAY_TMPDIR} --block
    " &

# Wait for the head node's Ray GCS to be ready before workers try to connect.
sleep 30

# Worker nodes: same pattern — install verl, then connect to the head.
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
    sleep 5
done

# ── Step 3: Launch training ─────────────────────────────────────────────
# --overlap: allows this srun to share node 0 with the Ray head process.
# pip install verl again in this session (same container, different process).
# RAY_ADDRESS tells ray.init() to connect to the existing cluster.
# RAY_TMPDIR ensures it finds the session sockets on the shared filesystem.
# train_spin.sh forwards "$@" so nnodes/n_gpus_per_node override the defaults.
PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" --environment=verl \
    bash -c "
        unset ROCR_VISIBLE_DEVICES && \
        cd ${SCRIPT_DIR}/verl && pip install -e . --quiet && cd ${SCRIPT_DIR} && \
        export RAY_ADDRESS=${ip_head} && \
        export RAY_TMPDIR=${RAY_TMPDIR} && \
        bash ${SCRIPT_DIR}/train_spin.sh \
            trainer.nnodes=${SLURM_NNODES} \
            trainer.n_gpus_per_node=${SLURM_GPUS_PER_NODE}
    "
