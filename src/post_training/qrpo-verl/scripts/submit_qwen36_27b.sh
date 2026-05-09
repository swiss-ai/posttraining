#!/usr/bin/env bash

set -euo pipefail

# ── Same-style config as colleague's launch.sh ──────────────────────────

SCRATCH="${SCRATCH:-/iopsstor/scratch/cscs/$(whoami)}"
MODEL_LAUNCH_DIR="${MODEL_LAUNCH_DIR:-${SCRATCH}/model-launch/legacy}"

LOGS_DIR="${LOGS_DIR:-/iopsstor/scratch/cscs/${USER}/qwen36-27b-smatrenok/logs}"
SERVER_LOGS_DIR="${LOGS_DIR}/serving"

mkdir -p "${SERVER_LOGS_DIR}"

ACCOUNT="${ACCOUNT:-infra01}"
#RESERVATION="${RESERVATION:-SD-69241-apertus-1-5}"
RESERVATION="${RESERVATION:-}"
PARTITION="${PARTITION:-normal}"
SERVER_TIME="${SERVER_TIME:-12:00:00}"

SERVER_MODEL="${SERVER_MODEL:-/iopsstor/scratch/cscs/smatreno/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9}"

SERVER_SERVED_NAME="${SERVER_SERVED_NAME:-Qwen/Qwen3.6-27B-smatrenok}"

SERVER_NODES="${SERVER_NODES:-64}"
SERVER_WORKERS="${SERVER_WORKERS:-64}"
SERVER_NODES_PER_WORKER="${SERVER_NODES_PER_WORKER:-1}"
SERVER_TP_SIZE="${SERVER_TP_SIZE:-4}"
SERVER_FRAMEWORK="${SERVER_FRAMEWORK:-vllm}"

WORKER_PORT="${WORKER_PORT:-8080}"
ROUTER_PORT="${ROUTER_PORT:-30000}"

VLLM_ENV="${VLLM_ENV:-${MODEL_LAUNCH_DIR}/serving/envs/${SERVER_FRAMEWORK}.toml}"
ROUTER_ENV="${ROUTER_ENV:-${MODEL_LAUNCH_DIR}/serving/envs/sglang.toml}"

# submit_job.py does not expose --slurm-reservation directly.
# Slurm should still honor this when model-launch calls sbatch.
if [[ -n "${RESERVATION}" ]]; then
    export SBATCH_RESERVATION="${RESERVATION}"
fi

echo "=== Launching Qwen/Qwen3.6-27B via model-launch ==="
echo "MODEL_LAUNCH_DIR:        ${MODEL_LAUNCH_DIR}"
echo "SERVER_LOGS_DIR:         ${SERVER_LOGS_DIR}"
echo "SERVER_MODEL:            ${SERVER_MODEL}"
echo "SERVER_SERVED_NAME:      ${SERVER_SERVED_NAME}"
echo "SERVER_NODES:            ${SERVER_NODES}"
echo "SERVER_WORKERS:          ${SERVER_WORKERS}"
echo "SERVER_NODES_PER_WORKER: ${SERVER_NODES_PER_WORKER}"
echo "SERVER_TP_SIZE:          ${SERVER_TP_SIZE}"
echo "SERVER_FRAMEWORK:        ${SERVER_FRAMEWORK}"
echo "VLLM_ENV:                ${VLLM_ENV}"
echo "ROUTER_ENV:              ${ROUTER_ENV}"
echo ""

if [[ ! -f "${MODEL_LAUNCH_DIR}/serving/submit_job.py" ]]; then
    echo "ERROR: Cannot find ${MODEL_LAUNCH_DIR}/serving/submit_job.py"
    exit 1
fi

if [[ ! -f "${VLLM_ENV}" ]]; then
    echo "ERROR: Cannot find ${VLLM_ENV}"
    exit 1
fi

if [[ ! -f "${ROUTER_ENV}" ]]; then
    echo "ERROR: Cannot find ${ROUTER_ENV}"
    exit 1
fi

FRAMEWORK_ARGS="--model ${SERVER_MODEL} \
--host 0.0.0.0 \
--port ${WORKER_PORT} \
--served-model-name ${SERVER_SERVED_NAME} \
--tensor-parallel-size ${SERVER_TP_SIZE}"

# Same harmless special case as colleague's scripts.
if [[ "${SERVER_FRAMEWORK}" == "vllm" && "${SERVER_MODEL,,}" == *"mistral"* ]]; then
    FRAMEWORK_ARGS+=" --tokenizer_mode mistral --load_format mistral --config_format mistral"
fi

cd "${SERVER_LOGS_DIR}"

python -u "${MODEL_LAUNCH_DIR}/serving/submit_job.py" \
    --slurm-job-name "qwen36-27b-smatrenok" \
    --slurm-nodes "${SERVER_NODES}" \
    --slurm-partition "${PARTITION}" \
    --slurm-time "${SERVER_TIME}" \
    --slurm-account "${ACCOUNT}" \
    --workers "${SERVER_WORKERS}" \
    --nodes-per-worker "${SERVER_NODES_PER_WORKER}" \
    --use-router \
    --router-port "${ROUTER_PORT}" \
    --serving-framework "${SERVER_FRAMEWORK}" \
    --worker-port "${WORKER_PORT}" \
    --slurm-environment "${VLLM_ENV}" \
    --router-environment "${ROUTER_ENV}" \
    --disable-ocf \
    --framework-args "${FRAMEWORK_ARGS}"

echo ""
echo "Submitted serving job."
echo ""
echo "Find the serving job id with:"
echo "  ls ${SERVER_LOGS_DIR}/logs"
echo ""
echo "Find the router URL with:"
echo "  grep -R \"Router URL:\" ${SERVER_LOGS_DIR}/logs/*/log.out"
echo ""
echo "Then use:"
echo "  OPENAI_BASE_URL=http://<router-ip>:${ROUTER_PORT}/v1"
echo "  OPENAI_MODEL=${SERVER_SERVED_NAME}"
echo "  OPENAI_API_KEY=EMPTY"