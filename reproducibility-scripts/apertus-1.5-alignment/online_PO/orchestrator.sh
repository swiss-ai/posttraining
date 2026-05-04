#!/bin/bash
# ── Orchestrator: launch inference server, wait for URL, submit training ─
# This runs as a 1-node SLURM job. It:
#   1. Submits an inference server job via submit_job.py (inside srun --environment)
#   2. Polls the server log for the URL
#   3. Health-checks the server
#   4. Submits the training job with JUDGE_BASE_URL set
#   5. Exits (server + training continue independently)
#
# All configuration is passed via environment variables from launch.sh.

set -xeuo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
SCRATCH="${SCRATCH:-/iopsstor/scratch/cscs/$(whoami)}"
MODEL_LAUNCH_DIR="${SCRATCH}/model-launch/legacy"
SERVER_LOGS_DIR="/iopsstor/scratch/cscs/dmelikidze/online-dpo/logs/serving"
mkdir -p "${SERVER_LOGS_DIR}"

# ── Step 1: Launch inference server ─────────���───────────────────────────
echo "=== Launching inference server ==="
echo "Model: ${SERVER_MODEL}"
echo "Framework: ${SERVER_FRAMEWORK}"
echo "Nodes: ${SERVER_NODES}, Workers: ${SERVER_WORKERS}, NPW: ${SERVER_NODES_PER_WORKER}"

FRAMEWORK_ARGS="--model ${SERVER_MODEL} --host 0.0.0.0 --port 8080 --served-model-name ${SERVER_SERVED_NAME} --tensor-parallel-size ${SERVER_TP_SIZE}"

if [[ "${SERVER_FRAMEWORK}" == "vllm" && "${SERVER_MODEL,,}" == *"mistral"* ]]; then
    FRAMEWORK_ARGS+=" --tokenizer_mode mistral --load_format mistral --config_format mistral"
fi

SERVER_SUBMIT_OUTPUT=$(srun --overlap --nodes=1 --ntasks=1 \
    --environment=activeuf --container-writable --container-workdir="${SERVER_LOGS_DIR}" \
    bash -c "cd '${SERVER_LOGS_DIR}' && python -u '${MODEL_LAUNCH_DIR}/serving/submit_job.py' \
        --slurm-nodes ${SERVER_NODES} \
        --workers ${SERVER_WORKERS} \
        --nodes-per-worker ${SERVER_NODES_PER_WORKER} \
        --use-router \
        --serving-framework '${SERVER_FRAMEWORK}' \
        --worker-port 8080 \
        --slurm-environment '${MODEL_LAUNCH_DIR}/serving/envs/${SERVER_FRAMEWORK}.toml' \
        --router-environment '${MODEL_LAUNCH_DIR}/serving/envs/sglang.toml' \
        --disable-ocf \
        --framework-args '${FRAMEWORK_ARGS}'" 2>&1)

echo "${SERVER_SUBMIT_OUTPUT}"

SERVER_JOB_ID=""
while IFS= read -r line; do
    if [[ "${line}" == *"Job submitted successfully with ID:"* ]]; then
        SERVER_JOB_ID=$(echo "${line}" | awk '{print $NF}')
        break
    fi
done <<< "${SERVER_SUBMIT_OUTPUT}"

if [[ -z "${SERVER_JOB_ID}" ]]; then
    echo "ERROR: Failed to parse server job ID from output"
    exit 1
fi

echo "Server job ID: ${SERVER_JOB_ID}"

# ── Step 2: Wait for server URL in logs ───────────────��─────────────────
SERVER_LOG_FILE="${SERVER_LOGS_DIR}/logs/${SERVER_JOB_ID}/log.out"
TARGET_PREFIX="Router URL: "
BASE_URL=""

echo "Waiting for server URL in: ${SERVER_LOG_FILE}"

WAIT_ATTEMPTS=0
while [[ -z "${BASE_URL}" ]]; do
    if [[ -f "${SERVER_LOG_FILE}" ]]; then
        while IFS= read -r line; do
            if [[ "${line}" == "${TARGET_PREFIX}"* ]]; then
                RAW="${line#${TARGET_PREFIX}}"
                RAW=$(echo "${RAW}" | tr -d '[:space:]')
                BASE_URL="${RAW}/v1"
                echo "Found server URL: ${BASE_URL}"
                break
            fi
        done < "${SERVER_LOG_FILE}"
    else
        if (( WAIT_ATTEMPTS % 6 == 0 )); then
            echo "Still waiting... log file does not exist yet (attempt ${WAIT_ATTEMPTS})"
        fi
    fi

    if [[ -z "${BASE_URL}" ]]; then
        WAIT_ATTEMPTS=$((WAIT_ATTEMPTS + 1))
        sleep 5
    fi
done

# ── Step 3: Health check ─────────────���──────────────────────────────────
HEALTH_URL="${BASE_URL%/v1}/health"
echo "Health-checking: ${HEALTH_URL}"

HEALTH_ATTEMPTS=0
while true; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 --noproxy '*' "${HEALTH_URL}" 2>/dev/null || echo "000")
    if [[ "${HTTP_CODE}" == "200" ]]; then
        echo "Server is healthy!"
        break
    fi
    if (( HEALTH_ATTEMPTS % 3 == 0 )); then
        echo "Health check attempt ${HEALTH_ATTEMPTS}: HTTP ${HTTP_CODE}"
    fi
    HEALTH_ATTEMPTS=$((HEALTH_ATTEMPTS + 1))
    sleep 10
done

# ── Step 4: Submit training job ─────────────────���───────────────────────
echo "=== Submitting training job ==="
echo "JUDGE_BASE_URL=${BASE_URL}"
echo "Training nodes: ${TRAIN_NODES}"

# Append server job ID to output dir to avoid collisions
export OUTPUT_DIR="${OUTPUT_DIR}-${SERVER_JOB_ID}"

TRAIN_JOB_ID=$(sbatch \
    --job-name="train-${EXPERIMENT_NAME}" \
    --account="${ACCOUNT}" \
    --reservation="${RESERVATION}" \
    --partition="${PARTITION}" \
    --time="${JOB_TIME}" \
    --nodes="${TRAIN_NODES}" \
    --parsable \
    --export=ALL,\
JUDGE_BASE_URL="${BASE_URL}",\
JUDGE_API_KEY="${JUDGE_API_KEY}",\
JUDGE_MODEL="${JUDGE_MODEL}",\
MODEL_PATH="${MODEL_PATH}",\
OUTPUT_DIR="${OUTPUT_DIR}",\
EXPERIMENT_NAME="${EXPERIMENT_NAME}",\
LEARNING_RATE="${LEARNING_RATE}",\
DPO_BETA="${DPO_BETA}",\
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
LENGTH_NORMALIZE="${LENGTH_NORMALIZE}" \
    "${SCRIPT_DIR}/submit_multinode.sh")

echo "Training job submitted: ${TRAIN_JOB_ID}"
echo "Server job: ${SERVER_JOB_ID}"
echo "Orchestrator done. Exiting."
