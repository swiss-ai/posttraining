#!/bin/bash

#SBATCH -J swiss-alignment-run-accelerate
#SBATCH -t 12:00:00
#SBATCH -A a-infra01-1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1

# Variables used by the entrypoint script
export PROJECT_ROOT_AT=$HOME/projects/swiss-alignment/run
source $PROJECT_ROOT_AT/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/env-vars.sh
unset HF_TOKEN_AT

# Retry mechanism --------------------------
# Initialize retry counter
export MAX_RETRIES=${MAX_RETRIES:-3}
export RETRY_COUNT=${RETRY_COUNT:-0}

# Check retry limit
if [ "$RETRY_COUNT" -eq 0 ]; then
  echo "Original run. Retry: $RETRY_COUNT."
elif [ "$RETRY_COUNT" -gt "$MAX_RETRIES" ]; then
  echo "Max retries ($MAX_RETRIES) reached. Exiting."
  exit 1
else
  echo "Retry: $RETRY_COUNT."
fi

# Creating duplicate sbatch ----------------------
# Updating count for new sbatch
RETRY_COUNT=$((RETRY_COUNT + 1))

# Identifying SLURM specific options
JOB_INFO=$(scontrol show job "$SLURM_JOB_ID")
extract_field() {
  local key="$1"
  echo "$JOB_INFO" | tr ' ' '\n' | grep "^$key=" | cut -d= -f2-
}
SCRIPT_PATH=$(extract_field "Command")
OUTPUT_PATH=$(extract_field "StdOut")
WORKDIR=$(extract_field "WorkDir")
ACCOUNT=$(extract_field "Account")
PARTITION=$(extract_field "Partition")
TIME_LIMIT=$(extract_field "TimeLimit")
JOB_ID=$(extract_field "JobId")
JOB_NAME=$(extract_field "JobName")
NODES=$(extract_field "NumNodes")

# Updating OUTPUT_PATH to include retry count
OUTPUT_PATH=$(echo "$OUTPUT_PATH" | sed 's/\(-retry[0-9]*\)\?\.out$//')
OUTPUT_PATH="${OUTPUT_PATH}-retry$RETRY_COUNT.out"

job_id=$(sbatch \
    --job-name="$JOB_NAME" \
    --time="$TIME_LIMIT" \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --nodes="$NODES" \
    --chdir="$WORKDIR" \
    --output="$OUTPUT_PATH" \
    --dependency=afternotok:"$JOB_ID" \
    "$SCRIPT_PATH" "$@" | awk '{print $4}')

# Executing srun -------------------------------------
# Parse some args:
# accelerate config
for arg in "$@"; do
  if [[ $arg == accelerate_config=* ]]; then
    # remove the prefix “accelerate_config=”…
    ACCELERATE_CONFIG="${arg#accelerate_config=}"
    break
  fi
done

srun \
  --container-image=$CONTAINER_IMAGE \
  --environment=$CONTAINER_ENV_FILE \
  --container-mounts=\
$PROJECT_ROOT_AT,\
$SCRATCH,\
$SHARED_SCRATCH,\
$STORE,\
$WANDB_API_KEY_FILE_AT \
  --container-workdir=$PROJECT_ROOT_AT \
  --no-container-mount-home \
  --no-container-remap-root \
  --no-container-entrypoint \
  --container-writable \
  /opt/template-entrypoints/pre-entrypoint.sh \
  bash -c "\
    exec accelerate launch \
    --config-file $ACCELERATE_CONFIG \
    --num_machines $SLURM_NNODES \
    --num_processes $((4*$SLURM_NNODES)) \
    --main_process_ip $(hostname) \
    --machine_rank \$SLURM_NODEID \
    $*"
SRUN_EXIT_CODE=$?

# If srun successful delete duplicate run ------------
if [ $SRUN_EXIT_CODE -eq 0 ]; then
  scancel $job_id
fi

exit 0
