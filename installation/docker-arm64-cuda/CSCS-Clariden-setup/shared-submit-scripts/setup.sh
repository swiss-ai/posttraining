export SLURM_ONE_ENTRYPOINT_SCRIPT_PER_NODE=1
export HYDRA_FULL_ERROR=1
export PROJECT_NAME=posttraining
export PACKAGE_NAME=post_training
export STORE=/capstor/store/cscs/swissai/infra01/
export SHARED_SCRATCH=/iopsstor/scratch/cscs/ismayilz/projects/posttraining/
export CONTAINER_IMAGE=/capstor/store/cscs/swissai/infra01/swiss-alignment/container-images/swiss-alignment+apertus-vllm-38c6036.sqsh
export CONTAINER_ENV_FILE="${PROJECT_ROOT_AT}/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/edf.toml"

export HF_HOME=$SCRATCH/huggingface
export WANDB_API_KEY_FILE_AT=$HOME/.wandb-api-key
export WANDB_ENTITY=apertus

export CUDA_BUFFER_PAGE_IN_THRESHOLD_MS=0.001
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

if [ "$ENABLE_RETRY" -eq 1 ]; then
  # Retry mechanism --------------------------
  # Initialize retry counter
  export MAX_RETRIES=${MAX_RETRIES:-4}
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
  ERROR_PATH=$(extract_field "StdErr")
  WORKDIR=$(extract_field "WorkDir")
  ACCOUNT=$(extract_field "Account")
  PARTITION=$(extract_field "Partition")
  TIME_LIMIT=$(extract_field "TimeLimit")
  JOB_ID=$(extract_field "JobId")
  JOB_NAME=$(extract_field "JobName")
  NODES=$(extract_field "NumNodes")
  TASKS_PER_NODE=$(extract_field "NumTasks")
  # If TASKS_PER_NODE is empty, set it to 1
  if [ -z "$TASKS_PER_NODE" ]; then
    TASKS_PER_NODE=1
  fi

  # Updating OUTPUT_PATH to include retry count
  OUTPUT_PATH=$(echo "$OUTPUT_PATH" | sed 's/\(-retry[0-9]*\)\?\.out$//')
  OUTPUT_PATH="${OUTPUT_PATH}-retry$RETRY_COUNT.out"
  ERROR_PATH=$(echo "$ERROR_PATH" | sed 's/\(-retry[0-9]*\)\?\.err$//')
  ERROR_PATH="${ERROR_PATH}-retry$RETRY_COUNT.err"

  echo Command to restart job: sbatch \
      --job-name="$JOB_NAME" \
      --time="$TIME_LIMIT" \
      --account="$ACCOUNT" \
      --partition="$PARTITION" \
      --nodes="$NODES" \
      --tasks-per-node="$TASKS_PER_NODE" \
      --chdir="$WORKDIR" \
      --output="$OUTPUT_PATH" \
      --error="$ERROR_PATH" \
      --dependency=afternotok:"$JOB_ID" \
      "$SCRIPT_PATH" "$@"

  job_id=$(sbatch \
      --job-name="$JOB_NAME" \
      --time="$TIME_LIMIT" \
      --account="$ACCOUNT" \
      --partition="$PARTITION" \
      --nodes="$NODES" \
      --tasks-per-node="$TASKS_PER_NODE" \
      --chdir="$WORKDIR" \
      --output="$OUTPUT_PATH" \
      --error="$ERROR_PATH" \
      --dependency=afternotok:"$JOB_ID" \
      "$SCRIPT_PATH" "$@" | awk '{print $4}')

  export RETRY_JOB_ID="$job_id"
fi
