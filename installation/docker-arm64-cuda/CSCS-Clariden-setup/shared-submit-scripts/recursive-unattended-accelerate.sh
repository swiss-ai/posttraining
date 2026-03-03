#!/bin/bash

#SBATCH -J swiss-alignment-run-accelerate
#SBATCH -t 12:00:00
#SBATCH -A infra01
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --reservation=PA-2338-RL

# Variables used by the entrypoint script
export PROJECT_ROOT_AT=$HOME/projects/posttraining/run
export ENABLE_RETRY=0
source $PROJECT_ROOT_AT/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/setup.sh

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
$HOME/projects/posttraining/dev,\
$SCRATCH,\
$SHARED_SCRATCH,\
$STORE,\
/iopsstor,\
$WANDB_API_KEY_FILE_AT \
  --container-workdir=$PROJECT_ROOT_AT \
  --no-container-mount-home \
  --no-container-remap-root \
  --no-container-entrypoint \
  --container-writable \
  /opt/template-entrypoints/pre-entrypoint.sh \
  bash -c "\
    LOCK_FILE=/tmp/pip_install_datasets.lock; \
    DONE_FILE=/tmp/pip_install_datasets.done; \
    if mkdir \$LOCK_FILE 2>/dev/null; then \
      echo \"[Node \$SLURM_NODEID] Installing datasets...\"; \
      pip install --upgrade datasets && touch \$DONE_FILE; \
      echo \"[Node \$SLURM_NODEID] Install complete.\"; \
    else \
      echo \"[Node \$SLURM_NODEID] Waiting for install...\"; \
      while [ ! -f \$DONE_FILE ]; do sleep 5; done; \
      echo \"[Node \$SLURM_NODEID] Install detected, proceeding.\"; \
    fi; \
    exec accelerate launch \
    --config-file $ACCELERATE_CONFIG \
    --num_machines $SLURM_NNODES \
    --num_processes $((4*$SLURM_NNODES)) \
    --main_process_ip $(hostname) \
    --machine_rank \$SLURM_NODEID \
    $*"

# If srun successful delete duplicate run ------------
SRUN_EXIT_CODE=$?
if [ $SRUN_EXIT_CODE -eq 0 ] && [ -n "$RETRY_JOB_ID" ]; then
  scancel $RETRY_JOB_ID
fi

exit $SRUN_EXIT_CODE
