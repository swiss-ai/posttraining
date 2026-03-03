#!/bin/bash

#SBATCH -J swiss-alignment-ref-logprobs
#SBATCH -t 6:00:00
#SBATCH -A infra01
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --reservation=PA-2338-RL

# Variables used by the entrypoint script
# Change this to the path of your project (can be the /dev or /run copy)
export PROJECT_ROOT_AT=$HOME/projects/posttraining/run
export ENABLE_RETRY=0
source $PROJECT_ROOT_AT/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/setup.sh

srun \
  --container-image=$CONTAINER_IMAGE \
  --environment=$CONTAINER_ENV_FILE \
  --container-mounts=\
$PROJECT_ROOT_AT,\
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
    sleep 60; \
    LOCK_FILE=/tmp/pip_install_datasets.lock; \
    DONE_FILE=/tmp/pip_install_datasets.done; \
    if mkdir \$LOCK_FILE 2>/dev/null; then \
      echo \"[Task \$SLURM_PROCID] Installing datasets...\"; \
      pip install --upgrade datasets && touch \$DONE_FILE; \
      echo \"[Task \$SLURM_PROCID] Install complete.\"; \
    else \
      echo \"[Task \$SLURM_PROCID] Waiting for datasets install...\"; \
      while [ ! -f \$DONE_FILE ]; do sleep 5; done; \
      echo \"[Task \$SLURM_PROCID] Install detected, proceeding.\"; \
    fi; \
    exec python -m swiss_alignment.data_alignment.compute_ref_logprobs_swissaiformat \
    subpartition_number=\$SLURM_PROCID \
    $*"

# If srun successful delete duplicate run ------------
SRUN_EXIT_CODE=$?
if [ $SRUN_EXIT_CODE -eq 0 ] && [ -n "$RETRY_JOB_ID" ]; then
  scancel $RETRY_JOB_ID
fi

echo SRUN_EXIT_CODE=$SRUN_EXIT_CODE
echo "Done"

exit $SRUN_EXIT_CODE
