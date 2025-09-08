#!/bin/bash

#SBATCH -J post-trainingun-accelerate
#SBATCH -t 12:00:00
#SBATCH -A a-infra01-1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
# these nodes are large512 partition nodes, comment them out if using a different partition
# SBATCH --exclude=nid006539,nid007378,nid006931,nid006726,nid006521,nid007352,nid006959,nid006944,nid006904,nid006946,nid006966,nid007017,nid006968,nid007068

# Variables used by the entrypoint script
export PROJECT_ROOT_AT=$HOME/projects/posttraining/run
export ENABLE_RETRY=1
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

# If srun successful delete duplicate run ------------
SRUN_EXIT_CODE=$?
if [ $SRUN_EXIT_CODE -eq 0 ] && [ -n "$RETRY_JOB_ID" ]; then
  scancel $RETRY_JOB_ID
fi

exit $SRUN_EXIT_CODE
