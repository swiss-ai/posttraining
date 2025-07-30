#!/bin/bash

#SBATCH -J dpr-run-generate
#SBATCH -t 6:00:00
#SBATCH -A a-infra01-1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4

# Variables used by the entrypoint script
# Change this to the path of your project (can be the /dev or /run copy)
export PROJECT_ROOT_AT=$HOME/projects/swiss-alignment/run
source $PROJECT_ROOT_AT/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/env-vars.sh

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
    sleep 60; \
    exec python -m swiss_alignment.data_alignment.generate_ref_completions_with_vllm \
    subpartition_number=\$SLURM_PROCID \
    $*"

exit 0
