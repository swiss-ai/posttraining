#!/bin/bash

#SBATCH -J dpr-run-generate
#SBATCH -t 6:00:00
#SBATCH -A a-infra01-1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --exclude=nid006539,nid007378,nid006931,nid006726,nid006521,nid007352,nid006959,nid006944,nid006904,nid006946,nid006966,nid007017,nid006968

# Variables used by the entrypoint script
# Change this to the path of your project (can be the /dev or /run copy)
export PROJECT_ROOT_AT=$HOME/projects/posttraining/run
export ENABLE_RETRY=0
source $PROJECT_ROOT_AT/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/setup.sh
export VLLM_DISABLE_COMPILE_CACHE=1

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
    exec python -m post_training.data_alignment.generate_ref_completions_with_vllm \
    subpartition_number=\$SLURM_PROCID \
    $*"

exit 0
