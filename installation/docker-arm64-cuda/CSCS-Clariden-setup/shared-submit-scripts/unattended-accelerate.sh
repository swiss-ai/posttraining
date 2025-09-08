#!/bin/bash

#SBATCH -J post-training-run-accelerate
#SBATCH -t 12:00:00
#SBATCH -A a-infra01-1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1

# Variables used by the entrypoint script
# Change this to the path of your project (can be the /dev or /run copy)
export PROJECT_ROOT_AT=$HOME/projects/posttraining/run
export ENABLE_RETRY=0
source $PROJECT_ROOT_AT/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/setup.sh

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

exit 0
