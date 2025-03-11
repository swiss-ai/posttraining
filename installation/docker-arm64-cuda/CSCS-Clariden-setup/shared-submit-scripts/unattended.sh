#!/bin/bash

#SBATCH -J swiss-alignment-run
#SBATCH -t 12:00:00
#SBATCH -A a-a10
#SBATCH --output=sunattended.out
#SBATCH --nodes 1

# Variables used by the entrypoint script
export PROJECT_ROOT_AT=$HOME/projects/swiss-alignment/run
export PROJECT_NAME=swiss-alignment
export PACKAGE_NAME=swiss_alignment
export SLURM_ONE_ENTRYPOINT_SCRIPT_PER_NODE=1
export WANDB_API_KEY_FILE_AT=$HOME/.wandb-api-key
export OPENAI_API_KEY_AT=$HOME/.openai-api-key
export HF_TOKEN_AT=$HOME/.hf-token
export HF_HOME=$SCRATCH/huggingface

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

OPENAI_API_KEY=$(cat $OPENAI_API_KEY_AT)
export OPENAI_API_KEY

srun \
  --container-image=$CONTAINER_IMAGES/$(id -gn)+$(id -un)+swiss-alignment+arm64-cuda-root-latest.sqsh \
  --environment="${PROJECT_ROOT_AT}/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/edf.toml" \
  --container-mounts=\
$PROJECT_ROOT_AT,\
$SCRATCH,\
$WANDB_API_KEY_FILE_AT,\
$HF_TOKEN_AT,\
$OPENAI_API_KEY_AT \
  --container-workdir=$PROJECT_ROOT_AT \
  --no-container-mount-home \
  --no-container-remap-root \
  --no-container-entrypoint \
  --container-writable \
  /opt/template-entrypoints/pre-entrypoint.sh \
  "$@"

# additional options for pyxis
# --container-env to override environment variables defined in the container

exit 0
