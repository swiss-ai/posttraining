#!/bin/bash

#SBATCH -J template-unattended
#SBATCH -t 0:30:00

# Variables used by the entrypoint script
# Change this to the path of your project (can be the /dev or /run copy)
export PROJECT_ROOT_AT=$SCRATCH/swiss-alignment/run
export PROJECT_NAME=swiss-alignment
export PACKAGE_NAME=swiss_alignment
export SLURM_ONE_ENTRYPOINT_SCRIPT_PER_NODE=1
# For wandb, huggingface, etc. look at the remote-development.sh

srun \
  --container-image=$CONTAINER_IMAGES/claire+smoalla+swiss-alignment+arm64-cuda-root-latest.sqsh \
  --environment="${PROJECT_ROOT_AT}/installation/docker-arm64-cuda/CSCS-Clariden-setup/submit-scripts/edf.toml" \
  --container-mounts=$SCRATCH \
  --container-workdir=$PROJECT_ROOT_AT \
  --container-env=PROJECT_NAME,PACKAGE_NAME \
  --no-container-mount-home \
  --no-container-remap-root \
  --no-container-entrypoint \
  --container-writable \
  /opt/template-entrypoints/pre-entrypoint.sh \
  python -m swiss_alignment.template_experiment some_arg=some_value wandb.mode=offline

# additional options for pyxis
# --container-env to override environment variables defined in the container

exit 0
