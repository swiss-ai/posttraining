#!/bin/bash

#SBATCH -J swiss-alignment-run-dist
#SBATCH -t 12:00:00
#SBATCH -A a-a10
#SBATCH --output=sunattended-distributed.out
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1

# Variables used by the entrypoint script
# Change this to the path of your project (can be the /dev or /run copy)
export PROJECT_ROOT_AT=$HOME/projects/swiss-alignment/run
export PROJECT_NAME=swiss-alignment
export PACKAGE_NAME=swiss_alignment
export SLURM_ONE_ENTRYPOINT_SCRIPT_PER_NODE=1
export WANDB_API_KEY_FILE_AT=$HOME/.wandb-api-key
export HF_TOKEN_AT=$HOME/.hf-token
export HF_HOME=$SCRATCH/huggingface

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

srun \
  --container-image=$CONTAINER_IMAGES/$(id -gn)+$(id -un)+swiss-alignment+arm64-cuda-root-latest.sqsh \
  --environment="${PROJECT_ROOT_AT}/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/edf.toml" \
  --container-mounts=\
$SCRATCH,\
$WANDB_API_KEY_FILE_AT,\
$HF_TOKEN_AT \
  --container-workdir=$PROJECT_ROOT_AT \
  --no-container-mount-home \
  --no-container-remap-root \
  --no-container-entrypoint \
  --container-writable \
  /opt/template-entrypoints/pre-entrypoint.sh \
  bash -c "exec accelerate launch --config-file src/swiss-alignment/configs/accelerate/ddp-4xN.yaml \
  --num_machines $SLURM_NNODES \
  --num_processes $((4*$SLURM_NNODES)) \
  --main_process_ip $(hostname) \
  --machine_rank \$SLURM_NODEID \
  $*"

exit 0
