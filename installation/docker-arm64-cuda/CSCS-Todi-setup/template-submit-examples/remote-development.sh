#!/bin/bash

#SBATCH -J template-remote-development
#SBATCH -t 12:00:00

# Variables used by the entrypoint script
# Change this to the path of your project (can be the /dev or /run copy)
export PROJECT_ROOT_AT=$SCRATCH/swiss-alignment/dev
export SLURM_ONE_ENTRYPOINT_SCRIPT_PER_NODE=1
export WANDB_API_KEY_FILE_AT=$HOME/.wandb-api-key
export HF_TOKEN_AT=$HOME/.hf-token
export HF_HOME=$SCRATCH/huggingface
export SSH_SERVER=1
export NO_SUDO_NEEDED=1
export JETBRAINS_SERVER_AT=$SCRATCH/jetbrains-server
# export JETBRAINS_IDE_AT=744eea3d4045b_pycharm-professional-2024.1.6-aarch64
# or
# export VSCODE_SERVER_AT=$SCRATCH/vscode-server
unset PROMPT_COMMAND

srun \
  --container-image=$CONTAINER_IMAGES/swiss-alignment+arm64-cuda-root-latest.sqsh \
  --environment="${PROJECT_ROOT_AT}/installation/docker-arm64-cuda/CSCS-Todi-setup/submit-scripts/edf.toml" \
  --container-mounts=\
$SCRATCH,\
$WANDB_API_KEY_FILE_AT,\
$HF_TOKEN_AT,\
$HOME/.gitconfig,\
$HOME/.ssh/authorized_keys \
  --container-workdir=$PROJECT_ROOT_AT \
  --no-container-mount-home \
  --no-container-remap-root \
  --no-container-entrypoint \
  --container-writable \
  /opt/template-entrypoints/pre-entrypoint.sh \
  sleep infinity

# additional options
# --container-env to override environment variables defined in the container

# Draft.
# Here can connect to the container with
# Get the job id (and node id if multinode)
#
# Connect to the allocation
#   srun --overlap --pty --jobid=JOBID bash
# Inside the job find the container name
#   enroot list -f
# Exec to the container
#   enroot exec <container-pid> zsh
