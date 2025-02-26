#!/bin/bash

#SBATCH -J template-remote-development
#SBATCH -t 12:00:00
#SBATCH -A a-a10
#SBATCH --output=template-remote-development-%j.out

# Variables used by the entrypoint script
# Change this to the path of your project (can be the /dev or /run copy)
export PROJECT_ROOT_AT=$HOME/projects/swiss-alignment/dev
export PROJECT_NAME=swiss-alignment
export PACKAGE_NAME=swiss_alignment
export SLURM_ONE_ENTRYPOINT_SCRIPT_PER_NODE=1
export WANDB_API_KEY_FILE_AT=$HOME/.wandb-api-key
export HF_TOKEN_AT=$HOME/.hf-token
export HF_HOME=$SCRATCH/huggingface
export SSH_SERVER=1
export NO_SUDO_NEEDED=1
export JETBRAINS_SERVER_AT=$HOME/jetbrains-server
#export PYCHARM_IDE_AT=9cd12b7762067_pycharm-professional-2024.3.1.1-aarch64
# or
# export VSCODE_SERVER_AT=$SCRATCH/vscode-server

srun \
  --container-image=$CONTAINER_IMAGES/$(id -gn)+$(id -un)+swiss-alignment+arm64-cuda-root-latest.sqsh \
  --environment="${PROJECT_ROOT_AT}/installation/docker-arm64-cuda/CSCS-Clariden-setup/submit-scripts/edf.toml" \
  --container-mounts=\
$SCRATCH,\
$WANDB_API_KEY_FILE_AT,\
$HF_TOKEN_AT,\
$HOME/.gitconfig,\
$JETBRAINS_SERVER_AT,\
$HOME/.ssh \
  --container-workdir=$PROJECT_ROOT_AT \
  --container-env=PROJECT_NAME,PACKAGE_NAME \
  --no-container-mount-home \
  --no-container-remap-root \
  --no-container-entrypoint \
  --container-writable \
  /opt/template-entrypoints/pre-entrypoint.sh \
  sleep infinity

# additional options
# --container-env to override environment variables defined in the container

# You may also want to mount $HOME/.ssh/config if you bypass the proxy for GitHub.

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
