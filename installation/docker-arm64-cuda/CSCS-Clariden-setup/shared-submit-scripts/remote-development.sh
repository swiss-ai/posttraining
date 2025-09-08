#!/bin/bash

#SBATCH -J post-training-dev
#SBATCH -t 12:00:00
#SBATCH -A a-infra01-1
#SBATCH --output=sremote-development.out
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
# these nodes are large512 partition nodes, comment them out if using a different partition
##SBATCH --exclude=nid006539,nid007378,nid006931,nid006726,nid006521,nid007352,nid006959,nid006944,nid006904,nid006946,nid006966,nid007017,nid006968,nid007068

# Variables used by the entrypoint script
export PROJECT_ROOT_AT=$HOME/projects/posttraining/dev
export ENABLE_RETRY=0
source $PROJECT_ROOT_AT/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/setup.sh
export HF_TOKEN_AT=$HOME/.hf-token
export SLURM_ONE_REMOTE_DEV=1
export PRE_COMMIT_HOME=$SCRATCH/pre-commit
export SSH_SERVER=1
export NO_SUDO_NEEDED=1
export JETBRAINS_SERVER_AT=$HOME/jetbrains-server
export PYCHARM_IDE_AT=a72a92099e741_pycharm-professional-2024.3.3-aarch64
# Lazy hack.
if [ "$(whoami)" = "smoalla" ]; then
  export PYCHARM_IDE_AT=926eda376fbe6_pycharm-2025.1.3.1-aarch64
fi
# export VSCODE_SERVER_AT=$SCRATCH/vscode-server
mkdir -p $JETBRAINS_SERVER_AT
# mkdir -p $VSCODE_SERVER_AT

srun \
  --container-image=$CONTAINER_IMAGE \
  --environment=$CONTAINER_ENV_FILE \
  --container-mounts=\
"$(
  printf "%s," \
    "$PROJECT_ROOT_AT" \
    "$SCRATCH" \
    "$SHARED_SCRATCH" \
    "$STORE" \
    "$WANDB_API_KEY_FILE_AT" \
    "$HF_TOKEN_AT" \
    "$HOME/.gitconfig" \
    "$HOME/.bashrc" \
    "$HOME/.oh-my-bash" \
    "$HOME/.ssh" \
    "$JETBRAINS_SERVER_AT" \
  | tr ',' '\n' \
  | while read -r m; do
      [ -e "$m" ] && printf "%s," "$m"
    done \
  | sed 's/,$//'
)" \
  --container-workdir=$PROJECT_ROOT_AT \
  --no-container-mount-home \
  --no-container-remap-root \
  --no-container-entrypoint \
  --container-writable \
  /opt/template-entrypoints/pre-entrypoint.sh \
  sleep infinity

# Here can connect to the container with ssh or attach a terminal:
# Get the job id (and node id if multinode)
# Connect to the allocation
#   srun --overlap --pty --jobid=JOBID bash
# Inside the job find the container name
#   enroot list -f
# Exec to the container
#   enroot exec <container-pid> zsh
