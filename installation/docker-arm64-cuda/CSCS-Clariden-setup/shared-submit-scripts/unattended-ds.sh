#!/bin/bash

#SBATCH -J swiss-alignment-run-dist
#SBATCH -t 12:00:00
#SBATCH -A a-a10
#SBATCH --output=sunattended-distributed.out
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1

# Variables used by the entrypoint script
export PROJECT_ROOT_AT=$HOME/projects/swiss-alignment/run
export PROJECT_NAME=swiss-alignment
export PACKAGE_NAME=swiss_alignment
export SLURM_ONE_ENTRYPOINT_SCRIPT_PER_NODE=1
export WANDB_API_KEY_FILE_AT=$HOME/.wandb-api-key
export HF_TOKEN_AT=$HOME/.hf-token
export HF_HOME=$SCRATCH/huggingface

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export TORCH_DISTRIBUTED_DEBUG=INFO

#export NCCL_P2P_LEVEL=NVL

parse_for_deepspeed_plugin() {
  for arg in "$@"; do
    if [[ $arg =~ training_args\.gradient_accumulation_steps=([0-9]+) ]]; then
      deepspeed_acc=${BASH_REMATCH[1]}
    fi
  done
}
parse_for_deepspeed_plugin "$@"

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
  bash -c "exec accelerate launch --config-file src/swiss-alignment/configs/accelerate/ds1-acc${deepspeed_acc}-4xN.yaml \
  --num_machines $SLURM_NNODES \
  --num_processes $((4*$SLURM_NNODES)) \
  --main_process_ip $(hostname) \
  --machine_rank \$SLURM_NODEID \
  $*"

# limitation have to manually edit the grad_accumulation_steps in the config


# additional options for pyxis
# --container-env to override environment variables defined in the container

exit 0
