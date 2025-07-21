#!/bin/bash

#SBATCH -J swiss-alignment-run-dist
#SBATCH -t 12:00:00
#SBATCH -A a-infra01-1
#SBATCH --output=sunattended-distributed.out
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1

# Variables used by the entrypoint script
# Change this to the path of your project (can be the /dev or /run copy)
export PROJECT_ROOT_AT=$HOME/projects/swiss-alignment/dev
source $PROJECT_ROOT_AT/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/env-vars.sh $@
export HF_TOKEN_AT=$HOME/.hf-token
export SLURM_ONE_ENTRYPOINT_SCRIPT_PER_NODE=1

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

srun \
  --container-image=/capstor/store/cscs/swissai/infra01/swiss-alignment/container-images/swiss-alignment+apertus-vllm.sqsh \
  --environment="${PROJECT_ROOT_AT}/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/edf.toml" \
  --container-mounts=\
$PROJECT_ROOT_AT,\
$SCRATCH,\
$SWISS_AI_STORAGE,\
/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/,\
$WANDB_API_KEY_FILE_AT,\
$HF_TOKEN_AT \
  --container-workdir=$PROJECT_ROOT_AT \
  --no-container-mount-home \
  --no-container-remap-root \
  --no-container-entrypoint \
  --container-writable \
  /opt/template-entrypoints/pre-entrypoint.sh \
  bash -c "\
    pip install bitsandbytes && \
    bash ${PROJECT_ROOT_AT}/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/hot-pip-install.sh && \
    exec accelerate launch \
    --config-file ${PROJECT_ROOT_AT}/src/swiss_alignment/configs/accelerate/ds-zero3.yaml \
    --num_machines $SLURM_NNODES \
    --num_processes $((4*$SLURM_NNODES)) \
    --main_process_ip $(hostname) \
    --machine_rank \$SLURM_NODEID \
    $*"

# limitation have to manually edit the grad_accumulation_steps in the config


# additional options for pyxis
# --container-env to override environment variables defined in the container

exit 0
