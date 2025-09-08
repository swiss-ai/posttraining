#!/bin/bash

#SBATCH -J post-training-ref-completions
#SBATCH -t 6:00:00
#SBATCH -A a-infra01-1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --exclude=nid006539,nid007378,nid006931,nid006726,nid006521,nid007352,nid006959,nid006944,nid006904,nid006946,nid006966,nid007017,nid006968

# Variables used by the entrypoint script
# Change this to the path of your project (can be the /dev or /run copy)
export PROJECT_ROOT_AT=$HOME/projects/posttraining/run
export ENABLE_RETRY=1
source $PROJECT_ROOT_AT/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/setup.sh

# Limit vllm to 1 node and disable NCCL looking for inter-node AWS libfabric
export VLLM_DISABLE_COMPILE_CACHE=1
export NCCL_IB_DISABLE=1
export NCCL_NET="Socket"

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
    exec python -m post_training.data_alignment.generate_ref_completions_vllm_swissaiformat \
    subpartition_number=\$SLURM_PROCID \
    $*"

# If srun successful delete duplicate run ------------
SRUN_EXIT_CODE=$?
if [ $SRUN_EXIT_CODE -eq 0 ] && [ -n "$RETRY_JOB_ID" ]; then
  scancel $RETRY_JOB_ID
fi

echo SRUN_EXIT_CODE=$SRUN_EXIT_CODE
echo "Done"

exit $SRUN_EXIT_CODE
