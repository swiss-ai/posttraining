#!/usr/bin/env bash

#SBATCH -J online-qrpo-run
#SBATCH -t 12:00:00
#SBATCH -A infra01
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH -o /users/smatreno/projects/posttraining/dev/artifacts/private/logs/online_qrpo_ray/%x-%j.out
#SBATCH -e /users/smatreno/projects/posttraining/dev/artifacts/private/logs/online_qrpo_ray/%x-%j.err

set -euo pipefail

# This is close to cscs-shared-submit-scripts/unattended.sh, but
# QRPO needs an explicit Ray cluster across nodes before the Python entrypoint.
# The default allocation is 2 nodes; override with `sbatch -N <nodes> ...`.

export PROJECT_ROOT_AT="${PROJECT_ROOT_AT:-/users/smatreno/projects/posttraining/dev}"
export ENABLE_RETRY="${ENABLE_RETRY:-0}"
source "${PROJECT_ROOT_AT}/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/setup.sh"

# The CSCS GH200 nodes are allocated as full nodes. Advertise the full CPU
# capacity to Ray so VERL placement groups and auxiliary workers are not blocked
# by an artificial CPU scheduling cap.
export RAY_NUM_CPUS_PER_NODE="${RAY_NUM_CPUS_PER_NODE:-288}"

CONTAINER_MOUNTS="${PROJECT_ROOT_AT},${SCRATCH},${SHARED_SCRATCH},${STORE},${WANDB_API_KEY_FILE_AT}"
if [[ -n "${HF_TOKEN_AT:-}" ]]; then
  if [[ ! -f "${HF_TOKEN_AT}" ]]; then
    echo "ERROR: HF_TOKEN_AT is set but the token file does not exist: ${HF_TOKEN_AT}" >&2
    exit 1
  fi
  CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${HF_TOKEN_AT}"
fi

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --environment="${CONTAINER_ENV_FILE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --container-workdir="${PROJECT_ROOT_AT}" \
  --no-container-mount-home \
  --no-container-remap-root \
  --no-container-entrypoint \
  --container-writable \
  /opt/template-entrypoints/pre-entrypoint.sh \
  "${PROJECT_ROOT_AT}/src/post_training/qrpo-verl/scripts/run_online_qrpo_ray.sh"

exit 0
