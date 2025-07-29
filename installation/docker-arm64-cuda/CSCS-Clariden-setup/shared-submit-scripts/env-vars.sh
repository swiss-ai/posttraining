export SLURM_ONE_ENTRYPOINT_SCRIPT_PER_NODE=1
export HYDRA_FULL_ERROR=1
export PROJECT_NAME=swiss-alignment
export PACKAGE_NAME=swiss_alignment
export STORE=/capstor/store/cscs/swissai/infra01/
export SHARED_SCRATCH=/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/
export CONTAINER_IMAGE=/capstor/store/cscs/swissai/infra01/swiss-alignment/container-images/swiss-alignment+apertus-vllm-38c6036.sqsh
export CONTAINER_ENV_FILE="${PROJECT_ROOT_AT}/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/edf.toml"

export HF_HOME=$SCRATCH/huggingface
export WANDB_API_KEY_FILE_AT=$HOME/.wandb-api-key
export WANDB_ENTITY=apertus

export CUDA_BUFFER_PAGE_IN_THRESHOLD_MS=0.001
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
