export PROJECT_NAME=swiss-alignment
export PACKAGE_NAME=swiss_alignment
export SWISS_AI_STORAGE=/capstor/store/cscs/swissai/

export WANDB_API_KEY_FILE_AT=$HOME/.wandb-api-key
export HF_TOKEN_AT=$HOME/.hf-token
export HF_HOME=$SCRATCH/huggingface
export OPENAI_API_KEY_AT=$HOME/.openai-api-key
export OPENAI_API_KEY=$(cat $OPENAI_API_KEY_AT)

export WANDB_API_KEY_FILE_AT=$HOME/.wandb-api-key
if [[ "$@" == *"wandb.anonymize=true"* ]]; then
    export WANDB_API_KEY_FILE_AT=$HOME/.wandb-api-key-anonymous
fi

export CUDA_BUFFER_PAGE_IN_THRESHOLD_MS=0.001
