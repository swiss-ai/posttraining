#!/bin/bash

#SBATCH -J pre-dpo
#SBATCH -t 12:00:00
#SBATCH -A infra01
#SBATCH -p normal
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --nodes 16
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --output=reproducibility-scripts/apertus-1.5-alignment/SFT/logs/%j.out
#SBATCH --error=reproducibility-scripts/apertus-1.5-alignment/SFT/logs/%j.err

set -euo pipefail

# --- Environment setup (same as recursive-unattended-accelerate.sh) ---
export PROJECT_ROOT_AT=$HOME/projects/posttraining/run
export ENABLE_RETRY=0
source $PROJECT_ROOT_AT/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/setup.sh

# --- Paths ---
SFT_SCRIPT="$PROJECT_ROOT_AT/reproducibility-scripts/apertus-1.5-alignment/SFT/sft_trainer.py"
ACCELERATE_CONFIG="src/post_training/configs/accelerate/ds-zero2.yaml"

# --- Hyperparameters (override via env vars) ---
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"swiss-ai/Apertus-8B-2509"}
DATASET_NAME=${DATASET_NAME:-"allenai/Dolci-Instruct-SFT"}
DATASET_SPLIT=${DATASET_SPLIT:-"train"}
RUN_NAME=${RUN_NAME:-"apertus-8b-sft-dolci"}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-8192}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}
LEARNING_RATE=${LEARNING_RATE:-5e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-"cosine"}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
SAVE_STEPS=${SAVE_STEPS:-2000}
EVAL_STEPS=${EVAL_STEPS:-512}
LOGGING_STEPS=${LOGGING_STEPS:-10}
SUBSET_FRAC=${SUBSET_FRAC:-1.0}

# --- Caches ---
CACHE_DIR="$SCRATCH/cache"
export OUTPUT_DIR="$SCRATCH/apertus_sft/${RUN_NAME}/${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"/{hf,datasets,hub,tmp}

export HF_HOME="$CACHE_DIR/hf"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$CACHE_DIR/hub"
export TMPDIR="$CACHE_DIR/tmp"
export PYTHONPATH="$PROJECT_ROOT_AT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export NCCL_TIMEOUT=7200

# Load API keys from .env if present
if [[ -f "$PROJECT_ROOT_AT/.env" ]]; then
  set -a
  source "$PROJECT_ROOT_AT/.env"
  set +a
fi

# --- Multinode srun + accelerate launch (pattern from recursive-unattended-accelerate.sh) ---
START=$(date +%s)

srun \
  --container-image=$CONTAINER_IMAGE \
  --environment=$CONTAINER_ENV_FILE \
  --container-mounts=\
$PROJECT_ROOT_AT,\
$HOME/projects/posttraining/dev,\
$SCRATCH,\
$SHARED_SCRATCH,\
$STORE,\
/iopsstor,\
$WANDB_API_KEY_FILE_AT \
  --container-workdir=$PROJECT_ROOT_AT \
  --no-container-mount-home \
  --no-container-remap-root \
  --no-container-entrypoint \
  --container-writable \
  /opt/template-entrypoints/pre-entrypoint.sh \
  bash -c "\
    exec accelerate launch \
    --config-file $ACCELERATE_CONFIG \
    --num_machines $SLURM_NNODES \
    --num_processes $((4*$SLURM_NNODES)) \
    --main_process_ip $(hostname) \
    --main_process_port $((20000 + SLURM_JOB_ID % 10000)) \
    --machine_rank \$SLURM_NODEID \
    $SFT_SCRIPT \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name $DATASET_NAME \
    --dataset_split $DATASET_SPLIT \
    --output_dir $OUTPUT_DIR \
    --run_name $RUN_NAME \
    --max_seq_length $MAX_SEQ_LENGTH \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_grad_norm $MAX_GRAD_NORM \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS \
    --subset_frac $SUBSET_FRAC"

SRUN_EXIT_CODE=$?
END=$(date +%s)
DURATION=$(( END - START ))

echo "Training complete. Output saved to: $OUTPUT_DIR"
echo "Total execution time: $DURATION seconds"

exit $SRUN_EXIT_CODE
