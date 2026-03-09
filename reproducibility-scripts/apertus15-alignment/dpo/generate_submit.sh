#!/bin/bash
#
# Hyperparameter sweep generator for DPO training.
# Submits jobs via recursive-unattended-accelerate.sh (handles container env).
#
# Usage:
#   bash generate_submit.sh              # Generate submit.sh only (dry-run)
#   bash generate_submit.sh --submit     # Generate and submit all jobs
#

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

# --- Paths ---
# The shared submit script that handles container, env, and accelerate launch
SBATCH_SCRIPT="$HOME/projects/posttraining/run/cscs-shared-submit-scripts/recursive-unattended-accelerate.sh"

# Training script
TRAINING_SCRIPT="reproducibility-scripts/apertus15-alignment/dpo/training.py"

# Accelerate config
ACCELERATE_CONFIG="src/swiss_alignment/configs/accelerate/ds-zero2.yaml"

# Dataset
DATASET_PATHS=(
#  "$SCRATCH/posttraining-data/preference_acquisition/datasets/Qwen3-32B_vs_0.6B"
 "$SCRATCH/posttraining-data/preference_acquisition/datasets/MaxMin"
)

echo "Using datasets: ${DATASET_PATHS[*]}"

# WandB
WANDB_PROJECT_NAME="apertus-1.5-post-training-dpo"
WANDB_TAGS="prod,dpo-sweep"

# --- Model ---
# MODEL_PATH="$HOME/projects/posttraining/run/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-7-ln-v2-ademamix/checkpoints/7fea1f8c44336360/checkpoint-8925"
# MODEL_PATH="$HOME/projects/posttraining/run/artifacts/shared/outputs/train_sft/final-run/apertus1-base-sft-stage1/global_step_2406/huggingface"
MODEL_PATH="/iopsstor/scratch/cscs/dmelikidze/huggingface/hub/models--allenai--Olmo-3-7B-Instruct-SFT/snapshots/e1452fc572d51966ff4aaeb25118b891eb93e549"
MODEL_SHORTNAME="olmo-3-7b-sft"

# --- Sweep Hyperparameters ---
BETAS=(0.1) #(5.0)
LEARNING_RATES=(1e-6 1e-7 5e-7 8e-8)
SEEDS=(42)
NORMALIZE_LOGPS=("False")
MAX_LENGTHS=(4096)
WARMUP_RATIOS=(0.1)
NUM_TRAIN_EPOCHS=(1)
PER_DEVICE_TRAIN_BATCH_SIZE=(2)
MAX_GRAD_NORM=(20)
LR_SCHEDULER_TYPES=("linear")

# --- Effective Batch Size Sweep ---
# The gradient_accumulation_steps will be computed automatically as:
#   gas = effective_batch_size / (SLURM_NODES * NUM_DEVICES_PER_NODE * per_device_train_batch_size)
EFFECTIVE_BATCH_SIZES=(128)

# --- SLURM ---
SLURM_NODES=4
SLURM_TIME="12:00:00"
SLURM_PARTITION="normal"

# Number of GPUs per node (used to compute global batch size info)
NUM_DEVICES_PER_NODE=4

# --- Output / Logging ---
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
SWEEP_NAME="dpo-sweep-${TIMESTAMP}"
SWEEP_DIR="$(cd "$(dirname "$0")" && pwd)/sweeps/${SWEEP_NAME}"
LOG_DIR="${SWEEP_DIR}/logs"
OUTPUT_BASE_DIR="$HOME/projects/posttraining/run/artifacts/private/outputs/dpo-sweep-final/${SWEEP_NAME}"

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

SUBMIT=false
if [[ "${1:-}" == "--submit" ]]; then
    SUBMIT=true
fi

# ============================================================================
# GENERATE COMMANDS
# ============================================================================

mkdir -p "${SWEEP_DIR}"
mkdir -p "${LOG_DIR}"

SUBMIT_FILE="${SWEEP_DIR}/submit.sh"
> "${SUBMIT_FILE}"

cat >> "${SUBMIT_FILE}" << 'HEADER'
#!/bin/bash
# Auto-generated sweep submission script
# Run each line to submit a job, or execute this script directly.
set -e

HEADER

COUNT=0

for dataset_path in "${DATASET_PATHS[@]}"; do
    DATASET_BASENAME=$(basename "$dataset_path")
for beta in "${BETAS[@]}"; do
for lr in "${LEARNING_RATES[@]}"; do
for seed in "${SEEDS[@]}"; do
for normalize in "${NORMALIZE_LOGPS[@]}"; do
for max_len in "${MAX_LENGTHS[@]}"; do
for warmup in "${WARMUP_RATIOS[@]}"; do
for epochs in "${NUM_TRAIN_EPOCHS[@]}"; do
for pbs in "${PER_DEVICE_TRAIN_BATCH_SIZE[@]}"; do
for ebs in "${EFFECTIVE_BATCH_SIZES[@]}"; do
for mgn in "${MAX_GRAD_NORM[@]}"; do
for lr_scheduler in "${LR_SCHEDULER_TYPES[@]}"; do

    # --- Compute gradient accumulation steps from effective batch size ---
    TOTAL_DEVICES=$((SLURM_NODES * NUM_DEVICES_PER_NODE))
    DEVICE_BATCH=$((TOTAL_DEVICES * pbs))
    gas=$((ebs / DEVICE_BATCH))

    COUNT=$((COUNT + 1))

    # --- Unique job identifier including all hps---
    JOB_ID="dpo-model-beta${beta}-lr${lr}-norm${normalize}-ebs${ebs}-epochs${epochs}-${DATASET_BASENAME}-${MODEL_SHORTNAME}"
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${JOB_ID}"
    RUN_NAME="${SWEEP_NAME}/${JOB_ID}"

    # Global batch size (should equal ebs)
    GLOBAL_BS=$((TOTAL_DEVICES * pbs * gas))

    # --- Build the sbatch command ---
    # recursive-unattended-accelerate.sh expects:
    #   1. accelerate_config=<path>    (parsed specially by the script)
    #   2. <script>                    (passed to accelerate launch)
    #   3. All remaining args          (passed to accelerate launch -> training script)
    #
    # The script uses "$*" so all positional args after the sbatch script
    # are forwarded into the accelerate launch command.

    SBATCH_CMD="sbatch"
    SBATCH_CMD+=" --job-name=${JOB_ID}"
    SBATCH_CMD+=" --nodes=${SLURM_NODES}"
    SBATCH_CMD+=" --time=${SLURM_TIME}"
    SBATCH_CMD+=" --output=${LOG_DIR}/${JOB_ID}.%j.out"
    SBATCH_CMD+=" --error=${LOG_DIR}/${JOB_ID}.%j.err"

    if [[ -n "${SLURM_PARTITION}" ]]; then
        SBATCH_CMD+=" --partition=${SLURM_PARTITION}"
    fi

    # Pass accelerate config as env var so it doesn't pollute $* in the shared script
    SBATCH_CMD+=" --export=ALL,ACCELERATE_CONFIG=${ACCELERATE_CONFIG}"

    # The sbatch script itself
    SBATCH_CMD+=" ${SBATCH_SCRIPT}"

    # --- Everything after the script path becomes positional args ($*) ---

    # all training script arguments
    SBATCH_CMD+=" ${TRAINING_SCRIPT}"
    SBATCH_CMD+=" --dataset_path ${dataset_path}"
    SBATCH_CMD+=" --output_dir ${OUTPUT_DIR}"
    SBATCH_CMD+=" --seed ${seed}"
    SBATCH_CMD+=" --beta ${beta}"
    SBATCH_CMD+=" --learning_rate ${lr}"
    SBATCH_CMD+=" --lr_scheduler_type ${lr_scheduler}"
    SBATCH_CMD+=" --normalize_logps ${normalize}"
    SBATCH_CMD+=" --max_length ${max_len}"
    SBATCH_CMD+=" --warmup_ratio ${warmup}"
    SBATCH_CMD+=" --num_train_epochs ${epochs}"
    SBATCH_CMD+=" --per_device_train_batch_size ${pbs}"
    SBATCH_CMD+=" --gradient_accumulation_steps ${gas}"
    SBATCH_CMD+=" --max_grad_norm ${mgn}"
    SBATCH_CMD+=" --model_name_or_path ${MODEL_PATH}"
    SBATCH_CMD+=" --run_name ${RUN_NAME}"
    SBATCH_CMD+=" --report_to wandb"
    SBATCH_CMD+=" --logging_steps 1"
    SBATCH_CMD+=" --bf16 True"
    SBATCH_CMD+=" --gradient_checkpointing True"

    # Write to submit file
    echo "# effective_batch_size=${ebs}, per_device_bs=${pbs}, grad_accum=${gas}, total_gpus=${TOTAL_DEVICES}" >> "${SUBMIT_FILE}"
    echo "${SBATCH_CMD}" >> "${SUBMIT_FILE}"
    echo "" >> "${SUBMIT_FILE}"

done
done
done
done
done
done
done
done
done
done
done
done

chmod +x "${SUBMIT_FILE}"

echo "============================================"
echo " Sweep Summary"
echo "============================================"
echo " Total jobs:           ${COUNT}"
echo " Submit script:        ${SUBMIT_FILE}"
echo " Log directory:        ${LOG_DIR}"
echo " Output base dir:      ${OUTPUT_BASE_DIR}"
echo " Datasets:             ${DATASET_PATHS[*]}"
echo " Model:                ${MODEL_PATH}"
echo " Accelerate config:    ${ACCELERATE_CONFIG}"
echo " Sbatch script:        ${SBATCH_SCRIPT}"
echo " Nodes per job:        ${SLURM_NODES}"
echo " GPUs per job:         $((SLURM_NODES * NUM_DEVICES_PER_NODE))"
echo " Effective batch sizes: ${EFFECTIVE_BATCH_SIZES[*]}"
echo " Total nodes needed:   $((COUNT * SLURM_NODES))"
echo "============================================"

# --- Optionally submit ---
if [[ "${SUBMIT}" == true ]]; then
    echo ""
    echo "Submitting ${COUNT} jobs..."
    echo ""
    bash "${SUBMIT_FILE}"
    echo ""
    echo "All ${COUNT} jobs submitted."
else
    echo ""
    echo "Dry run complete. To submit, either:"
    echo "  1. Run:  bash ${SUBMIT_FILE}"
    echo "  2. Or:   bash $(basename "$0") --submit"
fi