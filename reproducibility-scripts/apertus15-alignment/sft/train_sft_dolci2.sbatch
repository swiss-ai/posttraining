#!/bin/bash

#SBATCH -J sft
#SBATCH -t 12:00:00
#SBATCH -A infra01
#SBATCH -p normal
##SBATCH --reservation=PA-2338-RL
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --output=logs/apertus_sft/output/sft/%j.log
#SBATCH --error=logs/apertus_sft/output/sft/%j.err

set -euo pipefail

# Fix SSL issues on HPC nodes
unset SSL_CERT_FILE SSL_CERT_DIR

# --- Paths ---
PROJECT_ROOT_AT=$HOME/projects/posttraining/run
SFT_SCRIPT="$PROJECT_ROOT_AT/reproducibility-scripts/apertus15-alignment/sft/sft_trainer.py"
ACCELERATE_CONFIG="$PROJECT_ROOT_AT/src/swiss_alignment/configs/accelerate/ds-zero2.yaml"

# --- Caches ---
CACHE_DIR="$SCRATCH/cache"
export OUTPUT_DIR="$SCRATCH/apertus_sft/${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"/{hf,datasets,hub,tmp}

export HF_HOME="$CACHE_DIR/huggingface"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$CACHE_DIR/hub"
export TMPDIR="$CACHE_DIR/tmp"
export PYTHONPATH="$PROJECT_ROOT_AT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=7200

# Load API keys from .env if present
if [[ -f "$PROJECT_ROOT_AT/.env" ]]; then
  set -a
  source "$PROJECT_ROOT_AT/.env"
  set +a
fi

# --- Multinode accelerate setup ---
MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MAIN_PROCESS_PORT=29500
NUM_PROCESSES=$(expr $SLURM_NNODES \* $SLURM_GPUS_ON_NODE)

CMD="accelerate launch \
    --config_file $ACCELERATE_CONFIG \
    --num_processes $NUM_PROCESSES \
    --num_machines $SLURM_NNODES \
    --machine_rank \$SLURM_NODEID \
    --main_process_ip $MAIN_PROCESS_IP \
    --main_process_port $MAIN_PROCESS_PORT \
    $SFT_SCRIPT \
    --model swiss-ai/Apertus-8B-2509 \
    --out $OUTPUT_DIR \
    --max_length 4096 \
    --epochs 1 \
    --lr ${LR:-2e-5} \
    --bsz 4 \
    --gas 1 \
    --save_steps 500 \
    --logging_steps 10 \
    --subset_frac ${SUBSET_FRAC:-1.0}"

echo $CMD

START=$(date +%s)

srun --environment=sample-efficient-rlvr bash -c "$CMD"

SRUN_EXIT_CODE=$?
END=$(date +%s)
DURATION=$(( END - START ))

echo "Training complete. Output saved to: $OUTPUT_DIR"
echo "Total execution time: $DURATION seconds"

exit $SRUN_EXIT_CODE
