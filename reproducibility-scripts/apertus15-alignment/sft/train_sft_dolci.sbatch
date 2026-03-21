#!/bin/bash

#SBATCH -J sft
#SBATCH -t 12:00:00
#SBATCH -A infra01
#SBATCH -p normal
##SBATCH --reservation=PA-2338-RL
#SBATCH --nodes 16
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --output=logs/apertus_sft/output/sft/%j.log
#SBATCH --error=logs/apertus_sft/output/sft/%j.err

set -euo pipefail

# --- Environment setup (same as recursive-unattended-accelerate.sh) ---
export PROJECT_ROOT_AT=$HOME/projects/posttraining/run
export ENABLE_RETRY=0
source $PROJECT_ROOT_AT/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/setup.sh

# --- SFT-specific config ---
# Paths relative to PROJECT_ROOT_AT (the container workdir)
SFT_SCRIPT="$PROJECT_ROOT_AT/reproducibility-scripts/apertus15-alignment/sft/sft_trainer.py"
ACCELERATE_CONFIG="src/swiss_alignment/configs/accelerate/ds-zero2.yaml"

CACHE_DIR="$SCRATCH/cache"
export OUTPUT_DIR="$SCRATCH/apertus_sft/${SLURM_JOB_ID}"
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
    LOCK_FILE=/tmp/pip_install_datasets.lock; \
    DONE_FILE=/tmp/pip_install_datasets.done; \
    if mkdir \$LOCK_FILE 2>/dev/null; then \
      echo \"[Node \$SLURM_NODEID] Installing datasets...\"; \
      pip install --upgrade \"transformers<5.0.0\" datasets && touch \$DONE_FILE; \
      echo \"[Node \$SLURM_NODEID] Install complete.\"; \
    else \
      echo \"[Node \$SLURM_NODEID] Waiting for install...\"; \
      while [ ! -f \$DONE_FILE ]; do sleep 5; done; \
      echo \"[Node \$SLURM_NODEID] Install detected, proceeding.\"; \
    fi; \
    exec accelerate launch \
    --config-file $ACCELERATE_CONFIG \
    --num_machines $SLURM_NNODES \
    --num_processes $((4*$SLURM_NNODES)) \
    --main_process_ip $(hostname) \
    --machine_rank \$SLURM_NODEID \
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

SRUN_EXIT_CODE=$?
END=$(date +%s)
DURATION=$(( END - START ))

echo "Training complete. Output saved to: $OUTPUT_DIR"
echo "Total execution time: $DURATION seconds"

exit $SRUN_EXIT_CODE

