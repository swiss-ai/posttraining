#!/bin/bash

#SBATCH -A a-infra01-1
#SBATCH --job-name=dpo
#SBATCH --output=logs/dpo/O-%x.%j
#SBATCH --error=logs/dpo/E-%x.%j
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00

# --- ARGUMENT PARSING ---
# 1. Dataset Path (required)
if [[ -n "$MyDatasetPath" ]]; then
    export DATASET_PATH="$MyDatasetPath"
    echo "Using Custom Dataset Path: $DATASET_PATH"
else
    echo "Error: 'MyDatasetPath' environment variable is required."
    exit 1
fi

# 2. Output Directory
if [[ -n "$MyOutputDir" ]]; then
    export OUTPUT_DIR="$MyOutputDir"
    echo "Using Custom Output Dir: $OUTPUT_DIR"
else
    export OUTPUT_DIR="$SCRATCH/models/dpo"
    echo "Using Default Output Dir: $OUTPUT_DIR"
fi

# 3. Seed
if [[ -n "$MySeed" ]]; then
    export SEED="$MySeed"
    echo "Using Custom Seed: $SEED"
else
    export SEED="42"
    echo "Using Default Seed: $SEED"
fi

# 4. Extra training arguments (forwarded directly to the python training script)
if [[ -n "$MyExtraArgs" ]]; then
    export EXTRA_ARGS="$MyExtraArgs"
    echo "Using Extra Args: $EXTRA_ARGS"
else
    export EXTRA_ARGS=""
    echo "No extra args provided."
fi

echo "Dataset Path: $DATASET_PATH"
echo "Seed:         $SEED"
echo "Output Dir:   $OUTPUT_DIR"
echo "Extra Args:   $EXTRA_ARGS"
# ------------------------

BASE_CACHE_DIR="$SCRATCH"
export HF_HOME=$BASE_CACHE_DIR/hf_home
export VLLM_CACHE_DIR=$BASE_CACHE_DIR/vllm_cache
export WANDB_PROJECT=DPO
export ACCELERATE_DIR="${ACCELERATE_DIR:-/accelerate}"

# Project root: resolve from home directory to where the code lives
PROJECT_ROOT="/users/dmelikidze/projects/posttraining/run"

MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MAIN_PROCESS_PORT=29500
NUM_PROCESSES=$(expr $SLURM_NNODES \* $SLURM_GPUS_ON_NODE)

CMD="accelerate launch \
    --config_file=$SCRATCH/ActiveUltraFeedback/configs/accelerate/deepspeed2.yaml \
    --num_processes $NUM_PROCESSES \
    --num_machines $SLURM_NNODES \
    --machine_rank \$SLURM_NODEID \
    --main_process_ip $MAIN_PROCESS_IP \
    --main_process_port $MAIN_PROCESS_PORT \
    -m reproducibility-scripts.apertus15-alignment.dpo.training \
    --config_path $SCRATCH/ActiveUltraFeedback/configs/dpo_training.yaml \
    --output_dir $OUTPUT_DIR \
    --dataset_path $DATASET_PATH \
    --seed $SEED \
    $EXTRA_ARGS"

echo $CMD

START=$(date +%s)

# Change to project root so module resolution works from home directory
cd $PROJECT_ROOT
srun --environment=activeuf_dev bash -c "$CMD"

END=$(date +%s)
DURATION=$(( END - START ))

echo "Job ended at: $(date)"
echo "Total execution time: $DURATION seconds"