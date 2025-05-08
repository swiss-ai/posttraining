#!/bin/bash

#SBATCH -J apertus-checkpoint
#SBATCH -t 01:30:00
#SBATCH -A a-a10
#SBATCH --output=sunattended-distributed.out
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition debug

# Variables used by the entrypoint script
export PROJECT_ROOT_AT=$HOME/projects/Megatron-LM
export PROJECT_NAME=Megatron-Clariden

export CUDA_BUFFER_PAGE_IN_THRESHOLD_MS=0.001
export SLURM_ONE_ENTRYPOINT_SCRIPT_PER_NODE=1

export OMP_NUM_THREADS=1

srun \
  --container-image="/capstor/store/cscs/swissai/a06/containers/NGC-PyTorch/ngc_pt_jan.sqsh" \
  --environment="/iopsstor/scratch/cscs/ahernnde/ncg_pt.toml" \
  --container-mounts=\
$PROJECT_ROOT_AT,\
$SCRATCH,\
$HOME/projects/,\
/capstor/store/cscs/swissai/,\
/iopsstor/scratch/cscs/ahernnde \
  bash -c "\
  pip install git+https://github.com/swiss-ai/transformers.git && \
  CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=$PROJECT_ROOT_AT torchrun --nproc-per-node=4 $PROJECT_ROOT_AT/scripts/conversion/torchdist_2_torch.py \
   --bf16 \
   --pipeline-model-parallel-size=4 \
   --load /capstor/scratch/cscs/schlag/main_run_70B_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-70b-512-nodes-1e-5lr/checkpoints/ \
   --ckpt-convert-save /capstor/store/cscs/swissai/a10/swiss-alignment/meditron_checkpoints/intermediate_checkpoint && \
  python $PROJECT_ROOT_AT/tools/checkpoint/convert.py \
    --model-type GPT \
    --loader core \
    --saver swissai_hf \
    --load-dir /capstor/store/cscs/swissai/a10/swiss-alignment/meditron_checkpoints/intermediate_checkpoint/torch \
    --save-dir /capstor/store/cscs/swissai/a10/swiss-alignment/meditron_checkpoints/hf_checkpoint/ \
    --hf-tokenizer alehc/swissai-tokenizer
  "

exit 0

# sbatch --partition debug --time 01:30:00 --output=checkpoint.out ./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/megatron-to-hf-checkpoint.sh
