#!/bin/bash
set -euo pipefail

VERL_DIR="/iopsstor/scratch/cscs/dmelikidze/verl"
CKPT_DIR="/iopsstor/scratch/cscs/dmelikidze/verl-training/Apertus-70B-Instruct-2509-SFT-online-lr2e-6-beta0.35-bs512-lenNormfalse-maxPL2048-rollout8-images-2630131-2630528/global_step_192/actor"
OUTPUT_DIR="/iopsstor/scratch/cscs/dmelikidze/ap_mo/ap_1p5_sft/final/10/Apertus-70B-Instruct-2509-SFT-online-lr2e-6-beta0.35-bs512-lenNormfalse-maxPL2048-rollout8-images-2630131-2630528"

python "${VERL_DIR}/scripts/legacy_model_merger.py" merge \
    --backend fsdp \
    --local_dir "${CKPT_DIR}" \
    --target_dir "${OUTPUT_DIR}"
