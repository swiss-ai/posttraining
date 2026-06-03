#!/bin/bash
set -euo pipefail

VERL_DIR="/iopsstor/scratch/cscs/dmelikidze/verl"
CKPT_DIR="/iopsstor/scratch/cscs/dmelikidze/verl-training/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_6500-online-lr5e-6-beta0.1-bs256-lenNormfalse-maxPL2048-rollout8-images-2453606-2453633/global_step_948/actor"
OUTPUT_DIR="/iopsstor/scratch/cscs/dmelikidze/ap_mo/ap_1p5_sft/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_6500-online-lr5e-6-beta0.1-bs256-lenNormfalse-maxPL2048-rollout8-images-2453606-2453633"

python "${VERL_DIR}/scripts/legacy_model_merger.py" merge \
    --backend fsdp \
    --local_dir "${CKPT_DIR}" \
    --target_dir "${OUTPUT_DIR}"
