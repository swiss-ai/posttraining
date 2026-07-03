#!/bin/bash
set -euo pipefail

VERL_DIR="/iopsstor/scratch/cscs/dmelikidze/verl"
CKPT_DIR="/iopsstor/scratch/cscs/dmelikidze/verl-training/rl_1p5-70b_notools_nothink_0107_340it-online-lr2e-6-beta0.25-bs512-lenNormfalse-maxPL2048-rollout8-images-2670791-2670864/global_step_250/actor"
OUTPUT_DIR="/iopsstor/scratch/cscs/dmelikidze/ap_mo/ap_1p5_70B_mid/rl_1p5-70b_notools_nothink_0107_340it-online-lr2e-6-beta0.25-bs512-lenNormfalse-maxPL2048-rollout8-images-2670791-2670864"

python "${VERL_DIR}/scripts/legacy_model_merger.py" merge \
    --backend fsdp \
    --local_dir "${CKPT_DIR}" \
    --target_dir "${OUTPUT_DIR}"
