#!/bin/bash
set -euo pipefail

# VERL_DIR was /iopsstor/scratch/cscs/dmelikidze/verl until 2026-08-18 -- that path no
# longer exists (pre scratch-rebuild leftover; the tree now lives under dmelikidze/projects).
VERL_DIR="/iopsstor/scratch/cscs/dmelikidze/dmelikidze/projects/verl"

# Final checkpoint of the online-DPO run: 3107071 (steps 1-414, TIMEOUT) resumed as
# 3113892 -> completed all 474 steps, tracker=474, 1538 shards.
CKPT_DIR="/iopsstor/scratch/cscs/dmelikidze/verl-training/rl-init-dpo-step1921-online-lr2.5e-6-beta0.1-bs512-lenNormfalse-maxPL2048-rollout8-images-3106949-3107071/global_step_474/actor"
OUTPUT_DIR="/iopsstor/scratch/cscs/dmelikidze/ap_mo/lot_depends5"

python "${VERL_DIR}/scripts/legacy_model_merger.py" merge \
    --backend fsdp \
    --local_dir "${CKPT_DIR}" \
    --target_dir "${OUTPUT_DIR}"
