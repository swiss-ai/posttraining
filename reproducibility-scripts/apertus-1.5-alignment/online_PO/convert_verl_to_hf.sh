#!/bin/bash
set -euo pipefail

VERL_DIR="/iopsstor/scratch/cscs/dmelikidze/verl"
CKPT_DIR="/iopsstor/scratch/cscs/dmelikidze/verl-training/online-DPO-lr1e-5-beta0.1-2021096-2021220/global_step_889/actor"
OUTPUT_DIR="/iopsstor/scratch/cscs/dmelikidze/ap_mo/active_dpo_new/apertus-v1.5-sft-1.5--online-DPO-R30-beta0.1-bs256-lnF-lr1e-5"

python "${VERL_DIR}/scripts/legacy_model_merger.py" merge \
    --backend fsdp \
    --local_dir "${CKPT_DIR}" \
    --target_dir "${OUTPUT_DIR}"
