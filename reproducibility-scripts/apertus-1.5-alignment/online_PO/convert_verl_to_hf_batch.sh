#!/bin/bash
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir>"
    echo "  input_dir:  directory containing model subdirectories with /actor subfolders"
    echo "  output_dir: base directory for converted HF models"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
VERL_DIR="/iopsstor/scratch/cscs/dmelikidze/verl"

for subdir in "${INPUT_DIR}"/*/; do
    name="$(basename "${subdir}")"

    if [ -f "${subdir}latest_checkpointed_iteration.txt" ]; then
        iter="$(cat "${subdir}latest_checkpointed_iteration.txt")"
        step_dir="${subdir}global_step_${iter}"
    else
        echo "Skipping ${name}: no latest_checkpointed_iteration.txt"
        continue
    fi

    if [ -d "${step_dir}/actor" ]; then
        echo "Converting ${name} (global_step_${iter}) ..."
        python "${VERL_DIR}/scripts/legacy_model_merger.py" merge \
            --backend fsdp \
            --local_dir "${step_dir}/actor" \
            --target_dir "${OUTPUT_DIR}/${name}"
        echo "Done: ${OUTPUT_DIR}/${name}"
    else
        echo "Skipping ${name}: ${step_dir}/actor not found"
    fi
done
