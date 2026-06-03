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
    if [ -d "${subdir}actor" ]; then
        name="$(basename "${subdir}")"
        echo "Converting ${name} ..."
        python "${VERL_DIR}/scripts/legacy_model_merger.py" merge \
            --backend fsdp \
            --local_dir "${subdir}actor" \
            --target_dir "${OUTPUT_DIR}/${name}"
        echo "Done: ${OUTPUT_DIR}/${name}"
    fi
done
