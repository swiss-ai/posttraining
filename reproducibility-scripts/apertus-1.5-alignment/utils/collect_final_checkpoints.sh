#!/bin/bash
# Collect the final (highest-numbered) checkpoint from each model run
# and move it into a flat output directory, named after the model run.

set -euo pipefail

BASE_DIR="/iopsstor/scratch/cscs/dmelikidze/dmelikidze/projects/posttraining/run/artifacts/private/outputs/train_preference/apertus-first-sweep"
OUTPUT_DIR="/iopsstor/scratch/cscs/dmelikidze/ap_mo/suite8"

mkdir -p "$OUTPUT_DIR"

for model_dir in "$BASE_DIR"/*/; do
    model_name=$(basename "$model_dir")
    ckpt_base="$model_dir/checkpoints"

    if [[ ! -d "$ckpt_base" ]]; then
        echo "SKIP $model_name: no checkpoints dir"
        continue
    fi

    # Find the hash directory that contains checkpoint-* folders
    target_hash=""
    for hash_dir in "$ckpt_base"/*/; do
        if ls "$hash_dir" | grep -q '^checkpoint-'; then
            target_hash="$hash_dir"
            break
        fi
    done

    if [[ -z "$target_hash" ]]; then
        echo "SKIP $model_name: no hash with checkpoint-* folders"
        continue
    fi

    # Find the highest-numbered checkpoint
    final_ckpt=$(ls -d "$target_hash"/checkpoint-* | sort -t'-' -k2 -n | tail -1)

    if [[ -z "$final_ckpt" ]]; then
        echo "SKIP $model_name: no checkpoint found in $target_hash"
        continue
    fi

    dest="$OUTPUT_DIR/$model_name"
    echo "MOVE $final_ckpt -> $dest"
    mv "$final_ckpt" "$dest"
done

echo "Done. Final checkpoints are in: $OUTPUT_DIR"