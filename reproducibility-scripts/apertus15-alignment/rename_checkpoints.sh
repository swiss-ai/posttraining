#!/bin/bash
# Rename model checkpoint directories to condensed names.
# Usage:
#   ./rename_checkpoints.sh          # dry-run (just prints)
#   ./rename_checkpoints.sh --apply  # actually renames

set -euo pipefail

TARGET_DIR="/iopsstor/scratch/cscs/dmelikidze/aper_mods9"
DRY_RUN=true

if [[ "${1:-}" == "--apply" ]]; then
    DRY_RUN=false
fi

for dir in "$TARGET_DIR"/*/; do
    old_name=$(basename "$dir")

    # Extract algo (dpo or qrpo), seed, and run number
    if [[ "$old_name" =~ -SFT-(dpo|qrpo)- ]]; then
        algo="${BASH_REMATCH[1]}"
    else
        echo "SKIP: cannot parse algo from $old_name"
        continue
    fi

    if [[ "$old_name" =~ seed([0-9]+)-run([0-9]+)$ ]]; then
        seed="${BASH_REMATCH[1]}"
        run="${BASH_REMATCH[2]}"
    else
        echo "SKIP: cannot parse seed/run from $old_name"
        continue
    fi

    new_name="apertus-8b-${algo}-s${seed}-r${run}"

    if $DRY_RUN; then
        echo "$old_name -> $new_name"
    else
        mv "$TARGET_DIR/$old_name" "$TARGET_DIR/$new_name"
        echo "RENAMED $old_name -> $new_name"
    fi
done

if $DRY_RUN; then
    echo ""
    echo "(dry run — re-run with --apply to rename)"
fi
