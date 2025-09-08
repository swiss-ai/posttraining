# To back up data to capstor from time to time.
# Ideally should be automated to be performed at the end of each training run.

### Delete cache

# This should be done in  artifacts/shared/train_dpr from time to time.
find . -mindepth 1 -maxdepth 1 -type d | parallel -j32 'find {} -type f -name "cache-*" -print0' |   parallel -0 -j32 'rm -v {}'

### Refreshing/touching files

# Touch to keep the files we still need and avoid automatic cleanup
# Delete anything not needed and only touch the files we need to keep.
# This can only be done by the file owner.
find /iopsstor/scratch/cscs/smoalla/projects/posttraining/artifacts/shared/ -type f ! -name 'cache-*' -exec touch -a {} +

# For urgent cases, can touch with not the file owner but will re-trigger rsync
find  /iopsstor/scratch/cscs/smoalla/projects/posttraining/artifacts/shared/ -type f ! -name 'cache-*' \
  -exec bash -c '
        for f; do
            touch -a "$f" || touch "$f"
        done
  ' _ {} +


### Backing up data/shared and outputs/shared

# Parallel rsync

# For example from
cd /iopsstor/scratch/cscs/smoalla/projects/posttraining/artifacts/shared/outputs/train_preference/olmo2-qrpo && \
find . -mindepth 1 -maxdepth 1 -type d -print0   | \
  parallel --line-buffer -0 -j10 \
  rsync -av --info=progress2 {} \
  --exclude='**/cache-*' \
  /capstor/store/cscs/swissai/infra01/swiss-alignment/artifacts/outputs/train_preference/olmo2-qrpo/
