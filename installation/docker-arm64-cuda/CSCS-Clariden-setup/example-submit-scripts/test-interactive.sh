#!/bin/bash

# Enroot + Pyxis

srun \
  -J template-test \
  --pty \
  --container-image=$CONTAINER_IMAGES/$(id -gn)+$(id -un)+swiss-alignment+arm64-cuda-root-latest.sqsh \
  --no-container-mount-home \
  --no-container-remap-root \
  --no-container-entrypoint \
  --container-writable \
  bash

exit 0
