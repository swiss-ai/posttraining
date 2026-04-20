#!/usr/bin/env bash
set -euo pipefail

BACKEND="${1:-}"
NO_CACHE=""
if [[ "${2:-}" == "--no-cache" ]]; then
    NO_CACHE="--no-cache"
fi

if [[ "${BACKEND}" != "vllm" && "${BACKEND}" != "sglang" ]]; then
    echo "Usage: $0 <vllm|sglang> [--no-cache]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TAG="verl-${BACKEND}:latest"
LOG_FILE="${SCRIPT_DIR}/build.log"

echo "Building ${IMAGE_TAG} (logs: ${LOG_FILE})"
echo "Follow live with: tail -f ${LOG_FILE}"

{
    echo "=== Build started at $(date) ==="
    echo "IMAGE_TAG=${IMAGE_TAG} BACKEND=${BACKEND}"
    echo ""

    podman build \
        --platform linux/arm64 \
        ${NO_CACHE} \
        -t "${IMAGE_TAG}" \
        -f "${SCRIPT_DIR}/Dockerfile.${BACKEND}" \
        "${SCRIPT_DIR}"

    echo ""
    echo "Built: localhost/${IMAGE_TAG}"
    echo "=== Build finished at $(date) ==="
} &> "${LOG_FILE}"

echo "Build done. Exporting to enroot..."

enroot import -o "${SCRIPT_DIR}/verl-${BACKEND}.sqsh" "podman://${IMAGE_TAG}" &>> "${LOG_FILE}"

echo "Done: ${SCRIPT_DIR}/verl-${BACKEND}.sqsh"
