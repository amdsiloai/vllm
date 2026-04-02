#!/usr/bin/env bash
# Run vLLM with fused shared experts enabled for Kimi-K2.5 MXFP4 on ROCm.
#
# Usage:
#   # Build the Docker image first (from the vLLM repo root):
#   docker build -f docker/Dockerfile.rocm_kimi_fused_se \
#       -t vllm-rocm-fused-se:latest .
#
#   # Run with defaults (amd/Kimi-K2.5-MXFP4, TP4, GPUs 4-7):
#   ./scripts/run_kimi_k25_fused_se.sh
#
# Environment variables:
#   MODEL   - HuggingFace model ID (default: amd/Kimi-K2.5-MXFP4)
#   TP      - Tensor parallel size (default: 4)
#   GPUS    - Comma-separated GPU indices (default: 4,5,6,7)
#   PORT    - API server port (default: 8000)
#   IMAGE   - Docker image name (default: vllm-rocm-fused-se:latest)
#   HF_HOME - HuggingFace cache directory (default: ~/.cache/huggingface)

set -euo pipefail

MODEL="${MODEL:-amd/Kimi-K2.5-MXFP4}"
TP="${TP:-4}"
GPUS="${GPUS:-4,5,6,7}"
PORT="${PORT:-8000}"
IMAGE="${IMAGE:-vllm-rocm-fused-se:latest}"
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

echo "=========================================="
echo " vLLM + ROCm Fused Shared Experts"
echo "=========================================="
echo " Model:    ${MODEL}"
echo " TP:       ${TP}"
echo " GPUs:     ${GPUS}"
echo " Port:     ${PORT}"
echo " Image:    ${IMAGE}"
echo " HF cache: ${HF_HOME}"
echo "=========================================="

docker run --rm -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --shm-size 64g \
    --security-opt seccomp=unconfined \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    -p "${PORT}:8000" \
    -e ROCR_VISIBLE_DEVICES="${GPUS}" \
    -e VLLM_ROCM_USE_AITER=1 \
    -e VLLM_ROCM_USE_AITER_MOE=1 \
    -e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 \
    "${IMAGE}" \
    --model "${MODEL}" \
    --tensor-parallel-size "${TP}" \
    --trust-remote-code \
    --enforce-eager
