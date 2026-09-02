#!/usr/bin/env bash
# Serve the local SUFE Fin-R1 checkpoint for frozen Recovery V3.9.

set -euo pipefail

VLLM_BIN="/DATA/lianjh/miniconda3/envs/casevo/bin/vllm"
MODEL_PATH="/storage/lianjh/modelzoos/SUFE-AIFLM-Lab/Fin-R1"
VLLM_ENV_BIN="/DATA/lianjh/miniconda3/envs/casevo/bin"

exec env CUDA_VISIBLE_DEVICES=4 PATH="${VLLM_ENV_BIN}:${PATH}" \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
    --served-model-name Fin-R1 \
    --host 127.0.0.1 \
    --port 31520 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --generation-config vllm \
    --seed 0 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --trust-remote-code
