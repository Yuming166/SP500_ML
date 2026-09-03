#!/usr/bin/env bash
# Frozen local deployment for the V3.14 held-out Qwen3.6 target.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p results/recovery_v3_14_qwen36/runtime

export CUDA_VISIBLE_DEVICES=3,4
exec /DATA/lianjh/miniconda3/envs/casevo/bin/vllm serve \
  /storage/lianjh/modelzoos/Qwen/Qwen3.6-35B-A3B-FP8 \
  --host 127.0.0.1 \
  --port 31521 \
  --served-model-name Qwen3.6-35B-A3B \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --trust-remote-code \
  --enable-prefix-caching \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 16 \
  --enable-chunked-prefill \
  --default-chat-template-kwargs '{"enable_thinking":false}'
