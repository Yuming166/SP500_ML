#!/usr/bin/env bash
# Start Ling on an otherwise unused local port/GPU without touching Qwen or Hy.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_ENV="/DATA/lianjh/miniconda3/envs/casevo"
VLLM_PYTHON="${VLLM_ENV}/bin/python"
VLLM_BIN="${VLLM_ENV}/bin/vllm"
MODEL_PATH="/storage/lianjh/modelzoos/inclusionAI/Ling-3.0-tiny-int4"

test -x "${VLLM_PYTHON}"
test -x "${VLLM_BIN}"
test -d "${MODEL_PATH}"

export PATH="${VLLM_ENV}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=4
export HF_HOME="/storage/gaoym/.cache/huggingface"
export VLLM_CACHE_ROOT="/storage/gaoym/.cache/vllm"
export XDG_CACHE_HOME="/storage/gaoym/.cache"
export TRITON_CACHE_DIR="/storage/gaoym/.cache/triton"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"${VLLM_PYTHON}" - <<'PY'
from importlib.metadata import version
from vllm.model_executor.models import ModelRegistry

expected = {
    "vllm": "0.28.0",
    "torch": "2.13.0",
    "transformers": "5.16.1",
    "compressed-tensors": "0.17.0",
}
actual = {name: version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"frozen Ling runtime mismatch: {actual!r} != {expected!r}")
if "BailingMoeV3ForCausalLM" not in ModelRegistry.get_supported_archs():
    raise SystemExit("vLLM runtime does not support BailingMoeV3ForCausalLM")
PY

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
    --served-model-name Ling-3.0-tiny \
    --host 127.0.0.1 \
    --port 31520 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.80 \
    --dtype bfloat16 \
    --quantization compressed-tensors \
    --generation-config vllm \
    --seed 0 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --trust-remote-code
