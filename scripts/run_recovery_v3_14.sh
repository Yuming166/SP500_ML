#!/usr/bin/env bash
# Frozen V3.14 zero-shot Qwen3.6 model-holdout driver.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
EMBED_PYTHON="/DATA/lianjh/miniconda3/envs/casevo/bin/python"

"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_14 prepare-selection
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_14 freeze-router
CUDA_VISIBLE_DEVICES=4 "${EMBED_PYTHON}" scripts/embed_recovery_v3_11_development.py \
  --selection results/recovery_v3_14_qwen36/selection_manifest.json \
  --output results/recovery_v3_14_qwen36/router_inputs.npz \
  --batch-size 24 \
  --max-length 768 \
  --device cuda \
  --inference-only
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_14 freeze-protocol
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_14 endpoint-check
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_14 smoke --workers 2
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_14 formal-actions --workers 8
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_14 freeze-routes
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_14 evaluate

