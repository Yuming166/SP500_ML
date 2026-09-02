#!/usr/bin/env bash
# Frozen V3.12 selective cross-model co-sign driver.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
EMBED_PYTHON="/DATA/lianjh/miniconda3/envs/casevo/bin/python"

"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_12 prepare-selection
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_12 development-teacher --workers 8
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_12 freeze-router
CUDA_VISIBLE_DEVICES=4 "${EMBED_PYTHON}" scripts/embed_recovery_v3_11_development.py \
  --selection results/recovery_v3_12_hy18/selection_manifest.json \
  --output results/recovery_v3_12_hy18/router_inputs.npz \
  --batch-size 24 \
  --max-length 768 \
  --device cuda \
  --inference-only
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_12 freeze-protocol
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_12 endpoint-check
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_12 smoke --workers 2
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_12 formal-actions --workers 8
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_12 freeze-provisional
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_12 formal-teacher --workers 8
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_12 freeze-routes
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_12 evaluate
