#!/usr/bin/env bash
# Frozen V3.11 unseen-Hy dual-head provenance-repair driver.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
EMBED_PYTHON="/DATA/lianjh/miniconda3/envs/casevo/bin/python"

"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_11 prepare-selection
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_11 fit-router
CUDA_VISIBLE_DEVICES=4 "${EMBED_PYTHON}" scripts/embed_recovery_v3_11_development.py \
  --selection results/recovery_v3_11_hy/selection_manifest.json \
  --output results/recovery_v3_11_hy/router_inputs.npz \
  --batch-size 24 \
  --max-length 768 \
  --device cuda \
  --inference-only
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_11 freeze-protocol
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_11 endpoint-check
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_11 smoke --workers 2
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_11 formal-actions --workers 8
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_11 freeze-routes
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_11 evaluate
