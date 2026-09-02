#!/usr/bin/env bash
# Frozen V3.13 provenance-gated counter-consensus cascade driver.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
EMBED_PYTHON="/DATA/lianjh/miniconda3/envs/casevo/bin/python"

"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_13 prepare-selection
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_13 freeze-router
CUDA_VISIBLE_DEVICES=3 "${EMBED_PYTHON}" scripts/embed_recovery_v3_11_development.py \
  --selection results/recovery_v3_13_hy18/selection_manifest.json \
  --output results/recovery_v3_13_hy18/router_inputs.npz \
  --batch-size 24 \
  --max-length 768 \
  --device cuda \
  --inference-only
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_13 freeze-protocol
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_13 endpoint-check
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_13 smoke --workers 2
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_13 formal-actions --workers 8
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_13 freeze-provisional
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_13 formal-teacher --workers 8
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_13 freeze-routes
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_13 evaluate
