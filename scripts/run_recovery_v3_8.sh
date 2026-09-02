#!/usr/bin/env bash
# Frozen zero-shot Qwen-to-Ling ELAR driver. The Ling service is external.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
RESULT_ROOT="${PROJECT_ROOT}/results/recovery_v3_8_ling"
mkdir -p "${RESULT_ROOT}"

"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_8 prepare
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_8 endpoint-check
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_8 smoke --workers 2
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_8 formal-actions --workers 8
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_8 formal-certificates --workers 8
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_8 formal-ledgers --workers 8
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_8 evaluate
