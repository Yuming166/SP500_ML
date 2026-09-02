#!/usr/bin/env bash
# Frozen V3.9.1 Qwen-to-Fin-R1 ELAR driver.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"

"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_9_1 prepare
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_9_1 endpoint-check
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_9_1 smoke --workers 2
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_9_1 formal-actions --workers 8
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_9_1 formal-certificates --workers 8
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_9_1 formal-ledgers --workers 8
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_9_1 evaluate
