#!/usr/bin/env bash
# Frozen zero-shot Qwen-to-SUFE Fin-R1 ELAR replication.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"

"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_9 prepare
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_9 endpoint-check
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_9 smoke --workers 2
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_9 formal-actions --workers 8
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_9 formal-certificates --workers 8
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_9 formal-ledgers --workers 8
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_9 evaluate
