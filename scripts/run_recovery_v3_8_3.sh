#!/usr/bin/env bash
# Frozen V3.8.3 Qwen-to-Ling ELAR driver with final closed conformance.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"

"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_8_3 prepare
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_8_3 endpoint-check
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_8_3 smoke --workers 2
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_8_3 formal-actions --workers 8
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_8_3 formal-certificates --workers 8
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_8_3 formal-ledgers --workers 8
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_8_3 evaluate
