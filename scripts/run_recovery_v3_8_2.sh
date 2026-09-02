#!/usr/bin/env bash
# Frozen V3.8.2 Qwen-to-Ling ELAR driver with semantic JSON conformance.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"

"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_8_2 prepare
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_8_2 endpoint-check
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_8_2 smoke --workers 2
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_8_2 formal-actions --workers 8
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_8_2 formal-certificates --workers 8
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_8_2 formal-ledgers --workers 8
"${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_8_2 evaluate
