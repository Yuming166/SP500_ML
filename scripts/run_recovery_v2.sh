#!/usr/bin/env bash
# Recovery V2.2: page-root-disjoint paired recovery and frozen-router test.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
RESULT_ROOT="results/recovery_v2_2"
LOG_PATH="${RESULT_ROOT}/all.log"
RUNTIME_ENDPOINT="http://10.63.0.82:31518/v1/chat/completions"
mkdir -p "${RESULT_ROOT}"

run_stage() {
    echo "[recovery-v2.2] $*" | tee -a "${LOG_PATH}"
    "$@" 2>&1 | tee -a "${LOG_PATH}"
}

run_stage "${PYTHON_BIN}" -m sp500_forecastability.recovery_v2 prepare
run_stage "${PYTHON_BIN}" -m sp500_forecastability.recovery_v2 audit
run_stage "${PYTHON_BIN}" -m sp500_forecastability.recovery_v2 smoke --split train --workers 2 --endpoint "${RUNTIME_ENDPOINT}"
run_stage "${PYTHON_BIN}" -m sp500_forecastability.recovery_v2 run --split train --workers 4 --endpoint "${RUNTIME_ENDPOINT}"
run_stage "${PYTHON_BIN}" -m sp500_forecastability.recovery_v2 train-audit
run_stage "${PYTHON_BIN}" -m sp500_forecastability.recovery_v2 run --split dev --workers 4 --endpoint "${RUNTIME_ENDPOINT}"
run_stage "${PYTHON_BIN}" -m sp500_forecastability.recovery_v2 fit
run_stage "${PYTHON_BIN}" -m sp500_forecastability.recovery_v2 run --split test --workers 4 --endpoint "${RUNTIME_ENDPOINT}"
run_stage "${PYTHON_BIN}" -m sp500_forecastability.recovery_v2 evaluate
