#!/usr/bin/env bash
# Recovery V3.2/V3.3: CAPE development followed by the prospective CEW router.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
RESULT_ROOT="results/recovery_v3_2"
LOG_PATH="${RESULT_ROOT}/all.log"
mkdir -p "${RESULT_ROOT}"

run_stage() {
    echo "[recovery-v3.2] $*" | tee -a "${LOG_PATH}"
    "$@" 2>&1 | tee -a "${LOG_PATH}"
}

run_stage "${PYTHON_BIN}" -m sp500_forecastability.recovery_v3 prepare
run_stage "${PYTHON_BIN}" -m sp500_forecastability.recovery_v3 audit
run_stage "${PYTHON_BIN}" -m sp500_forecastability.recovery_v3 run --split train --workers 4
run_stage "${PYTHON_BIN}" -m sp500_forecastability.recovery_v3 train-audit
run_stage "${PYTHON_BIN}" -m sp500_forecastability.recovery_v3 run --split policy_dev --workers 4
run_stage "${PYTHON_BIN}" -m sp500_forecastability.recovery_v3 run --split calibration --workers 4
run_stage "${PYTHON_BIN}" -m sp500_forecastability.recovery_v3 fit
run_stage "${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_3 fit
run_stage "${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_3 test --workers 4
run_stage "${PYTHON_BIN}" -m sp500_forecastability.recovery_v3_3 evaluate
