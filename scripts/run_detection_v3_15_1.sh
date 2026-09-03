#!/usr/bin/env bash
# Frozen V3.15.1 bounded Ling conformance amendment driver.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"

"${PYTHON_BIN}" -m sp500_forecastability.detection_v3_15_1 freeze-protocol
"${PYTHON_BIN}" -m sp500_forecastability.detection_v3_15_1 endpoint-check
"${PYTHON_BIN}" -m sp500_forecastability.detection_v3_15_1 smoke
"${PYTHON_BIN}" -m sp500_forecastability.detection_v3_15_1 formal
"${PYTHON_BIN}" -m sp500_forecastability.detection_v3_15_1 freeze-preoutcome
"${PYTHON_BIN}" -m sp500_forecastability.detection_v3_15_1 evaluate

