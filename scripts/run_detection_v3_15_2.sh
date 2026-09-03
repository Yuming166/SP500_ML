#!/usr/bin/env bash
# Frozen V3.15.2 Ling detection replication after transport-only amendments.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"

"${PYTHON_BIN}" -m sp500_forecastability.detection_v3_15_2 freeze-protocol
"${PYTHON_BIN}" -m sp500_forecastability.detection_v3_15_2 endpoint-check
"${PYTHON_BIN}" -m sp500_forecastability.detection_v3_15_2 smoke
"${PYTHON_BIN}" -m sp500_forecastability.detection_v3_15_2 formal
"${PYTHON_BIN}" -m sp500_forecastability.detection_v3_15_2 freeze-preoutcome
"${PYTHON_BIN}" -m sp500_forecastability.detection_v3_15_2 evaluate

