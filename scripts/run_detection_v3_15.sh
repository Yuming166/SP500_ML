#!/usr/bin/env bash
# Frozen V3.15 Ling replication driver. The Ling endpoint must already be up.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"

"${PYTHON_BIN}" -m sp500_forecastability.detection_v3_15 freeze-protocol
"${PYTHON_BIN}" -m sp500_forecastability.detection_v3_15 endpoint-check
"${PYTHON_BIN}" -m sp500_forecastability.detection_v3_15 smoke
"${PYTHON_BIN}" -m sp500_forecastability.detection_v3_15 formal
"${PYTHON_BIN}" -m sp500_forecastability.detection_v3_15 freeze-preoutcome
"${PYTHON_BIN}" -m sp500_forecastability.detection_v3_15 evaluate

