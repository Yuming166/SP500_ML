#!/usr/bin/env bash
# V11.1: inherit V11 selection/597 valid rewrites, repair exactly three, then confirm.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
RESULTS="results/pilot_llm_v11_1"
mkdir -p "${RESULTS}"
LOG="${RESULTS}/all.log"
: > "${LOG}"
SKIP_FORMAL=0
for arg in "$@"; do
    case "${arg}" in
        --skip-formal) SKIP_FORMAL=1 ;;
        *) echo "[driver] unknown argument: ${arg}" >&2; exit 2 ;;
    esac
done
PY="${ROOT}/.venv/bin/python"
[[ -x "${PY}" ]] || PY="$(command -v python3)"
echo "[driver] V11.1 starts $(date -Iseconds); PY=${PY}" | tee -a "${LOG}"
echo "[driver] step 1/5: inherit exact V11 selection" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v11_1 prepare 2>&1 | tee -a "${LOG}"
echo "[driver] step 2/5: inherit 597 valid rewrites and repair exactly three" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v11_1 substitute-generation 2>&1 | tee -a "${LOG}"
echo "[driver] step 3/5: audit frozen V11.1 run manifest" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v11_1 audit 2>&1 | tee -a "${LOG}"
echo "[driver] step 4/5: smoke exactly 40 logical calls" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v11_1 smoke 2>&1 | tee -a "${LOG}"
if [[ "${SKIP_FORMAL}" -eq 1 ]]; then
    echo "[driver] step 5/5: formal skipped" | tee -a "${LOG}"
    exit 0
fi
echo "[driver] step 5/5: formal 4,000 logical calls inline" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v11_1 run 2>&1 | tee -a "${LOG}"
