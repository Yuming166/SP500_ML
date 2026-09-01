#!/usr/bin/env bash
# V12.1: inherit V12 selection/1,073 rewrites, repair one item, then run four workers.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
RESULTS="results/pilot_llm_v12_1"
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
echo "[driver] V12.1 starts $(date -Iseconds); PY=${PY}" | tee -a "${LOG}"
echo "[driver] step 1/5: inherit exact V12 selection" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v12_1 prepare 2>&1 | tee -a "${LOG}"
echo "[driver] step 2/5: inherit 1,073 rewrites and repair exactly one" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v12_1 substitute-generation 2>&1 | tee -a "${LOG}"
echo "[driver] step 3/5: audit frozen amendment and four shards" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v12_1 audit 2>&1 | tee -a "${LOG}"
echo "[driver] step 4/5: smoke four question blocks, 80 logical calls" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v12_1 smoke 2>&1 | tee -a "${LOG}"
if [[ "${SKIP_FORMAL}" -eq 1 ]]; then
    echo "[driver] step 5/5: formal skipped" | tee -a "${LOG}"
    exit 0
fi
echo "[driver] step 5/5: formal 7,160 logical calls over four workers" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v12_1 run 2>&1 | tee -a "${LOG}"
