#!/usr/bin/env bash
# V12: exhaustive untouched BoolQ replication with four endpoint workers.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
RESULTS="results/pilot_llm_v12"
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
echo "[driver] V12 starts $(date -Iseconds); PY=${PY}" | tee -a "${LOG}"
echo "[driver] step 1/5: freeze all 358 untouched eligible roots" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v12 prepare 2>&1 | tee -a "${LOG}"
echo "[driver] step 2/5: four-worker auxiliary rewrites with one frozen repair" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v12 substitute-generation 2>&1 | tee -a "${LOG}"
echo "[driver] step 3/5: audit frozen selection, rewrites, score, and shards" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v12 audit 2>&1 | tee -a "${LOG}"
echo "[driver] step 4/5: smoke four question blocks, 80 logical calls" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v12 smoke 2>&1 | tee -a "${LOG}"
if [[ "${SKIP_FORMAL}" -eq 1 ]]; then
    echo "[driver] step 5/5: formal skipped" | tee -a "${LOG}"
    exit 0
fi
echo "[driver] step 5/5: formal 7,160 logical calls over four workers" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v12 run 2>&1 | tee -a "${LOG}"
