#!/usr/bin/env bash
# V11 held-out BoolQ validation confirmation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
RESULTS="results/pilot_llm_v11"
mkdir -p "${RESULTS}"
LOG="${RESULTS}/all.log"
: > "${LOG}"
SKIP_SMOKE=0
SKIP_FORMAL=0
for arg in "$@"; do
    case "${arg}" in
        --skip-smoke) SKIP_SMOKE=1 ;;
        --skip-formal) SKIP_FORMAL=1 ;;
        *) echo "[driver] unknown argument: ${arg}" >&2; exit 2 ;;
    esac
done
PY="${ROOT}/.venv/bin/python"
[[ -x "${PY}" ]] || PY="$(command -v python3)"
DATASET="${ROOT}/data/boolq/validation.parquet"
[[ -f "${DATASET}" ]] || { echo "[driver] missing ${DATASET}" | tee -a "${LOG}"; exit 1; }
echo "[driver] V11 starts $(date -Iseconds); PY=${PY}" | tee -a "${LOG}"
echo "[driver] step 1/5: freeze N=200 held-out validation selection" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v11 prepare 2>&1 | tee -a "${LOG}"
echo "[driver] step 2/5: 600 fresh substitute rewrites" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v11 substitute-generation 2>&1 | tee -a "${LOG}"
echo "[driver] step 3/5: audit selection, substitutes, and risk contract" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v11 audit 2>&1 | tee -a "${LOG}"
if [[ "${SKIP_SMOKE}" -eq 1 && -f "${RESULTS}/smoke/records.jsonl" ]]; then
    echo "[driver] step 4/5: smoke reused" | tee -a "${LOG}"
else
    echo "[driver] step 4/5: smoke exactly 40 logical calls" | tee -a "${LOG}"
    "${PY}" -m sp500_forecastability.pilot_llm_v11 smoke 2>&1 | tee -a "${LOG}"
fi
if [[ "${SKIP_FORMAL}" -eq 1 ]]; then
    echo "[driver] step 5/5: formal skipped" | tee -a "${LOG}"
    exit 0
fi
echo "[driver] step 5/5: formal 4,000 logical calls inline" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v11 run 2>&1 | tee -a "${LOG}"
