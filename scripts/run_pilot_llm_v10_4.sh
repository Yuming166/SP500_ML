#!/usr/bin/env bash
# V10.4: exact-target rewrite with a preregistered deterministic short suffix.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
RESULTS="results/pilot_llm_v10_4"
mkdir -p "${RESULTS}"
LOG="${RESULTS}/all.log"
: > "${LOG}"
BG=0
SKIP_SMOKE=0
SKIP_FORMAL=0
for arg in "$@"; do
    case "${arg}" in
        --bg) BG=1 ;;
        --skip-smoke) SKIP_SMOKE=1 ;;
        --skip-formal) SKIP_FORMAL=1 ;;
        *) echo "[driver] unknown argument: ${arg}" >&2; exit 2 ;;
    esac
done
PY="${ROOT}/.venv/bin/python"
[[ -x "${PY}" ]] || PY="$(command -v python3)"
DATASET="/storage/gaoym/sp500-forecastability-lab/data/boolq/train.parquet"
PARENT="results/pilot_llm_v10_1/selection_manifest.json"
[[ -f "${DATASET}" ]] || { echo "[driver] missing dataset ${DATASET}" | tee -a "${LOG}"; exit 1; }
[[ -f "${PARENT}" ]] || { echo "[driver] missing frozen parent ${PARENT}" | tee -a "${LOG}"; exit 1; }
echo "[driver] V10.4 starts $(date -Iseconds); bg=${BG}; PY=${PY}" | tee -a "${LOG}"
echo "[driver] step 1/5: inherit and validate V10.1 selection" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v10_4 prepare 2>&1 | tee -a "${LOG}"
echo "[driver] step 2/5: one rewrite per frozen item + deterministic short normalization" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v10_4 substitute-generation 2>&1 | tee -a "${LOG}"
echo "[driver] step 3/5: pre-formal audit" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v10_4 audit 2>&1 | tee -a "${LOG}"
if [[ "${SKIP_SMOKE}" -eq 1 && -f "${RESULTS}/smoke/records.jsonl" ]]; then
    echo "[driver] step 4/5: smoke skipped (completed records present)" | tee -a "${LOG}"
else
    echo "[driver] step 4/5: smoke (40 logical calls)" | tee -a "${LOG}"
    "${PY}" -m sp500_forecastability.pilot_llm_v10_4 smoke 2>&1 | tee -a "${LOG}"
fi
if [[ "${SKIP_FORMAL}" -eq 1 ]]; then
    echo "[driver] step 5/5: formal skipped" | tee -a "${LOG}"
    exit 0
fi
CMD=("${PY}" -m sp500_forecastability.pilot_llm_v10_4 run)
if [[ "${BG}" -eq 1 ]]; then
    echo "[driver] step 5/5: formal in background" | tee -a "${LOG}"
    mkdir -p "${RESULTS}/formal"
    nohup "${CMD[@]}" > "${RESULTS}/formal/run.log" 2>&1 &
    PID=$!
    echo "${PID}" > "${RESULTS}/formal/run.pid"
    echo "[driver] formal PID=${PID}" | tee -a "${LOG}"
else
    echo "[driver] step 5/5: formal inline" | tee -a "${LOG}"
    "${CMD[@]}" 2>&1 | tee -a "${LOG}"
fi
