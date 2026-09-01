#!/usr/bin/env bash
# One-shot driver for Pilot-LLM V6.
#
# Sequence:
#   1. prepare                (writes the V6 manifest, N=100, fresh salt)
#   2. substitute-generation  (≤300 LLM calls; writes manifest to cache/;
#                              ~13K realized due to per-sentence cardinality)
#   3. audit                  (re-runs gates; non-zero exit on any failure)
#   4. smoke                  (8 LLM calls; ~30 s wall-clock)
#   5. formal                 (2,000 LLM calls; ~10-20 min wall-clock; resumable)
#
# Usage:
#     bash scripts/run_pilot_llm_v6.sh                # inline 1..5
#     bash scripts/run_pilot_llm_v6.sh --bg           # inline 1..4; background formal
#     bash scripts/run_pilot_llm_v6.sh --skip-smoke
#     bash scripts/run_pilot_llm_v6.sh --skip-formal
#
# Logs:
#   results/pilot_llm_v6/all.log
#   results/pilot_llm_v6/formal/run.log
#   results/pilot_llm_v6/formal/progress.json  (live; cat for status)
#   results/pilot_llm_v6/formal/run.pid        (when --bg)
#
# Companion script:
#     bash scripts/wait_pilot_llm_v6.sh             # blocks on background formal

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

mkdir -p results/pilot_llm_v6
LOG="results/pilot_llm_v6/all.log"
: > "${LOG}"

BG=0
SKIP_SMOKE=0
SKIP_FORMAL=0
for arg in "$@"; do
    case "${arg}" in
        --bg)          BG=1 ;;
        --skip-smoke)  SKIP_SMOKE=1 ;;
        --skip-formal) SKIP_FORMAL=1 ;;
        *) ;;
    esac
done

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
    PY="$(command -v python3)"
fi
echo "[driver] starting at $(date -Iseconds); bg=${BG} skip_smoke=${SKIP_SMOKE} skip_formal=${SKIP_FORMAL}; PY=${PY}" | tee -a "${LOG}"

# 1. prepare (manifest construction under fresh salt)
echo "[driver] step 1/5: prepare (V6 manifest, N=100, salt=pilot-llm-v6-2026-09-01)" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v6 prepare 2>&1 | tee -a "${LOG}"

# 2. substitute-generation (idempotent if cache/substitute_manifest.json exists)
echo "[driver] step 2/5: substitute-generation (≤300 LLM calls preregistered; ~13K realized)" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v6 substitute-generation 2>&1 | tee -a "${LOG}"

# 3. audit
echo "[driver] step 3/5: pre-formal audit" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v6 audit 2>&1 | tee -a "${LOG}"

# 4. smoke
echo "[driver] step 4/5: smoke (8 calls)" | tee -a "${LOG}"
if [[ "${SKIP_SMOKE}" -eq 1 ]] && [[ -f results/pilot_llm_v6/smoke/records.jsonl ]]; then
    echo "[driver] skipping smoke (records.jsonl present)" | tee -a "${LOG}"
else
    "${PY}" -m sp500_forecastability.pilot_llm_v6 smoke 2>&1 | tee -a "${LOG}"
fi

if [[ "${SKIP_FORMAL}" -eq 1 ]]; then
    echo "[driver] step 5/5: formal skipped (--skip-formal)" | tee -a "${LOG}"
    exit 0
fi

# 5. formal (foreground or background)
CMD="${PY} -m sp500_forecastability.pilot_llm_v6 run"
if [[ "${BG}" -eq 1 ]]; then
    echo "[driver] step 5/5: formal in background (~2,000 calls, ~10-20 min)" | tee -a "${LOG}"
    mkdir -p results/pilot_llm_v6/formal
    nohup ${CMD} > results/pilot_llm_v6/formal/run.log 2>&1 &
    PID=$!
    echo "${PID}" > results/pilot_llm_v6/formal/run.pid
    echo "[driver] forked formal pid=${PID}; tail -f results/pilot_llm_v6/formal/run.log" | tee -a "${LOG}"
else
    echo "[driver] step 5/5: formal (~2,000 calls; inline)" | tee -a "${LOG}"
    ${CMD} 2>&1 | tee -a "${LOG}"
fi