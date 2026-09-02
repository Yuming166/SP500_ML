#!/usr/bin/env bash
# One-shot driver for LLM-S&P500 V1.
#
# Sequence:
#   1. prepare    (writes the frozen manifest; zero API calls)
#   2. audit      (re-runs the offline gates; non-zero exit on any failure)
#   3. smoke      (~5-8 LLM calls; verifies the client + parser)
#   4. formal     (~2,500 LLM calls; the main experiment)
#
# Usage:
#     bash scripts/run_llm_sp500_v1.sh                # inline
#     bash scripts/run_llm_sp500_v1.sh --bg           # inline prepare+audit+smoke; background formal
#     bash scripts/run_llm_sp500_v1.sh --skip-smoke   # reuse the last smoke records
#     bash scripts/run_llm_sp500_v1.sh --skip-formal  # stop after smoke
#     bash scripts/run_llm_sp500_v1.sh --no-resume    # ignore any partial records from a prior run
#
# Logs:
#   results/llm_sp500_v1/all.log                          (everything)
#   results/llm_sp500_v1/formal/run.log                   (formal only, when --bg)
#   results/llm_sp500_v1/formal/progress.json             (live progress)
#
# Env:
#   OPENAI_API_KEY     REQUIRED for smoke / formal.  Never logged.
#                      The operator must rotate this key after the run
#                      because it was exposed in the planning transcript
#                      (prereg §3, D4_v1).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

mkdir -p results/llm_sp500_v1
LOG="results/llm_sp500_v1/all.log"
: > "${LOG}"  # truncate

BG=0
SKIP_SMOKE=0
SKIP_FORMAL=0
NO_RESUME=0
for arg in "$@"; do
    case "${arg}" in
        --bg)          BG=1 ;;
        --skip-smoke)  SKIP_SMOKE=1 ;;
        --skip-formal) SKIP_FORMAL=1 ;;
        --no-resume)   NO_RESUME=1 ;;
        *) ;;
    esac
done

if [[ "${SKIP_SMOKE}" -eq 0 ]] || [[ "${SKIP_FORMAL}" -eq 0 ]]; then
    if [[ -z "${OPENAI_API_KEY:-}" ]] && [[ -z "${OPENAI_KEY:-}" ]]; then
        echo "[driver] OPENAI_API_KEY is not set; export it before smoke/formal." | tee -a "${LOG}"
    fi
fi

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
    PY="$(command -v python3)"
fi
echo "[driver] starting at $(date -Iseconds); bg=${BG} skip_smoke=${SKIP_SMOKE} skip_formal=${SKIP_FORMAL}; PY=${PY}" | tee -a "${LOG}"

echo "[driver] step 1/4: prepare" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.llm_sp500_v1 prepare 2>&1 | tee -a "${LOG}"

echo "[driver] step 2/4: audit" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.llm_sp500_v1 audit 2>&1 | tee -a "${LOG}"

echo "[driver] step 3/4: smoke" | tee -a "${LOG}"
if [[ "${SKIP_SMOKE}" -eq 1 ]] && [[ -f results/llm_sp500_v1/smoke/records.jsonl ]]; then
    echo "[driver] skipping smoke (records.jsonl present)" | tee -a "${LOG}"
else
    if [[ "${NO_RESUME}" -eq 1 ]]; then
        "${PY}" -m sp500_forecastability.llm_sp500_v1 smoke --no-resume 2>&1 | tee -a "${LOG}"
    else
        "${PY}" -m sp500_forecastability.llm_sp500_v1 smoke 2>&1 | tee -a "${LOG}"
    fi
fi

if [[ "${SKIP_FORMAL}" -eq 1 ]]; then
    echo "[driver] step 4/4: formal skipped (--skip-formal)" | tee -a "${LOG}"
    exit 0
fi

if [[ "${BG}" -eq 1 ]]; then
    echo "[driver] step 4/4: forking formal into the background" | tee -a "${LOG}"
    nohup "${PY}" -m sp500_forecastability.llm_sp500_v1 formal \
        $([[ "${NO_RESUME}" -eq 1 ]] && echo "--no-resume") \
        > results/llm_sp500_v1/formal/run.log 2>&1 &
    echo $! > results/llm_sp500_v1/formal/run.pid
    echo "[driver] formal PID $(cat results/llm_sp500_v1/formal/run.pid); tail results/llm_sp500_v1/formal/run.log" | tee -a "${LOG}"
    exit 0
fi

echo "[driver] step 4/4: formal" | tee -a "${LOG}"
if [[ "${NO_RESUME}" -eq 1 ]]; then
    "${PY}" -m sp500_forecastability.llm_sp500_v1 formal --no-resume 2>&1 | tee -a "${LOG}"
else
    "${PY}" -m sp500_forecastability.llm_sp500_v1 formal 2>&1 | tee -a "${LOG}"
fi

echo "[driver] step 5/5: report" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.llm_sp500_v1 report 2>&1 | tee -a "${LOG}"

echo "[driver] done at $(date -Iseconds)" | tee -a "${LOG}"
