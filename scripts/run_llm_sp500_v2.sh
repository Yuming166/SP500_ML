#!/usr/bin/env bash
# One-shot driver for LLM-S&P500 V2.
#
# V2 inherits the frozen V1 manifest (sha 6fd3c3ed...) and all V1
# machinery (packet builder, routers, AURC CI); it differs only in the
# prompts (abstention-must-cite + verbatim-evidence-first) and in the
# backend being injectable from the CLI/env.
#
# Sequence:
#   1. prepare    (verifies the inherited manifest; zero API calls)
#   2. audit      (re-runs the offline gates; non-zero exit on any failure)
#   3. smoke      (5 LLM calls; verifies the parser on the abstain path)
#   4. formal     (2,500 LLM calls; the main experiment)
#   5. report     (routers + AURC CI + report.md)
#
# Usage:
#     bash scripts/run_llm_sp500_v2.sh                     # inline, local vLLM
#     bash scripts/run_llm_sp500_v2.sh --bg                # background formal
#     bash scripts/run_llm_sp500_v2.sh --endpoint URL --model NAME
#     bash scripts/run_llm_sp500_v2.sh --relay             # openapi.center gpt-5.4-mini
#
# Env:
#   OPENAI_API_KEY   REQUIRED for smoke / formal.  Never logged.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

VERSION="v2"
RES="results/llm_sp500_${VERSION}"
mkdir -p "${RES}"
LOG="${RES}/all.log"
: > "${LOG}"

BG=0
SKIP_SMOKE=0
SKIP_FORMAL=0
NO_RESUME=0
ENDPOINT="http://localhost:31519/v1/chat/completions"
MODEL="Hy-MT2-7B"
API_BASE="${OPENAI_BASE_URL:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bg)          BG=1 ;;
        --skip-smoke)  SKIP_SMOKE=1 ;;
        --skip-formal) SKIP_FORMAL=1 ;;
        --no-resume)   NO_RESUME=1 ;;
        --endpoint)    ENDPOINT="$2"; shift ;;
        --model)       MODEL="$2"; shift ;;
        --relay)
            ENDPOINT="${API_BASE%/}/chat/completions"
            MODEL="gpt-5.4-mini"
            ;;
        *) echo "[driver] unknown arg: $1"; exit 2 ;;
    esac
    shift
done

if [[ -z "${OPENAI_API_KEY:-}" ]] && [[ -z "${OPENAI_KEY:-}" ]]; then
    echo "[driver] OPENAI_API_KEY is not set; export it before smoke/formal." | tee -a "${LOG}"
fi

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
    PY="$(command -v python3)"
fi
echo "[driver] starting at $(date -Iseconds); bg=${BG} skip_smoke=${SKIP_SMOKE} skip_formal=${SKIP_FORMAL}; endpoint=${ENDPOINT}; model=${MODEL}" | tee -a "${LOG}"

run_step () {
    local step="$1"; shift
    echo "[driver] step: ${step}" | tee -a "${LOG}"
    "${PY}" -m "sp500_forecastability.llm_sp500_${VERSION}" "$@" 2>&1 | tee -a "${LOG}"
}

run_step "1/5 prepare (inherit V1 manifest)" prepare --endpoint "${ENDPOINT}" --model "${MODEL}"
run_step "2/5 audit" audit

echo "[driver] step 3/5: smoke" | tee -a "${LOG}"
if [[ "${SKIP_SMOKE}" -eq 1 ]] && [[ -f "${RES}/smoke/records.jsonl" ]]; then
    echo "[driver] skipping smoke (records.jsonl present)" | tee -a "${LOG}"
else
    if [[ "${NO_RESUME}" -eq 1 ]]; then
        run_step "smoke" smoke --no-resume --endpoint "${ENDPOINT}" --model "${MODEL}"
    else
        run_step "smoke" smoke --endpoint "${ENDPOINT}" --model "${MODEL}"
    fi
fi

if [[ "${SKIP_FORMAL}" -eq 1 ]]; then
    echo "[driver] step 4/5: formal skipped (--skip-formal)" | tee -a "${LOG}"
    exit 0
fi

if [[ "${BG}" -eq 1 ]]; then
    echo "[driver] step 4/5: forking formal into the background" | tee -a "${LOG}"
    mkdir -p "${RES}/formal"
    nohup "${PY}" -m "sp500_forecastability.llm_sp500_${VERSION}" formal \
        $([[ "${NO_RESUME}" -eq 1 ]] && echo "--no-resume") \
        --endpoint "${ENDPOINT}" --model "${MODEL}" \
        > "${RES}/formal/run.log" 2>&1 &
    echo $! > "${RES}/formal/run.pid"
    echo "[driver] formal PID $(cat "${RES}/formal/run.pid"); tail ${RES}/formal/run.log" | tee -a "${LOG}"
    exit 0
fi

echo "[driver] step 4/5: formal" | tee -a "${LOG}"
if [[ "${NO_RESUME}" -eq 1 ]]; then
    run_step "formal" formal --no-resume --endpoint "${ENDPOINT}" --model "${MODEL}"
else
    run_step "formal" formal --endpoint "${ENDPOINT}" --model "${MODEL}"
fi

run_step "5/5 report" report

echo "[driver] done at $(date -Iseconds)" | tee -a "${LOG}"
