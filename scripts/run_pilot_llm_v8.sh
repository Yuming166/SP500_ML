#!/usr/bin/env bash
# One-shot driver for Pilot-LLM V8.
#
# Cross-model replication of V7 on Ling-3.0-tiny-int4 (inclusionAI).
# Same V5 salt + V7 selection + V7 protocol. Ling is served by SGLang
# on http://localhost:31520 (GPU 3 RTX 4090).
#
# PREREQUISITES:
#   1. SGLang server must already be running (see scripts/start_sglang_ling.sh).
#      This driver does NOT start SGLang itself.
#   2. /storage/lianjh/modelzoos/inclusionAI/Ling-3.0-tiny-int4/ must be
#      readable (yes, gaoym can read; confirmed earlier).
#
# Sequence:
#   1. prepare                (writes V8 manifest under V5's salt → V5 ⊂ V7 ⊂ V8)
#   2. substitute-generation  (reuses V5's substitute manifest via cache)
#   3. audit                  (re-runs gates; non-zero exit on any failure)
#   4. smoke                  (8 LLM calls; ~1-2 min)
#   5. formal                 (2,000 LLM calls; ~15-30 min; resumable)
#
# Usage:
#     bash scripts/run_pilot_llm_v8.sh                # inline 1..5
#     bash scripts/run_pilot_llm_v8.sh --bg           # inline 1..4; background formal
#     bash scripts/run_pilot_llm_v8.sh --skip-smoke
#     bash scripts/run_pilot_llm_v8.sh --skip-formal
#
# Logs:
#   results/pilot_llm_v8/all.log
#   results/pilot_llm_v8/formal/run.log
#   results/pilot_llm_v8/formal/progress.json  (live; cat for status)
#   results/pilot_llm_v8/formal/run.pid        (when --bg)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

mkdir -p results/pilot_llm_v8
LOG="results/pilot_llm_v8/all.log"
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

# Pre-flight: verify SGLang endpoint reachable
echo "[driver] checking SGLang endpoint http://localhost:31520/v1/models ..." | tee -a "${LOG}"
if ! curl -sf http://localhost:31520/v1/models > /dev/null 2>&1; then
    echo "[driver] ERROR: SGLang endpoint not reachable. Run scripts/start_sglang_ling.sh first." | tee -a "${LOG}"
    exit 1
fi
echo "[driver] SGLang endpoint OK" | tee -a "${LOG}"

echo "[driver] starting at $(date -Iseconds); bg=${BG} skip_smoke=${SKIP_SMOKE} skip_formal=${SKIP_FORMAL}; PY=${PY}" | tee -a "${LOG}"

# 1. prepare
echo "[driver] step 1/5: prepare (V8 manifest, V5 salt, V5 ⊂ V8)" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v8 prepare 2>&1 | tee -a "${LOG}"

# 2. substitute-generation (idempotent via cache)
echo "[driver] step 2/5: substitute-generation (reuses V5's manifest)" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v8 substitute-generation 2>&1 | tee -a "${LOG}"

# 3. audit
echo "[driver] step 3/5: pre-formal audit" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v8 audit 2>&1 | tee -a "${LOG}"

# 4. smoke
echo "[driver] step 4/5: smoke (8 calls, Ling via SGLang)" | tee -a "${LOG}"
if [[ "${SKIP_SMOKE}" -eq 1 ]] && [[ -f results/pilot_llm_v8/smoke/records.jsonl ]]; then
    echo "[driver] skipping smoke (records.jsonl present)" | tee -a "${LOG}"
else
    "${PY}" -m sp500_forecastability.pilot_llm_v8 smoke 2>&1 | tee -a "${LOG}"
fi

if [[ "${SKIP_FORMAL}" -eq 1 ]]; then
    echo "[driver] step 5/5: formal skipped (--skip-formal)" | tee -a "${LOG}"
    exit 0
fi

# 5. formal (foreground or background)
CMD="${PY} -m sp500_forecastability.pilot_llm_v8 run"
if [[ "${BG}" -eq 1 ]]; then
    echo "[driver] step 5/5: formal in background (~2,000 calls via SGLang)" | tee -a "${LOG}"
    mkdir -p results/pilot_llm_v8/formal
    nohup ${CMD} > results/pilot_llm_v8/formal/run.log 2>&1 &
    PID=$!
    echo "${PID}" > results/pilot_llm_v8/formal/run.pid
    echo "[driver] forked formal pid=${PID}; tail -f results/pilot_llm_v8/formal/run.log" | tee -a "${LOG}"
else
    echo "[driver] step 5/5: formal (~2,000 calls; inline)" | tee -a "${LOG}"
    ${CMD} 2>&1 | tee -a "${LOG}"
fi