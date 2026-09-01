#!/usr/bin/env bash
# One-shot driver for Pilot-LLM V10.1.
#
# Cross-domain replication of V7's protocol on BoolQ (Wikipedia yes/no),
# with sentence-level same-source evidence and a selection frozen before
# any rewrite or evaluation-model call.
# Tests whether R2 (AUROC-weighted vote) router generalizes from V4
# (TQA) to other diverse-consensus domains.
#
# PREREQUISITES:
#   1. BoolQ train.parquet (3.6 MB) and validation.parquet (1.3 MB) at
#      /storage/gaoym/sp500-forecastability-lab/data/boolq/
#   2. lianjh's vLLM endpoint running at http://10.63.0.88:31519
#
# Sequence (same as V5/V7):
#   1. prepare                (writes V10.1 text-only selection, 50/50 balanced)
#   2. substitute-generation  (writes exactly 300 frozen evidence rewrites)
#   3. audit                  (writes final run manifest; fails closed on any gap)
#   4. smoke                  (40 logical calls; cache-backed)
#   5. formal                 (2,000 LLM calls; ~10-20 min; resumable)
#
# Usage:
#     bash scripts/run_pilot_llm_v10.sh                # inline 1..5
#     bash scripts/run_pilot_llm_v10.sh --bg           # inline 1..4; background formal
#     bash scripts/run_pilot_llm_v10.sh --skip-smoke
#     bash scripts/run_pilot_llm_v10.sh --skip-formal

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

mkdir -p results/pilot_llm_v10_1
LOG="results/pilot_llm_v10_1/all.log"
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

# Pre-flight: BoolQ files exist
for f in train.parquet; do
    if [[ ! -f "/storage/gaoym/sp500-forecastability-lab/data/boolq/$f" ]]; then
        echo "[driver] ERROR: missing /storage/gaoym/sp500-forecastability-lab/data/boolq/$f" | tee -a "${LOG}"
        exit 1
    fi
done
echo "[driver] BoolQ dataset files OK" | tee -a "${LOG}"

# 1. prepare
echo "[driver] step 1/5: prepare (V10.1 text-only selection, N=100)" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v10 prepare 2>&1 | tee -a "${LOG}"

# 2. substitute-generation (idempotent via cache)
echo "[driver] step 2/5: substitute-generation (exactly 300 frozen rewrites)" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v10 substitute-generation 2>&1 | tee -a "${LOG}"

# 3. audit
echo "[driver] step 3/5: pre-formal audit" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v10 audit 2>&1 | tee -a "${LOG}"

# 4. smoke
echo "[driver] step 4/5: smoke (40 logical calls)" | tee -a "${LOG}"
if [[ "${SKIP_SMOKE}" -eq 1 ]] && [[ -f results/pilot_llm_v10_1/smoke/records.jsonl ]]; then
    echo "[driver] skipping smoke (records.jsonl present)" | tee -a "${LOG}"
else
    "${PY}" -m sp500_forecastability.pilot_llm_v10 smoke 2>&1 | tee -a "${LOG}"
fi

if [[ "${SKIP_FORMAL}" -eq 1 ]]; then
    echo "[driver] step 5/5: formal skipped (--skip-formal)" | tee -a "${LOG}"
    exit 0
fi

# 5. formal (foreground or background)
CMD="${PY} -m sp500_forecastability.pilot_llm_v10 run"
if [[ "${BG}" -eq 1 ]]; then
    echo "[driver] step 5/5: formal in background (~2,000 calls via vLLM)" | tee -a "${LOG}"
    mkdir -p results/pilot_llm_v10_1/formal
    nohup ${CMD} > results/pilot_llm_v10_1/formal/run.log 2>&1 &
    PID=$!
    echo "${PID}" > results/pilot_llm_v10_1/formal/run.pid
    echo "[driver] forked formal pid=${PID}; tail -f results/pilot_llm_v10_1/formal/run.log" | tee -a "${LOG}"
else
    echo "[driver] step 5/5: formal (~2,000 calls; inline)" | tee -a "${LOG}"
    ${CMD} 2>&1 | tee -a "${LOG}"
fi
