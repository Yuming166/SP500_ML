#!/usr/bin/env bash
# One-shot driver for Pilot-LLM V4.
#
# Sequence:
#   1. prepare      (writes the frozen manifest, fail-fast on substitute yield)
#   2. audit        (re-runs the gates; non-zero exit on any failure)
#   3. smoke        (8 LLM calls; ~30 s wall-clock)
#   4. formal       (1,000 LLM calls; ~5-10 min wall-clock; resumable)
#
# All four phases run inline by default. Pass --bg to fork the formal run
# into the background (you can poll progress.json or run scripts/wait_pilot_llm_v4.sh
# to block on it).
#
# Usage:
#     bash scripts/run_pilot_llm_v4.sh                # inline prepare+audit+smoke+formal
#     bash scripts/run_pilot_llm_v4.sh --bg           # inline prepare+audit+smoke; background formal
#     bash scripts/run_pilot_llm_v4.sh --skip-smoke   # reuse the last smoke records
#     bash scripts/run_pilot_llm_v4.sh --skip-formal  # stop after smoke
#
# Logs:
#   results/pilot_llm_v4/all.log                          (everything)
#   results/pilot_llm_v4/formal/run.log                   (formal only, when --bg)
#   results/pilot_llm_v4/formal/progress.json             (live progress; poll with cat)
#   results/pilot_llm_v4/formal/run.pid                   (when --bg)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

mkdir -p results/pilot_llm_v4
LOG="results/pilot_llm_v4/all.log"
: > "${LOG}"  # truncate

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

# 1+2: prepare + audit (atomic at the python level: `all` runs both)
CHAIN_ARGS=(--yes)
[[ "${SKIP_SMOKE}"  -eq 1 ]] && CHAIN_ARGS+=(--skip-smoke)

# 3: smoke
SMOKE_ARGS=()
[[ "${SKIP_SMOKE}" -eq 1 ]] && SMOKE_ARGS+=(--skip-smoke)

# We use the `audit` subcommand for the hard gate, then run `all` for prepare+smoke,
# so the formal step can be controlled independently of the others.
echo "[driver] step 1/4: prepare" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v4 prepare 2>&1 | tee -a "${LOG}"

echo "[driver] step 2/4: pre-formal audit" | tee -a "${LOG}"
"${PY}" -m sp500_forecastability.pilot_llm_v4 audit 2>&1 | tee -a "${LOG}"

echo "[driver] step 3/4: smoke (8 calls)" | tee -a "${LOG}"
if [[ "${SKIP_SMOKE}" -eq 1 ]] && [[ -f results/pilot_llm_v4/smoke/records.jsonl ]]; then
    echo "[driver] skipping smoke (records.jsonl present)" | tee -a "${LOG}"
else
    "${PY}" -m sp500_forecastability.pilot_llm_v4 smoke 2>&1 | tee -a "${LOG}"
fi

if [[ "${SKIP_FORMAL}" -eq 1 ]]; then
    echo "[driver] step 4/4: formal skipped (--skip-formal)" | tee -a "${LOG}"
    exit 0
fi

echo "[driver] step 4/4: formal (~1,000 calls; resumable)" | tee -a "${LOG}"

# Make sure the formal directory exists before nohup writes its log there.
mkdir -p results/pilot_llm_v4/formal

if [[ "${BG}" -eq 1 ]]; then
    nohup "${PY}" -m sp500_forecastability.pilot_llm_v4 run \
        > results/pilot_llm_v4/formal/run.log 2>&1 &
    FORMAL_PID=$!
    echo "${FORMAL_PID}" > results/pilot_llm_v4/formal/run.pid
    echo "[driver] formal launched in background, PID=${FORMAL_PID}" | tee -a "${LOG}"
    echo "[driver] tail:    tail -f results/pilot_llm_v4/formal/run.log" | tee -a "${LOG}"
    echo "[driver] poll:    cat results/pilot_llm_v4/formal/progress.json" | tee -a "${LOG}"
    echo "[driver] wait:    bash scripts/wait_pilot_llm_v4.sh" | tee -a "${LOG}"
    echo "[driver] done at $(date -Iseconds)" | tee -a "${LOG}"
    exit 0
fi

# Foreground formal: just call `run`. Resume is on by default.
"${PY}" -m sp500_forecastability.pilot_llm_v4 run 2>&1 | tee -a "${LOG}"
echo "[driver] done at $(date -Iseconds)" | tee -a "${LOG}"
