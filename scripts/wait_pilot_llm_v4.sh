#!/usr/bin/env bash
# Block until the background Pilot-LLM V4 formal run finishes.
# Usage: bash scripts/wait_pilot_llm_v4.sh [--interval SECONDS]
#
# Exit code: the run's exit code, or 1 if no run was found.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

INTERVAL=15
if [[ "${1:-}" == "--interval" && -n "${2:-}" ]]; then
    INTERVAL="${2}"
fi

PID_FILE="results/pilot_llm_v4/formal/run.pid"
if [[ ! -f "${PID_FILE}" ]]; then
    echo "[wait] no run.pid at ${PID_FILE}; nothing to wait on" >&2
    exit 1
fi

PID=$(cat "${PID_FILE}")
echo "[wait] blocking on PID=${PID}; polling every ${INTERVAL}s"
while kill -0 "${PID}" 2>/dev/null; do
    if [[ -f results/pilot_llm_v4/formal/progress.json ]]; then
        echo "---- $(date -Iseconds) ----"
        cat results/pilot_llm_v4/formal/progress.json
    fi
    sleep "${INTERVAL}"
done

echo "[wait] process ${PID} has exited"
if [[ -f results/pilot_llm_v4/formal/records.jsonl ]]; then
    echo "[wait] records.jsonl present; final summary at results/pilot_llm_v4/formal/summary.json"
    .venv/bin/python -c "
import json
s = json.load(open('results/pilot_llm_v4/formal/summary.json'))
m = s['metrics']['D_OR__harmful_fc']
print('D_OR__harmful_fc:')
print('  auroc:', m['auroc'], 'ci:', m['auroc_ci'])
print('  auprc:', m['auprc'], 'ci:', m['auprc_ci'])
print('  risk_at_80:', m['risk_at_80'], 'ci:', m['risk_at_80_ci'])
print('  n_questions:', m['n_questions'])
print()
print('Calibration:', s['metrics'].get('D_OR__calibration'))
print('LOAO:', s['loao_robustness'])
"
else
    echo "[wait] no records.jsonl; check results/pilot_llm_v4/formal/run.log"
    tail -50 results/pilot_llm_v4/formal/run.log
    exit 2
fi
