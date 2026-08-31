#!/usr/bin/env bash
# Block until the background Pilot-LLM V5 formal run finishes.
# Usage: bash scripts/wait_pilot_llm_v5.sh [--interval SECONDS]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

INTERVAL=15
if [[ "${1:-}" == "--interval" && -n "${2:-}" ]]; then
    INTERVAL="${2}"
fi

PID_FILE="results/pilot_llm_v5/formal/run.pid"
if [[ ! -f "${PID_FILE}" ]]; then
    echo "[wait] no run.pid at ${PID_FILE}; nothing to wait on" >&2
    exit 1
fi

PID=$(cat "${PID_FILE}")
echo "[wait] blocking on PID=${PID}; polling every ${INTERVAL}s"
while kill -0 "${PID}" 2>/dev/null; do
    if [[ -f results/pilot_llm_v5/formal/progress.json ]]; then
        echo "---- $(date -Iseconds) ----"
        cat results/pilot_llm_v5/formal/progress.json
    fi
    sleep "${INTERVAL}"
done

echo "[wait] process ${PID} has exited"
if [[ -f results/pilot_llm_v5/formal/records.jsonl ]]; then
    echo "[wait] records.jsonl present; final summary at results/pilot_llm_v5/formal/summary.json"
    PY="${ROOT}/.venv/bin/python"
    [[ ! -x "${PY}" ]] && PY="$(command -v python3)"
    "${PY}" -c "
import json
s = json.load(open('results/pilot_llm_v5/formal/summary.json'))
print('Co-primary verdict:', s.get('co_primary_verdict'))
print()
for key in ('D_OR__harmful_fc', 'shared_weighted__harmful_fc', 'D_majority__harmful_fc'):
    m = s['metrics'].get(key)
    if not m:
        continue
    print(f'{key}:')
    print(f'  auroc: {m[\"auroc\"]}  ci: {m[\"auroc_ci\"]}')
    print(f'  risk_at_80: {m[\"risk_at_80\"]}  ci: {m[\"risk_at_80_ci\"]}')
    print()
print('LOAO:', s['loao_robustness'])
"
else
    echo "[wait] no records.jsonl; check results/pilot_llm_v5/formal/run.log"
    tail -50 results/pilot_llm_v5/formal/run.log
    exit 2
fi
