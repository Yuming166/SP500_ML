#!/usr/bin/env bash
# Block until the background Pilot-LLM V6 formal run finishes.
# Usage: bash scripts/wait_pilot_llm_v6.sh [--interval SECONDS]
#
# V6 has a single co-primary (D_OR); shared_weighted is reported as
# secondary S4_v6 (no longer co-primary, see §9 of the V6 preregistration).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

INTERVAL=15
if [[ "${1:-}" == "--interval" && -n "${2:-}" ]]; then
    INTERVAL="${2}"
fi

PID_FILE="results/pilot_llm_v6/formal/run.pid"
if [[ ! -f "${PID_FILE}" ]]; then
    echo "[wait] no run.pid at ${PID_FILE}; nothing to wait on" >&2
    exit 1
fi

PID=$(cat "${PID_FILE}")
echo "[wait] blocking on PID=${PID}; polling every ${INTERVAL}s"
while kill -0 "${PID}" 2>/dev/null; do
    if [[ -f results/pilot_llm_v6/formal/progress.json ]]; then
        echo "---- $(date -Iseconds) ----"
        cat results/pilot_llm_v6/formal/progress.json
    fi
    sleep "${INTERVAL}"
done

echo "[wait] process ${PID} has exited"
if [[ -f results/pilot_llm_v6/formal/records.jsonl ]]; then
    echo "[wait] records.jsonl present; final summary at results/pilot_llm_v6/formal/summary.json"
    PY="${ROOT}/.venv/bin/python"
    [[ ! -x "${PY}" ]] && PY="$(command -v python3)"
    "${PY}" -c "
import json
s = json.load(open('results/pilot_llm_v6/formal/summary.json'))
print('Co-primary verdict:', s.get('co_primary_verdict'))
print()
print('=== Single co-primary: D_OR__harmful_fc ===')
m = s['metrics'].get('D_OR__harmful_fc')
if m:
    print(f'  auroc: {m[\"auroc\"]}  ci: {m[\"auroc_ci\"]}')
    print(f'  auprc: {m[\"auprc\"]}  ci: {m[\"auprc_ci\"]}')
    print(f'  risk_at_80: {m[\"risk_at_80\"]}  ci: {m[\"risk_at_80_ci\"]}')
    print(f'  n_questions: {m[\"n_questions\"]}')
print()
print('=== Secondary (reported, not gating): shared_weighted__harmful_fc ===')
sw = s['metrics'].get('shared_weighted__harmful_fc')
if sw:
    print(f'  auroc: {sw[\"auroc\"]}  ci: {sw[\"auroc_ci\"]}')
    print(f'  (S4_v6: point estimate >= 0.5; CI lo > 0.5 is bonus, not §9.2 criterion)')
print()
print('=== Baseline collapse detectors ===')
for key in ('D_majority__harmful_fc', 'shared_citation_signal__harmful_fc'):
    m = s['metrics'].get(key)
    if m:
        print(f'  {key}: auroc={m[\"auroc\"]}  ci={m[\"auroc_ci\"]}')
print()
print('LOAO robustness:', s['loao_robustness'])
"
else
    echo "[wait] no records.jsonl; check results/pilot_llm_v6/formal/run.log"
    tail -50 results/pilot_llm_v6/formal/run.log
    exit 2
fi