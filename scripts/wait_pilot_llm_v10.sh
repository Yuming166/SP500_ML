#!/usr/bin/env bash
# Block until the background Pilot-LLM V10.1 formal run finishes.
# Usage: bash scripts/wait_pilot_llm_v10.sh [--interval SECONDS]
#
# V10.1 special: prints the cross-model × cross-domain comparison
# (V4 vs V7 vs V10) when §9.2 verdict is reached.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

INTERVAL=15
if [[ "${1:-}" == "--interval" && -n "${2:-}" ]]; then
    INTERVAL="${2}"
fi

PID_FILE="results/pilot_llm_v10_1/formal/run.pid"
if [[ ! -f "${PID_FILE}" ]]; then
    echo "[wait] no run.pid at ${PID_FILE}; nothing to wait on" >&2
    exit 1
fi

PID=$(cat "${PID_FILE}")
echo "[wait] blocking on PID=${PID}; polling every ${INTERVAL}s"
while kill -0 "${PID}" 2>/dev/null; do
    if [[ -f results/pilot_llm_v10_1/formal/progress.json ]]; then
        echo "---- $(date -Iseconds) ----"
        cat results/pilot_llm_v10_1/formal/progress.json
    fi
    sleep "${INTERVAL}"
done

echo "[wait] process ${PID} has exited"
if [[ -f results/pilot_llm_v10_1/formal/records.jsonl ]]; then
    echo "[wait] records.jsonl present; final summary at results/pilot_llm_v10_1/formal/summary.json"
    PY="${ROOT}/.venv/bin/python"
    [[ ! -x "${PY}" ]] && PY="$(command -v python3)"
    "${PY}" -c "
import json
s = json.load(open('results/pilot_llm_v10_1/formal/summary.json'))
print('=== V10.1 Co-primary verdict (BoolQ on Qwen3.5-4B) ===')
cpv = s.get('co_primary_verdict')
if cpv:
    print(f'  Verdict: {cpv[\"verdict\"]}')
    for k in ('D_OR', 'shared_weighted'):
        v = cpv.get(k, {})
        print(f'  {k}: AUROC={v.get(\"auroc\", \"NA\"):.4f}  CI=[{v.get(\"ci_lo\", \"NA\"):.3f}, {v.get(\"ci_hi\", \"NA\"):.3f}]  '
              f'passes_lo>0.5={v.get(\"passes_lower_bound_above_0_5\", False)}')
print()
print('=== V10.1 cross-domain shared_weighted comparison ===')
print(f'  V4 (TQA, N=50, V5 salt): 0.698 [0.359, 1.000]  CI wide')
print(f'  V7 (FEVER, N=100, V5 salt): 0.816 [0.567, 1.000]  CI lo > 0.5')
v10_sw = s['metrics'].get('shared_weighted__harmful_fc', {})
if v10_sw:
    print(f'  V10.1 (BoolQ, N=100, V10.1 salt): {v10_sw.get(\"auroc\", 0):.3f} '
          f'[{v10_sw.get(\"auroc_ci\", [0, 0])[0]:.3f}, {v10_sw.get(\"auroc_ci\", [0, 0])[1]:.3f}]  '
          f'{\"✅\" if v10_sw.get(\"auroc_ci\", [0, 0])[0] > 0.5 else \"❌\"}')
print()
print('LOAO:', s['loao_robustness'])
"
else
    echo "[wait] no records.jsonl; check results/pilot_llm_v10_1/formal/run.log"
    tail -50 results/pilot_llm_v10_1/formal/run.log
    exit 2
fi
