# V10.2 pre-formal instrumentation abort

**Recorded:** 2026-09-01
**Protocol:** `pilot-llm-v10.2-2026-09-01`

The frozen inherited selection contained 100 BoolQ questions and 300 evidence
sentences. Substitute instrumentation completed with 292 usable initial
rewrites. Eight fixed `length_repair` calls were attempted for the remaining
eight items; none passed the unchanged 0.5--1.5x whitespace-token window.

| field | value |
| --- | ---: |
| initial usable | 292 / 300 |
| repair attempted | 8 |
| repair usable | 0 |
| final usable | 292 / 300 |
| final unusable | 8 / 300 |

Per the frozen V10.2 protocol, no item was replaced and the workflow stopped
before the audit, smoke, or formal stages. Therefore this directory contains
auxiliary rewrite calls only; it contains no evaluation-agent, confidence,
intervention, router, metric, or outcome record. V10.2 rewrite text and cache
records are not inputs to V10.3.
