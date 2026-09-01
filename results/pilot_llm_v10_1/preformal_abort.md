# Pilot-LLM V10.1 pre-formal abort

**Observed:** 2026-09-01
**Status:** `PRE_FORMAL_ABORT` — no smoke or formal evaluation-model calls ran.

## Frozen gate outcome

V10.1 froze 100 BoolQ questions (50 yes / 50 no) and 300 sentence-evidence
units before substitute generation. Its preregistration requires a usable,
in-length-window substitute for every selected evidence unit; failed units may
not be replaced by later-ranked questions.

| Field | Observed value |
|---|---:|
| Selected evidence units | 300 |
| Usable substitutes | 217 |
| Unusable substitutes | 83 |
| Unusable fraction | 0.2767 |
| HTTP response cache entries | 300 |
| Smoke calls | 0 |
| Formal calls | 0 |

All 300 substitute requests received cached HTTP 200 responses. The 83 rejected
responses were not transport failures or multi-line parser failures: each
exceeded the frozen upper substitute-length ratio of 1.5 relative to its source
sentence. The runner therefore exited with code 2 before `audit`, `smoke`, or
`run`.

## Consequence

This is an instrumentation/constraint failure, not a result for or against the
router hypothesis. V10.1 must remain aborted. Continuing would require a new
pre-registered V10.2 protocol with a newly stated substitute-generation rule
and a separate cache namespace; it must not silently relax V10.1 or replace
frozen questions.
