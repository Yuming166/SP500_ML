# V10.4 post-run metric-validity audit

**Recorded:** 2026-09-01, after all 2,000 formal logical calls completed.
**Status:** This audit invalidates the stored `shared_weighted` co-primary
interpretation; it does not change, remove, rerun, or select any record.

## Complete run integrity

- Formal records: 2,000 / 2,000 unique `(cqid, agent_index, condition)` tuples.
- All records parsed successfully under `pilot-llm-v10.4-2026-09-01`.
- Conditions: 500 each for `original`, `remove`, `reverse`, and `substitute`.
- Agents: 400 records each for all five fixed personas.
- The pre-formal selection/audit had 100 frozen questions (50/50 labels) and
  300 / 300 usable substitutes (295 initial, 5 deterministic short
  normalizations).

## Critical outcome-leakage finding

The formal implementation computes, for each question,

```text
shared_weighted = frac_shared * (1 - correct) + 0.5 * frac_shared * correct
harmful_fc = int(correct == 0 and agreement >= 0.8)
```

(`src/sp500_forecastability/pilot_llm_v10.py`, `_per_question_risks`.) Thus the
stored `shared_weighted` score directly contains `correct`, which is observed
only after comparing the consensus with the native BoolQ outcome. For a
harmful false-consensus positive, `correct` is necessarily zero, so the score
is exactly `frac_shared`; for a correct consensus with the same citations it
is half as large. That makes the score outcome-dependent and therefore
ineligible as a pre-outcome routing/risk signal.

The stored co-primary result `shared_weighted` AUROC 0.959 [0.890, 0.997] and
its `PASS_SINGLE_SHARED_WEIGHTED` verdict must **not** be used as evidence for
the hypothesis. This is an implementation validity finding, not a decision to
discard an unfavorable result.

## Valid frozen summaries

| Registered field | Target | AUROC [95% bootstrap CI] | Valid interpretation |
| --- | --- | --- | --- |
| `D_OR` | harmful false consensus | 0.680 [0.496, 0.834] | Co-primary but does not clear its lower-bound rule. |
| `shared_citation_signal` (`frac_shared`, no outcome factor) | harmful false consensus | 0.623 [0.460, 0.768] | Non-leaky registered secondary; does not clear 0.5. |
| `D_inert` | harmful false consensus | 0.726 [0.530, 0.900] | Registered secondary endpoint; suggestive, but cannot be promoted post hoc. |

Harmful false-consensus prevalence was 12 / 100 (12.0%), which is below the
preregistered [20%, 70%] structural-interpretation band. Together with the
invalid `shared_weighted` metric, V10.4 does **not** establish the registered
cross-domain co-primary claim.

## Additional documentation deviation

The inherited report renderer labels `report.md` as “V10.1” although all formal
records and `summary.json` identify V10.4. `summary.json`, `records.jsonl`, and
this audit are the canonical V10.4 provenance; the report header is a cosmetic
inherited formatting defect. The separate smoke coverage mismatch (8 executed
versus 40 described) is recorded in `protocol_deviation.md`.

## Required boundary for any follow-up

V10.4 is frozen. A future test must preregister an outcome-independent
replacement for `shared_weighted` before inspecting a fresh held-out split or
new corpus. It must not relabel another V10.4 metric as co-primary, retune a
threshold on V10.4, or use the V10.4 positive-looking leaked score as support.
