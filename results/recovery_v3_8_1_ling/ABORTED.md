# Recovery V3.8.1 pre-evaluation abort

Status: `ABORTED_FORMAL_ACTION_SCHEMA_BEFORE_OUTCOME_EVALUATION`

The frozen V3.8.1 manifest has SHA-256
`10aa1823db2244f4782667c05ac3cc9ee3640276a1bfb5c23beaf3f0fee10655`.
The two-example development smoke completed with 16/16 terminal action rows
and 4/4 terminal certificate rows. Neither smoke example produced a
proof-eligible ledger candidate.

The formal action run was stopped after 130 complete example bundles (1,040
rows), before formal certificates, ledgers, route selection, or outcome
evaluation. One bundle contained one action row that remained invalid after
the frozen single repair. Both attempts returned a string instead of an array
for an empty citation set:

```json
{"answer": "No", "confidence": 0.0, "cited_evidence_ids": ""}
```

The terminal error was
`TypeError: cited_evidence_ids must be a list of strings`. No accuracy, gain,
label-stratified result, route, or action outcome was computed. The partial
records and content-addressed response cache are preserved under this result
root and are forbidden as V3.8.2 inference inputs.

V3.8.2 is a separately frozen transport-only amendment. In addition to the
V3.8.1 canonicalizations, its closed conformance layer maps an empty citation
string to an empty list and a single exact allowed evidence-ID string to a
singleton list. It does not change semantic prompts, source router, policy
thresholds, target model, formal examples, seeds, metrics, or gates, and it
reruns all target calls into a fresh cache.
