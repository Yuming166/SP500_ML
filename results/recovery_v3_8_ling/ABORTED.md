# Recovery V3.8 pre-evaluation abort

Status: `ABORTED_FORMAL_ACTION_SCHEMA_BEFORE_OUTCOME_EVALUATION`

The frozen V3.8 manifest has SHA-256
`f0043bd23b67e1977d6a7fce1d8a68916100cef61861a3bbbecc84ba7a36d06e`.
The two-example development smoke completed with 16/16 terminal action rows
and 4/4 terminal certificate rows. Neither fixed smoke example produced a
proof-eligible ledger candidate, so the ledger smoke artifact is empty.

The formal action run was stopped after 80 complete example bundles (640
rows), before formal certificates, ledgers, route selection, or outcome
evaluation. Three bundles contained one action row that remained invalid after
the frozen single repair. All three terminal errors were the same
outcome-independent schema alias:

```text
ValueError: response fields mismatch; unknown=['evidence_ids'],
missing=['cited_evidence_ids']
```

Across all 640 rows, first-attempt parsing produced 34 answer-casing errors,
three evidence-key alias errors, and two citation-type errors. The inherited
repair recovered 33 rows, but not the three alias rows. No accuracy, gain,
label-stratified result, route, or action outcome was computed. The partial
records and content-addressed response cache are preserved under this result
root and are forbidden as V3.8.1 inference inputs.

V3.8.1 is a separately frozen transport-only amendment. It accepts only
case-insensitive `yes`/`no` values and the unambiguous `evidence_ids` alias when
`cited_evidence_ids` is absent. It does not change semantic prompts, source
router, policy thresholds, target model, formal examples, seeds, metrics, or
gates, and it reruns all target calls into a fresh cache.
