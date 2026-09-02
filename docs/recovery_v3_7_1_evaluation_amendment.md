# Recovery V3.7.1 evaluation-only amendment

Status: **recorded after formal calls but before pre-outcome route materialization
and before construction of any gold-derived metric**.

The frozen evaluator successfully constructed the primary ELAR route map, then
raised `AttributeError: 'NoneType' object has no attribute 'get'` while
constructing the prespecified `atomic_proof_only` ablation. This happened when
a fail-closed ledger row existed with `ledger: null`. No pre-outcome file,
oracle, accuracy, fix, harm, label-group gain, bootstrap interval, or gate had
been constructed.

The amendment changes only the proof-only selector's local display values from
`ledger_row.get("ledger", {})` to `(ledger_row.get("ledger") or {})`. The
proof-only selector has `require_ledger=False`; these values are used only as
tie-break defaults and are not an eligibility gate. Primary ELAR still requires
a successful non-null ledger and uses the original frozen code path.

The original implementation and router manifest remain unchanged and retain
their pre-formal hashes. A separate amendment module applies the one-line null
normalization in memory, adds its own hash and this document's hash to the
pre-outcome payload, and then invokes the frozen evaluator. No formal result is
used by this change.
