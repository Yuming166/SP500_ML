# Recovery V3.7.1 preregistration: contradiction-aware ELAR

Protocol version: `recovery-v3.7.1-elar-fever-train-2026-09-02`

Status: **frozen before full development-ledger collection and before any
V3.7.1 formal model call**.

This protocol inherits every dataset, exclusion, selection, action,
certificate, threshold grid, development unlock, formal metric, formal gate,
novelty boundary, and claim boundary in
`docs/recovery_v3_7_preregistration.md`.

## Single bounded amendment

V3.7 called exactly four proof-eligible development candidates. All parsed.
One requested a contradiction and correctly quoted an entity mismatch, but the
model also marked that same mismatch as a hostile challenge. Under the frozen
gate this rejects valid Refuted proofs: the instruction did not distinguish a
counterexample to the claim from a counterexample to the proposed proof.

V3.7.1 changes only two explanatory sentences in the ledger system prompt:

- an explicitly contradicted claim term counts as established, not
  unsupported; and
- for a requested contradiction, the mismatch that proves the contradiction
  is not itself a challenge. A challenge means the quote fails to establish
  the requested semantic relation.

The JSON schema, exact-quote parser, seeds, token limit, one-repair rule,
threshold grid, prior PACE tie-break, fail-closed behavior, selection salts,
400-example formal sample, and five success gates are unchanged. V3.7 records
remain under `results/recovery_v3_7/` and are never mixed with V3.7.1 records.

No full development fitting and no formal call preceded this amendment. If the
V3.7.1 smoke parses, the prompt is frozen for all remaining development and
formal calls. Any later semantic weakness is an experimental result, not a
reason for another in-place change.
