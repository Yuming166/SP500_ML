# Recovery V3.9.2 preregistration: uniform JSON envelope conformance

Protocol version: `recovery-v3.9.2-qwen-to-finr1-elar-2026-09-03`

Status: **frozen before any V3.9.2 task-bearing Fin-R1 call**.

## 1. Development-only qualification

V3.9.1 made no formal target calls. On its two fixed development smoke
examples, actions were 16/16 valid, while two of four certificate rows failed
only because Fin-R1 appended prose to a complete JSON certificate. All four
failed attempts passed the unchanged certificate validator when their leading
JSON object was isolated. No target correctness, action outcome, route,
ledger, gain, or subgroup metric was computed.

The V3.9.2 manifest content-addresses the V3.9.1 manifest, abort record, smoke
action/certificate records, and four failed response-cache artifacts. None is
reused at inference.

## 2. Uniform envelope rule

V3.9.2 applies the V3.9.1 envelope rule uniformly to action decisions, atomic
certificates, and entailment ledgers. Strict parsing is always attempted first.
Fallback requires exactly one complete mapping at character zero followed by
nonempty text containing neither `{` nor `}`. The leading mapping then enters
the original artifact-specific validator without semantic relaxation.

Accepted certificate and ledger artifacts record
`transport_parse_mode=leading_json_with_trailing_text`. Multiple objects,
leading prose, braces in trailing text, partial JSON, schema errors, invalid
claim spans, nonlocal evidence IDs, inexact quotes, and invalid entailment
relations remain rejected. No further transport rule is permitted after
V3.9.2 freezes.

## 3. Frozen formal experiment

Everything scientific remains V3.9: the Qwen3.5-4B V3.7.1 ELAR router and
thresholds, Fin-R1 target checkpoint/runtime, exact 400 balanced formal
examples and 1,200 roots, five baselines, three actions, two atomic
certificates, proof-eligible ledgers, pre-outcome routing, bootstrap seed and
replicates, matched baselines, and all five pass gates. Ling outcomes are not
used for Fin-R1 fitting, calibration, or selection.

Terminal action failure aborts; certificate and ledger failures fail closed.
A pass supports a cross-checkpoint claim within the broad Qwen lineage, not a
cross-architecture or universal transfer claim.
