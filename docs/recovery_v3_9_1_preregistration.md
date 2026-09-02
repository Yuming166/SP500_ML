# Recovery V3.9.1 preregistration: Fin-R1 trailing-prose transport amendment

Protocol version: `recovery-v3.9.1-qwen-to-finr1-elar-2026-09-03`

Status: **frozen before any V3.9.1 task-bearing Fin-R1 call**.

## 1. Outcome-blind reason

V3.9 stopped during its two-example development schema smoke, before any
formal call. Twelve of 16 cached baseline attempts began with a complete JSON
decision and then appended explanatory prose; four were already strict JSON.
No target correctness, action outcome, route, certificate, ledger, gain, or
subgroup metric was computed.

The V3.9.1 manifest content-addresses the V3.9 manifest, abort record, and all
16 smoke cache artifacts. Those responses qualify transport only and are not
reused for V3.9.1 inference.

## 2. Sole added rule

V3.9.1 first tries the exact frozen V3.8.3 parser. If it fails, the amendment
accepts a response only when Python's standard JSON decoder finds one complete
mapping at character zero and all remaining nonempty text contains neither
`{` nor `}`. The leading mapping is then serialized and passed through the
unchanged V3.8.3 conformance and strict action schema validator. The accepted
decision records `leading_json_with_trailing_text` in `parse_mode`.

This permits model commentary after the requested machine-readable answer but
does not choose between multiple JSON answers, salvage a partial object, or
alter answer polarity, confidence, or evidence identity. No further parser
extension is allowed after V3.9.1 freezes.

## 3. Unchanged formal science

All V3.9 choices remain frozen: the Qwen3.5-4B V3.7.1 ELAR router and
thresholds, Fin-R1 checkpoint/runtime, 400 balanced formal examples, fresh
five-person baseline and three actions, certificates, ledgers, pre-outcome
selection, 10,000 bootstrap replicates, and five primary gates. Ling outcomes
are not used for fitting or selecting V3.9.1.

Any terminal action failure aborts before evaluation. Certificate and ledger
failures fail closed. A pass supports only cross-checkpoint transfer within
the broad Qwen lineage, not cross-architecture or universal transfer.
