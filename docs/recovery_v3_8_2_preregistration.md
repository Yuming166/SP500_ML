# Recovery V3.8.2 preregistration: closed semantic JSON conformance

Protocol version: `recovery-v3.8.2-qwen-to-ling-elar-2026-09-02`

Status: **frozen by `recovery_v3_8_2 prepare` before any V3.8.2 target-model
task call**.

## 1. Reason for the amendment

V3.8.1 was stopped before outcome evaluation after one of the first 130 formal
action bundles remained schema-invalid after the inherited single repair. The
row represented an empty citation set as the empty string rather than an empty
JSON array. No correctness, gain, route, subgroup result, certificate, or
ledger was computed.

The V3.8.2 manifest content-addresses the V3.8.1 manifest, abort record,
1,040-row partial action artifact, and the two cached responses behind the
failure. Its audit reads only transport status, error text, and response JSON
types and values; it never reads correctness or action outcomes. V3.8.1
artifacts remain immutable and are never reused for V3.8.2 inference.

## 2. Closed conformance layer

Before the inherited strict action parser, V3.8.2 applies exactly these
semantic-preserving canonicalizations:

1. a string answer is stripped and case-folded only when the result is exactly
   `yes` or `no`;
2. `evidence_ids` is renamed to `cited_evidence_ids` only when the former is
   present and the latter is absent;
3. an empty-string citation value is mapped to an empty list; and
4. a single citation string is mapped to a singleton list only when it exactly
   equals one of the evidence IDs in that request's packet.

Each applied transformation is recorded in the accepted decision's
`parse_mode`. All other unknown fields, missing fields, values, types,
duplicate citations, and out-of-packet citations retain strict rejection.
These transformations cannot alter answer polarity, confidence, or evidence
identity.

This is an infrastructure compatibility layer, not a claimed modeling
contribution. The scientific contribution remains the frozen ELAR policy and
its evidence/proof-conditioned route selection.

## 3. Everything else remains frozen

V3.8.2 inherits V3.8 verbatim except for Section 2:

- Qwen3.5-4B V3.7.1 ELAR router and thresholds `0.8`, `0.0`, and `1`;
- Ling-3.0-tiny-int4 through vLLM 0.28.0 on GPU 4;
- temperature 0, thinking disabled, prompts, seeds, and one repair attempt;
- the exact 400 formal examples, 200 per native label, and 1,200 unique roots;
- regeneration of five baseline decisions, all three actions, two atomic
  certificates, proof-eligible ledgers, and pre-outcome routes;
- zero Ling fitting, calibration, target-label use, or action-outcome use;
- 10,000-replicate question bootstrap with seed `20261102`; and
- all five V3.8 primary pass gates and all registered matched baselines.

The two fixed development smoke examples and all 400 formal examples are
rerun into a fresh cache. No V3.8 or V3.8.1 response is reused at inference.

## 4. Claim boundary

V3.8.2 may support a zero-shot Qwen-to-Ling recovery claim only if all five
pre-existing primary gates pass. Conformance counts and both prior aborts must
be reported. The result cannot support universal model invariance, live
retrieval robustness, or target-model tuning.
