# Recovery V3.8.1 preregistration: transport-only cross-model amendment

Protocol version: `recovery-v3.8.1-qwen-to-ling-elar-2026-09-02`

Status: **frozen by `recovery_v3_8_1 prepare` before any V3.8.1 target-model
task call**.

## 1. Reason for the amendment

V3.8 was stopped before outcome evaluation after three of the first 80 formal
action bundles remained schema-invalid after the inherited single repair. The
three terminal failures used `evidence_ids` in place of
`cited_evidence_ids`. Across the 640 generated action rows, the only other
frequent first-pass issue was answer casing (`Yes` or `No`). No correctness,
gain, route, or subgroup result was computed.

V3.8 artifacts remain immutable under `results/recovery_v3_8_ling`. V3.8.1
uses a new result root and content-addressed cache and does not reuse any V3.8
model response at inference.

The V3.8.1 manifest content-addresses the V3.8 manifest, abort record, and
640-row partial action artifact. It validates only the outcome-blind abort
facts (row and complete-bundle counts, terminal-failure count, and error text),
never correctness or action outcomes.

## 2. Sole permitted change

Before the inherited strict action parser, V3.8.1 applies exactly two
outcome-independent canonicalizations:

1. a string answer is stripped and case-folded only when the result is exactly
   `yes` or `no`;
2. `evidence_ids` is renamed to `cited_evidence_ids` only when the former is
   present and the latter is absent.

Both transformations are recorded in each accepted decision's `parse_mode`.
All other unknown fields, missing fields, values, types, duplicate citations,
and out-of-packet citations retain the inherited strict or fail-closed
behavior. The transformations cannot alter the answer polarity, confidence,
or cited evidence values.

## 3. Everything else remains frozen

V3.8.1 inherits V3.8 verbatim except for Section 2:

- Qwen3.5-4B V3.7.1 ELAR router and thresholds `0.8`, `0.0`, and `1`;
- Ling-3.0-tiny-int4 through vLLM 0.28.0 on GPU 4;
- temperature 0, thinking disabled, prompts, seeds, and one repair attempt;
- the exact 400 formal examples, 200 per native label, and 1,200 unique roots;
- regeneration of five baseline decisions, all three actions, two atomic
  certificates, proof-eligible ledgers, and pre-outcome routes;
- zero Ling fitting, calibration, target-label use, or action-outcome use;
- 10,000-replicate question bootstrap with seed `20261102`; and
- all five V3.8 primary pass gates and all registered matched baselines.

The two fixed development smoke examples are rerun into the fresh V3.8.1
cache. The 400 formal examples are then rerun in full; neither the 80 V3.8
partial examples nor their cached completions receive special treatment.

## 4. Claim boundary

V3.8.1 may support a zero-shot Qwen-to-Ling recovery claim only if all five
pre-existing primary gates pass. Parser normalization counts and the V3.8
abort must be reported alongside the result. This amendment does not support
universal model invariance, live retrieval robustness, or target-model tuning.
