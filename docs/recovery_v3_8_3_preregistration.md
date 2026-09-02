# Recovery V3.8.3 preregistration: final closed transport conformance

Protocol version: `recovery-v3.8.3-qwen-to-ling-elar-2026-09-03`

Status: **frozen by `recovery_v3_8_3 prepare` before any V3.8.3 target-model
task call**.

## 1. Development-only schema qualification

V3.8.2 made no formal target calls. Its fixed two-example development schema
smoke found one row in which Ling serialized a probability as the percentage
`95`. No target correctness, action outcome, route, certificate, ledger, gain,
or subgroup result was computed. The V3.8.3 manifest content-addresses the
V3.8.2 manifest, smoke-abort record, 16-row partial action artifact, and the
two cached responses behind the failure.

Earlier V3.8 and V3.8.1 formal runs were stopped before outcome access for
transport-schema failures, and their records are not reused. The scientific
protocol, examples, and gates have never been changed in response to a target
accuracy or gain result.

## 2. Final closed conformance layer

Before strict action validation, V3.8.3 permits exactly five
semantic-preserving transformations:

1. strip and case-fold a string answer only when it becomes exactly `yes` or
   `no`;
2. rename `evidence_ids` to `cited_evidence_ids` only when unambiguous;
3. map an empty citation string to an empty list;
4. map one exact in-packet evidence-ID string to a singleton list; and
5. divide a finite, non-Boolean numeric confidence in `(1, 100]` by 100.

Applied transformations are recorded in `parse_mode`. Unknown evidence IDs,
ambiguous fields, other types, duplicate citations, nonfinite confidence,
negative confidence, and confidence above 100 remain invalid. No further
parser extension is permitted after freezing V3.8.3. Any terminal formal
action failure aborts the protocol without outcome evaluation.

This conformance layer is infrastructure rather than the paper's claimed
modeling novelty. The contribution under test is the evidence-ledger action
router (ELAR): frozen, evidence-root-aware, proof-carrying, pre-outcome route
selection transferred without target-model fitting.

## 3. Frozen formal experiment

V3.8.3 otherwise inherits V3.8 verbatim:

- Qwen3.5-4B V3.7.1 ELAR router and thresholds `0.8`, `0.0`, and `1`;
- Ling-3.0-tiny-int4 through vLLM 0.28.0 on GPU 4;
- temperature 0, thinking disabled, prompts, seeds, and one repair attempt;
- 400 fixed formal examples, 200 per native label, with 1,200 unique roots;
- five fresh baseline decisions, three actions, two atomic certificates,
  proof-eligible ledgers, and pre-outcome routes per applicable example;
- no Ling fitting, calibration, target-label use, or action-outcome use;
- 10,000-replicate question bootstrap with seed `20261102`; and
- all five V3.8 primary pass gates and registered matched baselines.

All development smoke and formal target responses use a fresh V3.8.3 cache.

## 4. Claim boundary

A zero-shot Qwen-to-Ling recovery claim requires all five existing gates to
pass. The final paper/artifact must report conformance counts and the prior
schema-abort sequence. This experiment cannot establish universal model
invariance, live retrieval robustness, or transfer to an untested model.
