# Pilot-LLM V10.1 preregistration — BoolQ sentence-evidence replication

**Date frozen:** 2026-09-01
**Status:** Frozen before every V10/V10.1 evaluation-model call. There are no
V10 result records. V10.1 supersedes the unexecuted V10 design; the original
V10 implementation and preregistration remain recoverable at Git commit
`971989e`.

## 1. Purpose and permitted pre-run optimisation

V9 established a structural boundary on FEVER: evidence redundancy creates
saturated agent consensus, leaving no meaningful answer diversity for an
agent-level router. V10.1 tests the registered converse in an independent
domain:

> Under related-but-distinct evidence units from the same source, does the
> paired-intervention risk signal (`D_OR` and/or `shared_weighted`) rank harmful
> false consensus above chance, and does R2 beat simple majority aggregation?

The design improves **construct validity and test sensitivity** before model
calls. It does not select examples using an evaluation answer, confidence,
intervention response, router score, or outcome. This gives the hypothesis a
fairer test without post-hoc outcome selection.

## 2. Why V10.1 supersedes V10

The unexecuted V10 code audit found four pre-run defects:

1. The task text exposed gold answers and full evidence.
2. Page clustering used a guessed title from passage-leading words.
3. Substitute generation and audit used incompatible fields/loaders.
4. Sample selection depended on substitute availability.

V10.1 corrects all four before any evaluation call. It is a new version rather
than a silent edit of a frozen protocol.

## 3. Dataset snapshot and text-only eligibility rule

**Dataset:** `data/boolq/train.parquet` from `google/boolq`.

- SHA-256: `4f028e992c0bd4df30b9f056f4946b64f5c23028034ff0ed5ea467d8538cc623`
- Raw rows: 9,427.
- This parquet mirror contains no title field, so V10.1 does not infer one.

Each candidate is one native BoolQ `question`/`passage`/`answer` record. Its
provenance root is `SHA256(question + "\n" + raw_passage)`, not a named-entity
heuristic. The agent receives the one native BoolQ question and never sees an
answer label in the prompt.

An item is eligible only when all following text-only gates hold:

1. Split the raw passage on `(?<=[.!?])\s+`; retain sentences with 8–80 tokens.
2. Take the first three retained sentences in source order. No semantic
   re-ranking, LLM selection, or outcome-dependent filtering is allowed.
3. Compute pairwise TF-IDF cosine: lowercase, English stop words, 1–2 grams.
   Require mean pairwise cosine >= 0.08 and maximum pairwise cosine in
   [0.10, 0.60].
4. Retain all three sentences as `E01`, `E02`, `E03`; inherit the native BoolQ
   yes/no label unchanged.

This yields 1,584 candidates before selection: 987 yes and 597 no. These are
offline text-feasibility counts, not model results. The gates keep evidence at
one verifiable source while rejecting nearly duplicate and unrelated packets.

## 4. Frozen selection and interventions

- Salt: `pilot-llm-v10.1-2026-09-01`.
- Rank eligible candidates independently within native yes/no labels by
  `SHA256(salt || cqid)`.
- Select exactly 50 yes and 50 no questions: **N = 100**.
- All three evidence units of one question share its provenance root.

Five agents receive the existing 2-of-3 partition table. The evaluation task
contains only the native BoolQ question; evidence appears only in the
agent-specific packet. Conditions remain `original`, `remove`, `reverse`, and
`substitute`.

The mandatory sequence is:

1. `prepare`: write and validate the text-only 100-question selection manifest.
2. `substitute-generation`: rewrite exactly its 300 frozen evidence units.
3. `audit`: require a usable, in-length-window rewrite for all 300 units and
   write the final run manifest.
4. `smoke`: 2 questions x 5 agents x 4 conditions = 40 logical calls.
5. `run`: 100 x 5 x 4 = 2,000 logical calls, resumable through the cache.

If any substitute is unusable, V10.1 aborts; it does not replace that question
with a later-ranked candidate. The audit also aborts on a changed dataset
digest, unbalanced labels, mixed source roots, failed overlap gates, or leakage.

## 5. Frozen inference, metrics, and verdicts

The endpoint, Qwen3.5-4B model, temperature 0, JSON repair limit, five fixed
agent personas, 1,000 question bootstrap replicates, `D_OR`,
`shared_weighted`, calibration, leave-one-agent-out reporting, and all router
variants are inherited unchanged from V7/V10.

The co-primary verdict is unchanged: pass if the 95% bootstrap CI lower bound
for either `AUROC(D_OR, harmful_fc)` or
`AUROC(shared_weighted, harmful_fc)` exceeds 0.5. `harmful_fc` remains an
incorrect consensus with agreement >= 0.8. Correctness is native BoolQ answer
equality (`yes` iff the original BoolQ answer is true).

All primary/secondary metrics and all five router/baseline cells are reported.
If the intervention target is single-class or harmful-false-consensus prevalence
lies outside [0.20, 0.70], V10.1 reports a structural boundary result rather
than tuning or silently re-sampling.

## 6. Interpretation boundary

A positive result supports only the narrow claim that provenance-aware,
paired-intervention routing is viable in a controlled heterogeneous BoolQ
evidence regime. It would not establish cross-model generalisation, general
factuality, financial predictability, or overturn the FEVER boundary.
