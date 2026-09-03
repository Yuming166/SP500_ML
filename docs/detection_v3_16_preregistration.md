# Detection V3.16 development preregistration: label-symmetric contrastive transfer

Date: 2026-09-03 (Asia/Shanghai)

Protocol: `detection-v3.16-vitaminc-symmetric-development-2026-09-03`

Status: **development protocol only**. It freezes dataset construction,
observable intervention coordinates, pilot budgets, and the development/formal
firewall. It does not authorize the 500-item formal experiment. A successful
Qwen development run and frozen Ling transfer pilot are required before a new
formal protocol can be registered.

## 1. Research question

V12.1 and the Ling V3.15 replication passed aggregate error-detection and
Risk@80 gates, but both had a severe label-direction reversal. V3.16 asks:

> Can evidence interventions rank wrong high-consensus decisions under an
> exactly paired SUPPORTS/REFUTES construction, and does the same frozen risk
> transfer from Qwen3.5-4B to the non-Qwen Ling-3.0-tiny family?

The experiment is designed to falsify answer-prior and intervention-polarity
explanations. Every contrastive pair contributes two items with the identical
claim: one has naturally supporting evidence and one has a minimally edited,
naturally refuting evidence sentence.

## 2. Dataset and license

Dataset: VitaminC real contrastive fact verification test data from the
official NAACL 2021 release. Only naturally occurring Wikipedia revisions are
used; the VitaminC synthetic/FEVER-derived split is excluded.

- Official archive SHA-256:
  `04cce67d000a61fd83885d68924210dd08f3d0ac91fde9f5a4b0bfb768339418`
- Extracted `test.jsonl` SHA-256: recorded in the generated manifest.
- License: Wikipedia terms or CC BY-SA 3.0, as stated in the release.
- Local data path: `/storage/gaoym/datasets/vitaminc_real/`; dataset contents
  are not copied into the project repository.

## 3. Deterministic pair gate

Rows are grouped by `(case_id, normalized_claim)`. A candidate must have exactly
one `SUPPORTS` row and exactly one `REFUTES` row. The pair passes only if:

- claim length is 5--40 alphanumeric tokens;
- each evidence sentence is 10--120 alphanumeric tokens;
- neither claim nor evidence contains a Unicode replacement character;
- supporting and refuting evidence strings differ;
- character-level `SequenceMatcher` ratio is at least 0.97; and
- token-set Jaccard similarity is at least 0.90.

At most one pair per Wikipedia page is kept, chosen by salted SHA-256 ranking.
The page is the provenance/source root, so no page can occur in more than one
split.

## 4. Fixed split sizes

Target pages are ranked by
`SHA256(protocol_salt || page || case_id || normalized_claim)`.

- smoke: 4 pairs = 8 label-balanced items;
- Qwen development: 30 pairs = 60 label-balanced items;
- held-out formal candidate set: 250 pairs = 500 label-balanced items.

For every pair, one distinct distractor page is selected from outside all
target pages. Distractor assignment is one-to-one and requires target-claim vs
distractor-evidence token Jaccard at most 0.05. Target and distractor pages are
disjoint across smoke, development, and formal partitions.

Formal pages are frozen before any V3.16 model call. Development code must
refuse to call formal items.

## 5. Label-symmetric item construction

Each natural contrastive pair produces two items:

| Item | Original evidence | Gold label | Reverse evidence |
| --- | --- | --- | --- |
| `support` | natural supporting revision | SUPPORTS | natural refuting revision |
| `refute` | natural refuting revision | REFUTES | natural supporting revision |

Both items share the same claim and the same irrelevant substitute evidence.
Consequently each pair contains exactly one item per native label and reverse
has identical semantics in both directions.

## 6. Conditions and agents

Five fixed personas are used: literal evidence user, skeptical auditor,
consistency checker, counterfactual auditor, and minimal judge. Every
question--agent pair receives four conditions:

1. `original`: the item-label-consistent natural evidence;
2. `remove`: no evidence sentence;
3. `reverse`: the naturally paired opposite-label evidence;
4. `substitute`: an unrelated, source-disjoint evidence sentence.

The prompt, schema, temperature, seed rule, and model remain constant across
conditions. Agents output exactly one of `SUPPORTS` or `REFUTES`, confidence,
and citations restricted to the visible packet. `remove` requires an empty
citation list. Hidden chain-of-thought is neither requested nor stored.

## 7. Staged call budget

Small calls are permitted only on registered non-formal roots:

- transport smoke: 8 items x 5 agents x 4 conditions = 160 calls per model;
- Qwen development: 60 x 5 x 4 = 1,200 calls;
- Ling transfer pilot: the same 60 items and frozen score = 1,200 calls;
- formal, if separately authorized: 500 x 5 x 4 = 10,000 calls per model.

The 60-item development set cannot be enlarged after observing outcomes.
Formal execution is all-or-nothing; no optional stopping based on partial
formal metrics is permitted.

## 8. Intervention-only risk coordinates

For each item, compute from paired agent outputs before label access:

- `reverse_inertia`: fraction of agents whose answer does not flip from
  original to reverse;
- `remove_inertia`: fraction whose answer is unchanged after evidence removal;
- `substitute_inertia`: fraction unchanged under unrelated substitution;
- `reverse_confidence_nonresponse`: fraction whose confidence fails to respond
  to the natural contrastive reversal;
- `intervention_disagreement`: dispersion of per-agent intervention responses.

Original agreement and confidence are baselines, not inputs to the primary
intervention-only score.

On Qwen development only, nonnegative weights are chosen from the 0.1-spaced
simplex. Selection uses pair-grouped cross-validation and maximizes, in order:

1. worst-label mean held-out AUROC;
2. macro-label mean held-out AUROC;
3. overall held-out Risk@80 error reduction;
4. lexicographically smallest weight tuple.

The selected weights are frozen in a new version before Ling calls.

## 9. Ling pilot go/no-go

The Qwen-selected score advances to formal registration only if the untouched
Ling development pilot satisfies all of:

- final valid-response rate at least 0.98;
- first-pass valid-response rate at least 0.95;
- at least 20 high-consensus items and at least 4 errors per native label;
- overall AUROC above 0.55;
- macro-label AUROC above 0.55;
- worst-label AUROC above 0.50; and
- overall Risk@80 error reduction nonnegative.

These are development qualification gates, not paper results. Failure stops
V3.16; it does not authorize trying formal items or changing the score on Ling
outcomes.

## 10. Proposed formal endpoints

If the pilot qualifies, a new formal protocol will freeze weights, parsers,
model fingerprints, thresholds, bootstrap seed, and manifest hashes. The
formal primary report will include, for Qwen and Ling separately:

- high-consensus error AUROC;
- native-label macro AUROC and worst-label AUROC;
- Risk@80 retained error and absolute reduction;
- same-prediction/same-coverage comparisons with agreement, confidence,
  self-consistency, citation-only, and deterministic random baselines; and
- pair/page-cluster bootstrap intervals.

The cross-family headline will require both models to pass the registered
macro-label and Risk@80 gates. Aggregate pooling cannot rescue a failed model
or label group.

## 11. Claim boundary

V3.16 tests a label-symmetric, Wikipedia-revision fact-verification regime. It
does not establish universal factuality, arbitrary-domain transfer, independent
agents, investment performance, or cross-model repair. A positive result would
support cross-family selective error detection under natural contrastive
evidence. Repair remains a separate experiment.
