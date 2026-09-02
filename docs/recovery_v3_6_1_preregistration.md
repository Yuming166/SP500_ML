# Recovery V3.6.1 preregistration: proof-carrying Atomic-PACE

Protocol version:
`recovery-v3.6.1-atomic-pace-ex-fever-development-2026-09-02`

Status: **frozen before full development certificate collection/fitting and
before any formal EX-FEVER test call**.

## Prior and development boundary

V3.6.1 leaves V1--V3.6 unchanged. V3.5.2 failed its development unlock: scalar
certificates usually selected the annotated root but overlooked isolated false
attributes in long conjunctions. V3.6 introduced atomic certificates, but its
fixed first-40 pilot stopped because the inherited 160-token completion limit
truncated 59/80 JSON responses. V3.6.1 changes only atomic certificate
`max_tokens` to 512; ordinary action calls remain at 160.

The first 40 development examples have been exposed for prompt development. At
512 tokens, all 80 certificates parsed first-pass. The direct proof gate made
ten routes, selected the annotated root nine times, repaired eight errors, and
caused zero harms. Among Refuted examples, 26/28 candidate certificates derived
a grounded contradiction. These figures are development diagnostics only. The
accepted prompt, parser, and output budget are frozen without further changes.
No formal item has been called.

## Frozen data and selection

The official EX-FEVER development CSV and Wikipedia database have SHA-256
values
`75eb05d4b9a7f1d16a672312cd9c5203f5e603e2effda173a9a32ba58b09db34`
and
`8f23ace9b7242bc94fcfdf31607d0038ae7085ead09c7358a0339a9e35ec0940`.
Eligibility, sentence extraction, anchors, retrieval-matched distractors, and
candidate ordering are unchanged from V3.5.2.

The V3.6.1 selection manifest has SHA-256
`55a040788544866ad993d5bd250a7cafb6cb6f4a850e01999a6636fc842eb171`.
Development has 600 examples, 300 per native label. The prospective test has
500 examples, 250 per label. Development/test claims and page roots are
disjoint; all 1,000 annotated test roots are unique across test items and do
not overlap CLIMATE-FEVER roots. Candidate position is exactly balanced.

Raw retrieval score has oriented annotated-role AUROC 0.823 on development and
0.888 on test. Retrieval score, title overlap, root/source identity, annotation
role, label, gold, and all action outcomes remain forbidden router features.

## Proof-carrying action certificate

For each candidate root separately, Qwen receives only the claim, anchor
packet, and candidate packet. It must return 2--8 material atomic checks, each
containing an exact claim span, a status in `supported`, `contradicted`, or
`unresolved`, and packet-local evidence IDs; it also returns
`coverage_complete` and confidence. The prompt highlights entity, nationality,
date/year, number, negation, occupation, organization/studio, and relation
direction mismatches.

A deterministic checker projects evidence IDs onto exact packet IDs, drops
duplicate IDs, drops nonexact/duplicate claim spans, and never fuzzy-remaps.
It derives the relation rather than accepting a model verdict:

```text
one grounded contradicted atom                         -> refutes
complete coverage, at least two atoms, all supported  -> supports
otherwise                                              -> insufficient
```

The proposed root itself must supply evidence for the derived relation. This
creates a locally auditable proof obligation for every eligible route. Atomic
certificates use 512 completion tokens; baseline/recovery calls use 160.

## Paired action learner and unlock

All 600 development examples receive two atomic certificate calls. Their
already executed baseline and three recovery interventions are regenerated
from content-addressed caches under the V3.6.1 record schema. For each
single-root action:

```text
fix  = 1[baseline wrong and action correct]
harm = 1[baseline correct and action wrong]
```

Features contain only anchor consensus/agreement/confidence summaries and
machine-checked certificate statistics. Separate fix and harm predictors each
average a standardized class-balanced logistic model and a class-balanced
random forest with 500 trees, depth 7, and minimum leaf 8. Seed: `20260985`.

Three stratified example-level folds produce out-of-fold predictions. Threshold
grids are unchanged:

```text
fix threshold:     0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50
harm cap:          0.05, 0.10, 0.15, 0.20, 0.30, 0.40
utility threshold: 0.00, 0.05, 0.10, 0.15, 0.20
utility = p_fix - p_harm
```

Every held-out fold must have at least five routes, five net fixes, and five
repairs citing annotated-root evidence; damage among initially correct high-
consensus examples must be at most 5%; both native-label gains must be
nonnegative. Feasible policies are ranked lexicographically by worst-fold macro
gain, worst-fold net fixes, total net fixes, negative harms, negative routes,
then thresholds. No feasible policy means the formal test stays locked.

The frozen policy considers only anchor agreement at least 0.8, requires a
new-root-grounded atomic relation opposite the consensus, applies all learned
fix/harm/utility thresholds, and acquires at most one root.

## Implementation and execution lock

The implementation SHA-256 is:

- `recovery_v3_6_1.py`:
  `3a477b173b51adb0c442ef3c9d864501491f32b5054741aa3c6871fcc15cd947`;
- `pilot_llm_v1.py`:
  `e1976878d8f6df12a00e384121ec86604999a81e541f8920c61233d214c3b3c5`;
- `recovery_v2.py`:
  `3a458b5a9f03326ab989cf74239d1104fee3df0cdcadbf45c7d7c9b930202c35`;
- `recovery_v3.py`:
  `cf12e563e282fa93fc4406a89459f6b4bc5db271e5bcb265e2d19b20541b7804`;
- `recovery_v3_4.py`:
  `c0213b226f91881eb5083f1b4a062578eabaf5686a87416360ec489edfe6c61b`.

The endpoint is the verified `Qwen3.5-4B` service at
`http://10.63.0.82:31518/v1/chat/completions`. Certificate seeds are 20261031
and 20261041; baseline/action seeds are 20260991, 20261001, 20261011, and
20261021. Calls use temperature zero and one fixed repair attempt.

After development unlock, the selected policy, models, feature schema, code and
data hashes, preregistration, and development records are hashed into the router
manifest. A two-item formal smoke checks only transport/schema. Full routes and
probabilities are serialized before any gold-derived outcome is constructed.

## Frozen formal analyses and gates

Primary is Atomic-PACE versus KEEP. Prespecified ablations are the same learned
policy without proof gating and proof-only routing. Retrieval, deterministic
hash-random, candidate-0, candidate-1, and both actions are reported unlimited
and truncated to the realized Atomic-PACE root budget. The available-action
oracle is diagnostic.

The endpoint is native-label-macro accuracy gain. A label-stratified paired
bootstrap uses 10,000 replicates and seed `20260986`. Atomic-PACE passes only if:

1. the macro-gain 95% interval lower bound is above zero;
2. high-consensus-correct damage is at most 5%;
3. net gain is nonnegative in both native-label groups;
4. at least ten repairs cite acquired annotated-root evidence; and
5. net fixes exceed KEEP and every root-budget-matched nonlearned comparator.

Any failed gate is a retained negative result. Test outcomes cannot change the
sample, proof checker, features, models, thresholds, or gates.

## Claim and novelty boundary

The proposed contribution is a proof-carrying consensus state transition: a
candidate action must produce a machine-checked, provenance-local atomic witness
and pass learned fix and harm estimates from paired interventions. The novelty
claim is this composition, not atomic fact checking, evidence retrieval, or
selective prediction individually.

A pass supports prospective page-root-disjoint repair for one Qwen deployment
on EX-FEVER. It does not establish cross-model transfer, publisher independence,
live-retrieval robustness, or a formal safety guarantee.
