# Recovery V3.5 preregistration: PACE action certificates

Protocol version: `recovery-v3.5-pace-ex-fever-2026-09-02`

Status: **frozen before full development fitting and before any formal EX-FEVER
test call**.

## Prior-result and development boundary

V3.5 does not modify V1--V3.4.1. In particular, the negative prospective
V3.4.1 result remains unchanged: its static witness router did not route any
initially wrong high-consensus case. V3.5 tests a new hypothesis: successful
repair is better represented as the conditional value and risk of a concrete
evidence-acquisition action than as a static prediction that consensus is
wrong.

A prespecified first-20 development pilot has been exposed. It was used only to
check parsing, source grounding, and whether the proposed action mechanism can
produce both fixes and harms. The accepted prompt was not amended: all 200
calls parsed first-pass, and the certificate-only gate produced six fixes and
one harm among ten actions. These are development diagnostics and cannot be
reported as prospective evidence. Pilot examples remain in the full
development set; no example is filtered by outcome.

## Frozen data and structural gates

The official EX-FEVER development CSV has SHA-256
`75eb05d4b9a7f1d16a672312cd9c5203f5e603e2effda173a9a32ba58b09db34`.
The accompanying Wikipedia SQLite database has SHA-256
`8f23ace9b7242bc94fcfdf31607d0038ae7085ead09c7358a0339a9e35ec0940`.
Eligibility requires a SUPPORT or REFUTE claim, exactly two distinct annotated
Wikipedia roots, nonempty documents for both roots, and a unique normalized
claim. Each root packet contains at most three database sentences selected by
token overlap with the official explanation. Explanation and label are never
router features.

The frozen selection manifest has SHA-256
`03dd81075430edca1a2d130a94c21b2e20230d823a9e822d00e81b1afb51c378`.
It contains 600 development examples (300/300 per native label) and 500 test
examples (250/250). Test examples come from singleton two-root components, so
their 1,000 annotated roots are unique across test items. The selection has
zero normalized-claim and page-root overlap between development and test, zero
test claim overlap with the prior AVeriTeC/CLIMATE-FEVER selections, and zero
test page-root overlap with CLIMATE-FEVER.

One annotated root is the anchor, the other is the hidden repair root, and a
retrieval-matched root from another claim is the distractor. Candidate position
is exactly balanced. Raw retrieval score has oriented role AUROC 0.823 on
development and 0.888 on test. Therefore retrieval score, title overlap, root
identity, source identity, annotation role, and all gold/outcome fields are
excluded from PACE. This diagnostic is retained to show that the primary router
cannot exploit the shortcut.

## Frozen method

PACE means Provenance-grounded Action Certificates for Error repair. For each
candidate separately, Qwen receives the claim, anchor packet, and proposed new
root. It returns exactly:

```text
relation in {supports, refutes, insufficient, conflicted}
support_strength, refute_strength, confidence in [0,1]
new_evidence_ids drawn only from the proposed root
missing_bridge as a boolean
```

The route features are the five-agent anchor consensus/agreement/confidence
summary and the semantic certificate fields. Feature construction must work
with only baseline action rows and therefore cannot inspect recovery decisions.
The implementation frozen here is
`src/sp500_forecastability/recovery_v3_5.py`, SHA-256
`9e90f4a9fd603d0ba053f479a10189a004211d73e7fbcf13d4592e9c841c39d3`.

On 600 development examples, paired interventions define for each single-root
action:

```text
fix  = 1[baseline wrong and action correct]
harm = 1[baseline correct and action wrong]
```

Separate fix and harm predictors average a standardized class-balanced logistic
model and a class-balanced random forest (500 trees, depth 7, minimum leaf 8).
The seed is `20260985`. Three stratified example-level folds generate out-of-
fold probabilities. Policy thresholds are selected only from:

```text
fix threshold:     0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50
harm cap:          0.05, 0.10, 0.15, 0.20, 0.30, 0.40
utility threshold: 0.00, 0.05, 0.10, 0.15, 0.20
utility = p_fix - p_harm
```

Every held-out development fold must have at least five routes, at least five
net fixes, at least five repairs that cite annotated-root evidence, no more than
5% damage among initially correct high-consensus cases, and nonnegative gain in
both native-label groups. Among feasible configurations, selection maximizes,
lexicographically: worst-fold macro gain, worst-fold net fixes, total net fixes,
negative total harms, negative total routes, and then the numeric thresholds.
If no configuration is feasible, the formal test remains locked.

The final policy considers only examples with at least 0.8 anchor agreement. A
candidate is eligible only when its certificate cites the new root, reports no
missing bridge, and its relation is opposite the consensus (`supports` for a
`no` consensus, `refutes` for `yes`). It must also satisfy all learned fix,
harm, and utility thresholds. At most one candidate root is acquired, chosen by
utility, fix probability, lower harm probability, then deterministic action
name. PACE never routes to the two-root action.

## Execution and prospective analysis

The verified endpoint is `Qwen3.5-4B` at
`http://10.63.0.82:31518/v1/chat/completions`. Certificate seeds are 20261031
and 20261041. Baseline and recovery seeds are 20260991, 20261001, 20261011, and
20261021. Calls are temperature zero with one schema-repair retry and immutable
content-addressed caches.

After development unlock, the full-data models, selected thresholds, selection
hash, development-record hashes, implementation hash, and preregistration hash
are frozen in a router manifest. A two-example test smoke checks transport and
parsing only. The complete 500-example test is then executed once. Test routes,
probabilities, and record hashes are serialized before outcome construction.

The primary comparison is KEEP. Prespecified ablations are PACE without the
certificate gate and certificate-only routing. Nonlearned comparisons are
retrieval score, deterministic hash-random candidate, and each fixed action,
reported both unlimited and truncated to PACE's realized root budget. The
available-action oracle is diagnostic only.

The primary endpoint is native-label-macro accuracy gain over KEEP. Its 95%
interval uses 10,000 label-stratified paired-bootstrap replicates with seed
`20260986`. PACE passes only if all gates hold:

1. the macro-gain interval lower bound is above zero;
2. damage among initially correct high-consensus examples is at most 5%;
3. net gain is nonnegative in both native-label groups;
4. at least ten repairs cite acquired annotated-root evidence; and
5. net fixes exceed KEEP and every root-budget-matched nonlearned comparator.

Any failed gate is a negative result. The sample, thresholds, router, or gates
will not be changed after formal outcomes are inspected.

## Claim and novelty boundary

PACE combines pre-action, source-grounded semantic certificates with paired
intervention learning of fix and harm. The intended distinction is that it
learns whether a particular acquisition action is safe and useful, not merely
whether evidence is relevant, whether consensus is unreliable, or whether an
evidence set entails the claim. Novelty will be stated as this specific design,
not as an unverified claim that no prior work has ever combined related ideas.

A pass supports prospective page-root-disjoint repair for one Qwen deployment
within EX-FEVER. It does not establish cross-model transfer, publisher
independence, live-retrieval robustness, or a distribution-free safety
guarantee.
