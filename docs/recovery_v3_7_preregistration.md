# Recovery V3.7 preregistration: entailment-ledger action routing

Protocol version: `recovery-v3.7-elar-fever-train-2026-09-02`

Status: **development-smoke abort; no V3.7 formal model call occurred**.

Four proof-eligible development candidates were called after this document was
written. All four parsed, but a Refuted example showed that the hostile-review
instruction ambiguously treated the mismatch that proves contradiction as a
challenge to the proof. No threshold fitting or formal call occurred. V3.7.1
retains the data, parser, grids, gates, and method, and makes only that prompt
distinction explicit.

## 1. Amendment boundary and hypothesis

V3.6.2 remains immutable. Its primary Atomic-PACE policy achieved a significant
+8.00 percentage-point macro gain, 33 fixes, one harm, and four of five gates.
Its proof-only ablation achieved +11.75 points and numerical passes on all five
analogous gates. The failed Refuted gate came from a structurally valid atomic
certificate that called `Tolkien's fantasy.` evidence that Tolkien was an
American writer. This exposes a precise gap: local evidence IDs and exact claim
spans do not by themselves establish semantic entailment.

V3.7 tests whether an **entailment ledger with an adversarial challenge** can
retain proof-only coverage while removing semantically ungrounded transitions.
All 1,000 previously evaluated EX-FEVER items are now explicitly development
data. No V3.7 formal response or outcome may select the ledger prompt, parser,
thresholds, sample, or gates.

## 2. Formal data and controlled false-consensus construction

The formal universe is `data/fever/train.jsonl`, SHA-256
`bfa7b19109af675e5ce914dd96779b6382ca5c8c60d32c38e26f0072dd32f1c1`,
downloaded from the public `copenlu/fever_gold_evidence` train split. Eligible
rows are verifiable `SUPPORTS` or `REFUTES` claims with exactly one annotated
Wikipedia page root and nonempty gold evidence.

Every normalized claim and page root exposed in Pilot-LLM V5--V9, Recovery
V2.2, or Recovery V3.6.2 is excluded before retrieval. Page roots are
protocol-hash partitioned into target and retrieval pools. For each target:

- the initial anchor is an unannotated retrieval root;
- one candidate is the annotated gold-evidence root;
- the other candidate is a different unannotated retrieval root;
- the two retrieval roots are chosen by label-blind HashingVectorizer cosine
  matching within 0.08 of the gold evidence score;
- candidate order is exactly balanced and reveals no annotation role.

The formal selection has 400 examples, 200 per native label. Each target,
anchor, and candidate page root is used once, yielding 1,200 globally distinct
formal roots. There is zero claim/root overlap with development. The oriented
candidate-role AUROC of raw retrieval score must be at most 0.65. Retrieval
score, root identity, annotation role, native label, gold answer, and action
outcomes are forbidden router inputs.

This is a controlled false-consensus environment: the anchor is intentionally
an evidence-retrieval path rather than the annotated answer path. It evaluates
whether a router can recognize and acquire a semantically corrective path; it
does not estimate naturally occurring web-search prevalence.

## 3. Frozen actions and atomic certificates

The five-persona anchor-only consensus and complete paired actions are
unchanged: `KEEP`, `candidate_0`, `candidate_1`, and `both`. Actions execute for
all formal examples after the router is frozen. The model is `Qwen3.5-4B` at
`http://10.63.0.82:31518/v1/chat/completions`, with temperature zero, frozen
seeds, and one fixed schema-repair attempt.

Each single-root candidate receives the V3.6.2 atomic proof obligation. A
candidate is considered further only when it produces a new-root-grounded
relation opposite the high anchor consensus. Certificate failures are KEEP.

## 4. ELAR: an entailment ledger plus hostile challenge

For each proof-eligible action only, Qwen receives the claim, packet, and the
decisive atoms already produced by the atomic certificate. It must emit one
ledger entry per decisive atom with:

- the exact atom index;
- a certificate-local evidence ID;
- the shortest exact evidence quote;
- `entailed`, `contradicted`, or `insufficient`;
- confidence; and
- any unsupported terms copied from the atom.

The same structured call must then act as a hostile reviewer and report either
no problem or one of: unsupported attribute, entity mismatch, numeric mismatch,
negation mismatch, relation-direction mismatch, or insufficient context.

A deterministic checker rejects nonexact quotes, nonlocal IDs, missing or
duplicate decisive atoms, schema drift, and invalid confidence. The router
requires the expected semantic direction, no challenge, a minimum confidence,
a minimum atom-to-quote token coverage, and a cap on unsupported terms. Any
failure deterministically maps to KEEP.

The prior V3.6.2 `p_fix` and `p_harm` models remain unchanged. They do not veto
a ledger-valid action. They are used only as the third tie-break when both
candidates pass, after ledger confidence and quote coverage. This preserves
the paired-intervention router while removing the V3.6.2 hard gate that blocked
many proof-supported repairs.

## 5. Development-only selection and execution lock

The 600-example V3.6.1 development split and the 400-example V3.6.2 formal
split form two separately binding development folds. Only proof-eligible
candidate actions receive ledger calls. The only grid is:

```text
minimum ledger confidence: 0.50, 0.60, 0.70, 0.80, 0.90
minimum lexical coverage:  0.00, 0.10, 0.20, 0.30, 0.40, 0.50
unsupported-term cap:      0, 1, 2
```

A setting is feasible only if each source fold independently has damage at
most 5%, nonnegative gain in both native labels, at least five routes, at least
five net fixes, and at least five annotation-supported repairs. Feasible
settings lexicographically maximize worst-fold macro gain, worst-fold net
fixes, total net fixes, fewer harms, fewer routes, then stricter thresholds. If
none is feasible, no formal call is allowed.

After development unlock, code/dependency hashes, data/record hashes, selected
thresholds, prior-router hash, and feature boundary are serialized before any
formal model call. A two-example smoke checks transport and schemas only. Full
formal routes and probabilities are written before gold-derived outcomes are
constructed.

## 6. Frozen formal analyses and gates

Primary evaluation is ELAR versus KEEP. Native-label-macro gain uses 10,000
label-stratified paired bootstrap replicates and seed 20261102. Required
baselines are atomic proof-only, retrieval-score, hash-random, candidate 0,
candidate 1, and both; nonlearned baselines are reported unlimited and capped
to ELAR's realized root budget. The available-action oracle is diagnostic.

ELAR passes only if all hold:

1. the 95% macro-gain interval lower bound is above zero;
2. damage among initially correct high-consensus items is at most 5%;
3. net gain is nonnegative in both native-label groups;
4. at least ten repairs cite the acquired annotated root; and
5. net fixes exceed KEEP and every root-budget-matched nonlearned comparator.

Any failure is retained as a negative result. Formal outcomes cannot revise
this protocol or be reused to promote an ablation to primary.

## 7. Novelty and claim boundary

Exact attribution checking, LLM fact verification, minimal evidence groups,
and selective prediction already exist; none is claimed as new in isolation.
The proposed contribution is the action-conditioned composition: complete
paired evidence interventions train the benefit/harm state, an atomic
provenance witness proposes a counter-consensus transition, and an exact-quote
ledger with an explicit hostile challenge authorizes or rejects the transition
before its outcome exists. Related boundaries are documented by prior work on
[automatic attribution evaluation](https://aclanthology.org/2023.findings-emnlp.307/),
[LLMs as fact verifiers](https://aclanthology.org/2024.naacl-long.62/), and
[minimal evidence groups](https://aclanthology.org/2025.trustnlp-main.8/).

A pass supports prospective, page-root-disjoint corrective evidence routing
for one Qwen deployment in a static Wikipedia environment. It does not prove
publisher independence, cross-model transfer, live-search robustness, or a
formal semantic safety guarantee.
