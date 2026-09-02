# Recovery V3.2 preregistration: CAPE provenance-action routing

Protocol version: `recovery-v3.2-cape-averitec-2026-09-02`

Status: **design frozen before any V3 Qwen outcome call**.

Implementation clarification logged on 2026-09-02 while the model-train action
matrix was still being collected, before its outcomes were audited and before
any policy-selection, calibration, or prospective-test call: the 95% interval
uses linear empirical 2.5%/97.5% quantiles; a retrieval-score baseline spends
its matched budget in descending retrieval-score order with a salted-hash tie
break, while fixed and random baselines use only salted-hash order. This closes
two implementation ambiguities without changing a grid, gate, split, candidate,
prompt, model response, or test decision.

The first router-fit attempt later aborted before serialization because the
installed scikit-learn version rejects direct three-class use of the
`liblinear` solver. The intended one-versus-rest logistic encoder was made
explicit with `OneVsRestClassifier`; no model prediction, threshold, router
artifact, or prospective-test call existed when this compatibility fix was
made.

V3.0 ended in a pre-formal loader audit before a selection manifest or Qwen
call existed. The first feasibility count treated one empty answer as usable;
the frozen strict loader correctly rejected it, leaving 365 rather than 366
eligible Supported development items. V3.1 changes only the expected count and
the model-train Supported allocation. No model response or action outcome
informed this amendment.

V3.1 then created a selection manifest and failed its pre-call claim-overlap
gate. The official files contain nine duplicate eligible claims within train
and one eligible train claim repeated in dev. V3.2 adds only deterministic
claim-text deduplication (lowest source-row index retained) and removes any
development-pool claim present in the prospective test. V3.1 artifacts are
retained under `results/recovery_v3_1/`; it made zero Qwen calls.

## 1. Motivation and immutable boundary

Recovery V2.2 is complete and remains unchanged. Its test outcomes are now
exposed and may motivate V3, but no V2.2 example may be used as a formal V3
test item. V3 uses the public AVeriTeC train split for training, policy
selection, and safety calibration, and the separate public AVeriTeC dev split
as the untouched prospective test set.

V2.2 established a small, statistically positive low-harm routing effect, but
its shallow router under-routed and failed to beat an unlimited fixed two-root
action. V3 changes the scientific question from unconstrained accuracy to
safe, budget-aware provenance acquisition while retaining the original
unlimited comparison as a strict secondary result.

## 2. Contribution hypothesis

V3 evaluates **CAPE-Router: Counterfactual Action Policy with Evidence
provenance**. CAPE has four inseparable components:

1. complete paired potential outcomes for KEEP, candidate 0, candidate 1, and
   both on every development example;
2. an out-of-fold three-way evidence-stance encoder trained only on annotated
   development roots (`supports`, `refutes`, `irrelevant`);
3. action-conditioned benefit and harm models that combine consensus,
   retrieval, provenance, and stance features; and
4. a held-out safety shield that can only make the tuned policy more
   conservative before the prospective test.

This is full-information offline policy learning, not PPO over an opaque reward
and not ordinary model routing. Complete interventions make propensity
correction unnecessary and expose negative treatment effects directly.

## 3. Relation to current work

- Decision-Aware Memory Cards scores and packs context units using action
  shift, uplift, and negative-transfer risk. CAPE instead targets recovery from
  false multi-agent consensus, learns among provenance-root actions from a
  complete potential-outcome table, and freezes a harm shield.
- AgentAuditor searches reasoning-tree disagreement and trains an
  anti-consensus adjudicator. CAPE does not consume hidden reasoning traces;
  it routes external evidence roots using short structured decisions.
- Conformal LLM routing controls failure when selecting a cheap versus costly
  model. CAPE controls damage from corrective evidence actions and evaluates
  source-budget Pareto performance.
- Doubly robust policy learning is useful with observational logged actions.
  CAPE deliberately queries every action during development, so direct paired
  effects are identifiable without an estimated behavior propensity.

## 4. Frozen data

Source files:

```text
data/averitec/train.parquet 41d08f99b3d3afbdbb81a655ccee23a4cddd3b4af4480100391e305300ee784f
data/averitec/dev.parquet   18e9397649d12f0a9b2e21b553c7b32e10b5cf58aff54c2a11e1ad6243af403d
```

An eligible item must:

- have native label `Supported` or `Refuted`;
- have at least two distinct usable evidence source domains after canonical
  URL normalization;
- have non-empty claim, question, and answer text; and
- exclude evidence whose source domain equals the item's fact-checking-article
  domain.

After the frozen deduplication rule, the expected universe is 1,163
development-pool items from AVeriTeC
train and 236 prospective-test items from AVeriTeC dev. AVeriTeC train is
stratified by native label and deterministic protocol hash into:

- model train: 70%, expected 814;
- policy selection: 15%, expected 174;
- safety calibration: 15%, expected 175.

The official AVeriTeC dev split is not subdivided or balanced using outcomes.
Native labels may define strata and evaluation groups, but are forbidden from
prompts and inference-time router features.

## 5. Provenance roots and candidates

A provenance root is the canonical registrable source domain after unwrapping
Wayback URLs and removing `www`. This is closer to publisher provenance than
V2's Wikipedia-page roots, but a domain is not proof of editorial independence.

For each item:

- one annotated source domain is selected as the anchor by salted hash;
- one different annotated domain is selected as the held-out corrective root;
- one hard retrieval distractor is selected without labels or outcomes from a
  different claim in the same partition;
- the corrective root and distractor are hash-randomized into candidate 0 and
  candidate 1; and
- root identities are replaced by packet-local aliases in Qwen prompts.

The retrieval candidate is not asserted false. Its formal role is
`unannotated_retrieval_candidate`; it may incidentally help.

Structural gates before calls:

- exact expected split counts;
- zero claim-text overlap across partitions;
- anchor and candidates have three distinct domains;
- no prompt contains the native label, annotation role, fact-checking article,
  justification, speaker, or reporting source;
- candidate 0 contains the annotated root in [45%, 55%] of each partition;
- each label is at least 25% of each partition;
- retrieval score cannot identify annotation role with orientation-free AUROC
  above 0.85; and
- prospective test contains at least 200 items and 300 distinct evidence
  domains.

Failure requires a new version before any model call.

## 6. Frozen Qwen action matrix

Model: `Qwen3.5-4B` at the documented relocated endpoint
`http://10.63.0.82:31518/v1/chat/completions`.

Five fixed personas see only the anchor and produce structured yes/no,
confidence, and packet-local citations. Majority vote defines KEEP. Recovery
is considered only when agreement is at least 0.8.

Every item receives all three recovery actions regardless of initial outcome:

- candidate 0: anchor plus candidate 0, one added root;
- candidate 1: anchor plus candidate 1, one added root;
- both: anchor plus both candidates, two added roots.

Temperature is 0, seeds and retry rules are fixed in code, and the bounded
V2.2 single-confidence-quote normalization is retained. Labels, correctness,
annotation roles, and action outcomes never enter prompts.

## 7. CAPE training

The stance encoder is a class-balanced word-and-character TF-IDF logistic
model. Model-train stance probabilities must be generated out of fold by five
fixed stratified folds; the final encoder is then fitted on all model-train
packets for later partitions. It receives claim and packet text, not source
identity or native label at inference.

For each non-KEEP action, CAPE predicts:

```text
gain(q,a) = correct(q,a) - correct(q,KEEP)
harm(q,a) = 1[correct(q,KEEP)=1 and correct(q,a)=0]
```

The frozen predictor is an equal-weight ensemble of a random forest and
histogram gradient booster for gain, plus the analogous pair for harm. Features
include initial vote statistics, confidence dispersion, citation counts,
packet lengths, lexical retrieval scores, action cost, and stance probabilities
and their agreement/opposition to the initial consensus. Gold labels,
correctness, annotation roles, raw source names, split names, and outcome
aliases are forbidden.

## 8. Policy selection and safety shield

On the policy-selection partition, a fixed grid searches separate gain
thresholds for initial yes and no consensuses and one harm cap. The objective is
net accuracy gain minus 0.01 per added root. Feasible policies must have damage
at most 5%, non-negative net gain in both native labels, and at least ten
non-KEEP routes. Ties prefer lower damage, fewer roots, and higher thresholds.

The safety-calibration partition is then evaluated once. A predetermined
monotone offset sequence `[0, .025, .05, .10, .15, .20, .30]` may only raise
both gain thresholds. The first offset satisfying the same damage and
label-group constraints is frozen. If none qualifies, CAPE falls back to KEEP.
No parameter is selected from the prospective test.

The stance encoder, gain/harm models, selected thresholds, hashes, and feature
schema must be serialized before any AVeriTeC-dev Qwen call.

At evaluation time, test features and all non-oracle route choices are computed
from the five baseline decisions and frozen evidence packets only. They are
written to a hashed `preoutcome_routes.json` before the evaluator constructs
gold-derived action outcomes, oracle choices, or policy metrics.

## 9. Evaluation and success gates

Primary evaluation uses 2,000 native-label-stratified, question-level paired
bootstrap replicates with seed 20260942 and the linear 2.5%/97.5% empirical
quantiles. The main CAPE policy must pass all:

1. macro-average label gain has a 95% paired-bootstrap lower bound above zero;
2. overall damage among initially correct high-consensus cases is at most 5%;
3. net gain is non-negative in both native-label groups;
4. at least ten repairs cite evidence from the acquired annotated domain; and
5. at CAPE's realized root budget, net fixes exceed KEEP, retrieval-score
   routing, hash-random acquisition, and each fixed-action baseline truncated
   to the identical root budget without using outcomes.

Secondary results retain the V2 strict checks, including raw net fixes versus
unlimited fixed-both, as well as unrestricted CAPE, no-stance ablation,
no-safety-shield ablation, oracle action selection, cost curves at 0.25, 0.5,
1.0, and 2.0 roots/item, calibration, and publisher-seen/unseen subgroups.

## 10. Claim boundary

A pass would support safe, budget-aware recovery on a new real-world
fact-checking dataset using source-domain provenance. It would not establish
publisher independence, cross-model transfer, online search robustness, or a
distribution-free per-item safety guarantee. All failures and ablations remain
reportable; V3 thresholds may not be retuned after prospective-test outcomes.
