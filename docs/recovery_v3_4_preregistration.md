# Recovery V3.4 preregistration: publisher-blocked witness jury

Protocol version: `recovery-v3.4-pbwj-climate-fever-2026-09-02`

Status: **frozen before any formal CLIMATE-FEVER Qwen call**.

## Amendment boundary

V3.3 and its AVeriTeC-dev result remain immutable. V3.3 passed four of five
primary gates: its macro-label gain was positive, but the paired-bootstrap 95%
lower bound was below zero. Its test is now exposed and is used only as source
development evidence for V3.4. No V3.4 claim will be made from re-filtering that
test.

V3.4 is a new, prospective, cross-dataset test. The router is trained only on
the already exposed AVeriTeC evidence packets. It is evaluated once on the
previously unused public CLIMATE-FEVER data. External outcomes cannot alter the
router, selection rule, thresholds, sample, gates, or analysis.

## External selection

The input is the official public `climate-fever.jsonl` with SHA-256
`8a4b9032d861be482ffb49dddfd283ffa6089e654f1e968040011882c5eb6e0b`.
An item is eligible exactly when:

- its claim label is `SUPPORTS` or `REFUTES`;
- at least two distinct Wikipedia article titles contain evidence whose
  evidence label equals the claim label; and
- the claim and matching evidence text are nonempty.

All 483 eligible items are retained: 361 Supported and 122 Refuted. One matching
article is the anchor and a second is the hidden annotated root. A retrieval-
score-matched article packet from another claim is the distractor. Candidate
position is deterministically balanced. Selection must pass all frozen audits:
exact counts, unique claims, zero exact normalized-claim overlap with the local
AVeriTeC and FEVER material, three distinct packet roots, candidate-0 annotated
fraction in `[0.49, 0.51]`, oriented candidate-role retrieval AUC at most 0.85,
at least 100 examples per label, at least 450 total examples, and at least 400
distinct page roots.

CLIMATE-FEVER roots are Wikipedia article pages, not independent publishers.
This experiment therefore tests transfer across datasets and evidence-page
roots; it does not establish publisher independence.

## Method: PBWJ

Publisher-Blocked Witness Jury (PBWJ) extends CEW without changing any V3.3
artifact. Five word/character TF-IDF logistic stance witnesses are trained on
the union of the exposed AVeriTeC partitions. Source claims are grouped by their
fact-check publisher, publisher groups are greedily balanced into five folds,
and jury member `k` is trained while leaving fold `k` out. Every member sees the
retrieval-matched irrelevant packet as a hard negative.

At inference, members receive only the claim and packet text. For initial
consensus `no`, the counter-consensus stance is `supports`; for consensus `yes`,
it is `refutes`. Each member votes for the candidate with greater
counter-consensus probability. For the modal candidate, define:

```text
pessimistic_score = mean(member candidate probabilities) - sample_sd(...)
pessimistic_delta = mean(candidate - anchor) - sample_sd(candidate - anchor)
```

These are dispersion penalties, not frequentist confidence bounds and not
distribution-free guarantees. PBWJ acquires exactly one candidate root only if:

```text
initial five-agent agreement >= 0.8
jury candidate agreement >= 0.8  when consensus is no
jury candidate agreement == 1.0  when consensus is yes
pessimistic_score >= 0.4
pessimistic_delta >= -0.2
```

The score and delta thresholds are inherited unchanged from V3.3. The stricter
`yes -> refutes` quorum is a source-development safety amendment motivated by
the V3.3 error audit; it is fixed before external calls. The external native
label, gold answer, annotation role, action outcome, and source identity are
forbidden at route inference.

## Execution and analyses

The model is the verified `Qwen3.5-4B` service at
`http://10.63.0.82:31518/v1/chat/completions`. Each item receives five
anchor-only baseline calls and the three candidate-0, candidate-1, and both
counterfactual recovery calls. Seeds, messages, retries, cache behavior, and
parsing are frozen in code. A two-item smoke test checks transport and parsing
only and is not analyzed.

All policy routes and jury diagnostics are serialized and hashed before the
evaluation function constructs any gold-derived outcome. The primary policy is
PBWJ. Prespecified ablations remove the dispersion veto and all uncertainty
vetoes. Comparators are retrieval score, hash-random candidate, and each fixed
action, both unlimited and truncated to PBWJ's realized root budget. The
available-action oracle is diagnostic only.

## Frozen primary gates

PBWJ passes only if all five conditions hold on the complete 483-item external
test:

1. native-label-macro gain has a 10,000-replicate label-stratified paired-
   bootstrap 95% lower bound above zero;
2. damage among initially correct high-consensus examples is at most 5%;
3. net gain is nonnegative in both native-label groups;
4. at least ten repairs cite evidence from the acquired annotated root; and
5. net fixes exceed KEEP and every root-budget-matched nonlearned comparator.

The fixed bootstrap seed is `20260943`. Failure of any gate is reported as a
negative result; thresholds and samples will not be revised after inspection.

## Claim boundary

A pass would support prospective cross-dataset transfer of uncertainty-gated,
single-root repair for one Qwen deployment. It would not establish cross-model
transfer, live retrieval robustness, publisher independence, causal independence
between Wikipedia pages, or a per-query safety guarantee.
