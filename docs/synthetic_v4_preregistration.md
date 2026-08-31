# Synthetic V4 preregistration: matched-coverage monotonic routing

## Status and separation from prior results

V4 is a new experiment defined after inspection of the frozen V3 report. It
does not alter V1--V3 seeds, scores, thresholds, artifacts, or primary claims.
The protocol below is frozen before formal V4 outputs are generated. The
separate V3 matched-coverage appendix is explicitly post-hoc and cannot be used
to relabel the original V3 primary hypothesis as passed.

## Motivation

V3 compared train-frozen threshold errors at unequal held-out coverage:
conditional provenance retained 80.9% of rows, whereas quality-only retained
78.8%. V4 therefore makes coverage-matched selective risk and AURC primary and
treats error at a learned deployment threshold as secondary.

V4 also tests whether a small, interpretable monotonic router can learn how to
combine provenance components using only past mechanisms and matured outcomes.
It is not allowed to inspect the held-out mechanism or its outcomes.

## Frozen data-generation protocol

- Training base-seed clusters: `3301, 3407, 3511, 3607, 3701, 3803, 3907,
  4001, 4111, 4201`.
- Each of the four outer held-out mechanisms has ten distinct test-seed
  clusters. No test seed appears in training or another outer fold.
- Agent counts remain 3, 5, 7, and 9.
- Source-quality noise remains 0.00, 0.10, 0.20, and 0.35.
- Corruption strengths remain 0.40, 0.60, 0.80, and 1.00.
- Visible source identifiers remain aliased and transformation names remain
  opaque.
- Confidence retains 15% coupling to source quality and independent noise.

To avoid making the paired-intervention feature a deterministic copy of the
label, V4 freezes the following behavior noise before outcomes are generated:

- clean-agent action error probability: 0.10;
- corrupted-agent action error probability: 0.85;
- evidence-inertia action error probability: 0.80;
- ordinary paired-intervention failure probability: 0.10;
- evidence-inertia paired-intervention failure probability: 0.80.

Consequently, a faithful agent may still be wrong, an unfaithful agent may be
correct, and the intervention audit is informative but imperfect.

## Outer and inner evaluation

The outer evaluation leaves out one complete harmful mechanism. The outer
training set contains both controls and the other three harmful mechanisms on
the training seeds. The outer test set contains both controls and only the
held-out mechanism on that fold's disjoint test seeds.

Within each outer training set, five-fold cross-fitting is grouped by base
seed. Cross-fitted predictions are used for calibration and threshold
selection. The final router is then refit on every outer-training row and
applied once to the untouched outer test rows.

## Frozen methods

V4 retains Majority, Confidence, Agreement, Recent performance, Quality only,
Source overlap only, Temporal only, the fixed V3 conditional-provenance score,
and the diagnostic Oracle.

The new **Monotonic provenance V4** router is a non-negative logistic model of
four pre-outcome features:

1. source concentration times observed source-quality risk;
2. stale-evidence fraction;
3. temporal-violation fraction;
4. paired-intervention failure fraction.

All four coefficients are constrained to be non-negative. L2 regularization is
fixed at 0.01. A monotonic Platt calibrator is fitted to grouped cross-fitted
training logits; its slope is constrained to be non-negative. There is no V4
hyperparameter search on outer-test outcomes.

For secondary deployment-threshold metrics, every non-oracle method selects a
threshold from training-only predictions. Candidate thresholds must retain
between 80% and 82% of training rows; the candidate with the lowest training
selective error is chosen, with false rejection and distance from 80% as
predeclared tie-breakers. If score ties make the band impossible, the closest
coverage to 80% is used. Oracle retains its diagnostic threshold of 0.50.

## Metrics and decision rules

Primary metrics are:

- macro-average AURC across outer held-out mechanisms;
- pooled consensus error at matched coverage 60%, 70%, 80%, and 90%, with
  Risk@80 as the headline operating point;
- paired base-seed-cluster bootstrap differences between Monotonic provenance
  V4 and each non-oracle ablation, using 1,000 replicates.

Secondary metrics are AUROC, AUPRC, ECE, Brier score, selective error at the
train-selected threshold, false rejection, test coverage, and mechanism-wise
results. Results must always display selective error beside achieved coverage.

The primary hypothesis is supported only if Monotonic provenance V4 has lower
macro AURC and lower pooled Risk@80 than Quality only, Confidence, and the fixed
V3 score. A paired 95% interval below zero is required for a confirmatory
superiority claim. Otherwise the result is reported as mixed or negative.

## Interpretation boundary

V4 is still a controlled rule-agent benchmark. The added noise improves the
separation between observable provenance signals and hidden outcomes, but it
does not establish LLM faithfulness or S&P 500 predictability. External
validity requires repeated paired remove/reverse calls to an actual LLM under
the same frozen evaluation rules.
