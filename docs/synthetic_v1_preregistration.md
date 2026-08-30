# Synthetic V1 preregistration

## Scope

This document freezes the first formal evaluation of the controlled
false-consensus benchmark before results are generated. It evaluates
pre-outcome estimates of the probability that the consensus action is wrong
because its support is correlated, stale, or corrupted. It is not a financial
backtest and it does not establish real-world trading performance.

## Fixed generator protocol

- Base seeds: `101, 211, 307, 401, 509, 601, 709, 809, 907, 1009`.
- Mechanisms are generated independently with a deterministic
  mechanism/configuration-specific seed derived from each base seed. No row is
  copied from another mechanism.
- Clean controls: `independent_clean`, `shared_clean`.
- Corruption mechanisms: `shared_corruption`, `stale_evidence`,
  `partial_corruption`.
- Agent counts: `3, 5, 7, 9`.
- Source-quality noise standard deviations: `0.00, 0.10, 0.20, 0.35`.
- Corruption strengths: `0.40, 0.60, 0.80, 1.00`; strength is included only
  where a corruption mechanism uses a shared source.
- Provenance visibility is `aliased`; transformations use opaque names. The
  two curves that vary noise and agent count use the same fixed protocol, with
  the other axis pooled.

## Mechanism-held-out evaluation

Use leave-one-mechanism-out evaluation. For each corruption mechanism, the
test partition contains that mechanism plus both clean controls. The training
partition contains the other two corruption mechanisms plus both controls.
Configurations are identical across train and test, but every episode is
generated independently. No random row split is permitted.

## Frozen routing threshold

For every deployable method and held-out fold, calculate the **75th percentile**
of that method's pre-outcome risk scores on the training partition. Abstain on
a test episode only when `risk > threshold` (strict inequality makes the policy
deterministic under tied transparent scores). This is a coverage-targeting rule;
it does not inspect train or test outcomes. The selected thresholds are reported
verbatim. The non-deployable oracle instead uses its fixed probability threshold
of `0.50`, because it is defined as the post-outcome binary label itself.

## Methods

All methods use the same majority action. They differ only in their
pre-outcome risk score:

1. **Majority**: minority-vote fraction.
2. **Confidence**: one minus mean self-reported confidence.
3. **Agreement**: pairwise vote-disagreement rate.
4. **Recent performance**: one minus mean environment-provided, as-of
   source-quality estimate.
5. **Provenance**: the fixed source-concentration, source-quality, staleness,
   and temporal-validity score in `provenance_risk`.
6. **Oracle**: the post-outcome harmful-false-consensus label. It is a
   non-deployable diagnostic upper bound and is labelled separately in every
   result.

## Labels, metrics, and uncertainty

The primary label is `harmful_false_consensus`; outcomes are withheld from all
risk scores and threshold selection. AUROC, AUPRC, ECE (10 equal-width bins),
and Brier evaluate that binary risk label. AURC and risk at 80% coverage use
consensus-action error after retaining the least risky episodes. High-confidence
error is the routed consensus error at the frozen threshold. False rejection is
the rate of abstaining on correct consensus actions. We also report independent
correct-consensus rejection as a stricter control check.

All reported 95% confidence intervals use 1,000 deterministic nonparametric
bootstrap resamples of the ten base-seed clusters. Rows belonging to a base
seed are resampled together; confidence intervals are percentile intervals.

## Interpretation rule

Proceed to historical replay only if the deployable provenance method improves
high-confidence error or AURC over transparent baselines without a prohibitive
false-rejection cost, including on at least one held-out mechanism. The oracle
is excluded from this decision.
