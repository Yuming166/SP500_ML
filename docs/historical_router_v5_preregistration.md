# Historical Router V5: adaptive mechanism-invariant router

**Protocol version:** `historical-router-v5-2026-09-02`
**Frozen:** 2026-09-02, before fitting V5 or inspecting any V5 output
**Status:** post-V4 retrospective development experiment; not confirmatory

## 1. Motivation and hypothesis

V4 showed that the transferred intervention signature improved global selective-risk
ordering, but its fixed source anchor did not transfer reliably at the calibrated
80% operating point. Its soft stress penalty also produced no numerical separation
from the no-stress ablation. V5 therefore tests a narrower and more falsifiable
hypothesis:

> A source-domain intervention ranker can help target-domain selective routing when
> its influence is attenuated monotonically under source-target feature shift, while
> target ranking and risk calibration are learned by separate heads.

This hypothesis was designed after observing V4 and is not a fresh confirmatory
claim on the reused market period. V4 and all earlier artifacts remain unchanged.

## 2. Frozen data and temporal contract

V5 inherits V4 without modification:

- source records: `results/pilot_llm_v12_1/formal/records.jsonl`;
- source subset: complete high-consensus (`agreement >= 0.8`) V12.1 questions;
- market as-of builder, 11 agents, seven provenance roots, and five-day target;
- six expanding outer folds: 504 train, five-row label gap, 126 test, step 126;
- inner agent OOF: 252 train, five-row gap, 63 test, step 63;
- matured-label-only rolling quality features and outcome-free routing features.

Within each outer train, the earliest 70% of OOF timestamps fit the ranking heads.
The latest 30% remains temporally ordered and is split in half: the earlier half
fits the monotone calibration head and the later half fixes the 80% conformal
threshold. No outer-test label is available to any fit, gate, or threshold.

## 3. Source and target ranking heads

Both heads predict ordering of error risk rather than calibrated probabilities.
Within each declared robustness group, every error/correct pair contributes the
logistic ranking loss

```text
log(1 + exp(-(score_error - score_correct))).
```

At most 4,096 pairs per group are retained by deterministic sampling with seed
`20260903`. Group losses use the V4 smooth worst-group aggregation with
`tau = 0.10`; all non-intercept coefficients receive `0.01` L2 regularization.

The source head uses the three common features:

- `intervention_inertia`;
- `flip_inertia`;
- `source_concentration`.

Its coefficients are non-negative. The target head uses all seven V4 risk
features and two method offsets. All seven risk coefficients are non-negative.
Unlike V4, there is no coefficient-distance anchor between domains.

## 4. Hard graph-intervention ordering

For the full V5 target head, each of the three common mechanism coefficients has
the hard lower bound `0.25`. Therefore increasing any one common coordinate by
the frozen intervention increment `0.20`, holding all else fixed and before
clipping, must increase its target logit by at least `0.05`. This is an exact
optimization constraint, replacing V4's ineffective soft probability-margin
penalty. The no-hard-constraint ablation lowers these three bounds to zero.

The constraint encodes the direction of the declared mechanism, not its empirical
success. V5 will report coefficient bounds, violations, and the performance cost
of the constraint.

## 5. Shift-monotone source gate

Source means and standard deviations of the common features are fixed from V12.1.
For target candidate `x`, define

```text
d(x) = mean(((x_common - mean_source) / max(sd_source, 0.05))^2)
shift(x) = d(x) / (1 + d(x))
gate(x) = sigmoid(a - b * shift(x)),  b >= 0.
```

The source and target logits are each standardized using outcome-free score moments
from their own ranking-fit data, with each score standard deviation floored at
`0.05`. The routed ranking score is

```text
score(x) = gate(x) * source_score(x)
         + (1 - gate(x)) * target_score(x).
```

`a` and `b` are learned only on ranking-fit target pairs, with bounds
`a in [-6, 6]`, `b in [0, 10]` and L2 penalty `0.001`. Thus greater measured
source-target shift can never increase source reliance. A learned near-zero gate
is an admissible negative-transfer diagnosis rather than a fitting failure.

## 6. Separate monotone calibration head

Candidate selection uses the ranking score only. The first half of the ordered
calibration window fits a Platt head

```text
p_error = sigmoid(c0 + c1 * score),  c1 >= 0,
```

using labels already available inside the outer-training window. The second half
sets the finite-sample 80% score quantile. This separates method/case ordering from
probability and threshold calibration. Calibration Brier score and ten-bin ECE are
reported in addition to the inherited routing metrics.

## 7. Frozen comparisons and ablations

Report on every outer-test timestamp:

- `majority`, `confidence`, `recent_performance`, and `provenance`;
- full `amir_router` (Adaptive Mechanism-Invariant Router);
- `amir_target_only` (`gate = 0`);
- `amir_fixed_gate` (`gate = 0.5`);
- `amir_no_hard_constraint`;
- `amir_no_calibration` (identity sigmoid instead of the Platt head).

All candidate ties use V4's fixed order: provenance, recent performance, majority.
Every policy calibrates its threshold only from its own training-window scores.

## 8. Frozen endpoints

Primary exploratory endpoint:

```text
Delta_AURC = AURC(amir_router) - AURC(confidence).
```

Report a 95% paired moving-block bootstrap interval with 21-row circular blocks,
1,000 replicates, seed `20260903`. The directional target is an upper bound below
zero. AURC is primary because V4's strongest signal was global risk ordering and
because it avoids selecting one favorable post-hoc coverage point.

Secondary endpoints are routed error at the independently calibrated 80% target,
coverage, false rejection, risk Brier/ECE, selected action Brier, worst VIX-regime
error, and the inherited non-overlapping return diagnostics. Also report the point
difference from frozen V4 CPR AURC; it is descriptive because V4 influenced V5.

## 9. Audits and claim boundary

Report per-fold ranking coefficients, gate parameters and mean gate, correlation
between shift and gate, calibration parameters, thresholds, pair counts, and all
hard-constraint violations. Report every frozen ablation even if unfavorable.

No V5 result on this reused period may be called confirmatory, prospective, market
alpha, causal market impact, or cross-model generalization. An ACL-level claim
still requires frozen Qwen-to-Llama evaluation on TQA/BoolQ and preferably a later
or separately frozen financial window. V5's candidate novelty is the combined
contract of shift-monotone source transfer, mechanism-constrained ranking, and
temporally separated calibration—not any component in isolation.
