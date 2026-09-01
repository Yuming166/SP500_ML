# Historical Router V5: ACL assessment and next decision

**Assessed artifact:** `historical-router-v5-2026-09-02`
**Assessment status:** post-run interpretation; not preregistration

## Executive conclusion

V5 gives a useful but mixed result. AMIR has a favorable descriptive
risk--coverage operating point and calibrated risk estimates, but the frozen
primary AURC interval crosses zero. The current evidence supports presenting
finance as an external reliability case study; it does not support making the
S&P 500 result the central ACL claim.

## What improved

On 738 outer-test timestamps:

| Router | Coverage | Routed error | AURC | Risk Brier | Risk ECE |
|---|---:|---:|---:|---:|---:|
| confidence | 0.721 | 0.408 | 0.418 | 0.438 | 0.443 |
| provenance | 0.709 | 0.359 | 0.431 | 0.527 | 0.528 |
| V5 AMIR | 0.907 | 0.360 | 0.411 | 0.244 | 0.116 |

AMIR therefore descriptively dominates confidence at its independently
calibrated operating point: coverage is 18.6 percentage points higher and routed
error is 4.8 points lower. Relative to provenance, AMIR retains essentially the
same routed error (difference about 0.08 points) while covering 19.8 points more.
This Pareto-style reliability result is more informative than a favorable return
curve and is consistent with the paper's selective-agent framing.

The frozen primary difference was

```text
AURC(AMIR) - AURC(confidence) = -0.0078
95% paired moving-block CI = [-0.0607, 0.0459].
```

The point estimate favors the hypothesis, but the interval does not establish the
directional target. It must be reported as inconclusive.

The calibration head substantially improved the probability diagnostics relative
to the identity-sigmoid ablation: risk Brier decreased from `0.418` to `0.244` and
ten-bin ECE from `0.422` to `0.116`.

## What the ablations say

### Adaptive source transfer is not yet isolated

- Mean source gate on outer tests was `0.850`.
- Its shift slope was positive in only one of six folds and exactly zero in the
  remaining five. The monotonic contract held, but most folds learned a constant
  rather than genuinely shift-adaptive gate.
- `amir_target_only` and `amir_fixed_gate` had AURC about `0.409`, close to full
  AMIR's `0.411`. Full AMIR had lower routed error (`0.360` versus `0.370--0.371`),
  but at lower coverage.

Thus V5 demonstrates safe attenuation as a method contract, not empirical proof
that the learned shift variable drives the improvement.

### Hard intervention ordering is exact but not performance-critical here

All 18 fold-by-common-feature bounds passed. The minimum guaranteed target-logit
increase for an isolated `+0.20` mechanism intervention was exactly `0.05`.
However, the no-hard-constraint AURC (`0.4110`) was nearly identical to full AMIR
(`0.4105`). Several full coefficients sat at the lower bound. The constraint is a
useful faithfulness guarantee, but this replay does not show that it improves
ranking.

### Ranking and calibration expose a real tradeoff

The no-calibration ablation had the best V5 AURC (`0.3767`) but poor probability
calibration (Brier `0.418`, ECE `0.422`). The calibrated full model had much better
probability estimates but AURC `0.4105`. Per-fold Platt heads sometimes learned a
zero or near-zero slope, so scores were comparable as probabilities but lost
within-fold ordering; fold-specific transforms also change global cross-fold AURC.

This is not grounds to select the favorable ablation as the official result. It is
evidence that a future protocol should evaluate within-fold ranking separately
from cross-fold probability calibration and use more stable cross-fitted or
hierarchical calibration.

### The source signature remains partially sparse

The source ranking coefficients were approximately:

- intervention inertia: `1.615`;
- flip inertia: `1.174`;
- source concentration: `0.000`.

The source evidence therefore supports two intervention-response coordinates, not
all three. The paper should not describe source concentration as validated by V5.

## Eligible ACL wording

A defensible statement is:

> We introduce a shift-monotone, mechanism-constrained selective router that
> separates risk ranking from temporal calibration. In a leakage-controlled
> financial replay, the router reached a favorable coverage--error operating point
> relative to confidence and provenance baselines, while the preregistered AURC
> interval remained inconclusive.

Do not claim significant S&P 500 superiority, profitable forecasting, universal
cross-domain transfer, or confirmed shift adaptation.

## Recommended next experiment

The next main experiment should move to the originally intended cross-model LLM
setting rather than tune this market period again:

1. freeze TQA/BoolQ items, Qwen source records, Llama target records, model prompts,
   paired interventions, and bootstrap units;
2. use AMIR's mechanism representation and monotone source gate before observing
   Llama outcomes;
3. compare target-only, fixed transfer, adaptive transfer, and no-intervention
   ablations at matched coverage;
4. use finance as the secondary sequential validation of the same router contract;
5. reserve any rank-preserving hierarchical calibration change for a separately
   named protocol and a new target model or unseen date window.

This path gives the ACL paper a clearer novelty story: intervention-conditioned
transfer of agent reliability across foundation models, with finance providing a
hard temporal external test rather than carrying the central claim alone.
