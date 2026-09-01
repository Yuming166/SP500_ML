# Historical Router V4: ACL contribution assessment

**Written after the frozen V4 run.** This document interprets the result and
does not amend the V4 protocol.

## 1. Result in one sentence

The cross-domain CPR-Router improved **risk ordering across the full coverage
curve** but did not improve the pre-registered fixed-coverage error over the
strongest confidence baseline.

- CPR AURC: `0.366`
- majority / confidence / provenance AURC: `0.399 / 0.418 / 0.431`
- CPR routed error at realized coverage `0.824`: `0.387`
- confidence routed error at realized coverage `0.767`: `0.382`
- primary difference: `+0.0049`, 95% moving-block CI `[-0.0295, 0.0377]`

This is a ranking-transfer signal and a threshold-transfer failure, not an
overall router win.

## 2. What the ablations say

| Comparison | Observation | Mechanistic interpretation |
|---|---|---|
| Full vs no Group-DRO | AURC `0.366` vs `0.374`; error `0.387` vs `0.394` | Worst-group training is useful |
| Full vs no source anchor | AURC nearly tied; no-anchor error `0.371` | A fixed source anchor can cause negative transfer |
| Full vs no stress term | Numerically identical to reported precision | The soft stress penalty is too weak or redundant |
| Full vs fixed structural | AURC `0.366` vs `0.392` | Target-window learning improves global ranking |
| Source prior | source-concentration coefficient `0`; inertia coefficients positive | V12.1's transferable signal is intervention inertia, not citation sharing alone |

The result repeats a broader pattern already seen in the project: a structural
score can rank risk even when a frozen threshold or globally fixed transfer
weight does not generalize.

## 3. ACL novelty that is already defensible

The following is a potentially distinctive method contribution, subject to a
full related-work audit:

1. **Intervention signatures as the transfer unit.** The representation maps
   LLM evidence removal/reversal/substitution and market source-root operations
   into the same relational coordinates. It transfers behavior under evidence
   change rather than text embeddings, personas, or self-reported confidence.
2. **Agent-evidence graph semantics.** Duplicate agents sharing one provenance
   root do not count as independent support. The router observes the source
   topology maintained by the environment.
3. **Structure-preserving target adaptation.** Non-negative coefficients and
   grouped temporal optimization prevent target fitting from reversing the
   declared risk semantics.
4. **Cross-domain falsifiability.** Real-Qwen paired interventions are the
   linguistic source; leakage-aware S&P 500 replay is an external sequential
   stress test with an explicit negative-transfer outcome.

These components are stronger than presenting a generic MLP, PPO, debate
system, or financial backtest as the novelty. Individual ingredients such as
monotonic fitting, Group-DRO, or conformal calibration are not new on their own.

## 4. What cannot yet be claimed

- The LLM source anchor improved S&P 500 routing.
- The soft paired-stress loss added measurable value.
- The method transfers across model families; only Qwen3.5-4B supplied the LLM
  records.
- The market result is prospective or establishes investment alpha.
- The current hand-engineered signature is a complete learned graph encoder.

Hiding the no-anchor or no-stress ablations would weaken the paper and conflict
with the frozen protocol.

## 5. Next method version: adaptive mechanism-invariant router

V4 motivates, but does not itself test, a new version with three changes:

### 5.1 Adaptive source gate

Replace the fixed coefficient anchor by a target-train-only gate:

```text
risk = gate(signature) * source_ranker(signature)
     + (1 - gate(signature)) * target_ranker(signature)
```

The gate is trained to shrink the source component when source and target
intervention signatures disagree. This turns negative transfer into an
observable routing variable instead of assuming universal transfer.

### 5.2 Separate ranking and calibration heads

V4's AURC gain and threshold failure show that one scalar head is doing two
different jobs. The next version should train:

- a pairwise/listwise ranking head for harmful-consensus ordering;
- a temporally calibrated coverage head for abstention.

The primary NLP metric remains AURC/risk at fixed coverage; the calibration
head is evaluated separately by coverage drift and ECE.

### 5.3 Hard graph-intervention ordering

Replace the ineffective soft probability-margin penalty with a hard or
augmented-Lagrangian **logit ordering constraint**:

```text
logit_risk(corrupted graph) >= logit_risk(clean graph) + margin
```

Apply the constraint to held-out corruption mechanisms, not only synthetic
feature increments. Report violation rate as a first-class metric.

Because these changes were motivated by V4 outcomes, they require a new V5
preregistration and cannot be evaluated on the existing S&P 500 date range as a
fresh confirmatory result.

## 6. Required ACL experiment matrix

The core ACL evaluation should be linguistic; finance remains external
validation.

| Axis | Source | Target | Purpose |
|---|---|---|---|
| Cross-model | Qwen3.5-4B TQA/BoolQ | Llama-3.1-8B-Instruct on the same frozen items | Main model-family transfer claim |
| Cross-dataset | BoolQ development | untouched TQA Experiment or a frozen subset | Dataset transfer |
| Mechanism-held-out | remove/reverse | substitute/duplicate/alias holdout | Structural generalization |
| Cross-domain | real LLM QA | S&P 500 source-agent replay | External sequential stress test |

Minimum baselines are majority, self-confidence, recent performance,
provenance-only, fixed `R_PI`, unconstrained learned router, no-source-gate, and
no-intervention-constraint. All model/dataset cells must be reported.

## 7. Paper framing

A defensible working framing is:

> Multi-agent reliability should transfer through how decisions respond to
> evidence interventions and provenance structure, not through global agent
> identities or confidence. Such structural risk can preserve ordering while
> fixed transfer weights and thresholds fail under domain shift.

The negative market anchor result supports the problem statement but is not a
positive main result. A strong ACL submission still needs the frozen Llama
cross-model cell and a new target-only holdout before claiming generalization.
