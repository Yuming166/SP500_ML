# Synthetic V3 preregistration: conditional provenance faithfulness

## Status and separation from prior results

V3 is a new experiment defined after the frozen V2 result.  It does not alter
V1 or V2 seeds, score definitions, thresholds, artifacts, or claims.  The V3
base seeds and all score coefficients below are fixed before V3 outputs are
generated.

## Question

V2 established that source-quality, source-overlap, and temporal signals can
separate harmful consensus in a controlled setting.  V3 asks a narrower
mechanistic question:

> Does conditional provenance faithfulness add value beyond the same runtime
> source-quality, source-overlap, and temporal metadata in isolation?

In particular, shared evidence is not automatically unsafe.  A shared,
high-integrity, timely source should not be rejected merely because several
agents derive features from it.  Shared evidence should become risky when its
integrity is poor, when it is stale or temporally invalid, or when the agents
do not change their decisions under a paired intervention on the evidence they
cite.

## Frozen generation protocol

- Base-seed clusters: `2203, 2309, 2411, 2503, 2609, 2707, 2801, 2903, 3001,
  3109`.
- Agent counts: 3, 5, 7, and 9; all counts are odd to avoid vote ties.
- Source-quality noise: 0.00, 0.10, 0.20, and 0.35.
- Corruption strengths: 0.40, 0.60, 0.80, and 1.00 for each harmful mechanism.
- Visible source identifiers are aliased; feature and transformation names are
  opaque.  This prevents the detector from relying on textual source names.
- Confidence is independently noisy with 15% coupling to source quality;
  recent performance is sampled at the agent level.  Neither exposes the
  current source-integrity audit.
- Every outer fold holds out one complete harmful mechanism.  Both clean
  controls appear in every test fold.
- Each deployable method abstains only when its pre-outcome risk is strictly
  above its own 75th-percentile score on training-only rows.  Parameters and
  thresholds never inspect the held-out outcome.
- Uncertainty is a 1,000-replicate base-seed-cluster bootstrap.

The scenarios are independent clean evidence, shared clean evidence, shared
corruption, stale evidence, partial corruption, and **evidence inertia**.
Evidence inertia gives all agents a current, high-integrity shared source but
constructs paired remove/reverse responses that fail to change the cited
decision.  It is a controlled proxy for an LLM that names evidence in a
rationale without allowing that evidence to affect its action.  The hidden
outcome remains unavailable to every deployable method.

## Fixed V3 methods

All methods receive the same agent decisions.  The first four are behavior
baselines: Majority, self-reported Confidence, Agreement, and Recent
performance.  The following ablations receive the same environment-held audit
metadata as V3 provenance:

| Method | Pre-outcome risk signal |
| --- | --- |
| Quality only | Mean root-source integrity risk |
| Source overlap only | Largest root-source share among agents |
| Temporal only | Stale fraction plus temporal-violation fraction |
| Conditional provenance | The fixed score below |
| Oracle | Hidden harmful-false-consensus label; diagnostic only |

The conditional provenance score is frozen as:

```text
temporal-violation fraction
+ source concentration × source-integrity risk
+ 0.35 × stale fraction
+ 0.60 × paired-intervention failure fraction
```

The score is clipped to `[0, 1]`.  A future evidence violation is a hard risk
term: high scores from any other component cannot offset it.  The
paired-intervention term is generated before the outcome and corresponds to
the fraction of agents that fail to abstain after removal or to change direction
after reversal of the evidence they cited.

## Hypotheses and decision rules

Primary hypothesis: Conditional provenance has lower pooled held-out AURC and
high-confidence error than Confidence, Recent performance, Quality only,
Source overlap only, and Temporal only, without a higher false-rejection rate
than the best non-oracle ablation.

Mechanism-level hypothesis: Conditional provenance retains shared clean
controls while improving detection of shared corruption, stale evidence, and
evidence inertia.  It is not expected to solve independent sources that fail
simultaneously; those failures must be reported rather than relabeled as
provenance success.

The report must include AUROC, AUPRC, ECE, Brier score, AURC, risk at 80%
coverage, high-confidence error, false rejection, coverage, mechanism-wise
results, and all ablation rows.  If the hypothesis fails, preserve and report
the result; do not reselect coefficients, seeds, thresholds, or metrics.

## Interpretation boundary

V3 remains a controlled rule-agent benchmark.  It validates a scoring and
evaluation contract, not LLM faithfulness, S&P 500 predictability, or investment
performance.  Replacing the rule-agent intervention flag with paired Qwen
remove/reverse calls is the next external-validity step and must use the same
frozen metric and split principles.
