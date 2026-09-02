# Recovery V1 result: Qwen counterfactual-recovery router

Date: 2026-09-02
Status: **retrospective development; frozen negative result**

Recovery V1 asks whether the already-frozen Pilot-LLM V12.1 risk signal can be
extended from false-consensus detection to action selection. It leaves every
V12.1 artifact and conclusion unchanged. The Qwen endpoint is used as a frozen
generator of candidate recovery answers; the trained component is a small
action-specific uplift router, not Qwen itself.

## Experiment completed

- Parent cohort: 300 V12.1 high-consensus BoolQ questions.
- Frozen recovery gate: the highest-risk 20%, or 60 questions, under the
  inherited V12.1 score.
- Generator: Qwen3.5-4B through the existing OpenAI-compatible endpoint.
- Candidate actions: `full_evidence`, `counter_consensus`, and
  `intervention_ledger`, with `KEEP` as the no-action baseline.
- Collection: 900/900 formal action calls succeeded; every response was valid
  on the first attempt, with no retry or parse failure.
- Learning: five deterministic source-root folds; action-specific paired-gain
  models plus a harm constraint. No gold or post-outcome field is a router
  feature.

## Main result

The frozen development criterion was not met:

```text
NO_LEARNED_NET_RESCUE
```

All accuracies below are end-to-end over the 300-question cohort. Fixes and
harms are counted only where the frozen 60-question gate permits recovery.

| Policy | Accuracy | Fixes | Harms | Net | Paired net-gain 95% CI |
|---|---:|---:|---:|---:|---:|
| Keep V12.1 consensus | 0.780 | 0 | 0 | 0 | [0.000, 0.000] |
| Learned greedy uplift | 0.777 | 1 | 2 | -1 | [-0.013, 0.010] |
| Learned conservative uplift | 0.777 | 0 | 1 | -1 | [-0.010, 0.000] |
| Fixed `full_evidence` | 0.787 | 2 | 0 | +2 | [0.000, 0.017] |
| Fixed `counter_consensus` | 0.787 | 8 | 6 | +2 | [-0.017, 0.033] |
| Fixed `intervention_ledger` | 0.780 | 0 | 0 | 0 | [0.000, 0.000] |
| Available-action oracle (diagnostic) | 0.807 | 8 | 0 | +8 | [0.010, 0.047] |

The oracle shows that the generated actions contain limited repair capacity,
but the observable V12.1 features do not identify the correct question/action
pairs reliably enough. The strongest learned policy is worse than both `KEEP`
and the best fixed action.

## Structural diagnostic

The gate is label-asymmetric: its 34 baseline errors are all native `yes`
questions predicted as `no`. Blindly flipping every gated consensus therefore
fixes 34 cases but also damages 26 correct cases. Its net gain is +8/300, but the
paired interval is [-0.027, 0.077]. This is a dataset-specific polarity shortcut,
not evidence that the router learned reliable recovery.

The inherited remove/reverse/substitute intervention majorities also provide
no repair mechanism in this gate: their respective fix/harm counts are 0/1,
0/0, and 0/0.

## Claim boundary and next protocol

This result must not be reported as confirmatory recovery. V12.1 was inspected
before Recovery V1, and each BoolQ item supplies only one upstream passage root.
Consequently, the experiment does not establish provenance-disjoint evidence
acquisition or verified rescue.

The scientifically useful next step is a new, frozen Recovery V2 rather than
retuning V1. It should build a separate training pool with multiple genuinely
independent evidence roots, learn the causal value and harm of each acquisition
action, and test the frozen policy on untouched source roots and preferably a
different model family. The distinctive paper contribution would then be
paired intervention-uplift routing for false-consensus repair under provenance
and harm constraints—not generic reinforcement learning or generic LLM routing.

## Reproducibility anchors

```text
V12.1 records     1c9a1e4d01b198a5e4e6f7527500c44b9058a772a93afb9844c3ef3b22d7e5a8
V12.1 selection   ac1f15283ad719aaa6da969aabce34f59e1da6872a01890912c23fc1519d674b
Recovery manifest 80d174789f745d2068e8c685d3a2bfce8cce2ebd68edd2ede33737db1db9c995
Recovery records  b2cbcb245c7e26c93411b7fed56bfeb61a4e1804aed5c5715533c91da4563a58
Recovery summary  ae563f7e48f6e87e89799cf3a7c56fadc89b7e752a6e5e99d1c6494720a9dede
```

The executable implementation is in
`src/sp500_forecastability/recovery_v1.py`; the frozen protocol is in
`docs/recovery_v1_protocol.md`; machine-readable results are under
`results/recovery_v1/`.
