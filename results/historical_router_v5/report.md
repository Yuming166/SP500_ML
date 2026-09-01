# Historical Router V5: AMIR-Router

- Protocol: `historical-router-v5-2026-09-02`
- Status: **POST-V4 RETROSPECTIVE EXPLORATORY ONLY**
- Source LLM questions: 300
- Market outer folds: 6

## Primary exploratory endpoint

`AURC(AMIR) - AURC(confidence)` = -0.0078 (95% paired moving-block CI [-0.0607, 0.0459]).
Directional target (upper CI < 0): **False**.
This endpoint is not eligible for a confirmatory PASS claim.

## Routers and frozen ablations

| Router | Coverage | Routed error | AURC | Risk Brier | Risk ECE | Worst VIX error | Selected Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority | 0.610 | 0.387 | 0.399 | 0.328 | 0.270 | 0.433 | 0.375 |
| confidence | 0.721 | 0.408 | 0.418 | 0.438 | 0.443 | 0.437 | 0.362 |
| recent_performance | 0.537 | 0.407 | 0.427 | 0.468 | 0.470 | 0.444 | 0.245 |
| provenance | 0.709 | 0.359 | 0.431 | 0.527 | 0.528 | 0.388 | 0.242 |
| amir_router | 0.907 | 0.360 | 0.411 | 0.244 | 0.116 | 0.383 | 0.284 |
| amir_target_only | 0.996 | 0.370 | 0.409 | 0.244 | 0.119 | 0.388 | 0.300 |
| amir_fixed_gate | 0.991 | 0.371 | 0.409 | 0.244 | 0.119 | 0.388 | 0.301 |
| amir_no_hard_constraint | 0.911 | 0.362 | 0.411 | 0.244 | 0.116 | 0.388 | 0.285 |
| amir_no_calibration | 0.852 | 0.364 | 0.377 | 0.418 | 0.422 | 0.387 | 0.286 |

## V4-linked descriptive comparison

Frozen V4 CPR AURC = 0.3658; V5 AMIR AURC difference = 0.0447.
This comparison is descriptive because V4 directly motivated V5.

## Mechanism and transfer audits

- Hard target-bound violations: 0
- Minimum guaranteed target-logit rise for an isolated +0.20 common-feature intervention: 0.0500
- Mean source gate on outer tests: 0.8501
- Mean shift/gate correlation: -0.9999241629766037

## Interpretation boundary

AMIR tests whether cross-domain mechanism scores can be used conditionally rather than imposed as a fixed prior. All market results reuse a period already inspected by earlier versions; frozen cross-model LLM transfer and a new financial window remain necessary for an ACL generalization claim.
