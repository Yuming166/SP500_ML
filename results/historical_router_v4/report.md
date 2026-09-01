# Historical Router V4: CPR-Router

- Protocol: `historical-router-v4-2026-09-02`
- Status: **RETROSPECTIVE DEVELOPMENT ONLY**
- Source LLM questions: 300
- Market outer folds: 6
- Actual market agents / roots: 11 / 7

## Primary developmental endpoint

`routed_error(CPR) - routed_error(confidence)` = 0.0049 (95% moving-block CI [-0.0295, 0.0377]).
Coverage gate [0.70, 0.90]: **True**.
This endpoint is diagnostic, not eligible for a confirmatory PASS claim.

## All routers and frozen ablations

| Router | Coverage | Routed error | Brier | AURC | Worst VIX error | Mean 5d return | Max drawdown | Turnover | Net 5bps | Net 10bps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| majority | 0.684 | 0.404 | 0.369 | 0.399 | 0.456 | 0.0027 | -0.117 | 0.128 | 0.0027 | 0.0026 |
| confidence | 0.767 | 0.382 | 0.344 | 0.418 | 0.389 | 0.0034 | -0.140 | 0.223 | 0.0033 | 0.0032 |
| recent_performance | 0.652 | 0.407 | 0.245 | 0.427 | 0.426 | 0.0027 | -0.140 | 0.155 | 0.0027 | 0.0026 |
| provenance | 0.856 | 0.388 | 0.243 | 0.431 | 0.403 | 0.0037 | -0.102 | 0.088 | 0.0037 | 0.0036 |
| cpr_router | 0.824 | 0.387 | 0.338 | 0.366 | 0.421 | 0.0025 | -0.154 | 0.182 | 0.0024 | 0.0024 |
| cpr_no_anchor | 0.795 | 0.371 | 0.324 | 0.367 | 0.441 | 0.0028 | -0.135 | 0.128 | 0.0027 | 0.0027 |
| cpr_no_group_dro | 0.780 | 0.394 | 0.348 | 0.374 | 0.462 | 0.0018 | -0.154 | 0.196 | 0.0017 | 0.0016 |
| cpr_no_stress | 0.824 | 0.387 | 0.338 | 0.366 | 0.421 | 0.0025 | -0.154 | 0.182 | 0.0024 | 0.0024 |
| fixed_structural | 0.839 | 0.384 | 0.243 | 0.392 | 0.403 | 0.0035 | -0.117 | 0.169 | 0.0034 | 0.0033 |

## Structural audit

- Monotonicity violations: 0
- Mean paired-stress margin satisfaction: 0.475
- Mean risk increase under frozen paired stress: 0.0329

## Interpretation boundary

CPR-Router transfers an abstract intervention signature from real LLM-agent records and adapts it using only matured market-training labels. The current date range was previously inspected by V0-V3, so this run is evidence about method feasibility and failure modes, not a prospective financial result.
