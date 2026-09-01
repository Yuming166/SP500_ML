# Historical replay V3: cross-domain reliability routing

V3 is specified in [`docs/historical_replay_v3_protocol.md`](../docs/historical_replay_v3_protocol.md). It keeps the V0/V1/V2 reports frozen and adds 3 pre-registered routers (R_v10, R_brier, R_equal) to the V2 ablation. All 7 routers reported; no cherry-picking.

- As-of feature rows: 1247 (2021-05-10 to 2026-04-27).
- Splits: expanding 504-day train, five-day gap, 126-day test.
- Every abstention threshold is the train-only nested-OOF 75th-percentile risk.
- Portfolio figures use every fifth calendar decision, so five-day returns do not overlap.

**7 routers compared (pre-registered; all reported):**
- `majority` / `confidence` / `recent_performance` / `provenance`: inherited from V2
- `v10`: V10 per-agent AUROC_fragility (literal=0.423, skept=0.493, consist=0.439, cf=0.427, min=0.468), clipped to [0.5, 1.0]
- `brier`: market-OOF per-agent Brier, weight = 1 / (Brier + 0.01)
- `equal`: uniform weights (control = simple mean)

| Router | Coverage | Routed error | False rejection | Mean 5d trade return | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| majority | 0.953 | 0.377 | 0.058 | 0.0038 | -0.140 |
| confidence | 0.825 | 0.371 | 0.176 | 0.0046 | -0.140 |
| recent_performance | 0.898 | 0.380 | 0.110 | 0.0041 | -0.140 |
| provenance | 0.917 | 0.377 | 0.077 | 0.0046 | -0.140 |
| v10 | 0.879 | 0.385 | 0.136 | 0.0038 | -0.140 |
| brier | 0.898 | 0.380 | 0.110 | 0.0041 | -0.140 |
| equal | 0.879 | 0.385 | 0.136 | 0.0038 | -0.140 |

## Mean train-only risk thresholds

| Majority | Confidence | Recent performance | Provenance | V10 | Brier | Equal |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.273 | 0.882 | 0.951 | 1.026 | 0.952 | 0.951 | 0.952 |

## Boundary

No transaction costs or intraday release-time audit are included. This replay is evidence about selective routing under stated as-of assumptions, not investment advice or proof of causal market impact. R_v10 weights are inherited frozen from V10's per-agent AUROC; R_brier weights are computed from train-only OOF predictions; R_equal is a uniform-weight control. None of the three new routers is tuned on V3 outcomes.
