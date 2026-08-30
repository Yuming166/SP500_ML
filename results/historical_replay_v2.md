# Historical replay V2: CBOE/ICI source expansion

V2 is specified in [`docs/historical_replay_v2_protocol.md`](../docs/historical_replay_v2_protocol.md). It keeps the prior V0/V1 reports frozen.

- As-of feature rows: 1247 (2021-05-10 to 2026-04-27).
- Splits: expanding 504-day train, five-day gap, 126-day test.
- Every abstention threshold is the train-only nested-OOF 75th-percentile risk.
- Portfolio figures use every fifth calendar decision, so five-day returns do not overlap.

| Router | Coverage | Routed error | False rejection | Mean 5d trade return | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| majority | 0.953 | 0.377 | 0.058 | 0.0038 | -0.140 |
| confidence | 0.825 | 0.371 | 0.176 | 0.0046 | -0.140 |
| recent_performance | 0.898 | 0.380 | 0.110 | 0.0041 | -0.140 |
| provenance | 0.917 | 0.377 | 0.077 | 0.0046 | -0.140 |

## Mean train-only risk thresholds

| Majority | Confidence | Recent performance | Provenance |
| ---: | ---: | ---: | ---: |
| 0.273 | 0.882 | 0.951 | 1.026 |

## Boundary

No transaction costs or intraday release-time audit are included. This replay is evidence about selective routing under stated as-of assumptions, not investment advice or proof of causal market impact.
