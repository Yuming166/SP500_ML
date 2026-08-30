# Historical replay V0

Rule-agent, fixed-price expanding walk-forward replay under the assumptions in [`docs/historical_replay_protocol.md`](../docs/historical_replay_protocol.md).

- As-of feature rows: 1247 (2021-05-10 to 2026-04-27).
- Mature online-quality decisions: 1053 (2022-02-14 to 2026-04-27).
- Splits: expanding 504-day train, 5-day gap, 126-day test; each threshold is the train-only risk 75th percentile.
- Portfolio figures use every fifth routed decision to avoid overlapping five-day label returns; no transaction costs yet.

| Router | Coverage | Routed error | False rejection | Mean 5d trade return | Max drawdown | Turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | 1.000 | 0.504 | 0.000 | 0.0019 | -0.037 | 0.394 |
| confidence | 0.697 | 0.488 | 0.281 | 0.0028 | -0.056 | 0.434 |
| recent_performance | 0.616 | 0.528 | 0.415 | 0.0034 | -0.046 | 0.433 |
| provenance | 0.724 | 0.500 | 0.270 | 0.0022 | -0.070 | 0.468 |

## Boundary

This is a rule-agent replay, not an LLM evaluation or a claim that the historical data supplies audited Bloomberg publication timestamps.
