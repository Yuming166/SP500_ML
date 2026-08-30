# Historical replay V1: source grouping

Each agent is a logistic model fit only on its expanding training window. Three market agents share one root source and SPY/IVV flow agents share an ETF-flow family; Provenance averages within each source group before combining groups.

| Router | Coverage | Error | Mean non-overlapping 5d return | Max drawdown |
| --- | ---: | ---: | ---: | ---: |
| majority | 1.000 | 0.371 | 0.0040 | -0.140 |
| confidence | 1.000 | 0.371 | 0.0040 | -0.140 |
| provenance | 1.000 | 0.393 | 0.0040 | -0.140 |
| soft_root_cap | 1.000 | 0.400 | 0.0038 | -0.140 |

No transaction costs or LLM calls are included. This is a source-duplication ablation, not a final trading result.
