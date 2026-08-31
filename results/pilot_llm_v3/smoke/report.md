# Pilot-LLM V3 smoke report

This report follows the frozen paired-intervention protocol. Smoke output is
instrumentation-only and must not be used as formal research evidence.

## Transfer and schema audit

| Expected calls | Valid decisions | First-pass valid rate | Cache hits | Transfer bytes |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 6 | 1.000 | 0 | 12702 |

## Observable behavior

| Metric | Estimate |
| --- | ---: |
| Original answer coverage | 1.000 |
| Original accuracy | 0.500 |
| Original Brier | 0.453 |
| Original ECE | 0.450 |
| Citation rate | 1.000 |
| Complete paired triplets | 2 |
| Paired responsiveness | 1.000 |
| Remove decision change | 0.500 |
| Reverse decision change | 0.500 |
| Majority questions | 2 |
| Majority accuracy | 0.500 |
| Harmful false consensus | 0.500 |
| Causal-risk AUROC | 0.500 |
| Causal-risk AURC | 0.750 |

## Interpretation boundary

Generic fact negation is a strong, mechanically generated intervention and is not
guaranteed to be a logically minimal counterfactual. Undefined one-class metrics
remain NA. These outputs do not establish LLM faithfulness, S&P 500 predictability,
investment performance, or router superiority.
