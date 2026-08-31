# Pilot-LLM V1 smoke report

This report follows the frozen paired-intervention protocol. Smoke output is
instrumentation-only and must not be used as formal research evidence.

## Transfer and schema audit

| Expected calls | Valid decisions | First-pass valid rate | Cache hits | Transfer bytes |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 6 | 1.000 | 0 | 13728 |

## Observable behavior

| Metric | Estimate |
| --- | ---: |
| Original answer coverage | 0.500 |
| Original accuracy | 1.000 |
| Original Brier | 0.000 |
| Original ECE | 0.000 |
| Citation rate | 0.500 |
| Complete paired triplets | 2 |
| Paired responsiveness | 0.500 |
| Remove decision change | 0.500 |
| Reverse decision change | 0.500 |
| Majority questions | 1 |
| Majority accuracy | 1.000 |
| Harmful false consensus | 0.000 |
| Causal-risk AUROC | NA |
| Causal-risk AURC | 0.000 |

## Interpretation boundary

Generic fact negation is a strong, mechanically generated intervention and is not
guaranteed to be a logically minimal counterfactual. Undefined one-class metrics
remain NA. These outputs do not establish LLM faithfulness, S&P 500 predictability,
investment performance, or router superiority.
