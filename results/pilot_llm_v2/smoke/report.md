# Pilot-LLM V2 smoke report

This report follows the frozen paired-intervention protocol. Smoke output is
instrumentation-only and must not be used as formal research evidence.

## Transfer and schema audit

| Expected calls | Valid decisions | First-pass valid rate | Cache hits | Transfer bytes |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 6 | 1.000 | 0 | 13134 |

## Observable behavior

| Metric | Estimate |
| --- | ---: |
| Original answer coverage | 0.000 |
| Original accuracy | NA |
| Original Brier | NA |
| Original ECE | NA |
| Citation rate | 1.000 |
| Complete paired triplets | 2 |
| Paired responsiveness | 0.000 |
| Remove decision change | 0.000 |
| Reverse decision change | 0.000 |
| Majority questions | 0 |
| Majority accuracy | NA |
| Harmful false consensus | NA |
| Causal-risk AUROC | NA |
| Causal-risk AURC | NA |

## Interpretation boundary

Generic fact negation is a strong, mechanically generated intervention and is not
guaranteed to be a logically minimal counterfactual. Undefined one-class metrics
remain NA. These outputs do not establish LLM faithfulness, S&P 500 predictability,
investment performance, or router superiority.
