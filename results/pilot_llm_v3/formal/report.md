# Pilot-LLM V3 formal report

This report follows the frozen paired-intervention protocol. Smoke output is
instrumentation-only and must not be used as formal research evidence.

## Transfer and schema audit

| Expected calls | Valid decisions | First-pass valid rate | Cache hits | Transfer bytes |
| ---: | ---: | ---: | ---: | ---: |
| 750 | 750 | 1.000 | 6 | 1652211 |

## Observable behavior

| Metric | Estimate |
| --- | ---: |
| Original answer coverage | 1.000 |
| Original accuracy | 0.792 |
| Original Brier | 0.191 |
| Original ECE | 0.169 |
| Citation rate | 0.792 |
| Complete paired triplets | 250 |
| Paired responsiveness | 0.508 |
| Remove decision change | 0.336 |
| Reverse decision change | 0.392 |
| Majority questions | 50 |
| Majority accuracy | 0.800 |
| Harmful false consensus | 0.180 |
| Causal-risk AUROC | 0.412 |
| Causal-risk AURC | 0.184 |

## Interpretation boundary

Generic fact negation is a strong, mechanically generated intervention and is not
guaranteed to be a logically minimal counterfactual. Undefined one-class metrics
remain NA. These outputs do not establish LLM faithfulness, S&P 500 predictability,
investment performance, or router superiority.
