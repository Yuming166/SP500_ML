# LLM-S&P500 V1 report: LLM agents × trained AMIR router

- Protocol: `llm-sp500-v1-2026-09-03`
- Salt: `llm-sp500-v1-2026-09-03`
- Window: 2021-05-10 → 2026-04-27 (500 decision dates; 350 train / 150 test)
- Observed response model: `Hy-MT2-7B`
- Roles: literal, skeptic, consistency, cf, min
- Smoke accepted under D10_v1: 4/5 parsed (80.0%)
- Formal valid-response yield: 1705/2500 (68.2%); 795 calls failed closed

## Recorded final-response token usage

These totals cover the response retained for each call; earlier retry-attempt usage was not retained.

- Prompt tokens: 14,686,614
- Completion tokens: 565,936
- Total tokens: 15,252,550

## Per-router metrics (test window)

| Router | Coverage | Routed error | Risk Brier | Risk ECE | n_rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| majority | 1.000 | 0.400 | 0.295 | 0.174 | 150 |
| mean_confidence_long | 1.000 | 0.400 | 0.269 | 0.174 | 150 |
| v5_provenance_baseline | 0.933 | 0.379 | 0.258 | 0.156 | 150 |
| single_min_agent | 0.953 | 0.406 | 0.279 | 0.197 | 150 |
| amir_router_v5 | 0.920 | 0.406 | 0.240 | 0.055 | 150 |

## Primary endpoint

**H1_v1**: AURC(amir_router_v5) − AURC(majority) < 0

- Observed AURC difference: **-0.0874**
- 95% paired moving-block CI: [-0.1994, 0.0736]
- PASS criterion (upper CI < 0): **FAIL**

## Interpretation boundary

V1 is a confirmatory experiment under the V5 signpost. A PASS closes one open question (does AMIR work on real-time LLM outputs in a new financial window?) but does not establish S&P 500 predictability, investment performance, or cross-model generalization. All five routers are reported; the with-vs-without comparison is not selective.
