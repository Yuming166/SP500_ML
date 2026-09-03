# LLM-S&P500 V2 report: prompt-hardening re-run (paired with V1)

- Protocol: `llm-sp500-v2-2026-09-03`
- Manifest: inherited from V1 (salt `llm-sp500-v1-2026-09-03`, sha `6fd3c3edbfa8…`) — decision dates identical to V1
- Window: 2021-05-10 → 2026-04-27 (500 decision dates; 350 train / 150 test)
- Observed response model: `Hy-MT2-7B`
- Roles: literal, skeptic, consistency, cf, min
- Smoke: 5/5 parsed (100.0%)
- Formal valid-response yield: 1890/2500 (75.6%); 610 calls failed closed

## Recorded final-response token usage

- Prompt tokens: 15,388,392
- Completion tokens: 818,411
- Total tokens: 16,206,803

## Per-router metrics (test window)

| Router | Coverage | Routed error | Risk Brier | Risk ECE | n_rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| majority | 0.487 | 0.370 | 0.251 | 0.135 | 150 |
| mean_confidence_long | 0.940 | 0.376 | 0.276 | 0.195 | 150 |
| v5_provenance_baseline | 0.433 | 0.385 | 0.354 | 0.329 | 150 |
| single_min_agent | 0.940 | 0.383 | 0.280 | 0.192 | 150 |
| amir_router_v5 | 0.947 | 0.380 | 0.253 | 0.184 | 150 |

## Primary endpoint

**H1_v2** (frozen identical to H1_v1): AURC(amir_router_v5) − AURC(majority) < 0

- Observed AURC difference: **0.0317**
- 95% paired moving-block CI: [-0.0886, 0.1805]
- PASS criterion (upper CI < 0): **FAIL**

## D3_v2: paired V1 → V2 per-role yield (informational)

| Role | V1 valid | V2 valid | Δ |
| --- | ---: | ---: | ---: |
| literal | 434/500 | 386/500 | -48 |
| skeptic | 367/500 | 472/500 | +105 |
| consistency | 33/500 | 397/500 | +364 |
| cf | 390/500 | 166/500 | -224 |
| min | 481/500 | 469/500 | -12 |

## Interpretation boundary

V2 is a confirmatory re-run of V1 with prompts hardened against the two dominant V1 failure modes (empty-claims abstention, evidence_id hallucination). The decision dates are identical to V1, so the V1→V2 yield comparison is paired. A PASS closes the V5 signpost question under the V2 prompts but does not establish S&P 500 predictability, investment performance, or cross-model generalization. All five routers are reported.
