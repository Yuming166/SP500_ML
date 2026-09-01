# Agent-level router comparison (pre-registered 2026-09-01)

5 methods × 4 versions = 20 cells, all reported.

Pre-registered router variants: R1 = top-V7-AUROC agent, R2 = AUROC-weighted majority, R3 = min-fragility agent.

## Cross-version comparison (rows = Pilot-LLM version, cols = method)

| Version | N | BL_majority | BL_D_OR | R1_top_auroc | R2_weighted | R3_min_frag |
|---|---:|---:|---:|---:|---:|---:|
| V7 | 92 | 0.922 | 0.599 | 0.525 | 0.423 | 0.513 |
| V6 | 100 | 0.925 | 0.415 | 0.514 | 0.512 | 0.441 |
| V5 | 50 | 1.000 | 0.625 | 0.385 | 0.521 | 0.438 |
| V4 | 50 | 0.566 | 0.056 | 0.115 | 1.000 | 0.364 |


## V7 — N = 93 questions

| Method | AUROC | CI lo | CI hi |
|---|---:|---:|---:|
| BL_majority | 0.922 | 0.739 | 1.000 |
| BL_D_OR | 0.599 | 0.420 | 0.781 |
| R1_top_auroc | 0.525 | 0.360 | 0.739 |
| R2_weighted | 0.423 | 0.307 | 0.546 |
| R3_min_frag | 0.513 | 0.414 | 0.683 |

## V6 — N = 100 questions

| Method | AUROC | CI lo | CI hi |
|---|---:|---:|---:|
| BL_majority | 0.925 | 0.737 | 1.000 |
| BL_D_OR | 0.415 | 0.272 | 0.570 |
| R1_top_auroc | 0.514 | 0.352 | 0.723 |
| R2_weighted | 0.512 | 0.396 | 0.633 |
| R3_min_frag | 0.441 | 0.407 | 0.471 |

## V5 — N = 50 questions

| Method | AUROC | CI lo | CI hi |
|---|---:|---:|---:|
| BL_majority | 1.000 | 1.000 | 1.000 |
| BL_D_OR | 0.625 | 0.479 | 0.765 |
| R1_top_auroc | 0.385 | 0.323 | 0.446 |
| R2_weighted | 0.521 | 0.375 | 0.660 |
| R3_min_frag | 0.438 | 0.388 | 0.479 |

## V4 — N = 50 questions

| Method | AUROC | CI lo | CI hi |
|---|---:|---:|---:|
| BL_majority | 0.566 | 0.414 | 0.711 |
| BL_D_OR | 0.056 | 0.018 | 0.110 |
| R1_top_auroc | 0.115 | 0.053 | 0.184 |
| R2_weighted | 1.000 | 1.000 | 1.000 |
| R3_min_frag | 0.364 | 0.222 | 0.500 |


## Interpretation

This table reports **5 methods × 4 versions = 20 cells**, none hidden.

**Reading the table**
- AUROC = P(method's score higher on harmful_fc questions than on non-harmful_fc questions).
- Higher is better. CI lo > 0.5 means the method's bar at §9.2-equivalent is cleared.
- All methods use the same per-question `harmful_fc` label (V7 §8: consensus wrong AND agreement ≥ 0.8).

**Pre-registered router design (locked 2026-09-01)**
- **R1 (top-AUROC agent)**: pick the answer from the agent with the highest V7 per-agent AUROC_fragility (0.493, `skeptical_auditor`). Tie-break by per-question agent fragility.
- **R2 (weighted vote)**: per-agent weight = clip(V7 per-agent AUROC, [0.5, 1.0]). Weighted majority vote among 5 agents' original answers.
- **R3 (min-fragility agent)**: for each question, pick the answer from the agent with the lowest fragility (most inert/conf-stable) on that question.

**Baselines**
- **BL_majority**: unweighted majority of 5 agents (the implicit baseline of every prior version).
- **BL_D_OR**: per-question D_OR-style score = 1 - mean(inert OR conf_stable) across 5 agents.

**Reporting policy**
- All 5 methods reported for all 4 versions. No version hidden. No method hidden.
- If a router loses on a version, this is reported as-is. The methodological lesson
  (router efficacy is selection- and prevalence-dependent) is more interesting than
  cherry-picking a winning variant.
