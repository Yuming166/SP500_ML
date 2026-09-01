# Stratified consensus-strength analysis on V7 (pre-registered 2026-09-01)

3 strata × 5 methods = 15 cells, all reported.

**V7 baseline (full N=92, prevalence 92.4%): BL_majority=0.922, R2=0.42**

## Strata counts

- `S1_unanimous`: N = 82 questions
- `S2_strong`: N = 5 questions
- `S3_weak`: N = 5 questions

## Cross-strata × method AUROC

| Stratum | N | BL_majority | BL_D_OR | R1_top_auroc | R2_weighted | R3_min_frag |
|---|---:|---:|---:|---:|---:|---:|
| S1_unanimous | 82 | 0.500 | 0.938 | 0.938 | 0.667 | 0.938 |
| S2_strong | 5 | 0.500 | 0.500 | 0.500 | 0.125 | 0.500 |
| S3_weak | 0 | NA | NA | NA | NA | NA |
### S1_unanimous (N = 82)

| Method | AUROC | CI lo | CI hi |
|---|---:|---:|---:|
| BL_majority | 0.500 | 0.500 | 0.500 |
| BL_D_OR | 0.938 | 0.900 | 0.969 |
| R1_top_auroc | 0.938 | 0.900 | 0.969 |
| R2_weighted | 0.667 | 0.611 | 0.718 |
| R3_min_frag | 0.938 | 0.900 | 0.969 |

### S2_strong (N = 5)

| Method | AUROC | CI lo | CI hi |
|---|---:|---:|---:|
| BL_majority | 0.500 | 0.500 | 0.500 |
| BL_D_OR | 0.500 | 0.125 | 0.875 |
| R1_top_auroc | 0.500 | 0.500 | 0.500 |
| R2_weighted | 0.125 | 0.000 | 0.375 |
| R3_min_frag | 0.500 | 0.500 | 0.500 |

### S3_weak (N = 0)

| Method | AUROC | CI lo | CI hi |
|---|---:|---:|---:|
| BL_majority | NA | NA | NA |
| BL_D_OR | NA | NA | NA |
| R1_top_auroc | NA | NA | NA |
| R2_weighted | NA | NA | NA |
| R3_min_frag | NA | NA | NA |

## Interpretation (pre-registered)

**Pre-registered question**: "Does router AUROC depend on consensus strength
(agent agreement)?"

**Reading the strata**
- **S1_unanimous** (5/5 agree): strongest consensus. All agents say the same
  thing — no router can pick a different answer.
- **S2_strong** (4/5 or 5/5 minus 1): strong consensus with one dissenter.
  Some room for routers to pick the dissenter.
- **S3_weak** (3/2 or less): weak consensus. Multiple agents disagree, and
  the minority might be right.

**Pre-registered expected patterns**
- S1 should show BL_majority ≈ routers (no answer diversity).
- S3 should show R2 (AUROC-weighted vote) > BL_majority if routers work
  on diverse-agent questions.

**Reporting policy**
- All 15 cells reported, no stratum hidden.
- If S3 also loses, the failure mode is not consensus-strength but
  something deeper (e.g., wrong answer = "obvious" to all agents regardless
  of evidence removal).
