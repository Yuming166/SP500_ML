# Controlled-heterogeneity experiment on V4 (pre-registered 2026-09-01)

4 profiles × 5 methods = 20 AUROC cells. Profiles vary the agent-reliability distribution; methods are the V4-V7 router/baseline suite. All cells reported.

## Cross-profile × method AUROC (controlled heterogeneity)

Rows: 4 pre-registered profiles. Cols: 5 methods. All 20 cells reported.

| Profile | N | BL_majority | BL_D_OR | R1_top_auroc | R2_weighted | R3_min_frag |
|---|---:|---:|---:|---:|---:|---:|
| P1_homogeneous | 50 | 0.566 | 0.056 | 0.555 | 0.986 | 0.555 |
| P2_concentrated_best | 50 | 0.566 | 0.056 | 0.527 | 1.000 | 0.527 |
| P3_concentrated_worst | 50 | 0.566 | 0.056 | 0.601 | 1.000 | 0.601 |
| P4_realistic_FEVER | 50 | 0.566 | 0.056 | 0.500 | 1.000 | 0.500 |

### P1_homogeneous (N = 50 questions)

| Method | AUROC | CI lo | CI hi |
|---|---:|---:|---:|
| BL_majority | 0.566 | 0.414 | 0.711 |
| BL_D_OR | 0.056 | 0.018 | 0.110 |
| R1_top_auroc | 0.555 | 0.328 | 0.760 |
| R2_weighted | 0.986 | 0.955 | 1.000 |
| R3_min_frag | 0.555 | 0.328 | 0.760 |

### P2_concentrated_best (N = 50 questions)

| Method | AUROC | CI lo | CI hi |
|---|---:|---:|---:|
| BL_majority | 0.566 | 0.414 | 0.711 |
| BL_D_OR | 0.056 | 0.018 | 0.110 |
| R1_top_auroc | 0.527 | 0.333 | 0.702 |
| R2_weighted | 1.000 | 1.000 | 1.000 |
| R3_min_frag | 0.527 | 0.333 | 0.702 |

### P3_concentrated_worst (N = 50 questions)

| Method | AUROC | CI lo | CI hi |
|---|---:|---:|---:|
| BL_majority | 0.566 | 0.414 | 0.711 |
| BL_D_OR | 0.056 | 0.018 | 0.110 |
| R1_top_auroc | 0.601 | 0.412 | 0.783 |
| R2_weighted | 1.000 | 1.000 | 1.000 |
| R3_min_frag | 0.601 | 0.412 | 0.783 |

### P4_realistic_FEVER (N = 50 questions)

| Method | AUROC | CI lo | CI hi |
|---|---:|---:|---:|
| BL_majority | 0.566 | 0.414 | 0.711 |
| BL_D_OR | 0.056 | 0.018 | 0.110 |
| R1_top_auroc | 0.500 | 0.500 | 0.500 |
| R2_weighted | 1.000 | 1.000 | 1.000 |
| R3_min_frag | 0.500 | 0.500 | 0.500 |


## Interpretation (pre-registered)

**Pre-registered question**: "Does router AUC depend on agent reliability
distribution?"

**Reading the table**
- The 4 profiles are constructed to span the agent-reliability axis:
  - **P1 (homogeneous)**: every agent equally likely to be most-reliable
    per question. Router should be unable to extract signal.
  - **P2 (concentrated-best)**: agent 1 (skeptical_auditor, V7 AUROC
    0.493) is always most-reliable. Router should win.
  - **P3 (concentrated-worst)**: agent 0 (literal_evidence, V7 AUROC
    0.423) is always most-reliable. Router's "anti-best" behavior.
  - **P4 (realistic FEVER)**: V7's actual per-question fragility used as
    the "reliability" score. This is the V4-V7 control.

**Pre-registered expected patterns**
- If `P2_R1` > `P1_R1` and `P2_R1` > `P3_R1`: routers correctly
  identify the most-reliable agent under concentration.
- If `P2_R1` > `P4_R1`: routers extract more signal under concentration
  than under realistic FEVER — supports the V9 §2 finding that realistic
  FEVER is over-redundant.
- If `P2_R1` > `P2_BL_majority`: routers beat majority vote under
  concentration. If not, majority vote is always the best — confirming
  the V4-V7 finding at a more fundamental level.

**Reporting policy**
- All 20 cells reported, no method/version hidden.
- No profile cherry-picked. If P2 and P3 give similar results, that's
  a finding.
