# Controlled-heterogeneity experiment (pre-registered 2026-09-01)

4 profiles × 5 methods = 20 AUROC cells. Profiles vary the agent-reliability distribution; methods are the V4-V7 router/baseline suite. All cells reported.

## Cross-profile × method AUROC (controlled heterogeneity)

Rows: 4 pre-registered profiles. Cols: 5 methods. All 20 cells reported.

| Profile | N | BL_majority | BL_D_OR | R1_top_auroc | R2_weighted | R3_min_frag |
|---|---:|---:|---:|---:|---:|---:|
| P1_homogeneous | 92 | 0.922 | 0.599 | 0.459 | 0.415 | 0.459 |
| P2_concentrated_best | 92 | 0.922 | 0.599 | 0.430 | 0.423 | 0.430 |
| P3_concentrated_worst | 92 | 0.922 | 0.599 | 0.494 | 0.419 | 0.494 |
| P4_realistic_FEVER | 92 | 0.922 | 0.599 | 0.500 | 0.423 | 0.500 |

### P1_homogeneous (N = 92 questions)

| Method | AUROC | CI lo | CI hi |
|---|---:|---:|---:|
| BL_majority | 0.922 | 0.739 | 1.000 |
| BL_D_OR | 0.599 | 0.420 | 0.781 |
| R1_top_auroc | 0.459 | 0.230 | 0.699 |
| R2_weighted | 0.415 | 0.300 | 0.540 |
| R3_min_frag | 0.459 | 0.230 | 0.699 |

### P2_concentrated_best (N = 92 questions)

| Method | AUROC | CI lo | CI hi |
|---|---:|---:|---:|
| BL_majority | 0.922 | 0.739 | 1.000 |
| BL_D_OR | 0.599 | 0.420 | 0.781 |
| R1_top_auroc | 0.430 | 0.232 | 0.647 |
| R2_weighted | 0.423 | 0.308 | 0.546 |
| R3_min_frag | 0.430 | 0.232 | 0.647 |

### P3_concentrated_worst (N = 92 questions)

| Method | AUROC | CI lo | CI hi |
|---|---:|---:|---:|
| BL_majority | 0.922 | 0.739 | 1.000 |
| BL_D_OR | 0.599 | 0.420 | 0.781 |
| R1_top_auroc | 0.494 | 0.228 | 0.740 |
| R2_weighted | 0.419 | 0.303 | 0.543 |
| R3_min_frag | 0.494 | 0.228 | 0.740 |

### P4_realistic_FEVER (N = 92 questions)

| Method | AUROC | CI lo | CI hi |
|---|---:|---:|---:|
| BL_majority | 0.922 | 0.739 | 1.000 |
| BL_D_OR | 0.599 | 0.420 | 0.781 |
| R1_top_auroc | 0.500 | 0.500 | 0.500 |
| R2_weighted | 0.423 | 0.307 | 0.546 |
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
