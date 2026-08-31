# Pilot-LLM V5 formal report

## Transfer and schema audit

| Expected calls | Valid records | First-pass valid rate | Transfer bytes |
| ---: | ---: | ---: | ---: |
| 1000 | 1000 | 1.000 | 2620838 |

## Co-primary verdict (V5 §9.2: both endpoints must clear)
- **Verdict: PARTIAL_PASS**
- D_OR AUROC = 0.656 [0.508, 0.787] passes_lo>0.5: True
- shared_weighted AUROC = 0.698 [0.359, 1.000] passes_lo>0.5: False

## Outcomes
- N: 50
- Harmful false consensus: 48 (96.0%)
- Any wrong consensus: 49

## Per-condition flip rates (per agent)
- `remove`: 0.452
- `reverse`: 0.500
- `substitute`: 0.336

## Pre-registered metrics (with 95% question-cluster bootstrap CIs)

### D_inert__harmful_fc
- auroc: 0.615
- auroc_ci: [0.367, 0.806]
- auprc: 0.978
- auprc_ci: [0.941, 0.991]
- risk_at_80: 0.950
- risk_at_80_ci: [0.900, 1.000]
- n_questions: 50

### D_conf__harmful_fc
- auroc: 0.641
- auroc_ci: [0.480, 0.776]
- auprc: 0.980
- auprc_ci: [0.949, 0.993]
- risk_at_80: 0.950
- risk_at_80_ci: [0.875, 1.000]
- n_questions: 50

### D_OR__harmful_fc
- auroc: 0.656
- auroc_ci: [0.508, 0.787]
- auprc: 0.981
- auprc_ci: [0.951, 0.994]
- risk_at_80: 0.950
- risk_at_80_ci: [0.875, 1.000]
- n_questions: 50

### D_majority__harmful_fc
- auroc: 0.000
- auroc_ci: [0.000, 0.000]
- auprc: 0.937
- auprc_ci: [0.863, 0.970]
- risk_at_80: 0.950
- risk_at_80_ci: [0.875, 1.000]
- n_questions: 50

### shared_weighted__harmful_fc
- auroc: 0.698
- auroc_ci: [0.359, 1.000]
- auprc: 0.975
- auprc_ci: [0.923, 1.000]
- risk_at_80: 0.975
- risk_at_80_ci: [0.925, 1.000]
- n_questions: 50

### shared_citation_signal__harmful_fc
- auroc: 0.396
- auroc_ci: [0.340, 0.449]
- auprc: 0.952
- auprc_ci: [0.879, 0.978]
- risk_at_80: 0.950
- risk_at_80_ci: [0.875, 1.000]
- n_questions: 50

### D_OR__any_wrong
- auroc: 0.714
- auroc_ci: [0.583, 0.819]
- auprc: 0.992
- auprc_ci: [0.967, 0.995]
- risk_at_80: 0.975
- risk_at_80_ci: [0.925, 1.000]
- n_questions: 50

### D_majority__any_wrong
- auroc: 0.010
- auroc_ci: [0.000, 0.031]
- auprc: 0.959
- auprc_ci: [0.883, 0.970]
- risk_at_80: 0.975
- risk_at_80_ci: [0.900, 1.000]
- n_questions: 50

### D_OR__calibration
- brier_platt: 0.040
- ece_platt: 0.000
- brier_raw: 0.252
- n: 50
- prevalence: 0.960

## LOAO robustness
- Deterministic AUROC D_OR: 0.65625
- Deterministic AUROC shared_weighted: 0.6979166666666666
- LOAO median AUROC: 0.6979166666666666
- LOAO [p05, p95]: [0.6145833333333334, 0.7083333333333334]

> **Deviation:** loao_substitutes_for_preregistered_partition_permutation: v5 single-call-per-agent design cannot re-assign subsets without additional LLM calls; LOAO is the closest honest proxy.

## Interpretation boundary

These results test whether V4's provenance-aware methodology generalizes across domain (TQA -> FEVER) on a single model (Qwen3.5-4B). They do not establish LLM faithfulness in general, S&P 500 predictability, investment performance, or cross-model generalization.
