# Pilot-LLM V10.1 formal report

## Transfer and schema audit

| Expected calls | Valid records | First-pass valid rate | Transfer bytes |
| ---: | ---: | ---: | ---: |
| 2000 | 2000 | 1.000 | 4599686 |

## Co-primary verdict (V10.1: any-passes; D_OR + shared_weighted)
- **Verdict: PASS_SINGLE_SHARED_WEIGHTED**
- D_OR AUROC = 0.680 [0.496, 0.834] passes_lo>0.5: False
- shared_weighted AUROC = 0.959 [0.890, 0.997] passes_lo>0.5: True

## Outcomes
- N: 100
- Harmful false consensus: 12 (12.0%)
- Any wrong consensus: 21

## Per-condition flip rates (per agent)
- `remove`: 0.426
- `reverse`: 0.308
- `substitute`: 0.316

## Pre-registered metrics (with 95% question-cluster bootstrap CIs)

### D_inert__harmful_fc
- auroc: 0.726
- auroc_ci: [0.530, 0.900]
- auprc: 0.335
- auprc_ci: [0.144, 0.561]
- risk_at_80: 0.150
- risk_at_80_ci: [0.062, 0.212]
- n_questions: 100

### D_conf__harmful_fc
- auroc: 0.597
- auroc_ci: [0.417, 0.757]
- auprc: 0.156
- auprc_ci: [0.077, 0.282]
- risk_at_80: 0.138
- risk_at_80_ci: [0.062, 0.212]
- n_questions: 100

### D_OR__harmful_fc
- auroc: 0.680
- auroc_ci: [0.496, 0.834]
- auprc: 0.216
- auprc_ci: [0.101, 0.358]
- risk_at_80: 0.138
- risk_at_80_ci: [0.062, 0.212]
- n_questions: 100

### D_majority__harmful_fc
- auroc: 0.355
- auroc_ci: [0.261, 0.465]
- auprc: 0.107
- auprc_ci: [0.056, 0.175]
- risk_at_80: 0.075
- risk_at_80_ci: [0.050, 0.188]
- n_questions: 100

### shared_weighted__harmful_fc
- auroc: 0.959
- auroc_ci: [0.890, 0.997]
- auprc: 0.783
- auprc_ci: [0.546, 0.967]
- risk_at_80: 0.150
- risk_at_80_ci: [0.075, 0.237]
- n_questions: 100

### shared_citation_signal__harmful_fc
- auroc: 0.623
- auroc_ci: [0.460, 0.768]
- auprc: 0.164
- auprc_ci: [0.076, 0.281]
- risk_at_80: 0.125
- risk_at_80_ci: [0.062, 0.212]
- n_questions: 100

### D_OR__any_wrong
- auroc: 0.520
- auroc_ci: [0.384, 0.663]
- auprc: 0.227
- auprc_ci: [0.140, 0.350]
- risk_at_80: 0.225
- risk_at_80_ci: [0.125, 0.312]
- n_questions: 100

### D_majority__any_wrong
- auroc: 0.608
- auroc_ci: [0.478, 0.729]
- auprc: 0.277
- auprc_ci: [0.156, 0.415]
- risk_at_80: 0.188
- risk_at_80_ci: [0.138, 0.312]
- n_questions: 100

### D_OR__calibration
- brier_platt: 0.105
- ece_platt: 0.019
- brier_raw: 0.379
- n: 100
- prevalence: 0.120

## LOAO robustness
- Deterministic AUROC D_OR: 0.6799242424242424
- Deterministic AUROC shared_weighted: 0.9588068181818182
- LOAO median AUROC: 0.6785037878787878
- LOAO [p05, p95]: [0.662405303030303, 0.6893939393939394]

> **Deviation:** loao_substitutes_for_preregistered_partition_permutation: v6 single-call-per-agent design cannot re-assign subsets without additional LLM calls; LOAO is the closest honest proxy.

## Interpretation boundary

These results test the registered cross-domain BoolQ replication under same-source but non-redundant sentence evidence. They do not establish LLM faithfulness in general, financial predictability, investment performance, or cross-model generalization.
