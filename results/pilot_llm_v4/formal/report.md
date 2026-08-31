# Pilot-LLM V4 formal report

## Transfer and schema audit

| Expected calls | Valid records | First-pass valid rate | Transfer bytes |
| ---: | ---: | ---: | ---: |
| 1000 | 1000 | 1.000 | 787133 |

## Outcomes
- N: 50
- Harmful false consensus: 22 (44.0%)
- Any wrong consensus: 35

## Per-condition flip rates (per agent)
- `remove`: 0.396
- `reverse`: 0.396
- `substitute`: 0.404

## Pre-registered metrics (with 95% question-cluster bootstrap CIs)

### D_inert__harmful_fc
- auroc: 0.686
- auroc_ci: [0.524, 0.838]
- auprc: 0.644
- auprc_ci: [0.448, 0.832]
- risk_at_80: 0.500
- risk_at_80_ci: [0.300, 0.600]
- n_questions: 50

### D_conf__harmful_fc
- auroc: 0.625
- auroc_ci: [0.463, 0.761]
- auprc: 0.607
- auprc_ci: [0.429, 0.764]
- risk_at_80: 0.475
- risk_at_80_ci: [0.350, 0.650]
- n_questions: 50

### D_OR__harmful_fc
- auroc: 0.676
- auroc_ci: [0.515, 0.821]
- auprc: 0.583
- auprc_ci: [0.387, 0.772]
- risk_at_80: 0.475
- risk_at_80_ci: [0.325, 0.625]
- n_questions: 50

### D_majority__harmful_fc
- auroc: 0.159
- auroc_ci: [0.067, 0.270]
- auprc: 0.372
- auprc_ci: [0.249, 0.508]
- risk_at_80: 0.325
- risk_at_80_ci: [0.200, 0.525]
- n_questions: 50

### D_OR__any_wrong
- auroc: 0.570
- auroc_ci: [0.365, 0.750]
- auprc: 0.729
- auprc_ci: [0.574, 0.874]
- risk_at_80: 0.775
- risk_at_80_ci: [0.600, 0.875]
- n_questions: 50

### D_majority__any_wrong
- auroc: 0.521
- auroc_ci: [0.365, 0.685]
- auprc: 0.733
- auprc_ci: [0.591, 0.872]
- risk_at_80: 0.650
- risk_at_80_ci: [0.550, 0.825]
- n_questions: 50

### shared_citation_signal__harmful_fc
- auroc: 0.611
- auroc_ci: [0.473, 0.747]
- auprc: 0.508
- auprc_ci: [0.348, 0.677]
- risk_at_80: 0.450
- risk_at_80_ci: [0.325, 0.625]
- n_questions: 50

### D_OR__calibration
- brier_platt: 0.244
- ece_platt: 0.100
- brier_raw: 0.295
- n: 50
- prevalence: 0.440

## LOAO robustness (D_OR on harmful_fc)
- Deterministic AUROC: 0.676
- LOAO median AUROC: 0.664
- LOAO [p05, p95]: [0.627, 0.724]

> **Deviation:** loao_substitutes_for_preregistered_partition_permutation: v4 single-call-per-agent design cannot re-assign subsets without additional LLM calls; LOAO is the closest honest proxy.

## Interpretation boundary

These results test whether a single LLM agent's action responds to paired evidence interventions under partitioned packets and confusable substitutions. They do not establish LLM faithfulness in general, S&P 500 predictability, investment performance, or cross-model generalization.
