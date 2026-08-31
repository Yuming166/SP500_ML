# Pilot-LLM V4 smoke report

## Transfer and schema audit

| Expected calls | Valid records | First-pass valid rate | Transfer bytes |
| ---: | ---: | ---: | ---: |
| 8 | 8 | 1.000 | 6262 |

## Outcomes
- N: 0
- Harmful false consensus: 0 (0.0%)
- Any wrong consensus: 0

## Per-condition flip rates (per agent)
- `remove`: 0.500
- `reverse`: 0.500
- `substitute`: 0.500

## Pre-registered metrics (with 95% question-cluster bootstrap CIs)

### D_inert__harmful_fc
- auroc: None
- auroc_ci: [nan, nan]
- auprc: None
- auprc_ci: [nan, nan]
- risk_at_80: None
- risk_at_80_ci: [nan, nan]
- n_questions: 0

### D_conf__harmful_fc
- auroc: None
- auroc_ci: [nan, nan]
- auprc: None
- auprc_ci: [nan, nan]
- risk_at_80: None
- risk_at_80_ci: [nan, nan]
- n_questions: 0

### D_OR__harmful_fc
- auroc: None
- auroc_ci: [nan, nan]
- auprc: None
- auprc_ci: [nan, nan]
- risk_at_80: None
- risk_at_80_ci: [nan, nan]
- n_questions: 0

### D_majority__harmful_fc
- auroc: None
- auroc_ci: [nan, nan]
- auprc: None
- auprc_ci: [nan, nan]
- risk_at_80: None
- risk_at_80_ci: [nan, nan]
- n_questions: 0

### D_OR__any_wrong
- auroc: None
- auroc_ci: [nan, nan]
- auprc: None
- auprc_ci: [nan, nan]
- risk_at_80: None
- risk_at_80_ci: [nan, nan]
- n_questions: 0

### D_majority__any_wrong
- auroc: None
- auroc_ci: [nan, nan]
- auprc: None
- auprc_ci: [nan, nan]
- risk_at_80: None
- risk_at_80_ci: [nan, nan]
- n_questions: 0

### shared_citation_signal__harmful_fc
- auroc: None
- auroc_ci: [nan, nan]
- auprc: None
- auprc_ci: [nan, nan]
- risk_at_80: None
- risk_at_80_ci: [nan, nan]
- n_questions: 0

## LOAO robustness (D_OR on harmful_fc)
- NA (n too small)

## Interpretation boundary

These results test whether a single LLM agent's action responds to paired evidence interventions under partitioned packets and confusable substitutions. They do not establish LLM faithfulness in general, S&P 500 predictability, investment performance, or cross-model generalization.
