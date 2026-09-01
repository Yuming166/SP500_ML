# Pilot-LLM V7 smoke report

## Transfer and schema audit

| Expected calls | Valid records | First-pass valid rate | Transfer bytes |
| ---: | ---: | ---: | ---: |
| 8 | 8 | 1.000 | 6347 |

## Co-primary verdict (V7 §9.2: any-passes; D_OR + shared_weighted)
- NA

## Outcomes
- N: 0
- Harmful false consensus: 0 (0.0%)
- Any wrong consensus: 0

## Per-condition flip rates (per agent)
- `remove`: 0.000
- `reverse`: 0.000
- `substitute`: 0.000

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

### shared_weighted__harmful_fc
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

## LOAO robustness
- Deterministic AUROC D_OR: None
- Deterministic AUROC shared_weighted: None
- LOAO median AUROC: None
- LOAO [p05, p95]: [None, None]

## Interpretation boundary

These results test whether V5's signal (D_OR = 0.656 at N = 50) holds at N = 100 with V5's same questions plus 50 more drawn under the same selection rule (V5 ⊂ V7 by construction). They do not establish LLM faithfulness in general, S&P 500 predictability, investment performance, or cross-model generalization. Cross-model generalization is the V8 prereg.
