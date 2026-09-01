# Pilot-LLM V7 formal report

## Transfer and schema audit

| Expected calls | Valid records | First-pass valid rate | Transfer bytes |
| ---: | ---: | ---: | ---: |
| 2000 | 1848 | 0.924 | 3033326 |

## Co-primary verdict (V7 §9.2: any-passes; D_OR + shared_weighted)
- **Verdict: PASS_SINGLE_SHARED_WEIGHTED**
- D_OR AUROC = 0.621 [0.441, 0.793] passes_lo>0.5: False
- shared_weighted AUROC = 0.816 [0.567, 1.000] passes_lo>0.5: True

## Outcomes
- N: 93
- Harmful false consensus: 86 (92.5%)
- Any wrong consensus: 88

## Per-condition flip rates (per agent)
- `remove`: 0.398
- `reverse`: 0.406
- `substitute`: 0.290

## Pre-registered metrics (with 95% question-cluster bootstrap CIs)

### D_inert__harmful_fc
- auroc: 0.634
- auroc_ci: [0.501, 0.770]
- auprc: 0.963
- auprc_ci: [0.929, 0.989]
- risk_at_80: 0.905
- risk_at_80_ci: [0.865, 0.973]
- n_questions: 93

### D_conf__harmful_fc
- auroc: 0.607
- auroc_ci: [0.430, 0.784]
- auprc: 0.951
- auprc_ci: [0.899, 0.988]
- risk_at_80: 0.919
- risk_at_80_ci: [0.851, 0.973]
- n_questions: 93

### D_OR__harmful_fc
- auroc: 0.621
- auroc_ci: [0.441, 0.793]
- auprc: 0.953
- auprc_ci: [0.905, 0.989]
- risk_at_80: 0.919
- risk_at_80_ci: [0.851, 0.973]
- n_questions: 93

### D_majority__harmful_fc
- auroc: 0.078
- auroc_ci: [0.000, 0.263]
- auprc: 0.900
- auprc_ci: [0.836, 0.963]
- risk_at_80: 0.905
- risk_at_80_ci: [0.838, 0.973]
- n_questions: 93

### shared_weighted__harmful_fc
- auroc: 0.816
- auroc_ci: [0.567, 1.000]
- auprc: 0.971
- auprc_ci: [0.929, 1.000]
- risk_at_80: 0.973
- risk_at_80_ci: [0.932, 1.000]
- n_questions: 93

### shared_citation_signal__harmful_fc
- auroc: 0.537
- auroc_ci: [0.357, 0.781]
- auprc: 0.930
- auprc_ci: [0.865, 0.984]
- risk_at_80: 0.932
- risk_at_80_ci: [0.865, 0.986]
- n_questions: 93

### D_OR__any_wrong
- auroc: 0.618
- auroc_ci: [0.368, 0.859]
- auprc: 0.964
- auprc_ci: [0.920, 0.995]
- risk_at_80: 0.946
- risk_at_80_ci: [0.892, 0.986]
- n_questions: 93

### D_majority__any_wrong
- auroc: 0.123
- auroc_ci: [0.005, 0.381]
- auprc: 0.918
- auprc_ci: [0.859, 0.979]
- risk_at_80: 0.932
- risk_at_80_ci: [0.878, 0.986]
- n_questions: 93

### D_OR__calibration
- brier_platt: 0.071
- ece_platt: 0.000
- brier_raw: 0.244
- n: 93
- prevalence: 0.925

## LOAO robustness
- Deterministic AUROC D_OR: 0.6212624584717608
- Deterministic AUROC shared_weighted: 0.8156146179401993
- LOAO median AUROC: 0.6237541528239202
- LOAO [p05, p95]: [0.5988372093023255, 0.6420265780730897]

> **Deviation:** loao_substitutes_for_preregistered_partition_permutation: v6 single-call-per-agent design cannot re-assign subsets without additional LLM calls; LOAO is the closest honest proxy.

## Interpretation boundary

These results test whether V5's signal (D_OR = 0.656 at N = 50) holds at N = 100 with V5's same questions plus 50 more drawn under the same selection rule (V5 ⊂ V7 by construction). They do not establish LLM faithfulness in general, S&P 500 predictability, investment performance, or cross-model generalization. Cross-model generalization is the V8 prereg.
