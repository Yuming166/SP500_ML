# Pilot-LLM V6 formal report

## Transfer and schema audit

| Expected calls | Valid records | First-pass valid rate | Transfer bytes |
| ---: | ---: | ---: | ---: |
| 2000 | 2000 | 1.000 | 5236585 |

## Co-primary verdict (V6 §9.2: D_OR single co-primary)
- **Verdict: FAIL**
- D_OR AUROC = 0.388 [0.242, 0.552] passes_lo>0.5: False
- shared_weighted (V6 §9.3 S4 secondary) AUROC = 0.8202764976958525 [0.571, 0.995] — V6 §9.3 S4: reported, not gating; CI lo > 0.5 is bonus

## Outcomes
- N: 100
- Harmful false consensus: 93 (93.0%)
- Any wrong consensus: 96

## Per-condition flip rates (per agent)
- `remove`: 0.442
- `reverse`: 0.478
- `substitute`: 0.324

## Pre-registered metrics (with 95% question-cluster bootstrap CIs)

### D_inert__harmful_fc
- auroc: 0.445
- auroc_ci: [0.331, 0.561]
- auprc: 0.948
- auprc_ci: [0.901, 0.985]
- risk_at_80: 0.912
- risk_at_80_ci: [0.838, 0.975]
- n_questions: 100

### D_conf__harmful_fc
- auroc: 0.469
- auroc_ci: [0.297, 0.645]
- auprc: 0.932
- auprc_ci: [0.863, 0.982]
- risk_at_80: 0.912
- risk_at_80_ci: [0.850, 0.975]
- n_questions: 100

### D_OR__harmful_fc
- auroc: 0.388
- auroc_ci: [0.242, 0.552]
- auprc: 0.917
- auprc_ci: [0.846, 0.975]
- risk_at_80: 0.912
- risk_at_80_ci: [0.838, 0.975]
- n_questions: 100

### D_majority__harmful_fc
- auroc: 0.075
- auroc_ci: [0.000, 0.221]
- auprc: 0.904
- auprc_ci: [0.843, 0.966]
- risk_at_80: 0.925
- risk_at_80_ci: [0.838, 0.975]
- n_questions: 100

### shared_weighted__harmful_fc
- auroc: 0.820
- auroc_ci: [0.571, 0.995]
- auprc: 0.974
- auprc_ci: [0.930, 1.000]
- risk_at_80: 0.975
- risk_at_80_ci: [0.938, 1.000]
- n_questions: 100

### shared_citation_signal__harmful_fc
- auroc: 0.482
- auroc_ci: [0.364, 0.690]
- auprc: 0.928
- auprc_ci: [0.864, 0.977]
- risk_at_80: 0.925
- risk_at_80_ci: [0.863, 0.975]
- n_questions: 100

### D_OR__any_wrong
- auroc: 0.435
- auroc_ci: [0.224, 0.646]
- auprc: 0.957
- auprc_ci: [0.908, 0.995]
- risk_at_80: 0.950
- risk_at_80_ci: [0.900, 0.988]
- n_questions: 100

### D_majority__any_wrong
- auroc: 0.147
- auroc_ci: [0.005, 0.531]
- auprc: 0.934
- auprc_ci: [0.879, 0.982]
- risk_at_80: 0.963
- risk_at_80_ci: [0.900, 1.000]
- n_questions: 100

### D_OR__calibration
- brier_platt: 0.066
- ece_platt: 0.000
- brier_raw: 0.285
- n: 100
- prevalence: 0.930

## LOAO robustness
- Deterministic AUROC D_OR: 0.3878648233486943
- Deterministic AUROC shared_weighted: 0.8202764976958525
- LOAO median AUROC: 0.39093701996927804
- LOAO [p05, p95]: [0.34485407066052226, 0.42780337941628266]

> **Deviation:** loao_substitutes_for_preregistered_partition_permutation: v6 single-call-per-agent design cannot re-assign subsets without additional LLM calls; LOAO is the closest honest proxy.

## Interpretation boundary

These results test whether V5's methodology scales on the same model (Qwen3.5-4B) at N = 100. They do not establish LLM faithfulness in general, S&P 500 predictability, investment performance, or cross-model generalization. The `shared_weighted` secondary signal has a structural variance ceiling on FEVER (see V6 §13 and `results/pilot_llm_v5/scaling_check.json`); V6 reports it but does not gate §9.2 on it. Cross-model generalization is the V7 prereg.
