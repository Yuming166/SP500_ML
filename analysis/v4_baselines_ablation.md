# V4 baseline + ablation analysis (post-hoc)

**No new LLM calls.** This script reads the 1,000 formal records and recomputes baselines, ablations, and shared-citation detectors in-place. The frozen V4 preregistration is not modified; this is honestly labeled post-hoc analysis on the same records.

- n_questions: 50
- harmful_fc prevalence: 22/50 = 44.0%
- any_wrong prevalence: 35/50 = 70.0%

All CIs are 95% question-cluster bootstrap, seed 20260901, 1000 replicates.

## 1. Baselines vs method (target = harmful_fc)

| Score | AUROC [95% CI] | AUPRC [95% CI] | Risk@80 [95% CI] |
|---|---|---|---|
| D_majority (1-agreement) | 0.159 [0.067, 0.270] | 0.372 [0.249, 0.508] | 0.325 [0.200, 0.525] |
| D_confidence_risk (1-mean_orig_conf) | 0.390 [0.236, 0.563] | 0.437 [0.290, 0.644] | 0.375 [0.225, 0.525] |
| D_mean_conf_drop | 0.337 [0.199, 0.489] | 0.392 [0.257, 0.570] | 0.375 [0.225, 0.550] |
| D_OR_full (method) | 0.676 [0.515, 0.821] | 0.583 [0.387, 0.772] | 0.475 [0.325, 0.625] |
| D_inert_full | 0.686 [0.524, 0.838] | 0.644 [0.448, 0.832] | 0.500 [0.300, 0.600] |
| D_conf_full | 0.625 [0.463, 0.761] | 0.607 [0.429, 0.764] | 0.475 [0.350, 0.650] |

## 2. Condition ablations (target = harmful_fc)

| Variant | AUROC [95% CI] | AUPRC [95% CI] | Risk@80 [95% CI] |
|---|---|---|---|
| D_OR_full (all 3 conditions) | 0.676 [0.515, 0.821] | 0.583 [0.387, 0.772] | 0.475 [0.325, 0.625] |
| D_OR_no_substitute (remove+reverse only) | 0.670 [0.504, 0.812] | 0.578 [0.389, 0.766] | 0.450 [0.325, 0.625] |
| D_OR_no_remove (reverse+substitute only) | 0.697 [0.532, 0.846] | 0.608 [0.401, 0.794] | 0.450 [0.300, 0.625] |
| D_OR_no_reverse (remove+substitute only) | 0.665 [0.523, 0.794] | 0.559 [0.387, 0.728] | 0.475 [0.325, 0.650] |
| D_inert_substitute_only | 0.711 [0.529, 0.872] | 0.719 [0.512, 0.880] | 0.425 [0.275, 0.600] |
| D_inert_remove_only | 0.698 [0.520, 0.851] | 0.688 [0.488, 0.860] | 0.450 [0.275, 0.600] |
| D_inert_reverse_only | 0.621 [0.453, 0.776] | 0.556 [0.366, 0.759] | 0.425 [0.275, 0.600] |
| D_conf_substitute_only | 0.640 [0.494, 0.790] | 0.545 [0.365, 0.731] | 0.450 [0.300, 0.625] |

## 3. Shared-citation detectors (target = harmful_fc)

| Detector | AUROC [95% CI] | AUPRC [95% CI] | Risk@80 [95% CI] |
|---|---|---|---|
| shared_agents (V4 current) | 0.597 [0.470, 0.731] | 0.498 [0.337, 0.665] | 0.425 [0.325, 0.625] |
| shared_count_total | 0.468 [0.311, 0.634] | 0.421 [0.282, 0.595] | 0.425 [0.300, 0.600] |
| shared_id_count | 0.411 [0.263, 0.560] | 0.400 [0.263, 0.554] | 0.425 [0.300, 0.600] |
| shared_weighted | 0.785 [0.665, 0.897] | 0.647 [0.465, 0.827] | 0.550 [0.375, 0.725] |

## 4. LOAO robustness (AUROC across 5 leave-one-agent-out variants)

| Variant | median | [p05, p95] | deterministic |
|---|---|---|---|
| D_OR_full | 0.664 | [0.627, 0.724] | 0.676 |
| D_inert_full | 0.678 | [0.667, 0.713] | 0.686 |
| D_conf_full | 0.617 | [0.610, 0.666] | 0.625 |
| D_OR_no_substitute | 0.670 | [0.670, 0.670] | 0.670 |
| D_OR_no_remove | 0.697 | [0.697, 0.697] | 0.697 |
| D_OR_no_reverse | 0.665 | [0.665, 0.665] | 0.665 |

## 5. Honest interpretation

### Baselines
- **Baseline comparison:** D_majority AUROC=0.159, D_confidence_risk AUROC=0.390, D_mean_conf_drop AUROC=0.337, D_OR_full AUROC=0.676. D_OR is the strongest of the four. D_majority is anti-predictive (< 0.5) because high agreement is **part of** the harmful_fc definition (agreement ≥ 0.8); this confirms agreement alone cannot rank individual questions within the harmful subset.

### Condition attribution
- Dropping `substitute` changes D_OR AUROC by +0.006 (to 0.670).
- Dropping `remove` changes D_OR AUROC by -0.021 (to 0.697).
- Dropping `reverse` changes D_OR AUROC by +0.011 (to 0.665).
- The condition whose removal **hurts least** is `reverse` (+0.011); its presence in D_OR is the least cost-effective.
- The condition whose removal **hurts most** is `remove` (-0.021); dropping it loses the most signal.

### Single-condition inert (D_inert_{c}_only)
- substitute-only AUROC=0.711, remove-only=0.698, reverse-only=0.621.
- Ordering: substitute=0.711 > remove=0.698 > reverse=0.621. `substitute` is the most informative single condition.
- All three single conditions independently pass 0.5 AUROC with CI lower bounds above ~0.45; this is what the OR-combination D_OR capitalizes on.

### Shared-citation detectors
- Detector 1 (V4's current `shared_agents`): AUROC=0.597.
- Detector 4 (`shared_weighted = frac_shared × (1-correct) + 0.5 × frac_shared × correct`): AUROC=0.785.
- The weighted detector **beats the unweighted one** by +0.188 AUROC. This is the S4 detector that V3 diagnostic Adjustment 6 predicted would work *if* within-question citation variance were restored (V3 gave AUROC = 0.500 because V3 had no variance). On V4 partitioned packets, S4 is the strongest shared-citation signal we have.

## 6. Reproducibility

- Script: `analysis/v4_baselines_ablation.py` (reads only `results/pilot_llm_v4/formal/records.jsonl`).
- No LLM calls.
- All bootstrap CIs use question-level sampling, seed 20260901, 1000 replicates.
