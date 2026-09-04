# Detection V3.16.1 formal result: fresh natural-pair replication

Date: 2026-09-05 (Asia/Shanghai)

Protocol: `detection-v3.16.1-formal-vitaminc-qwen-ling-2026-09-05`

## Verdict

**Cross-family PASS.** Qwen3.5-4B and Ling-3.0-tiny each pass every
pre-registered model-level gate on the fresh natural-pair cohort. The primary
result is selective error detection under the frozen `R_sym` score, not answer
repair or universal factuality.

The experiment ran 11,560 calls per model (23,120 total), using five fixed
personas, four conditions, prompt-only JSON, a 256-token cap, and fresh
model-specific caches. The final response and first-pass validity gates were
also passed:

| Model | Final valid | First-pass valid | High-consensus N | Errors |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | 0.9978 | 0.9952 | 557 | 87 |
| Ling-3.0-tiny | 0.9993 | 0.9804 | 545 | 158 |

## Primary metrics

`R_sym = 0.3 * reverse_inertia + 0.7 * intervention_disagreement` was
selected before formal outcomes and held fixed for both models. The reported
intervals are 5,000 pair-bootstrap 95% intervals with seed `20261616`.

| Model | Overall AUROC | Macro label AUROC | Worst label AUROC | Risk@80 |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | 0.839 [0.802, 0.873] | 0.840 [0.802, 0.875] | 0.807 [0.750, 0.857] | 0.156 → 0.099 (+0.057 [0.038, 0.075]) |
| Ling-3.0-tiny | 0.727 [0.689, 0.764] | 0.738 [0.689, 0.784] | 0.607 [0.515, 0.692] | 0.290 → 0.248 (+0.042 [0.026, 0.055]) |

Native-label AUROCs remain above chance in both models. Qwen has 30
SUPPORTS and 57 REFUTES high-consensus errors; Ling has 30 SUPPORTS and 128
REFUTES errors. Thus both models satisfy the minimum of 20 errors per label,
including the previously unmet Ling SUPPORTS adequacy condition.

## Cross-model stability and controls

The two models share 526 high-consensus items, and their frozen risk scores
have Spearman correlation `0.294`. This is evidence that the aggregate
behavioral detection signal transfers across model families, while item-level
risk ordering is not stable enough to claim universal transfer.

The frozen composite outperforms simple vote-disagreement and confidence
baselines on the primary endpoint. Reverse inertia is a strong descriptive
component, while intervention disagreement alone is near chance for both
models; neither observation changes the registered composite.

## Cohort boundary

The official VitaminC audit yielded 573 page-disjoint natural pairs. V3.16
consumed the deterministic first 284 pairs. V3.16.1 uses all remaining 289
pairs, with exactly 289 SUPPORTS and 289 REFUTES items. Their natural target
pages are disjoint from prior natural target pages. Seventy target pages were
previously used as distractor pages in V3.16; they are retained to preserve the
full remaining pair pool and are explicitly recorded as controlled
cross-experiment exposure. New distractors are assigned one-to-one from pages
outside all prior 568 target/distractor pages and the new target pages.

This boundary prevents the stronger claim that every page shown in the new
cohort was unseen by the prior experiment. No prior model outputs, selected
errors, or pooled outcomes enter the V3.16.1 primary analysis.

## Reproducibility artifacts

- Selection and data-only audit: `results/detection_v3_16_1/selection_manifest.json`
  and `results/detection_v3_16_1/selection_audit.json`.
- Outcome-free public inputs: `results/detection_v3_16_1/public_manifest.json`.
- Frozen model protocols: `results/detection_v3_16_1/qwen/protocol_manifest.json`
  and `results/detection_v3_16_1/ling/protocol_manifest.json`.
- Pre-outcome routes: `results/detection_v3_16_1/qwen/evaluation/preoutcome_routes.json`
  and `results/detection_v3_16_1/ling/evaluation/preoutcome_routes.json`.
- Machine-readable evaluation: `results/detection_v3_16_1/evaluation/summary.json`.
