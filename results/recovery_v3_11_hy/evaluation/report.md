# Recovery V3.11 result: unseen-model dual-head provenance repair

Verdict: **NO_VERIFIED_UNSEEN_MODEL_DUAL_HEAD_TRANSFER_V3_11**.

| metric | KEEP | V3.11 router |
| --- | ---: | ---: |
| accuracy | 54.26% | 67.55% |
| macro gain | 0.00pp | +13.30pp |
| fixes / harms | 0 / 0 | 26 / 1 |
| routed roots | 0 | 27 |

Macro-gain 95% CI: [+9.04, +18.09]pp.

## Frozen gates

- macro_gain_ci_lower_above_zero: PASS
- damage_rate_at_most_005: FAIL
- both_label_groups_nonnegative: FAIL
- annotation_supported_repairs_at_least_10: PASS
- net_fixes_above_keep_and_all_matched_baselines: PASS
- provenance_path_accuracy_at_least_090: PASS

The relation head, thresholds, fresh roots, embedding model, prompts, seeds, and gates were frozen before Hy study-root calls. Pre-outcome routes were content-addressed before gold labels or annotation roles were used for evaluation.
