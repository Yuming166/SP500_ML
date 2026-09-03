# Recovery V3.14 result: zero-shot Qwen3.6 model holdout on HoVer

Verdict: **NO_VERIFIED_ZERO_SHOT_QWEN36_HOVER_TRANSFER_V3_14**.

| metric | KEEP | V3.14 router |
| --- | ---: | ---: |
| accuracy | 49.33% | 49.67% |
| macro gain | 0.00pp | +0.33pp |
| fixes / harms | 0 / 0 | 5 / 4 |
| routed bundles | 0 | 9 |

Macro-gain 95% CI: [-1.67, +2.33]pp.

## Frozen gates

- macro_gain_ci_lower_above_zero: FAIL
- zero_observed_harms: FAIL
- both_label_groups_nonnegative: FAIL
- annotation_supported_repairs_at_least_10: FAIL
- net_fixes_above_keep_and_all_matched_baselines: FAIL
- provenance_path_accuracy_at_least_090: PASS
- final_schema_yield_is_one: PASS
- first_pass_schema_yield_at_least_095: PASS

The V3.11 relation head and all three thresholds were reused byte-for-byte. Qwen3.6 supplied only target actions; no Qwen3.6 outcome was available before the pre-outcome routes were frozen. HoVer is a new task domain, while the target remains in the Qwen family, so this is not cross-family evidence.
