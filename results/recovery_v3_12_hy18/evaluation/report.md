# Recovery V3.12 result: selective cross-model co-sign repair

Verdict: **NO_VERIFIED_SELECTIVE_COSIGN_TRANSFER_V3_12**.

| metric | KEEP | V3.12 |
| --- | ---: | ---: |
| accuracy | 59.82% | 61.61% |
| macro gain | 0.00pp | +1.79pp |
| fixes / harms | 0 / 0 | 2 / 0 |
| teacher calls / final routes | 0 / 0 | 2 / 2 |

Macro-gain 95% CI: [+0.00, +4.46]pp.

## Frozen gates

- macro_gain_ci_lower_above_zero: FAIL
- zero_observed_harms: PASS
- both_label_groups_nonnegative: PASS
- annotation_supported_repairs_at_least_5: FAIL
- net_fixes_above_keep_and_all_matched_baselines: PASS
- provenance_path_accuracy_at_least_090: PASS
- teacher_calls_fewer_than_formal_examples: PASS
