# Recovery V3.10 result: schema-constrained Qwen-to-Fin-R1 ELAR

Verdict: **NO_VERIFIED_QWEN_TO_FINR1_ELAR_TRANSFER_V3_10**.

| metric | KEEP | zero-shot ELAR |
| --- | ---: | ---: |
| accuracy | 53.75% | 53.75% |
| native-label macro gain | 0.00pp | +0.00pp |
| fixes / harms | 0 / 0 | 0 / 0 |
| acquired roots | 0 | 0 |

Macro-gain 95% CI: [+0.00, +0.00]pp.

## Frozen gates

- macro_gain_ci_lower_above_zero: FAIL
- damage_rate_at_most_005: PASS
- both_label_groups_nonnegative: PASS
- annotation_supported_repairs_at_least_10: FAIL
- net_fixes_above_keep_and_all_matched_baselines: FAIL

The router, thresholds, prompts, seeds, and formal roots were frozen before
target formal calls. Ling outcomes were not used for fitting or calibration.
## Structured-output and claim boundary

All target artifacts used frozen vLLM JSON schemas and then the original semantic validators. Structured decoding is an interface control, not the claimed routing contribution. Fin-R1 shares the broad Qwen lineage with the source model.
