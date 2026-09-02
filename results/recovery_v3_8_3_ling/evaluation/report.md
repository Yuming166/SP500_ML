# Recovery V3.8.3 result: zero-shot Qwen-to-Ling ELAR

Verdict: **NO_VERIFIED_CROSS_MODEL_ELAR_TRANSFER_V3_8_3**.

| metric | KEEP | zero-shot ELAR |
| --- | ---: | ---: |
| accuracy | 58.50% | 58.50% |
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
## Final closed transport conformance

The fresh run used only the five preregistered transformations.
Accepted action parse modes: `{"strict": 3017, "v3_8_3_answer_casefold": 170, "v3_8_3_answer_casefold_and_empty_citation_string": 2, "v3_8_3_empty_citation_string": 10, "v3_8_3_singleton_citation_string": 1}`.
All prior schema aborts are preserved; none of their target responses were reused at inference.
## Analysis-only amendment

The registered atomic-proof-only diagnostic treats a fail-closed None ledger as empty; primary ELAR is unchanged.
Amendment: `recovery-v3.8.3-analysis-none-ledger-2026-09-03`.
