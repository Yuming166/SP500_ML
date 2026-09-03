# Detection V3.15: Ling cross-family BoolQ replication

Verdict: **PASS_CROSS_FAMILY_AGGREGATE_ONLY_DETECTION_V3_15**.

- High-consensus AUROC: 0.640 [0.558, 0.717]
- Risk@80 error: 0.194 -> 0.145
- Error reduction: 0.048 [0.022, 0.077]
- Macro-label AUROC: 0.538 [0.500, 0.582]
- Worst-label AUROC: 0.120

## Label subgroups

- yes: AUROC 0.957 [0.930, 0.980], n=224, wrong=25
- no: AUROC 0.120 [0.049, 0.203], n=86, wrong=35

## Frozen gates

- high_consensus_count_and_class_gate: PASS
- aggregate_auroc_ci_lower_above_0_5: PASS
- risk80_error_reduction_ci_lower_above_zero: PASS
- final_valid_rate_is_one: PASS
- first_pass_valid_rate_at_least_0_95: PASS
- both_label_count_gates: PASS
- macro_label_auroc_ci_lower_above_0_5: PASS
- worst_label_auroc_at_least_0_5: FAIL

Aggregate and label-robust verdicts are separate. An aggregate-only pass does not support label-invariant or universal transfer.
