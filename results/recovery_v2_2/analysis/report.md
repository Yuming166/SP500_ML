# Recovery V2 test report

- Verdict: **NO_VERIFIED_NET_RESCUE**
- Test examples: 230
- Page-root overlap with train/dev: 0 by construction and audit.
- Publisher independence: not tested (all roots are Wikipedia pages).

## Policies

- learned_conservative: accuracy=0.787, fixes=6, harms=1, net=5, CI=[0.004347826086956522, 0.04782608695652174], annotation-supported=6
- learned_unrestricted: accuracy=0.848, fixes=25, harms=6, net=19, CI=[0.0391304347826087, 0.13043478260869565], annotation-supported=24
- retrieval_score: accuracy=0.822, fixes=23, harms=10, net=13, CI=[0.008695652173913044, 0.10434782608695652], annotation-supported=17
- keep: accuracy=0.765, fixes=0, harms=0, net=0, CI=[0.0, 0.0], annotation-supported=0
- fixed_candidate_0: accuracy=0.809, fixes=23, harms=13, net=10, CI=[-0.008695652173913044, 0.09565217391304348], annotation-supported=20
- fixed_candidate_1: accuracy=0.778, fixes=11, harms=8, net=3, CI=[-0.02608695652173913, 0.04782608695652174], annotation-supported=7
- fixed_both: accuracy=0.861, fixes=31, harms=9, net=22, CI=[0.043478260869565216, 0.15217391304347827], annotation-supported=28
- available_action_oracle_diagnostic: accuracy=0.913, fixes=34, harms=0, net=34, CI=[0.10434782608695652, 0.1956521739130435], annotation-supported=30

## Frozen primary gates

- paired_ci_lower_above_zero: True
- net_fixes_above_every_fixed_action: False
- damage_rate_at_most_005: True
- both_label_groups_nonnegative: False
- annotation_supported_repairs_at_least_5: True
