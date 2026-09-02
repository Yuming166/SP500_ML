# Recovery V3.3 CEW prospective test report

- Verdict: **NO_VERIFIED_CEW_DOMINANCE**
- Test examples: 236
- CEW root budget: 60

## Policies

- cew: accuracy=0.847, fixes=14, harms=5, net=9, roots=60
- semantic_witness_unlimited: accuracy=0.839, fixes=21, harms=14, net=7, roots=231
- keep: accuracy=0.809, fixes=0, harms=0, net=0, roots=0
- matched_retrieval_score: accuracy=0.831, fixes=13, harms=8, net=5, roots=60
- unlimited_retrieval_score: accuracy=0.881, fixes=29, harms=12, net=17, roots=231
- matched_hash_random: accuracy=0.835, fixes=6, harms=0, net=6, roots=60
- unlimited_hash_random: accuracy=0.869, fixes=23, harms=9, net=14, roots=231
- matched_fixed_candidate_0: accuracy=0.809, fixes=4, harms=4, net=0, roots=60
- unlimited_fixed_candidate_0: accuracy=0.856, fixes=22, harms=11, net=11, roots=231
- matched_fixed_candidate_1: accuracy=0.814, fixes=4, harms=3, net=1, roots=60
- unlimited_fixed_candidate_1: accuracy=0.852, fixes=19, harms=9, net=10, roots=231
- matched_fixed_both: accuracy=0.818, fixes=3, harms=1, net=2, roots=60
- unlimited_fixed_both: accuracy=0.881, fixes=31, harms=14, net=17, roots=462
- available_action_oracle_diagnostic: accuracy=0.949, fixes=33, harms=0, net=33, roots=34

## Frozen primary gates

- macro_gain_ci_lower_above_zero: False
- damage_rate_at_most_005: True
- both_label_groups_nonnegative: True
- annotation_supported_repairs_at_least_10: True
- net_fixes_above_keep_and_all_matched_baselines: True
