# Recovery V3.4 PBWJ external test report

- Verdict: **NO_VERIFIED_PBWJ_DOMINANCE**
- External examples: 483
- PBWJ root budget: 71

## Policies

- pbwj: accuracy=0.547, macro_gain=-0.007, fixes=0, harms=5, net=-5, roots=71
- jury_without_dispersion_veto: accuracy=0.553, macro_gain=-0.003, fixes=3, harms=5, net=-2, roots=88
- mean_witness_without_uncertainty_veto: accuracy=0.553, macro_gain=-0.003, fixes=3, harms=5, net=-2, roots=147
- keep: accuracy=0.557, macro_gain=0.000, fixes=0, harms=0, net=0, roots=0
- matched_retrieval_score: accuracy=0.582, macro_gain=0.000, fixes=18, harms=6, net=12, roots=71
- unlimited_retrieval_score: accuracy=0.687, macro_gain=0.033, fixes=89, harms=26, net=63, roots=465
- matched_hash_random: accuracy=0.571, macro_gain=-0.001, fixes=11, harms=4, net=7, roots=71
- unlimited_hash_random: accuracy=0.681, macro_gain=0.034, fixes=82, harms=22, net=60, roots=465
- matched_fixed_candidate_0: accuracy=0.571, macro_gain=-0.004, fixes=12, harms=5, net=7, roots=71
- unlimited_fixed_candidate_0: accuracy=0.675, macro_gain=0.027, fixes=80, harms=23, net=57, roots=465
- matched_fixed_candidate_1: accuracy=0.578, macro_gain=0.008, fixes=14, harms=4, net=10, roots=71
- unlimited_fixed_candidate_1: accuracy=0.679, macro_gain=0.030, fixes=86, harms=27, net=59, roots=465
- matched_fixed_both: accuracy=0.576, macro_gain=0.012, fixes=9, harms=0, net=9, roots=70
- unlimited_fixed_both: accuracy=0.745, macro_gain=0.083, fixes=109, harms=18, net=91, roots=930
- available_action_oracle_diagnostic: accuracy=0.824, macro_gain=0.179, fixes=129, harms=0, net=129, roots=136

## Frozen primary gates

- macro_gain_ci_lower_above_zero: False
- damage_rate_at_most_005: True
- both_label_groups_nonnegative: False
- annotation_supported_repairs_at_least_10: False
- net_fixes_above_keep_and_all_matched_baselines: False
