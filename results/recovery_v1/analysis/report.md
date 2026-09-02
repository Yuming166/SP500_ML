# Recovery V1 retrospective development report

## Boundary

- This is not an untouched validation result.
- V12.1 and its frozen router remain unchanged.
- All recovery actions reuse the inherited BoolQ source root; verified provenance-disjoint repair is not measured.

## Cohort

- High-consensus questions: 300
- Frozen high-risk recovery gate: 60
- Gate native-label composition: `{'no': 22, 'yes': 38}`
- Gate label/correctness composition: `{'native_no__keep_correct': 22, 'native_yes__keep_wrong': 34, 'native_yes__keep_correct': 4}`
- Development verdict: **NO_LEARNED_NET_RESCUE**

## Recovery action quality

- full_evidence: all accuracy=0.773, gate fixes=2, gate harms=0
- counter_consensus: all accuracy=0.710, gate fixes=8, gate harms=6
- intervention_ledger: all accuracy=0.773, gate fixes=0, gate harms=0

## Policies

- learned_greedy_uplift: accuracy=0.777, fixes=1, harms=2, net=-1, net gain=-0.003 [-0.013333333333333334, 0.01], added calls=0.170
- learned_conservative_uplift: accuracy=0.777, fixes=0, harms=1, net=-1, net gain=-0.003 [-0.01, 0.0], added calls=0.107
- fixed_full_evidence: accuracy=0.787, fixes=2, harms=0, net=2, net gain=0.007 [0.0, 0.016666666666666666], added calls=0.200
- fixed_counter_consensus: accuracy=0.787, fixes=8, harms=6, net=2, net gain=0.007 [-0.016666666666666666, 0.03333333333333333], added calls=0.200
- fixed_intervention_ledger: accuracy=0.780, fixes=0, harms=0, net=0, net gain=0.000 [0.0, 0.0], added calls=0.200
- keep: accuracy=0.780, fixes=0, harms=0, net=0, net gain=0.000 [0.0, 0.0], added calls=0.000
- flip_consensus_diagnostic: accuracy=0.807, fixes=34, harms=26, net=8, net gain=0.027 [-0.02666666666666667, 0.07666666666666666], added calls=0.000
- inherited_remove_majority_diagnostic: accuracy=0.777, fixes=0, harms=1, net=-1, net gain=-0.003 [-0.01, 0.0], added calls=0.000
- inherited_reverse_majority_diagnostic: accuracy=0.780, fixes=0, harms=0, net=0, net gain=0.000 [0.0, 0.0], added calls=0.000
- inherited_substitute_majority_diagnostic: accuracy=0.780, fixes=0, harms=0, net=0, net gain=0.000 [0.0, 0.0], added calls=0.000
- available_action_oracle_diagnostic: accuracy=0.807, fixes=8, harms=0, net=8, net gain=0.027 [0.01, 0.04666666666666667], added calls=0.027

## Interpretation

Only a positive net result that beats fixed recovery actions without relying on the native-label asymmetry would motivate a new untouched protocol. This development run cannot support a paper claim by itself.
