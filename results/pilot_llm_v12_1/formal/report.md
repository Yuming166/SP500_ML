# Pilot-LLM V12.1 formal report

## Integrity

- Records: 7160 / 7160
- Valid rate: 1.000
- First-pass valid rate: 1.000

## Single confirmatory endpoint

- Verdict: **PASS**
- AUROC: 0.705
- 95% CI: [0.620, 0.781]
- Count gate passed: True
- High-consensus N: 300 (0.838)
- Wrong high consensus: 66 (0.220)

## Frozen pre-outcome score

`R_PI = 0.1 * D_inert + 0.3 * flip_inertia + 0.6 * frac_shared`

## Secondary metrics

- D_inert__wrong_high_consensus: AUROC 0.791 [0.724, 0.850]
- flip_inertia__wrong_high_consensus: AUROC 0.807 [0.744, 0.863]
- frac_shared__wrong_high_consensus: AUROC 0.498 [0.419, 0.571]
- R_PI__harmful_fc_all: AUROC 0.712 [0.631, 0.796]

## Rank router at 80% coverage

- Baseline high-consensus error: 0.220
- Routed retained error: 0.133
- Error reduction: 0.087
- Error-reduction CI: [0.04583333333333334, 0.0975]

## Interpretation boundary

This held-out result confirms or rejects only the frozen BoolQ validation paired-intervention score for this model and evidence regime. It does not establish cross-model, cross-domain, financial, or general factuality claims.
