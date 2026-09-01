# Pilot-LLM V11 formal report

## Integrity

- Records: 4000 / 4000
- Valid rate: 1.000
- First-pass valid rate: 1.000

## Single confirmatory endpoint

- Verdict: **FAIL**
- AUROC: 0.605
- 95% CI: [0.495, 0.721]
- Count gate passed: True
- High-consensus N: 180 (0.900)
- Wrong high consensus: 30 (0.167)

## Frozen pre-outcome score

`R_PI = 0.1 * D_inert + 0.3 * flip_inertia + 0.6 * frac_shared`

## Secondary metrics

- D_inert__wrong_high_consensus: AUROC 0.578 [0.466, 0.693]
- flip_inertia__wrong_high_consensus: AUROC 0.589 [0.480, 0.705]
- frac_shared__wrong_high_consensus: AUROC 0.552 [0.456, 0.650]
- R_PI__harmful_fc_all: AUROC 0.620 [0.498, 0.730]

## Rank router at 80% coverage

- Baseline high-consensus error: 0.167
- Routed retained error: 0.132
- Error reduction: 0.035
- Error-reduction CI: [0.0, 0.06944444444444443]

## Interpretation boundary

This held-out result confirms or rejects only the frozen BoolQ validation paired-intervention score for this model and evidence regime. It does not establish cross-model, cross-domain, financial, or general factuality claims.
