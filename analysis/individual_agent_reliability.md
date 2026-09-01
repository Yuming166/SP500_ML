# Individual-agent reliability analysis

Per-agent breakdown of Pilot-LLM V4/V5/V6/V7. Each row computes fragility = 1 - (inert OR conf_stable) for every (cqid, agent_index) pair, then computes AUROC of fragility → harmful_fc label (consensus answer is wrong with ≥0.8 agreement).

## V7 — per-agent reliability (N = 100 questions)

| Agent | n | correct_rate | conf_stable | flip_remove | flip_reverse | flip_substitute | AUROC_fragility | CI_lo | CI_hi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| literal_evidence | 93 | 0.075 | 0.441 | 0.473 | 0.473 | 0.301 | 0.423 | 0.236 | 0.628 |
| skeptical_auditor | 93 | 0.043 | 0.753 | 0.430 | 0.419 | 0.323 | 0.493 | 0.302 | 0.635 |
| consistency_checker | 92 | 0.065 | 0.674 | 0.457 | 0.446 | 0.348 | 0.439 | 0.201 | 0.649 |
| counterfactual_reasoner | 92 | 0.043 | 0.696 | 0.359 | 0.457 | 0.272 | 0.427 | 0.229 | 0.637 |
| minimal_judge | 92 | 0.054 | 0.587 | 0.435 | 0.402 | 0.326 | 0.468 | 0.255 | 0.674 |

## V6 — per-agent reliability (N = 100 questions)

| Agent | n | correct_rate | conf_stable | flip_remove | flip_reverse | flip_substitute | AUROC_fragility | CI_lo | CI_hi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| literal_evidence | 100 | 0.050 | 0.430 | 0.470 | 0.480 | 0.340 | 0.671 | 0.500 | 0.780 |
| skeptical_auditor | 100 | 0.060 | 0.700 | 0.460 | 0.480 | 0.370 | 0.486 | 0.289 | 0.644 |
| consistency_checker | 100 | 0.060 | 0.730 | 0.430 | 0.470 | 0.260 | 0.558 | 0.386 | 0.667 |
| counterfactual_reasoner | 100 | 0.030 | 0.660 | 0.410 | 0.480 | 0.320 | 0.595 | 0.432 | 0.700 |
| minimal_judge | 100 | 0.050 | 0.570 | 0.440 | 0.480 | 0.330 | 0.617 | 0.458 | 0.721 |

## V5 — per-agent reliability (N = 50 questions)

| Agent | n | correct_rate | conf_stable | flip_remove | flip_reverse | flip_substitute | AUROC_fragility | CI_lo | CI_hi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| literal_evidence | 50 | 0.060 | 0.400 | 0.520 | 0.520 | 0.300 | 0.240 | 0.170 | 0.312 |
| skeptical_auditor | 50 | 0.000 | 0.760 | 0.440 | 0.460 | 0.360 | 0.615 | 0.561 | 0.677 |
| consistency_checker | 50 | 0.040 | 0.640 | 0.500 | 0.520 | 0.380 | 0.417 | 0.122 | 0.713 |
| counterfactual_reasoner | 50 | 0.020 | 0.680 | 0.360 | 0.540 | 0.280 | 0.396 | 0.102 | 0.688 |
| minimal_judge | 50 | 0.040 | 0.520 | 0.440 | 0.460 | 0.360 | 0.448 | 0.153 | 0.745 |

## V4 — per-agent reliability (N = 50 questions)

| Agent | n | correct_rate | conf_stable | flip_remove | flip_reverse | flip_substitute | AUROC_fragility | CI_lo | CI_hi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| literal_evidence | 50 | 0.520 | 0.060 | 0.480 | 0.480 | 0.500 | 0.795 | 0.721 | 0.875 |
| skeptical_auditor | 50 | 0.660 | 0.200 | 0.340 | 0.360 | 0.360 | 0.885 | 0.816 | 0.946 |
| consistency_checker | 50 | 0.920 | 0.480 | 0.080 | 0.100 | 0.080 | 0.611 | 0.476 | 0.760 |
| counterfactual_reasoner | 50 | 0.540 | 0.100 | 0.500 | 0.620 | 0.500 | 0.718 | 0.637 | 0.795 |
| minimal_judge | 50 | 0.420 | 0.100 | 0.580 | 0.420 | 0.580 | 0.744 | 0.662 | 0.821 |

## Cross-version per-agent AUROC (fragility → harmful_fc)

| Agent | V7 (N=100, V5 salt) | V6 (N=100, V6 salt) | V5 (N=50, V5 salt) | V4 (N=50, V4 salt) |
|---|---|---|---|---|
| literal_evidence | 0.423 [0.236, 0.628] | 0.671 [0.500, 0.780] | 0.240 [0.170, 0.312] | 0.795 [0.721, 0.875] |
| skeptical_auditor | 0.493 [0.302, 0.635] | 0.486 [0.289, 0.644] | 0.615 [0.561, 0.677] | 0.885 [0.816, 0.946] |
| consistency_checker | 0.439 [0.201, 0.649] | 0.558 [0.386, 0.667] | 0.417 [0.122, 0.713] | 0.611 [0.476, 0.760] |
| counterfactual_reasoner | 0.427 [0.229, 0.637] | 0.595 [0.432, 0.700] | 0.396 [0.102, 0.688] | 0.718 [0.637, 0.795] |
| minimal_judge | 0.468 [0.255, 0.674] | 0.617 [0.458, 0.721] | 0.448 [0.153, 0.745] | 0.744 [0.662, 0.821] |

## Interpretation

Per-agent AUROC measures how well each agent's *fragility* (1 - (inert OR conf_stable)) predicts the question-level `harmful_fc` outcome.

**Reading the tables**
- Higher AUROC = this agent's fragility is a better per-question predictor of harm
- `correct_rate` reveals baseline accuracy (this is the original-condition answer only)
- `conf_stable_rate` reveals confidence discipline (how often this agent's confidence is robust to interventions regardless of answer flips)
- `per_condition_flip_rate[c]` decomposes fragility by intervention type

**Use cases for downstream paper sections**
- **Agent-level router (§11.b)**: weight answers by per-agent AUROC, or pick the most-reliable agent per question. Per-agent AUROC provides the diagnostic table that justifies the routing weights.
- **Persona engineering (§11.c)**: if `minimal_judge` is consistently more fragile (lower AUROC) than `skeptical_auditor`, this is empirical evidence that the `minimal` persona is too credulous under evidence removal.
- **Selection-fixed stability (§10)**: comparing per-agent AUROC across V5 (N=50) and V7 (N=100, V5 salt) tests whether each individual agent's reliability is selection-fixed-stable. If the SAME agent has consistent AUROC across V5 and V7 but D_OR (the unweighted mean) varies, this isolates the aggregation as the source of V5→V6 regression.
