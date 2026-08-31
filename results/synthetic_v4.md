# Synthetic V4 formal experiment

This report was generated under the frozen [V4 preregistration](../docs/synthetic_v4_preregistration.md).

## Protocol snapshot

- Training seed clusters: 3301, 3407, 3511, 3607, 3701, 3803, 3907, 4001, 4111, 4201.
- Four disjoint ten-seed outer test sets; no test seed appears in training or another fold.
- Outer leave-one-mechanism-out evaluation with five-fold base-seed grouped cross-fitting inside training.
- Non-negative logistic provenance weights, monotonic cross-fitted calibration, and training-only 80--82% coverage threshold selection.
- Imperfect actions and paired interventions are generated with the frozen V4 behavior-noise rates.

## Preregistered outcome

The primary hypothesis is not supported. Monotonic provenance V4 must not be described as superior to the fixed V3 score or every ablation; the tables below preserve the mixed result.

## Macro-average AURC across held-out mechanisms

| Method | Macro AURC |
| --- | ---: |
| Majority | 0.447 |
| Confidence | 0.468 |
| Agreement | 0.445 |
| Recent performance | 0.537 |
| Quality only | 0.352 |
| Source overlap only | 0.348 |
| Temporal only | 0.425 |
| Conditional provenance | 0.235 |
| Monotonic provenance V4 | 0.316 |
| Oracle (diagnostic) | 0.191 |

## Pooled held-out results

Values are estimate [95% cluster-bootstrap CI]. Selective error and coverage use each fold's train-selected threshold.

| Method | AUROC | AUPRC | ECE | Brier | AURC | Risk@80% | Selective error | False rejection | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority | 0.579 [0.566, 0.593] | 0.577 [0.557, 0.600] | 0.342 [0.320, 0.369] | 0.368 [0.354, 0.385] | 0.504 [0.452, 0.554] | 0.542 [0.528, 0.557] | 0.548 [0.530, 0.566] | 0.186 [0.159, 0.209] | 0.824 [0.802, 0.849] |
| Confidence | 0.596 [0.565, 0.626] | 0.625 [0.605, 0.647] | 0.325 [0.305, 0.345] | 0.349 [0.335, 0.363] | 0.480 [0.439, 0.518] | 0.511 [0.483, 0.534] | 0.512 [0.484, 0.538] | 0.120 [0.103, 0.136] | 0.826 [0.805, 0.846] |
| Agreement | 0.587 [0.572, 0.603] | 0.593 [0.572, 0.614] | 0.227 [0.204, 0.251] | 0.315 [0.303, 0.328] | 0.502 [0.452, 0.550] | 0.528 [0.506, 0.545] | 0.527 [0.507, 0.547] | 0.172 [0.153, 0.191] | 0.802 [0.784, 0.822] |
| Recent performance | 0.506 [0.487, 0.525] | 0.549 [0.530, 0.570] | 0.340 [0.322, 0.358] | 0.364 [0.353, 0.376] | 0.538 [0.514, 0.561] | 0.538 [0.516, 0.558] | 0.537 [0.515, 0.558] | 0.200 [0.184, 0.218] | 0.792 [0.779, 0.804] |
| Quality only | 0.756 [0.681, 0.830] | 0.824 [0.781, 0.866] | 0.226 [0.175, 0.283] | 0.243 [0.195, 0.295] | 0.448 [0.356, 0.528] | 0.442 [0.419, 0.468] | 0.457 [0.415, 0.497] | 0.023 [0.015, 0.033] | 0.824 [0.783, 0.864] |
| Source overlap only | 0.706 [0.696, 0.714] | 0.656 [0.631, 0.680] | 0.257 [0.249, 0.266] | 0.242 [0.235, 0.249] | 0.340 [0.316, 0.369] | 0.549 [0.497, 0.604] | 0.486 [0.462, 0.505] | 0.111 [0.052, 0.184] | 0.792 [0.667, 0.896] |
| Temporal only | 0.581 [0.539, 0.635] | 0.634 [0.589, 0.685] | 0.429 [0.360, 0.491] | 0.433 [0.368, 0.494] | 0.599 [0.495, 0.681] | 0.485 [0.446, 0.507] | 0.499 [0.455, 0.532] | 0.088 [0.045, 0.135] | 0.833 [0.750, 0.917] |
| Conditional provenance | 0.916 [0.908, 0.925] | 0.914 [0.901, 0.926] | 0.123 [0.103, 0.147] | 0.142 [0.134, 0.150] | 0.234 [0.219, 0.251] | 0.440 [0.416, 0.467] | 0.459 [0.425, 0.494] | 0.015 [0.008, 0.023] | 0.833 [0.808, 0.857] |
| Monotonic provenance V4 | 0.769 [0.718, 0.822] | 0.815 [0.781, 0.848] | 0.074 [0.042, 0.124] | 0.197 [0.172, 0.224] | 0.339 [0.289, 0.388] | 0.447 [0.421, 0.475] | 0.521 [0.498, 0.544] | 0.003 [0.001, 0.007] | 0.953 [0.943, 0.964] |
| Oracle (diagnostic) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.188 [0.175, 0.204] | 0.427 [0.406, 0.451] | 0.005 [0.002, 0.009] | 0.000 [0.000, 0.000] | 0.460 [0.442, 0.477] |

## Matched-coverage consensus error

| Method | Risk@60% | Risk@70% | Risk@80% | Risk@90% |
| --- | ---: | ---: | ---: | ---: |
| Majority | 0.496 | 0.525 | 0.542 | 0.543 |
| Confidence | 0.484 | 0.498 | 0.511 | 0.525 |
| Agreement | 0.496 | 0.524 | 0.528 | 0.530 |
| Recent performance | 0.541 | 0.538 | 0.538 | 0.539 |
| Quality only | 0.343 | 0.391 | 0.442 | 0.495 |
| Source overlap only | 0.525 | 0.586 | 0.549 | 0.493 |
| Temporal only | 0.477 | 0.445 | 0.485 | 0.499 |
| Conditional provenance | 0.295 | 0.376 | 0.440 | 0.495 |
| Monotonic provenance V4 | 0.374 | 0.401 | 0.447 | 0.496 |
| Oracle (diagnostic) | 0.237 | 0.346 | 0.427 | 0.491 |

## Paired primary-metric differences

Negative values favor Monotonic provenance V4.

| Baseline | Delta macro AURC [95% CI] | Delta Risk@80 [95% CI] |
| --- | ---: | ---: |
| Confidence | -0.152 [-0.169, -0.133] | -0.064 [-0.070, -0.057] |
| Quality only | -0.036 [-0.043, -0.029] | 0.005 [0.001, 0.008] |
| Conditional provenance | 0.081 [0.075, 0.086] | 0.006 [0.003, 0.009] |

## Learned outer-fold models

| Held-out mechanism | Shared-integrity | Stale | Temporal | Intervention | Calibration slope | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shared_corruption | 2.169 | 0.925 | 0.000 | 2.204 | 2.039 | 0.946 |
| stale_evidence | 2.320 | 0.000 | 0.000 | 2.033 | 1.981 | 0.950 |
| partial_corruption | 2.095 | 0.879 | 0.000 | 2.149 | 1.994 | 0.943 |
| evidence_inertia | 2.914 | 0.937 | 0.000 | 0.013 | 1.772 | 0.954 |

## Train-selected thresholds by held-out mechanism

| Held-out mechanism | Majority | Confidence | Agreement | Recent performance | Quality only | Source overlap only | Temporal only | Conditional provenance | Monotonic provenance V4 | Oracle (diagnostic) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shared_corruption | 0.333 | 0.255 | 0.571 | 0.220 | 0.614 | 1.000 | 0.600 | 0.814 | 0.946 | 0.500 |
| stale_evidence | 0.333 | 0.259 | 0.571 | 0.222 | 0.690 | 1.000 | 0.000 | 0.770 | 0.950 | 0.500 |
| partial_corruption | 0.333 | 0.257 | 0.571 | 0.220 | 0.614 | 1.000 | 0.600 | 0.812 | 0.943 | 0.500 |
| evidence_inertia | 0.400 | 0.261 | 0.600 | 0.221 | 0.746 | 0.889 | 0.600 | 0.892 | 0.954 | 0.500 |

## Mechanism-wise selective error and achieved coverage

| Held-out mechanism | Method | Selective error | Coverage | False rejection |
| --- | --- | ---: | ---: | ---: |
| shared_corruption | Quality only | 0.336 | 0.697 | 0.033 |
| shared_corruption | Conditional provenance | 0.424 | 0.824 | 0.007 |
| shared_corruption | Monotonic provenance V4 | 0.487 | 0.929 | 0.002 |
| stale_evidence | Quality only | 0.472 | 0.916 | 0.011 |
| stale_evidence | Conditional provenance | 0.348 | 0.733 | 0.021 |
| stale_evidence | Monotonic provenance V4 | 0.491 | 0.957 | 0.002 |
| partial_corruption | Quality only | 0.313 | 0.690 | 0.044 |
| partial_corruption | Conditional provenance | 0.410 | 0.821 | 0.023 |
| partial_corruption | Monotonic provenance V4 | 0.470 | 0.928 | 0.008 |
| evidence_inertia | Quality only | 0.629 | 0.996 | 0.000 |
| evidence_inertia | Conditional provenance | 0.616 | 0.955 | 0.008 |
| evidence_inertia | Monotonic provenance V4 | 0.629 | 0.998 | 0.000 |

## Diagnostic interpretation

The learned router improves probability calibration and beats generic confidence routing, but it does not extrapolate the intervention mechanism as well as the fixed V3 structural prior.

When evidence inertia is completely held out, the learned intervention coefficient is 0.013; held-out coverage rises to 0.998. The training-selected threshold therefore fails to preserve its intended 80--82% coverage under this mechanism shift.

This negative result supports retaining explicit provenance priors or adding mechanism-diverse training data rather than relying on an empirical advantage from known mechanisms.

## Figures

- [Risk-coverage curve](synthetic_v4/risk_coverage.png)
- [Reliability diagram](synthetic_v4/reliability_diagram.png)
- [Mechanism-wise threshold heatmap](synthetic_v4/mechanism_heatmap.png)
- [Source-quality-noise curve](synthetic_v4/provenance_noise_curve.png)
- [Agent-count curve](synthetic_v4/agent_count_curve.png)

## Interpretation boundary

These are controlled, noisy rule-agent results. They test a routing and evaluation contract; they do not demonstrate LLM faithfulness, S&P 500 predictability, or investment performance.
