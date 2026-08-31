# Synthetic V3 post-hoc matched-coverage audit

This appendix does not modify the frozen V3 protocol or primary conclusion. It was added after inspecting unequal threshold coverage and is exploratory.

## Matched-coverage consensus error

| Method | Risk@60% | Risk@70% | Risk@80% | Risk@90% |
| --- | ---: | ---: | ---: | ---: |
| Majority | 0.444 | 0.524 | 0.583 | 0.616 |
| Confidence | 0.496 | 0.512 | 0.533 | 0.554 |
| Agreement | 0.444 | 0.524 | 0.583 | 0.546 |
| Recent performance | 0.569 | 0.571 | 0.572 | 0.569 |
| Quality only | 0.342 | 0.408 | 0.470 | 0.525 |
| Source overlap only | 0.566 | 0.628 | 0.586 | 0.525 |
| Temporal only | 0.503 | 0.470 | 0.513 | 0.525 |
| Conditional provenance | 0.293 | 0.394 | 0.466 | 0.525 |
| Oracle (diagnostic) | 0.288 | 0.390 | 0.466 | 0.525 |

## Why the frozen-threshold errors differed

- Quality only retained 0.788, rejected 801 errors, and rejected 13 correct rows.
- Conditional provenance retained 0.809, rejected 732 errors, and rejected 1 correct rows.

At matched 80% coverage, conditional provenance reaches the same risk as the diagnostic Oracle. This supports its ranking quality, but it does not retroactively satisfy the original unequal-threshold V3 hypothesis.

## Paired cluster-bootstrap differences

Negative values favor Conditional provenance. Clusters are frozen V3 base seeds.

| Baseline | Delta AURC [95% CI] | Delta Risk@80 [95% CI] |
| --- | ---: | ---: |
| Confidence | -0.267 [-0.293, -0.244] | -0.067 [-0.073, -0.059] |
| Quality only | -0.238 [-0.250, -0.227] | -0.004 [-0.008, -0.001] |
| Source overlap only | -0.122 [-0.127, -0.116] | -0.120 [-0.120, -0.120] |
| Temporal only | -0.401 [-0.407, -0.395] | -0.047 [-0.049, -0.047] |

## Boundary

This is a post-hoc evaluation of the unchanged synthetic V3 rows. It is not an independent confirmation and does not establish LLM or market validity.
