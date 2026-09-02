# Recovery V3.3 result: contrastive evidence witnessing

Protocol: `recovery-v3.3-cew-averitec-2026-09-02`

Formal verdict: **NO_VERIFIED_CEW_DOMINANCE** (four of five gates passed).

## Prospective result

The official AVeriTeC dev split was untouched until the CEW model, two
thresholds, hashes, and inference boundary were frozen. All 1,888 Qwen calls
for 236 test items succeeded. The non-oracle route map was written and hashed
before the evaluator constructed gold-derived action outcomes.

| Policy | Accuracy | Fixes | Harms | Net fixes | 95% macro-gain CI | Added roots |
|---|---:|---:|---:|---:|---:|---:|
| KEEP | 80.93% | 0 | 0 | 0 | [0.00, 0.00] pp | 0 |
| CEW | 84.75% | 14 | 5 | **+9** | [-0.69, +11.03] pp | 60 |
| Matched retrieval score | 83.05% | 13 | 8 | +5 | [+1.58, +12.21] pp | 60 |
| Matched hash random | 83.47% | 6 | 0 | +6 | [+1.36, +7.62] pp | 60 |
| Matched fixed candidate 0 | 80.93% | 4 | 4 | 0 | [-1.94, +5.18] pp | 60 |
| Matched fixed candidate 1 | 81.36% | 4 | 3 | +1 | [-1.65, +4.49] pp | 60 |
| Matched fixed both | 81.78% | 3 | 1 | +2 | [-0.29, +5.18] pp | 60 |
| Unlimited semantic witness | 83.90% | 21 | 14 | +7 | [-0.29, +14.15] pp | 231 |
| Available-action oracle | 94.92% | 33 | 0 | +33 | [+16.88, +29.38] pp | 34 |

The oracle reads action outcomes and is diagnostic only.

## Frozen gates

| Gate | Result | Pass |
|---|---|---:|
| Macro label-gain CI lower bound above zero | -0.69 pp | No |
| Damage at most 5% | 2.66% | Yes |
| Nonnegative gain in both labels | Supported +7.81 pp; Refuted +2.33 pp | Yes |
| At least ten annotation-supported repairs | 11 | Yes |
| Net fixes above KEEP and all budget-matched baselines | +9 versus best baseline +6 | Yes |

## Interpretation

CEW made 60 single-root interventions. Accuracy rose by 3.81 points, with 14
repairs and five harms. Unlike V2.2, improvement was positive in both native
labels. Unlike V3.2's value router, the semantic witness policy remained
nontrivial after development safety checks and beat every baseline at the same
source budget.

The composite claim nevertheless fails because the stratified paired-bootstrap
interval slightly crosses zero. This is mainly a power and heterogeneity issue:
the Supported test subgroup contains only 64 items. The result must not be
retuned on this test or relabeled as a formal pass.

CEW selected the annotated candidate in exactly 30/60 routes, or 50%. Its gain
therefore does not come from recovering the hidden annotation-role bit; it
selects evidence by its semantic relationship to the current consensus. This
also means the experiment establishes useful evidence-path routing, not source
quality or publisher-independence prediction.

The seen/unseen-domain analysis shows +7 net fixes on 103 items whose two
candidate domains were both observed in model training, but only +2 on 133
items containing an unseen candidate domain. Cross-domain generalization is
therefore the clearest remaining weakness.

## Paper implication

The strongest defensible NAACL framing is the benchmark and method combination:
complete paired evidence interventions expose benefit and harm for every
action; CEW converts a three-way stance representation into a joint
intervene-and-source decision; and a pre-outcome route artifact makes leakage
auditable. The current single-model test is promising but insufficient for a
full superiority claim. A new external dataset or model can test the frozen
CEW policy; the exposed AVeriTeC dev set cannot be reused for tuning.

## Claim boundary

This is one Qwen3.5-4B deployment, source-domain provenance, static retrieved
evidence, and one prospective dataset. It does not establish publisher
independence, cross-model transfer, live-retrieval robustness, or a
distribution-free safety guarantee.
