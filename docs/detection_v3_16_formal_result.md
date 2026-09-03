# Detection V3.16 formal result: label-symmetric Qwen-to-Ling transfer

Date: 2026-09-03 (Asia/Shanghai)

Protocol: `detection-v3.16-formal-vitaminc-qwen-ling-2026-09-03`

Verdict: **`NO_CONFIRMED_CROSS_FAMILY_PASS_V3_16_DUE_TO_LING_EVENT_COUNT`**.

Both models passed every registered performance and transport gate. The joint
verdict nevertheless fails because Ling had 17 high-consensus SUPPORTS errors,
below the frozen minimum of 20 errors per native label. This adequacy gate is
not removed after outcome access.

## Formal population and transport

- 250 previously uncalled Wikipedia page roots;
- 250 natural contrastive pairs / 500 exactly balanced items;
- 10,000 calls per model;
- Qwen: 9,968/10,000 final-valid, 9,963 first-pass-valid;
- Ling: 9,973/10,000 final-valid, 9,757 first-pass-valid;
- no label or outcome field in either model's records;
- pre-outcome risk and retained IDs frozen before the outcome ledger was read.

## Primary metrics

| Model | High-consensus N | Errors | AUROC | Macro | Worst label | Risk@80 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | 482 | 87 | 0.808 | 0.811 | 0.792 | 0.180 -> 0.132 |
| Ling-3.0-tiny | 464 | 113 | 0.761 | 0.766 | 0.635 | 0.244 -> 0.202 |

Qwen pair-bootstrap intervals:

- overall AUROC: `[0.761, 0.853]`;
- macro-label AUROC: `[0.763, 0.856]`;
- worst-label AUROC: `[0.729, 0.844]`;
- Risk@80 error reduction: `[+0.025, +0.067]`.

Ling pair-bootstrap intervals:

- overall AUROC: `[0.722, 0.798]`;
- macro-label AUROC: `[0.719, 0.816]`;
- worst-label AUROC: `[0.546, 0.732]`;
- Risk@80 error reduction: `[+0.026, +0.058]`.

## Native-label metrics

| Model | SUPPORTS N/errors/AUROC | REFUTES N/errors/AUROC |
| --- | --- | --- |
| Qwen | 242 / 37 / 0.792 | 240 / 50 / 0.830 |
| Ling | 238 / 17 / 0.635 | 226 / 96 / 0.897 |

The label-direction reversal from BoolQ is substantially reduced: all four
model-label AUROCs exceed 0.63, and both worst-label confidence intervals clear
0.5. Ling's SUPPORTS event count, not its AUROC interval, causes the frozen
joint failure.

## Same-prediction baselines

The frozen score outperformed vote disagreement and confidence on the primary
risk endpoints. For Qwen, agreement/confidence overall AUROCs were 0.512/0.601;
for Ling they were 0.522/0.568. Confidence had severe label asymmetry on Ling
(worst-label AUROC 0.338).

Natural reverse inertia alone was stronger than the development-selected
composite in formal evaluation: 0.855 overall AUROC on Qwen and 0.837 on Ling.
The intervention-disagreement coordinate alone was weak in formal data. This
is an informative development-to-formal shift; the V3.16 score is not retuned.

Qwen--Ling per-item risk Spearman on 449 common high-consensus items was 0.295.
Aggregate and label-conditional error ranking transferred even though exact
item ordering was only weakly correlated.

## Claim boundary and next step

V3.16 supplies strong formal evidence that natural contrastive reversal ranks
false consensus across Qwen and Ling under a label-symmetric construction. It
does not receive the preregistered joint PASS label because one feasibility
gate failed. It does not establish answer repair or universal transfer.

Any follow-up must use new, globally root-disjoint pairs and the identical
frozen score. It may address event-count power through a separately registered
replication, but cannot pool selectively chosen errors or revise V3.16.

## Artifact hashes

- Qwen records: `436cf62ae6420836b5615e157e4dbf2933d3618d6ff699d621b4ca90e8891e5f`
- Ling records: `81a691c4b009560bbb11187623001d67668e09e8ed262975be54e73780eab804`
- Qwen pre-outcome routes: `8068dc707c5925a7f388c53848696544f36c22900c49d22163e3f4126eaa3505`
- Ling pre-outcome routes: `54fe39f5488fa6024d9a5d21c0791f1f080ca0655917a0994708bd157e668252`
- Evaluation summary: `23d144909ec4d1fd5001e584c8f9fd9fb1164a01ff068b78f2645056dc46c168`
