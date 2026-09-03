# Detection V3.16 formal: label-symmetric cross-family transfer

Cross-family verdict: **FAIL**.

| Model | N | Errors | AUROC | Macro | Worst | Risk@80 | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-4B | 482 | 87 | 0.808 | 0.811 | 0.792 | 0.180 -> 0.132 | PASS |
| Ling-3.0-tiny | 464 | 113 | 0.761 | 0.766 | 0.635 | 0.244 -> 0.202 | FAIL |

Qwen--Ling risk Spearman on common high-consensus items: 0.295.

This result concerns selective error detection under natural contrastive
evidence. It does not establish answer repair or universal transfer.
