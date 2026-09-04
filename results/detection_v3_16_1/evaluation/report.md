# Detection V3.16.1 formal: fresh-root label-symmetric replication

Cross-family verdict: **PASS**.

| Model | N | Errors | AUROC | Macro | Worst | Risk@80 | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-4B | 557 | 87 | 0.839 | 0.840 | 0.807 | 0.156 -> 0.099 | PASS |
| Ling-3.0-tiny | 545 | 158 | 0.727 | 0.738 | 0.607 | 0.290 -> 0.248 | PASS |

Qwen--Ling risk Spearman on common high-consensus items: 0.294.

This result concerns selective error detection under natural contrastive
evidence on a fresh natural-pair cohort. The 289 target pairs are disjoint
from prior natural target pairs; 70 target pages were prior distractor pages,
and this controlled cross-experiment exposure is a stated boundary. New
distractor pages are disjoint from all prior target/distractor pages. The
result does not establish answer repair or universal transfer, and it does
not pool prior V3.16 outcomes.
