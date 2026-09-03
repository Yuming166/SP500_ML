# Detection V3.15.2 result: Qwen-to-Ling BoolQ replication

Date: 2026-09-03 (Asia/Shanghai)

Protocol: `detection-v3.15.2-ling-boolq-v12.1-2026-09-03`

Verdict: **`PASS_CROSS_FAMILY_AGGREGATE_ONLY_DETECTION_V3_15`**.

## Formal result

V3.15.2 applied the frozen Qwen V12.1 questions, evidence packets,
interventions, personas, substitute strings, `R_PI` weights, high-consensus
threshold, and 80% coverage point to the non-Qwen `Ling-3.0-tiny` model family.
No Ling outcome was used for sample, prompt, weight, threshold, or parser
selection.

All 7,160 fresh formal calls completed successfully and were first-pass valid.
Target records contain no label or gold-answer fields. Before outcome access,
the protocol froze 310 high-consensus question IDs and the 248 questions
retained by Risk@80.

| Metric | Qwen3.5-4B V12.1 | Ling-3.0-tiny V3.15.2 |
| --- | ---: | ---: |
| High-consensus N | 300 | 310 |
| Wrong high consensus | 66 | 60 |
| `R_PI` AUROC | 0.705 [0.620, 0.781] | 0.640 [0.558, 0.717] |
| Baseline error | 0.220 | 0.194 |
| Risk@80 retained error | 0.133 | 0.145 |
| Error reduction | 0.087 [0.046, 0.098] | 0.048 [0.022, 0.077] |

All five frozen aggregate replication gates passed. This supports an aggregate
cross-model-family replication of intervention-tested false-consensus ranking
and the fixed Risk@80 operating point.

## Label robustness

The stronger label-robust verdict failed.

| Native BoolQ label | Qwen3.5-4B AUROC | Ling-3.0-tiny AUROC |
| --- | ---: | ---: |
| yes | 0.834 [0.747, 0.913] | 0.957 [0.930, 0.980] |
| no | 0.213 [0.112, 0.339] | 0.120 [0.049, 0.203] |

Ling's label-macro AUROC was `0.538 [0.500, 0.582]`, but worst-label AUROC was
`0.120`. The macro CI lower bound cleared 0.5 only narrowly; the registered
worst-label gate failed decisively. The aggregate pass must therefore be called
`aggregate-only`, never label-invariant or universal.

The same polarity reversal appears more strongly in a second model family. It
is unlikely to be only a Qwen-specific artifact; the leading explanations are
BoolQ answer priors, asymmetric evidence construction, or the polarity of the
remove/reverse operations. Distinguishing these mechanisms requires a new
label-symmetric protocol, not retuning V3.15.2.

## Cross-model behavior

The Qwen--Ling Spearman correlation of per-question `R_PI` on all 358 common
questions was `0.292`, showing that individual risk ordering is only weakly
stable even though the aggregate endpoint replicated.

Ling changed answers more often under removal (`0.617` vs. Qwen `0.470`) and
reversal (`0.525` vs. `0.327`), while substitute flip rates were similar
(`0.285` vs. `0.301`). These are prespecified descriptive, outcome-free
behavioral statistics and do not replace the primary result.

## Transport amendments

V3.15 stopped at smoke because Ling usually omitted the environment-known
`agent_id`. V3.15.1 inserted that fixed metadata field but then stopped after a
full transport run: 20 empty-packet `remove` responses cited placeholder IDs.
V3.15.2 cleared citations only when the environment packet was empty and reran
all 7,160 calls with a fresh cache.

The final parse modes were:

- insert expected `agent_id`: 7,126;
- insert expected `agent_id` and clear empty-packet citations: 33;
- strict without normalization: 1.

No amendment changed an answer, confidence, prompt, question, intervention,
score, threshold, metric, or outcome field. The V3.15 smoke abort and V3.15.1
formal transport abort remain preserved.

## Claim boundary

The result supports the paper's aggregate detection and selective-routing claim
across Qwen and Ling on the same BoolQ evidence regime. It does not support
label-invariant detection, cross-dataset transfer, universal factuality, or a
cross-model repair claim. The five personas remain prompted instances of one
checkpoint within each deployment.

## Frozen artifact hashes

- protocol manifest: `eb170ae781d38c912bf91a342e9bfc3e760eea6d4b5424c67d9b2924ec07637a`
- formal records: `d1f45da96b450b84dd490854f47decd6374cb8be43d61379dce6288ff3610dd7`
- pre-outcome routes: `0fa757345b8bcfcb21bc93287f1699a3ea06a46c353ddbb12652c16c88cdd951`
