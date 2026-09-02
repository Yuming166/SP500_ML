# Recovery V3.12 result: selective cross-model co-sign on Hy-MT2-1.8B

Protocol: `recovery-v3.12-selective-cosign-hy18-2026-09-03`

Status: **completed formal negative result; preserved without post-outcome tuning**.

## Formal result

The unseen Hy-MT2-1.8B target completed all 896 frozen action calls for 112
FEVER-validation examples. All calls were schema-valid on the first attempt.
The frozen provisional router escalated 2/112 examples to Qwen3.5-4B, and both
co-sign calls passed. Both executed routes repaired an incorrect target
consensus and neither harmed a correct consensus.

| Metric | KEEP | Selective co-sign |
|---|---:|---:|
| Accuracy | 59.82% | 61.61% |
| Net gain | 0.00 pp | +1.79 pp |
| Macro-label gain | 0.00 pp | +1.79 pp |
| 95% stratified-bootstrap CI | [0.00, 0.00] pp | [0.00, 4.46] pp |
| Fixes / harms | 0 / 0 | 2 / 0 |
| Annotation-supported repairs | 0 | 2 |

The provenance selector was correct on 112/112 examples. The two primary
repairs beat every registered same-budget baseline, all of which produced zero
net fixes.

## Gate verdict

Five of seven gates passed. The protocol failed:

- `macro_gain_ci_lower_above_zero`, because the lower bound was exactly zero;
- `annotation_supported_repairs_at_least_5`, because only two routes were
  eligible.

It passed zero observed harms, nonnegative gains in both native labels, strict
dominance over KEEP and all matched baselines, at least 90% provenance-path
accuracy, and sparse teacher use. The registered verdict is
`NO_VERIFIED_SELECTIVE_COSIGN_TRANSFER_V3_12`.

The result is useful evidence that independent co-signing can reduce damage,
but it is not a successful confirmatory test of cross-model repair. A later
version may use V3.12 only as declared development data and must evaluate any
new rule on fresh, non-overlapping examples.

## Frozen artifact hashes

- final protocol manifest: `3e3f642938bf134d04b52aac59ca47afd1f604901c8655a2b98ffecf4e1f15e0`
- selection: `68eb73b4b9afd9be53ae5fa74ce8e092178ce01b434a9c2e073c2fb11630a715`
- router manifest: `0927f852e0e82d684cb603cf887fa3148832dc2e67c0eea655ad4de1d650c882`
- inference-only router inputs: `aebf7ca854a1e4ca2ca3352bb6b80358d5f5f17ad9f6ab837eb854f424d88124`
- target action records: `c6ef3b9a57d0f6154123b66afca376258f178d05268349e51fea003c9e07f359`
- teacher records: `65915b63c93b9fc4acd10072ff1ec6987f441f75de6c5aca342d6bda539dc8ab`
- provisional routes: `7f390332c3047cb2d5dc96c7fba755465f3e045d5253d88367d1d8cbc0b81ae7`
- final pre-outcome routes: `1c80a292a6b623edd09efa737fcd7410a75be004d81e0cb9fa531adbe22a3719`
- evaluation summary: `f83fe8f3f9000cbf54a5a23399d37e19543f1fe79575815a2eed26960b65be51`

Two failed server starts were archived under `preformal_aborts/`. They made no
study-root calls. The final server used the readable, version-pinned `casevo`
runtime on GPU 4; this deployment-only amendment was re-frozen before the
first formal target call.
