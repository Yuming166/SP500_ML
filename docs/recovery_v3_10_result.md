# Recovery V3.10 result: Qwen-to-Fin-R1 zero-shot ELAR

Date: 2026-09-03 (Asia/Shanghai)

Protocol: `recovery-v3.10-qwen-to-finr1-guided-elar-2026-09-03`

Frozen protocol-manifest SHA-256:
`e305995fb76384e1a736b984125aec769156c8866024d2f58110e16863138620`.

## Verdict

`NO_VERIFIED_QWEN_TO_FINR1_ELAR_TRANSFER_V3_10`

The Qwen3.5-4B-trained ELAR was transferred without target-model fitting or
calibration to SUFE Fin-R1 on the same 400 formal FEVER roots. The target
checkpoint regenerated every action, certificate, and ledger observable.
Root identity was disjoint from router training, and route selection did not
access gold labels or action outcomes.

## Transport and semantic validity

V3.10 used the frozen vLLM `response_format=json_schema` contract only to
constrain artifact field names and JSON types. The original semantic validators
were applied afterward.

- Actions: 3,200 / 3,200 successful and first-pass valid.
- Certificates: 798 / 800 successful and first-pass valid. The two fail-closed
  rows belonged to one example and had no valid atomic checks.
- Proof ledgers: four candidates, three first-pass valid, one fail-closed.
- No response or cache from the earlier V3.9--V3.9.2 transport attempts was
  reused.

The structured-output intervention therefore resolved the Fin-R1 transport
failure. It is an interface control, not the claimed routing contribution.

## Primary outcome

| Metric | KEEP | Frozen ELAR |
| --- | ---: | ---: |
| Accuracy | 53.75% | 53.75% |
| Native-label macro gain | 0.00pp | +0.00pp |
| 95% macro-gain CI | -- | [+0.00, +0.00]pp |
| Fixes / harms | 0 / 0 | 0 / 0 |
| Routed examples | 0 | 0 |

Only two of five gates passed:

- `damage_rate_at_most_005`: pass;
- `both_label_groups_nonnegative`: pass;
- `macro_gain_ci_lower_above_zero`: fail;
- `annotation_supported_repairs_at_least_10`: fail;
- `net_fixes_above_keep_and_all_matched_baselines`: fail.

## Failure localization

The result is a proof-coverage failure, not evidence that Fin-R1 lacks useful
repair actions.

- Candidate actions existed for 393 / 400 examples.
- Only four candidate actions survived the atomic-certificate gate.
- Three ledgers were structurally valid, but none passed the frozen ELAR ledger
  gate: each either contained a found challenge or failed the expected-verdict
  requirement.
- The proof-only diagnostic routed three examples and produced two fixes and
  one harm: +0.25pp with a 95% CI of [-0.50, +1.25]pp.
- The outcome-aware available-action oracle, reported only as a diagnostic,
  achieved 82.50% accuracy: +28.75pp, 115 fixes, and zero harms. It is not a
  deployable policy and must not be presented as the primary result.

The paired Qwen source run remains positive at +7.25pp (29 fixes, zero harms;
95% CI [+4.75, +9.75]pp). Together, the source result and the Fin-R1 result show
that the frozen source router did not preserve proof coverage under checkpoint
shift, even within the broad Qwen lineage.

## Claim boundary

V3.10 does not support the claim that the current ELAR guides arbitrary target
models to correct answers. It supports a narrower conclusion: schema-constrained
decoding can make the cross-model interface reliable, while zero-shot
proof-carrying routing can still collapse because the target model generates too
few admissible certificates and ledgers.

Any learned cross-model repair selector must be evaluated under a new frozen
protocol and fresh roots. V3.10 thresholds, routes, and outcomes must not be
retuned or filtered.

## Artifact hashes

- Evaluation summary:
  `0dd2700de6c82f7d72bcea13d602d3b100a936b9eb1e9436c85a78004f501bb4`
- Pre-outcome routes:
  `59c198e608e00d5e5c07443d6b2455268e0e1c4ce675a74725aab19c5afaa975`
- Formal actions:
  `235310fe429bbbf057048f3b81700fba844e0a38a41b42e1bbd387edf9ec2f84`
- Formal certificates:
  `06f5c8fd12de8d52bbe4e5b95ce28505d16025fdea7ff19ab0e5a070b15b39d7`
- Formal ledgers:
  `605060605bd9fc7592820d7deda2fbdbcbc8d98bc81ec2ff838b9d1271a298e6`
