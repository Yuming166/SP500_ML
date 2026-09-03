# Recovery V3.14 result: zero-shot Qwen3.6 model holdout on HoVer

Date: 2026-09-03 (Asia/Shanghai)

Protocol: `recovery-v3.14-zero-shot-qwen36-hover-2026-09-03`

Verdict: **`NO_VERIFIED_ZERO_SHOT_QWEN36_HOVER_TRANSFER_V3_14`**.

## Formal result

The frozen V3.11 dual-head router was transferred without target fitting,
threshold selection, prompt adaptation, or label calibration to the previously
uncalled `Qwen3.6-35B-A3B` checkpoint. The formal set contains 300 fresh HoVer
claims, balanced 150 `Supported` and 150 `NotSupported`, with zero normalized
claim or packet-component-root overlap with the registered prior selections.

All 2,400 target action rows completed successfully and were valid on the first
attempt. The frozen router selected nine evidence bundles before outcomes were
accessed.

| Metric | KEEP | V3.14 zero-shot router |
| --- | ---: | ---: |
| Accuracy | 49.33% | 49.67% |
| Macro-label gain | 0.00 pp | +0.33 pp |
| 95% stratified-bootstrap CI | [0.00, 0.00] pp | [-1.67, +2.33] pp |
| Fixes / harms | 0 / 0 | 5 / 4 |
| Routed bundles | 0 | 9 |
| Annotation-supported repairs | 0 | 5 |

The damage rate among the 147 initially correct high-consensus cases was
`4/147 = 2.72%`.

## Native-label result

- `Supported`: 2.67% -> 6.00%, gain +3.33 pp.
- `NotSupported`: 96.00% -> 93.33%, gain -2.67 pp.

The target's anchor-only consensus was therefore extremely directional. The
zero-shot gate recovered five Supported cases but damaged four NotSupported
cases, reproducing the asymmetric safety failure seen in earlier unseen-model
experiments.

## Frozen gate accounting

Passed:

- provenance-path accuracy at least 90% (`270/300 = 90%`, and `9/9` routed);
- final structured-output yield exactly 100%; and
- first-pass structured-output yield at least 95% (`2400/2400 = 100%`).

Failed:

- macro-gain CI lower bound above zero;
- zero observed harms;
- both native-label groups nonnegative;
- at least ten annotation-supported repairs; and
- net fixes above KEEP and every matched-budget baseline.

The strongest matched baseline, fixed candidate 0 at the same nine-root budget,
made five fixes and zero harms. The primary zero-shot router made only one net
fix. The outcome-aware available-action oracle reached 90.00% accuracy with 122
fixes and zero harms; this is diagnostic only. It shows that Qwen3.6 generated
many useful recovery actions, while the frozen cross-domain router failed to
identify them safely.

## Interpretation boundary

V3.14 is a completed formal negative result. It does not support zero-shot
transfer of the current dual-head gate to Qwen3.6 on HoVer. The transport layer,
schema contract, and provenance ranking transferred; the relation/action safety
gate did not.

The target and one development model share the Qwen family, so even a pass would
not have established cross-family generalization. No threshold, route, sample,
or output may be retuned under V3.14. Any calibrated adaptation must be a new
version with a declared development set and fresh formal claims.

## Deployment record

The final service used GPUs 3 and 4 and the frozen local FP8 weights. Earlier
startup attempts stopped before the endpoint became ready because of cache
placement, a missing `ninja` PATH entry, and a GPU-allocation race. All are
recorded under `results/recovery_v3_14_qwen36/runtime/`; none made a target call.
The successful service was stopped after evaluation, returning both GPUs to the
free pool. Existing Qwen3.5 and Hy services on GPUs 5 and 6 were not changed.

## Frozen artifact hashes

- protocol manifest: `977424373678a92efd3be8a4fede3924be8c3dc09a33fa8d14c994bf5f3df65c`
- selection: `5f1c1dfbef7304ff9e7458c7612e8ed10bd5f3d5893433f7419b3f3ae5722d61`
- inference-only router inputs: `844c7d048e21604e0b3fa81544834378ef52a8551433b407c14b89bf7bf08561`
- target action records: `37d5c114d998a4f1e5cf19f7d439d7d5531f31ee5402d9c740b4d9abbca27314`
- pre-outcome routes: `c9bdc2fb773e2472a222bdb65fcda660176d050b0a02755d828cf37a42a0cf63`
- evaluation summary: `e3d0ef033df6e035f66032e84f00e2566a8e0783dc7ee2e90846d39bbaaa8dbe`
