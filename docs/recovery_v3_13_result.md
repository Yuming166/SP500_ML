# Recovery V3.13 result: calibrated counter-consensus cascade

Protocol: `recovery-v3.13-counter-consensus-cascade-2026-09-03`

Status: **all seven preregistered gates passed, with an analysis-only interface
amendment disclosed separately**.

## Result

On 80 fresh FEVER-validation examples, the provenance-gated
counter-consensus cascade improved accuracy from 60.00% to 68.75%: **+8.75
percentage points**. It made 7 fixes and 0 harms. The native-label-stratified
bootstrap 95% interval for macro-label gain was **[+3.75, +15.00] pp**.

| Metric | KEEP | V3.13 cascade |
|---|---:|---:|
| Accuracy | 60.00% | 68.75% |
| Net / macro-label gain | 0.00 pp | +8.75 pp |
| 95% macro-gain CI | [0.00, 0.00] pp | [+3.75, +15.00] pp |
| Fixes / harms | 0 / 0 | 7 / 0 |
| Annotation-supported repairs | 0 | 7 |

The frozen router produced 16 provisional escalations and Qwen accepted 7;
thus deployment uses 16 teacher calls for 80 target examples and overrides
only 7 answers. The provenance head selected the annotated root on 80/80
examples. All seven executed paths were annotation-supported repairs.

The strongest registered same-budget target-action baseline made one net fix;
the other registered baselines made zero. The primary cascade made seven.

## Native-label result

- Supported: 20.00% -> 37.50%, gain +17.50 pp.
- Refuted: 100.00% -> 100.00%, gain 0.00 pp.

The result therefore passes the nonnegative-per-label gate but is directional:
the Hy-MT2-1.8B anchor-only baseline was already perfect on Refuted and very
weak on Supported. The experiment does not establish symmetric repair of both
error directions.

## What this result supports

V3.13 supports a **calibrated sparse cross-model cascade**: a frozen
provenance selector and relation head detect a high-confidence contradiction
to the small-model consensus, and Qwen3.5-4B must independently agree and cite
the selected root before the system overrides that consensus.

This is not zero-shot transfer to Hy-MT2-1.8B. V3.12's 112 examples were used
after their formal negative verdict to choose relation margin `0.15`; the 80
V3.13 examples have zero claim and packet-root overlap with that calibration
set. Qwen is the answer source on accepted routes. Those facts must appear in
any paper claim.

The remaining FEVER-validation pool also has retrieval-role AUC 0.792 under
the stored lexical score. That score is forbidden from the primary router,
candidate position is balanced, and the statistic is reported as a limitation
rather than hidden. A cleaner next validation should use a new dataset/source
pool with balanced retrieval directions.

## Analysis amendment

The first evaluator invocation stopped after internally computing the primary
metric but before printing or writing any metric. The inherited matched-policy
function received raw JSONL rows instead of the required grouped mapping. The
frozen module, routes, answers, and manifest were left unchanged. A separate
analysis-only adapter applied `base._record_groups` and reran the original
evaluator. See `docs/recovery_v3_13_analysis_amendment.md`. This blemish should
be disclosed; the rerun is not described as a pristine one-shot analysis.

## Frozen artifact hashes

- protocol manifest: `299dbc15999164bcb4cdbedcfde14bbcb8b384e2f2c2a7be6e67c572a854df55`
- selection: `a8320c8ea2a2db710a01d1461c9909ff514c5d07b6e3f649b3362df54ceb5325`
- router manifest: `f8e0fa23bde6023c29731f17d1878b041f9dd7440ce597c5bfd7b6b0a817802f`
- inference-only router inputs: `2dfee0300ace2fdfc8f5883f7e4a46f2c16ca7f7942b198cfcaef16218a83df7`
- target records: `470b59ee95b85d94e5cb76b73d176bc78d41d5ffde9492336eeb9c61f8be1c85`
- teacher records: `85360de91f006011b766f9b1d793b22501bfae2dafdac92500f1459341a4b0c8`
- provisional routes: `3d59de2d76e68710334fcb9bbb4db79e957b6c5d028da66547fc60f6e6341488`
- final pre-outcome routes: `321a690705d17bcf3f62ed42985c8391458553464bef843d3de5c1536b582a25`
- evaluation summary: `431a7129e4f414f3506189e5449e24d27175d96ef0eeef865ec0686b45aafeb3`
- analysis adapter: `90595074bac5ae972ef9717dd117e438404ba3b8f12058ab5e751c5d2a5dae7a`
