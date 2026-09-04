# Detection V3.16.1 formal preregistration: fresh-root label-symmetric replication

Date: 2026-09-05 (Asia/Shanghai)

Protocol: `detection-v3.16.1-formal-vitaminc-qwen-ling-2026-09-05`

Status: **frozen before any V3.16.1 model call**.

## Purpose and claim boundary

V3.16.1 is a fresh-root replication of the V3.16 label-symmetric detection
protocol. It is designed to test whether the observed Qwen-to-Ling signal is
repeatable on the remaining deterministic VitaminC page-pair pool. It is not a
repair of the V3.16 verdict, and it does not pool selected errors from V3.16.

The primary headline passes only if Qwen3.5-4B and Ling-3.0-tiny each pass all
registered model-level gates on the new cohort. A failure remains a formal
boundary result. The experiment does not establish answer repair, universal
factuality, bidirectional transfer, independent agents, live retrieval
robustness, or financial performance.

## Fresh selection

The source is the official VitaminC real contrastive test archive used by V3.16.
The original deterministic audit found 573 page-disjoint natural
SUPPORTS/REFUTES pairs. The prior V3.16 selection consumed 284 pairs across
smoke, development, and formal partitions. V3.16.1 uses all remaining 289
page-disjoint pairs as its formal population.

For V3.16.1, the 289 natural target pairs are disjoint from every prior V3.16
natural target pair. The deterministic remainder contains 70 target pages that
were previously used only as distractor pages; those pages are retained rather
than silently discarded, and this limited cross-experiment exposure is recorded
as a boundary of the replication. A new one-to-one distractor assignment is
made only from pages outside both the prior 568 target/distractor pages and the
new target pages. The data-only audit must pass before any model call.

Each of the 289 pairs contributes two items with the same claim: one natural
SUPPORTS item and one natural REFUTES item. The formal population therefore
contains 578 exactly balanced items. Dataset labels are used only to construct
and audit this balanced design; they are not sent to either model and do not
enter pre-outcome risk or route selection.

## Models and calls

- Qwen3.5-4B at `http://127.0.0.1:31518/v1/chat/completions`;
- Ling-3.0-tiny at `http://127.0.0.1:31520/v1/chat/completions`;
- five fixed personas: literal, skeptic, consistency, counterfactual, minimal;
- four conditions per item: original, remove, reverse, substitute;
- 578 x 5 x 4 = 11,560 calls per model, 23,120 total;
- temperature 0, prompt-only JSON, strict local parser, 256-token cap;
- model-specific fresh caches and a new V3.16.1 seed protocol;
- no server-side `response_format`.

The existing Qwen service is left unchanged. Ling is started separately on an
otherwise unused GPU and local port. Endpoint model IDs must match the frozen
IDs before formal calls begin.

## Frozen risk and endpoints

V3.16.1 reuses the exact V3.16 risk manifest and weights:

```text
R_sym = 0.3 * reverse_inertia
      + 0.7 * intervention_disagreement
```

The score is not refit on the new cohort, on Ling, or on any V3.16 formal
outcome. For each model, the primary metrics are overall error-detection AUROC,
native-label macro and worst-label AUROC, and absolute error reduction at 80%
retained coverage. Five-thousand pair-bootstrap replicates use the frozen seed
`20261616`.

Each model must satisfy all of:

1. final response validity at least 0.98;
2. first-pass validity at least 0.95;
3. at least 400 high-consensus items;
4. at least 20 high-consensus errors in each native label;
5. overall AUROC CI lower bound above 0.5;
6. macro-label AUROC CI lower bound above 0.5;
7. worst-label AUROC CI lower bound above 0.5; and
8. Risk@80 error-reduction CI lower bound above 0.

The cross-family headline requires all gates for both models. Pooled data,
previous V3.16 outcomes, and favorable secondary analyses cannot rescue a
failed gate.

## Baselines and freeze order

All risk methods rank the same original-condition high-consensus errors. The
prespecified baselines are vote disagreement, low mean confidence, reverse
inertia alone, intervention disagreement alone, and deterministic hash-random
ordering. Baseline comparisons are secondary and cannot replace the gates.

The freeze order is:

1. construct and audit the remaining 289-pair selection manifest;
2. build the outcome-free public manifest and sealed outcome ledger;
3. freeze model fingerprints, endpoint IDs, prompt/call hashes, score hash, and
   protocol manifests;
4. verify endpoint models and transport on non-formal registered smoke inputs;
5. run all 11,560 formal calls per model into fresh caches;
6. freeze pre-outcome rows, high-consensus IDs, risks, and retained IDs for both
   models without reading the outcome ledger; and
7. only then evaluate labels and write the formal report.

Formal execution is all-or-nothing. No optional stopping, example replacement,
parser tuning, gate relaxation, score change, or selected-error pooling is
permitted.
