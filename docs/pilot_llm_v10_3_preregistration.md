# Pilot-LLM V10.3 preregistration — exact-source-length BoolQ replication

**Date frozen:** 2026-09-01
**Status:** Frozen before every V10.3 model call. V10.3 does not modify or
reuse V10.1/V10.2 rewrite outputs.

## 1. Version lineage and permitted change

V10.1 and V10.2 both froze the same 100-question, 300-evidence-unit BoolQ
selection before auxiliary substitute instrumentation. V10.1 stopped because
its broad rewrite prompt produced 83 out-of-window responses. V10.2 added one
fixed repair but stopped at 292/300 usable rewrites: all eight repair outputs
were single-line but below the unchanged lower token bound. Neither version
ran an audit, smoke, formal evaluation, agent, confidence, intervention,
router, metric, or outcome stage.

V10.3 changes only the **pre-evaluation generation format contract**. The
change is based on those instrumentation feasibility counts, not on any
evaluation or routing result. Earlier rewrite strings, substitute manifests,
and cache responses are not copied into V10.3.

## 2. Frozen inherited selection

V10.3 inherits exactly the V10.1 text-only selection manifest at
`results/pilot_llm_v10_1/selection_manifest.json`, whose dataset SHA-256 is
`4f028e992c0bd4df30b9f056f4946b64f5c23028034ff0ed5ea467d8538cc623`.
It has 50 native BoolQ `yes` and 50 `no` questions, each with the deterministic
first three eligible sentence units from one raw passage. V10.3 validates and
records the parent manifest SHA-256; it does not rerank candidates or replace
an item for any rewrite or model response.

## 3. Frozen exact-length substitute contract

For a source evidence sentence containing **N whitespace tokens**, each V10.3
initial rewrite request demands one plain-text, one-line counterfactual
evidence sentence containing **exactly N whitespace tokens**. The original
0.5--1.5x token rule remains unchanged: exact N is a stricter target entirely
inside that interval. The request still requires support for the opposite
native BoolQ answer, the same topic/named entities, no new entity, and no
meta-language.

Only if that first response is absent, multiline, or outside the unchanged
0.5--1.5x interval, V10.3 sends exactly one fixed `length_repair` prompt for
the same item. The repair repeats the exact-N target and displays the failed
candidate. There are no further retries, semantic rankings, adaptive prompts,
or sample replacement. The maximum auxiliary call count is 600 (300 initial,
at most 300 repairs), and the new cache namespace is
`results/pilot_llm_v10_3/cache`.

V10.3 advances only if all 300 rewrites are usable under the original length
window. Otherwise it records a pre-formal abort. This strict all-or-nothing
audit avoids conditioning the evaluation set on generation success.

## 4. Frozen evaluation and decision rule

After a successful audit, endpoint/model, temperature zero, five fixed 2-of-3
agent packets, four intervention conditions, 40-call smoke, 2,000-call formal
run, cache-backed resumption, 1,000-question bootstrap, co-primary `D_OR` and
`shared_weighted` endpoints, baseline/router cells, correctness rule, and
structural-boundary rules are unchanged from V10.1.

The co-primary criterion passes only if at least one lower 95% bootstrap bound
for `AUROC(D_OR, harmful_fc)` or `AUROC(shared_weighted, harmful_fc)` exceeds
0.5. All results, including a boundary or negative result, will be reported
without modifying V10.3.

## 5. Scope

This is a cross-domain evidence-routing test in one controlled BoolQ regime,
not evidence of general LLM faithfulness, cross-model generalization, finance
forecasting performance, or a reversal of the FEVER structural boundary.
