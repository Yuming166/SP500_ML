# Pilot-LLM V10.2 preregistration — BoolQ fixed-length rewrite replication

**Date frozen:** 2026-09-01
**Status:** Frozen before every V10.2 model call. V10.2 is a new protocol
version, not a modification of V10.1.

## 1. Reason for the new version

V10.1 froze a text-only, balanced 100-question BoolQ selection and then
aborted during its auxiliary substitute-evidence instrumentation. Of the 300
fixed evidence sentences, 217 first-pass rewrites satisfied the registered
0.5--1.5x whitespace-token length window and 83 were too long. No V10.1 smoke,
formal, agent, confidence, intervention, router, or outcome records were
created. The only information used to design V10.2 is that pre-evaluation
instrumentation feasibility failure.

V10.2 preserves the original length interval and replaces V10.1's single
unbounded-format rewrite request with a pre-specified, at-most-once length
repair. It does not relax a constraint, change selected samples, inspect an
evaluation result, or use any V10.1 rewrite text.

## 2. Inherited frozen selection

The V10.2 selection manifest is a byte-accounted inheritance of
`results/pilot_llm_v10_1/selection_manifest.json`:

- Parent protocol: `pilot-llm-v10.1-2026-09-01`.
- Parent dataset SHA-256:
  `4f028e992c0bd4df30b9f056f4946b64f5c23028034ff0ed5ea467d8538cc623`.
- Parent selection: exactly 50 native BoolQ `yes` and 50 `no` questions,
  each with the deterministic first three eligible same-passage evidence
  sentences.
- Parent selection salt: `pilot-llm-v10.1-2026-09-01`.

V10.2 verifies the parent manifest against the current dataset, records the
parent manifest SHA-256 in its own selection and run manifests, and fails if
the inherited examples differ. It does **not** rerank candidates or replace a
question after any rewrite or evaluation response.

## 3. Frozen two-stage substitute-evidence contract

For each of the 300 inherited evidence units, V10.2 submits one initial
counterfactual rewrite request. The request states the integer token lower and
upper bounds implied by the unchanged registered interval
`ceil(0.5 * source_tokens)` through `floor(1.5 * source_tokens)`, asks for a
single evidence sentence, preserves the topic/entity constraint, and forbids
meta text.

If and only if that response is absent, multiline, or outside the unchanged
0.5--1.5x interval, V10.2 sends exactly one fixed `length_repair` request for
that same frozen evidence unit. It contains the source sentence and failed
candidate and repeats the same opposite-answer and token-bound requirements.
There are no further retries, no adaptive prompts, and no replacement items.

The maximum auxiliary-instrumentation budget is 600 calls (300 initial plus at
most 300 length repairs). The cache namespace is
`results/pilot_llm_v10_2/cache`; no V10.1 rewrite response, cache record, or
substitute manifest is reused. A run advances only if all 300 rewrites pass the
same original length window. Otherwise V10.2 records a pre-formal abort.

## 4. Inherited evaluation protocol

After a successful substitute audit, V10.2 inherits V10.1's endpoint,
Qwen3.5-4B model, temperature zero, five fixed 2-of-3 evidence agents, four
conditions (`original`, `remove`, `reverse`, `substitute`), 40-call smoke run,
2,000-call formal run, cache-backed resumption, 1,000-question bootstrap,
co-primary `D_OR` and `shared_weighted` endpoints, and all baseline/router
reporting.

The co-primary result passes only when the 95% bootstrap lower bound for at
least one of `AUROC(D_OR, harmful_fc)` or
`AUROC(shared_weighted, harmful_fc)` exceeds 0.5. Correctness remains equality
to the native BoolQ label. The target-prevalence and single-class structural
boundary rules remain unchanged.

## 5. Interpretation boundary

V10.2 can test only the registered cross-domain claim under the inherited
same-source, related-but-distinct BoolQ sentence regime. A successful run does
not establish general LLM faithfulness, cross-model robustness, financial
predictability, or invalidate the FEVER structural boundary. A failed
instrumentation or evaluation result is reported as such without retuning this
version.
