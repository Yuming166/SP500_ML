# Pilot-LLM V10.4 preregistration — deterministic short-rewrite normalization

**Date frozen:** 2026-09-01
**Status:** Final auxiliary-instrumentation version, frozen before all V10.4
model calls.

## 1. Version boundary

V10.1, V10.2, and V10.3 never reached their evaluation stages. They froze the
same text-only 100-question BoolQ selection but stopped in substitute-evidence
instrumentation: V10.1 had 83 broad-prompt length failures; V10.2 had 8
short-length failures after a single fixed repair; V10.3 had 5 one-line
initial/repair failures, each one or two tokens below the unchanged lower
bound. No agent answer, confidence, intervention, router, metric, or outcome
was available when V10.4 was designed.

V10.4 is the final version for this auxiliary feasibility issue. It does not
change dataset, selected examples, labels, original token window, endpoint,
evaluation prompts, metric, or verdict. It does not read or reuse any prior
version's rewrite text, cache record, substitute manifest, or evaluation
record.

## 2. Frozen inherited selection

V10.4 inherits and validates exactly
`results/pilot_llm_v10_1/selection_manifest.json`: 50 native BoolQ `yes` and
50 `no` questions, three deterministic eligible sentences per question, and
dataset SHA-256
`4f028e992c0bd4df30b9f056f4946b64f5c23028034ff0ed5ea467d8538cc623`.
The V10.4 manifest records the parent manifest SHA-256 and fails if any
example differs. There is no re-ranking, response-dependent eligibility, or
replacement sample.

## 3. Frozen substitute rule

Each of the 300 inherited evidence sentences receives exactly one new
counterfactual rewrite call in the fresh namespace
`results/pilot_llm_v10_4/cache`. The request asks for a one-line, plain-text
sentence with exactly the source sentence's whitespace-token count; this exact
target is inside the inherited 0.5--1.5x length interval.

The parser accepts only a one-line result. It then applies the following fully
deterministic, response-local rule, before any evaluation call:

1. If the candidate already lies in the original 0.5--1.5x interval, retain it
   unchanged.
2. If it is below the original lower bound, remove only terminal sentence
   punctuation, append the fixed neutral qualifier `in the described local
   situation.`, and retain the result only if it now lies in the same original
   interval.
3. If it is multiline, empty, over the upper bound, or still outside the
   original interval after that one fixed append, mark it unusable.

The qualifier introduces no named entity or BoolQ answer and does not alter
the candidate proposition; it only qualifies the assertion to the supplied
task-local scenario. There are no model repair calls, additional prompts,
semantic scores, adaptive transformations, or retries. The audit requires
300/300 usable units or records a pre-formal abort.

## 4. Frozen evaluation and interpretation

On a successful audit, V10.4 inherits V10.1's fixed Qwen3.5-4B endpoint,
temperature zero, five 2-of-3 agent packets, four conditions, 40-call smoke,
2,000-call formal run, 1,000-replicate bootstrap, `D_OR` and
`shared_weighted` co-primary endpoints, all baseline/router cells, and
structural-boundary reporting. The criterion remains: at least one co-primary
AUROC 95% bootstrap lower bound must exceed 0.5.

This version tests only the controlled BoolQ cross-domain routing claim. It
does not establish general LLM faithfulness, cross-model robustness, financial
forecasting, or overturn the FEVER boundary.
