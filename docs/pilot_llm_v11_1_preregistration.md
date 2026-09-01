# Pilot-LLM V11.1 preregistration — auxiliary length-repair amendment

**Date frozen:** 2026-09-01
**Status:** Frozen before every V11.1 model call. No V11 validation agent or
outcome record exists.

## 1. Permitted amendment

V11 froze its held-out N=200 BoolQ validation selection and single
outcome-independent primary score, then stopped during auxiliary substitute
generation at 597/600 usable rewrites. The only observed information was the
three rewrite lengths and their one-line format. V11 did not run audit, smoke,
formal agents, consensus, metrics, or outcomes.

V11.1 inherits byte-accounted copies of the exact V11 examples and 597 usable
substitute strings. It does not rerank or replace a question, change a label,
inspect validation correctness, reuse a V10.4 rewrite, or change any evaluation
prompt, score, threshold, bootstrap, or verdict.

## 2. Frozen repair rule

The three frozen V11 unusable evidence IDs are:

- `boolq-1e583402fdb107ef-e01`: over the upper window;
- `boolq-095374f01188fd3e-e01`: below the lower window;
- `boolq-2e961908ed018a35-e02`: below the lower window.

Each receives exactly one new fixed repair request in
`results/pilot_llm_v11_1/cache`, with the original question/evidence, failed
candidate, unchanged integer token bounds, opposite-answer requirement, and an
exact source-token target. There are no further model calls.

The one-line repaired candidate is processed as follows:

1. If already in the original 0.5--1.5x whitespace-token interval, retain it.
2. If short, repeatedly append the fixed neutral qualifier
   `in the described local situation.` until it first enters the interval;
   reject it if the next append would exceed the upper bound.
3. If empty, multiline, or overlong, reject it. No truncation or token deletion
   is permitted.

V11.1 advances only with exactly 600/600 usable substitutes. A failed repair
records another pre-formal abort.

## 3. Unchanged confirmation contract

All V11 design elements remain frozen: official BoolQ validation split,
N=200 balanced salted selection, zero provenance-root overlap with V10.4,
five agents, four conditions, true 40-call smoke, 4,000-call formal run, and
the single primary score

```text
R_PI = 0.1 * D_inert + 0.3 * flip_inertia + 0.6 * frac_shared
```

evaluated only among original agreement >=0.8 questions. The pass gates remain
at least 80 high-consensus questions, at least 10 wrong and 10 correct, and a
1,000-bootstrap 95% AUROC lower bound strictly above 0.5. No
`shared_weighted` field is computed.
