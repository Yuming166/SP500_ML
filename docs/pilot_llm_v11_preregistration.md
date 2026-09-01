# Pilot-LLM V11 preregistration — held-out BoolQ validation confirmation

**Date frozen:** 2026-09-01
**Status:** Frozen before every V11 model call and before any V11 validation
agent output. V10.4 is used only as a disclosed development set.

## 1. Confirmatory purpose

V10.4 completed 2,000 formal calls on BoolQ `train.parquet`, but its registered
`shared_weighted` score was invalid because it directly included outcome
correctness. V11 excludes that score completely and tests a single
outcome-independent paired-intervention risk score on the untouched official
BoolQ validation split.

The confirmatory question is:

> Among high-consensus questions, does a provenance-aware paired-intervention
> score rank wrong consensus above correct consensus with AUROC reliably above
> chance?

V11 treats V10.4 as development evidence, not as a confirmatory success.

## 2. Frozen development fit

Three pre-outcome features were computed from the frozen V10.4 records:

- `D_inert`: fraction of agents that never flip under remove/reverse/substitute.
- `flip_inertia`: one minus the mean answer-flip rate over all 5 x 3 paired
  interventions.
- `frac_shared`: fraction of agents whose original-condition cited evidence
  overlaps at least one earlier fixed-order agent.

No feature reads the BoolQ label, correctness, `harmful_fc`, or any other
outcome. On the V10.4 development set, all 66 convex combinations on the
0.1-spaced simplex were evaluated. The selected and now frozen score is:

```text
R_PI = 0.1 * D_inert + 0.3 * flip_inertia + 0.6 * frac_shared
```

It had development AUROC 0.734 among the 74 high-consensus questions (12
wrong). These development results select one formula only; they are not part
of the V11 confirmatory p-value, confidence interval, or verdict.

## 3. Held-out dataset and text-only selection

**Dataset:** `data/boolq/validation.parquet` from `google/boolq`. Its SHA-256 is
recorded by `prepare` before any model call.

The V10.1 text-only eligibility gate is inherited unchanged: first three
8--80-whitespace-token passage sentences; TF-IDF mean cosine at least 0.08 and
maximum cosine in [0.10, 0.60]; one raw question/passage provenance root; no
model-based filtering. Offline feasibility is 558 questions (346 yes, 212 no).

- Salt: `pilot-llm-v11-2026-09-01`.
- Select by salted SHA-256 independently within native labels.
- Exactly 100 yes and 100 no questions: N = 200, 600 evidence units.
- Selection is frozen before substitute generation or evaluation calls.
- No V10.4 selected example, rewrite response, or cache file is reused.

Using gold only to balance the native dataset labels is part of the frozen
sampling design; no downstream model correctness or risk outcome can affect
eligibility.

## 4. Frozen instrumentation

Five personas, the fixed 2-of-3 evidence partition, and conditions
`original`, `remove`, `reverse`, and `substitute` are inherited. Substitute
generation uses one fresh exact-source-token rewrite request per evidence unit.
For a one-line response below the unchanged 0.5--1.5x window, it applies once
the V10.4 fixed neutral suffix `in the described local situation.` and retains
the result only if it then enters that same window. There are no model repair
calls, adaptive prompts, or sample replacements. The audit requires 600/600.

The smoke stage is corrected to execute exactly 2 questions x 5 agents x 4
conditions = 40 logical calls. The formal stage is 200 x 5 x 4 = 4,000 logical
calls. Both are cache-backed and resumable.

## 5. Single primary endpoint and verdict

For each question, consensus and agreement are computed from the five original
condition answers. The primary analysis subset is fixed as agreement >= 0.8,
defined without gold. Within that subset the target is `consensus_wrong`,
computed afterward from the native BoolQ label.

The single primary endpoint is
`AUROC(R_PI, consensus_wrong | agreement >= 0.8)` with a 1,000-replicate,
question-cluster percentile bootstrap using seed 20260902. V11 passes only if:

1. there are at least 80 high-consensus questions;
2. that subset contains at least 10 wrong and 10 correct consensus questions;
3. the 95% AUROC CI lower bound is strictly above 0.5.

There is no any-passes rule and no replacement co-primary. If class/count gates
fail, V11 reports an underpowered structural boundary rather than changing the
sample or threshold.

## 6. Frozen secondary reporting

Secondary, non-confirmatory outputs include AUROC/AUPRC for `D_inert`,
`flip_inertia`, and `frac_shared` in the same high-consensus subset; all-question
`R_PI` ranking of harmful false consensus; intervention flip rates; and an
outcome-free rank router that abstains on the top 20% `R_PI` scores and reports
the retained consensus error versus the unfiltered high-consensus baseline.
Bootstrap intervals and all negative/structural results are reported.

No V11 metric named `shared_weighted` is computed or emitted. A regression
test changes gold/correctness fields while holding pre-outcome features fixed
and requires `R_PI` to remain identical.

## 7. Interpretation boundary

A pass confirms transfer from the V10.4 development split to an untouched
BoolQ validation split for this model and evidence regime. It does not establish
cross-model, cross-domain, financial, or general factuality claims. A failure
does not authorize retuning V11.
