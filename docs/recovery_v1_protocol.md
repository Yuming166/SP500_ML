# Recovery V1: retrospective Qwen counterfactual-recovery development

Status: **retrospective development, frozen before new recovery calls**.

This protocol adds a recovery stage after the completed Pilot-LLM V12.1
reliability experiment.  It does not alter V12.1 records, manifests, risk
weights, endpoints, or conclusions.  V12.1 has already been inspected and is
therefore development data for this new question, not an untouched validation
set.

## Question

Can a policy trained on the observable V12.1 consensus and paired-intervention
behavior choose a Qwen recovery action that produces a net increase in correct
answers, after charging corrections of already-correct answers as harms?

## Frozen stage-A gate

Stage A is the existing V12.1 contract:

```text
R_PI = 0.1 * D_inert + 0.3 * flip_inertia + 0.6 * frac_shared
```

Only the highest-risk 20% of high-consensus questions are eligible for an
automatic recovery action.  The ordering and stable tie handling are inherited
unchanged from V12.1.  Other questions retain the original consensus.

## Recovery actions

Each eligible question has `KEEP` plus three one-call Qwen3.5-4B actions.  The
three calls are collected for every high-consensus development question so that
the outcome of every candidate action is observed during development.

1. `full_evidence`: independently answer from all three original evidence
   units without seeing the old consensus or agent ledger.
2. `counter_consensus`: treat the old consensus as a hypothesis to falsify and
   answer from all three original evidence units.
3. `intervention_ledger`: adjudicate using all original evidence plus the
   outcome-independent agent answer/confidence/citation ledger from original,
   remove, reverse, and substitute conditions.

No prompt receives the native label, gold binary value, correctness, harmful
consensus indicator, or any alias of those fields.  Outputs are restricted to
`action_id`, `answer`, `confidence`, and cited original evidence IDs.

These actions reuse the one BoolQ passage root.  Recovery V1 therefore tests
counterfactual action learning but **does not test provenance-disjoint evidence
acquisition or verified source-independent repair**.  A later protocol must add
new upstream roots to test that stronger claim.

## Paired learning target

For question `q` and action `a`, define:

```text
gain(q, a) = correct(q, a) - correct(q, KEEP)
```

Thus a wrong-to-correct change is `+1`, a correct-to-wrong change is `-1`, and
all unchanged outcomes are `0`.  Separate action-specific models predict this
paired gain from only stage-A pre-outcome features.  This is an uplift target,
not a classifier trained merely to imitate the answer label.

Five deterministic source-root folds produce out-of-fold predictions for all
questions.  Bootstrap ensembles estimate a prediction spread.  The conservative
policy changes an answer only when the lower estimate of predicted gain is
positive and the upper estimate of action harm is below the frozen safety cap.

## Outcomes

The report must include:

- wrong-to-correct fixes;
- correct-to-wrong harms;
- net fixes and paired bootstrap interval;
- end-to-end accuracy after the frozen gate and recovery policy;
- action frequency and mean added calls;
- every fixed-action baseline;
- a flip-consensus diagnostic exposing answer-polarity shortcuts;
- a per-question available-action oracle, clearly marked diagnostic;
- results by native label and by original predicted answer.

No result from this retrospective V1 is confirmatory.  A later version must
freeze the policy and evaluate it on untouched source roots, ideally with a
different model family and genuinely independent evidence roots.
