# NAACL Adversarial Review

Date: 2026-09-05 (Asia/Shanghai)

Review baseline: `b7144c5` (`Audit NAACL paper claims and originality boundary`).
This review treats the integrated manuscript as a submission candidate rather
than as a progress report.

## Overall assessment

The paper is viable as a bounded evaluation/methodology paper, but it is not
viable if presented as a new debate algorithm, a general faithfulness method,
or a successful routing system. The main acceptance risk is narrative
misclassification: the ingredients are familiar, while the paper's distinct
object is easy to miss unless the estimand is stated before the literature
comparisons.

The paper's independent contribution should remain:

> an outcome-firewalled audit of whether an already formed multi-agent
> consensus error can be ranked before the label is revealed, using paired
> responses to evidence conditions whose identities are controlled outside the
> model.

This is a protocol and estimand claim. It is not a claim that the individual
interventions, citations, evidence contracts, confidence signals, or abstention
mechanisms are new.

## Major reviewer attacks

### 1. “This is existing faithfulness under another name.”

Risk: high.

Prior work already uses deletion, negation, and contradiction to test semantic
or explanation faithfulness. The manuscript now cites the closest examples and
states the distinction: those works target a model prediction or explanation;
this paper targets the error of a fixed multi-agent consensus, with external
evidence identity and a pre-outcome ranking endpoint.

Required response: do not use “new intervention-based faithfulness method” or
imply that intervention itself supplies novelty. Use “paired evidence-response
audit” and “consensus-error estimand.”

### 2. “The intervention proves what the model internally used.”

Risk: high.

It does not. Invariance can mean robust inference, evidence irrelevance, or a
failure to process the intervention. A changed answer can reflect an artifact
of the transformed text. The formal result only shows predictive association
between observable intervention behavior and later consensus error.

Required response: explicitly call the test operational rather than causal;
state that it does not identify internal evidence use or entailment.

### 3. “V3.16 is a post-hoc rescue of the BoolQ label failure.”

Risk: high.

V3.16 was motivated by the observed BoolQ/V3.15.2 asymmetry and is therefore a
new registered protocol, not an independent replication of the old score. Its
selection uses balanced natural contrastive pairs, separate roots, and a new
frozen `R_sym`. That genealogy must be visible. The paper must not merge V3.15.2
and V3.16 into one PASS.

Required response: report the sequence as diagnosis → new protocol → bounded
cross-family result; keep `R_PI` and `R_sym` separate; disclose that transfer is
directional from Qwen development to Qwen/Ling formal evaluation.

### 4. “Why is the reported V3.16 result not a pass?”

Risk: medium-high.

The answer is the frozen feasibility gate: Ling has 17 high-consensus SUPPORTS
errors, below the required 20. All substantive performance intervals clear
their thresholds, but the joint gate is conjunctive. This is an adequacy result,
not an outcome to be repaired by adding three examples.

Required response: show the event-count rule, label-specific counts, per-model
status, and no-pooling rule in the main text.

### 5. “The label-symmetric construction still hides leakage.”

Risk: medium.

SUPPORTS/REFUTES labels necessarily enter pair construction and balance checks.
That is allowed design information, but the manuscript must distinguish it from
the unrevealed outcome used to evaluate consensus error. Model calls, risk
coordinates, route IDs, and the evaluator's pre-outcome snapshot must not see
the labels.

Required response: state this separation explicitly in the dataset and protocol
sections; never call the item construction “label-free.”

### 6. “The score is just confidence/disagreement routing.”

Risk: medium.

Confidence and vote disagreement are same-prediction baselines. V3.16 reports
their overall AUROCs as 0.601/0.512 for Qwen and 0.568/0.522 for Ling, versus
`R_sym` 0.808/0.761. The distinction is that the primary signal comes from
paired evidence conditions, not from confidence or debate scheduling.

Required response: include the matched baseline numbers and say that the paper
does not propose a debate scheduler or model-selection router.

### 7. “Reverse inertia is the real method, so the composite is cherry-picked.”

Risk: medium.

Reverse inertia is stronger descriptively in V3.16 (0.855 Qwen, 0.837 Ling),
but it was not the registered composite. Promoting it would be post-formal
tuning. The paper should treat this as a development-to-formal shift and a
limitation, not as an improved method version.

### 8. “There are too many experiments and too many chances to find a result.”

Risk: medium.

The paper has one primary endpoint per frozen protocol, while audits,
subgroups, components, S&P calibration, and recovery results are secondary or
boundary evidence. The main text must say this clearly and avoid pooled
significance or a single aggregate score across protocol versions.

Required response: keep the evidence-chain table, identify every result as
primary/secondary/diagnostic, and move answer-repair experiments to the
supplement rather than expanding the headline.

## Claim decision

### Allowed as main claims

- Environment-controlled paired evidence-response auditing for fixed
  multi-agent consensus error.
- Outcome-independent risk ranking with a frozen primary endpoint and explicit
  pre-outcome route snapshot.
- Qwen BoolQ `R_PI` replication.
- Strong per-model but not joint-pass V3.16 Qwen/Ling evidence under a balanced
  natural contrastive construction.
- A reproducible boundary in which calibration and protocol diagnostics are
  more stable than confirmatory temporal routing gain.

### Must remain out of the headline

- Universal factuality, label-invariant reliability, or bidirectional transfer.
- Internal causal evidence-use identification.
- New evidence-contract or citation-validation machinery.
- Universal answer repair or financial alpha.
- A confirmed V3.16 joint cross-family pass.
- Formal status for the S&P V3 development branch.

## Action taken in this review

- State the distinct estimand before related-work comparisons.
- Replace causal-sounding “potential outcomes” with paired intervention
  responses/conditions.
- Add an operational-not-causal intervention limitation.
- Make V3.16 directional, label-separated, and baseline-compared.
- Keep the V3.16 event-count failure binding and preserve the no-pooling rule.
- Retain the close-work citations only to establish boundaries, not as the
  paper's conceptual scaffold.

No new formal experiment is authorized by this review. A fresh-root V3.16.1
replication remains a separate future protocol decision.
