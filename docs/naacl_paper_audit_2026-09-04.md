# NAACL Paper Claim and Originality Audit

Date: 2026-09-04 (Asia/Shanghai)

Baseline before this audit: `63b4a3d` (`Integrate V3.16 transfer results into
NAACL paper`). The audit is performed on the independent
`codex/naacl-symmetric` worktree and does not modify `main`.

## Audit outcome

The paper should be presented as a protocol and measurement paper about a
specific pre-outcome object: the error risk of an already formed multi-agent
consensus under environment-controlled evidence interventions. It should not
be presented as a new debate algorithm, a new citation validator, a generic
faithfulness score, or a new LLM router.

The defensible central sentence is:

> We define and preregister an environment-controlled paired evidence-response
> audit that tests whether a high-consensus multi-agent decision remains
> behaviorally dependent on its assigned evidence, then evaluates the resulting
> risk signal against the unrevealed decision outcome.

The contribution is the estimand and its integrity protocol together: fixed
agent views, opaque environment-held evidence identities, paired intervention
responses, an outcome firewall, pre-outcome route snapshots, and adequacy gates
that remain binding after results are observed. The weighted scores are frozen
instantiations of that protocol, not claims of a new universal scoring model.

## Closest prior work and separation

| Existing direction | What it already establishes | Our non-overlapping object | Required wording |
|---|---|---|---|
| Input-intervention faithfulness: [Chaturvedi et al. 2024](https://aclanthology.org/2024.cl-1.5/) | Deletion and negation interventions test whether semantic content affects QA inference in individual language models | Multi-agent consensus error is the unit of analysis; evidence IDs and source roots are held by the environment, and paired responses become a pre-outcome risk feature | Do not claim that intervention-based faithfulness is new |
| Explanation intervention: [FaithLM](https://aclanthology.org/2026.eacl-long.177/) | Contradicting an explanation can measure and optimize explanation faithfulness | We do not score or optimize natural-language explanations; we perturb assigned evidence and rank wrong consensus before labels are revealed | Say “evidence-response audit,” not “new explanation-faithfulness method” |
| Evidence contracts: [GAVEL](https://aclanthology.org/2026.findings-acl.1789/) | Atomic subclaims, evidence binding, mechanized citation checks, and a judge improve open-book fact-checking | We do not validate sufficient quoted evidence or repair a fact-checking answer; we test whether the existing multi-agent decision changes under controlled evidence counterfactuals | Do not claim new citation validation or evidence-contract machinery |
| Selective debate/routing: [SELENE](https://aclanthology.org/2026.eacl-industry.7/) | Confidence/disagreement can decide whether to initiate debate and can improve efficiency | Our intervention signal is not a debate scheduler or confidence-only router; it audits dependence of a fixed consensus and separates ranking from deferral | Do not claim generic selective-routing novelty |
| Multi-agent conformity: [Choi et al. 2025](https://aclanthology.org/2025.findings-acl.265/) | Agents can converge toward dominant stances in social debate | We operationalize a different failure mechanism: correlated evidence roots and response behavior under evidence interventions | Treat conformity as motivation, not as our discovery |

## Claim allowlist

The main paper may claim:

1. An environment-controlled protocol for testing evidence dependence in a
   multi-agent consensus, without collecting hidden chain-of-thought.
2. A frozen, outcome-independent risk contract whose primary endpoint is error
   ranking inside a preregistered high-consensus subset.
3. A confirmatory BoolQ replication for Qwen3.5-4B (`R_PI`).
4. Strong per-model, label-symmetric VitaminC evidence for Qwen3.5-4B and
   Ling-3.0-tiny under V3.16 (`R_sym`), with the joint pass withheld because
   Ling has fewer than 20 native-label errors.
5. A reproducible boundary result: S&P 500 calibration and output-validity
   diagnostics transfer more readily than confirmatory routing gain.

The paper must not claim:

- that paired input intervention, evidence contracts, citation checking,
  selective prediction, or multi-agent conformity is individually novel;
- a universal or label-invariant truth detector;
- a confirmed V3.16 joint cross-family pass;
- answer repair, citation sufficiency, live retrieval robustness, or financial
  alpha;
- that `R_sym` is a post-formal replacement for the stronger reverse-inertia
  ablation; or
- that V3 development statistics are formal evidence.

## Reviewer attack points

### “Is this just faithfulness under another name?”

Answer: prior faithfulness work tests whether a single model's prediction or
explanation responds to manipulated semantic content. This paper tests a
different estimand: whether an already aggregated multi-agent consensus is
wrong, using environment-held source identity and paired intervention responses
available before the label. The target is consensus error risk, not explanation
quality.

### “Is this just citation validation?”

Answer: citations are one observable coordinate and are null or unstable in
several formal results. The positive signal comes from behavioral response to
remove/reverse/substitute conditions. The protocol does not certify that a quote
entails a claim and does not call an answer repaired.

### “Is this just confidence/disagreement routing?”

Answer: vote disagreement and confidence are frozen same-prediction baselines.
The intervention score is generated by controlled counterfactual evidence views;
the primary question is whether it ranks errors in fixed consensus, not whether
it schedules more debate or chooses a cheaper model.

### “Why is the V3.16 result not a pass?”

Answer: the adequacy gate was fixed before formal calls and requires at least 20
high-consensus errors in each native label per model. Ling has 17 SUPPORTS
errors. All substantive performance intervals clear their thresholds, but the
registered joint verdict remains withheld. No selected errors may be pooled into
V3.16.

## Writing actions

- Keep `R_PI` and `R_sym` as separate frozen protocol instantiations; use the
  common notation `A_q`, `y_q`, `z_q`, and “pre-outcome error ranking” across
  both.
- Replace broad “intervention-based” novelty language with “environment-held
  paired evidence-response audit” and “pre-outcome consensus-error estimand.”
- State explicitly that the paper is not proposing a new debate algorithm,
  evidence validator, explanation optimizer, or financial strategy.
- Cite the closest prior work in one compact Related Work paragraph, then spend
  the paper's space on the estimand, outcome firewall, label-symmetric control,
  and negative boundaries.
- Add one V3.16 error-tail example and one label-asymmetry diagram/table; do not
  add a new formal experiment solely to make the prose look positive.

## Decision on further experiments

Do not start S&P V3 formal calls during the audit. A V3.16.1 fresh-root
replication remains optional and must be separately registered, sized before
outcome access for the native-label event gate, and run with the identical
`R_sym`. The current paper can be submitted without it if the claim remains
bounded and the event-count failure is reported as a design boundary.
