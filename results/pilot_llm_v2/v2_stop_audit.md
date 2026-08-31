# Pilot-LLM V2 stop audit

## Decision

Pilot-LLM V2 stops after its predeclared six-call instrumentation smoke.  The formal
750-call run was not started.  This preserves V2's question-label alignment fix while
reporting that its agent-level abstention contract made the pilot uninformative.

## Frozen smoke outcome

- calls: 6/6;
- strict valid responses on first attempt: 6/6;
- retries: 0;
- transferred request plus response bytes: 13,134;
- complete question-agent triplets: 2/2;
- original answer coverage: 0/2;
- remove answer coverage: 0/2;
- reverse answer coverage: 0/2.

Both original calls cited the supplied evidence but abstained with confidence zero.
Every paired output also abstained, so answer accuracy, consensus, answer-flip
sensitivity, and causal-risk ranking were undefined.  The pipeline worked, but the
behavioral instrument did not elicit the decisions required by the research question.

## Mechanism and versioned correction

Synthetic V1--V4 agents always produce an action; the selective router subsequently
chooses whether to accept the consensus or abstain.  V2 instead allowed each LLM agent
to abstain before consensus formation.  This confounded agent behavior with the router
and created a safe default that can erase false-consensus observations.

Pilot-LLM V3 restores the layer separation: every agent must return yes or no, while
abstention remains a downstream router action.  V3 retains V2's use of the original
StrategyQA question and excludes the generated claim.  It uses a new selection salt and
manifest.  V2 outputs are not relabeled or merged into V3.
