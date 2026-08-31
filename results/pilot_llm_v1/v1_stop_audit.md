# Pilot-LLM V1 stop audit

## Decision

Pilot-LLM V1 stops after its predeclared six-call instrumentation smoke.  The formal
50-question, 750-call run was not started.  Smoke artifacts remain in `smoke/` and must
not be treated as research evidence.

## Reason discovered before the formal run

The local StrategyQA conversion stores an original yes/no `question`, its original
binary `valid` label, and a separately generated declarative `claim`.  Full-manifest
audit found examples where the generated claim changes the question's polarity while
the label remains attached to the original question.

For example, qid `d8bed090b4755e2f7b67` asks whether the Medieval Times knights are
**not** authentic.  Its label is `true`, while the generated claim says that the knights
**are** authentic.  Evaluating that claim against the unchanged label would mark a
correct model judgment as an error.  Other items contain more subjective claim
rewrites, so a keyword-only repair would not establish label alignment.

## Preserved smoke outcome

- endpoint/model: `http://10.63.0.88:31519/v1/chat/completions`, `Qwen3.5-4B`;
- calls: 6 of the maximum allowed 6;
- strict valid responses: 6/6 on the first attempt;
- retries: 0;
- transferred request plus response bytes: 13,728;
- complete question-agent triplets: 2/2.

One original question was answered and changed to abstention under both interventions;
the other was already an abstention in the original condition.  Two examples and one
agent are insufficient for any method claim.

## Versioned correction

Pilot-LLM V2 uses only the original yes/no `question` and its `valid` label.  It does not
expose or evaluate the generated `claim`.  V2 also uses a new selection salt and a new
manifest, so V1 outputs and inspected examples are not silently reused as a formal
sample.
