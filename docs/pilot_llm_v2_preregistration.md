# Pilot-LLM V2 preregistration: question-aligned paired evidence interventions

## Status and reason for the new version

This protocol is frozen on 2026-08-31 after the Pilot-LLM V1 six-call instrumentation
smoke and before any V2 model call.  V1 stopped before its formal run because the local
StrategyQA conversion sometimes changes polarity when turning an original yes/no
question into a declarative `claim`, while keeping the label attached to the original
question.  The failure and V1 smoke remain preserved in
`results/pilot_llm_v1/v1_stop_audit.md`.

V2 changes exactly two experimental inputs:

1. the model receives and answers the original StrategyQA `question`; the generated
   `claim` field is neither exposed nor evaluated;
2. deterministic selection uses the new salt `pilot-llm-v2-2026-08-31`, producing a
   new balanced 50-question manifest.

All endpoint, decoding, evidence intervention, agent, parser, cache, retry, metric,
bootstrap, gate, and interpretation rules below remain the same as V1.  Any substantive
change after V2 outputs are observed requires Pilot-LLM V3.

## Frozen model and transfer controls

- endpoint: `http://10.63.0.88:31519/v1/chat/completions`;
- model: `Qwen3.5-4B`;
- temperature: `0.0`;
- maximum completion tokens: `160`;
- timeout: 60 seconds;
- at most one fixed schema-repair or transport retry;
- content-addressed SHA-256 cache for every request;
- no model/data download and no hidden chain-of-thought request or storage.

The only permitted pre-formal V2 smoke is at most two questions, one agent, and all
three evidence conditions: six calls.  Smoke artifacts stay under
`results/pilot_llm_v2/smoke/` and never enter the formal report.

## Frozen data and question-aligned sample

The local source remains:

```text
/storage/gaoym/argumentative-llms/Datasets/StrategyQA/Prompt/data.jsonl
```

Its required SHA-256 remains:

```text
be345e0a08ea87fb5d1642c076bfb8bb186efd1cc39f9c07a812a26eef606760
```

The environment loads `qid`, original `question`, binary `valid`, and `facts`.  The
generated `claim` is retained only because it exists in the source row and is never
placed in a V2 request or used by evaluation.

Selection is outcome-blind and deterministic:

1. reject duplicate/empty qids, non-boolean labels, and rows without facts;
2. stratify by `valid`;
3. rank by `SHA256("pilot-llm-v2-2026-08-31\n" + qid)`;
4. retain 25 true and 25 false questions;
5. freeze qid, label, fact count, and selection digest in the V2 manifest.

The manifest must be recomputed from the source before every run.  A source digest,
sample, label, or fact-count mismatch is a hard failure.

## Frozen evidence interventions

Each fact becomes an environment-owned `E01`, `E02`, ... item.  IDs remain paired
across conditions:

- `original`: all facts verbatim;
- `remove`: empty evidence packet and no allowed citation IDs;
- `reverse`: every fact becomes
  `Task-local counterfactual: it is false that: <original fact>`.

Reverse calls measure observable behavior change, not counterfactual accuracy.  Generic
negation is a strong intervention and may not be a logically minimal counterfactual;
that limitation must remain in every report.  Call order is question, agent, then
original/remove/reverse.  Every call is stateless.

## Frozen agents and output

The five personas are `literal_evidence`, `skeptical_auditor`,
`consistency_checker`, `counterfactual_reasoner`, and `minimal_judge`.  They differ only
in a fixed one-sentence decision style and receive identical evidence access.

They answer the original yes/no question with exactly:

```json
{
  "agent_id": "literal_evidence",
  "decision": "answer",
  "answer": "yes",
  "confidence": 0.8,
  "cited_evidence_ids": ["E01"]
}
```

`answer` decisions require yes/no.  `abstain` requires `answer: null` and
`confidence: 0.0`.  Confidence is finite and in `[0,1]`.  Citation IDs are unique and
must belong to the condition-specific packet.  Unknown fields, prose outside an
optional JSON fence, schema drift, and invented citations are rejected.  The fixed
retry adds no task information.

## Frozen paired-effect and consensus definitions

A remove or reverse output is decision-changing when it changes yes to no, no to yes,
answer to abstain, or abstain to answer relative to the original.  A complete
question-agent triplet is paired responsive if either intervention is
decision-changing; otherwise it is evidence-inert.  Confidence changes are secondary
and cannot alter this binary definition.

For a complete five-agent question, original consensus is the majority of answered
yes/no decisions.  Agreement is majority votes divided by all five agents.  Consensus
error compares this answer with `valid`; harmful false consensus requires error and
agreement at least `0.8`.  Causal-effect risk is the fraction of complete agent triplets
that are evidence-inert.  The fixed V3 intervention contribution is
`min(1, 0.60 * causal-effect risk)`.  V2 does not invent StrategyQA source-quality,
staleness, or temporal-risk values and cannot call this a complete market score.

## Frozen metrics, uncertainty, and gates

Reports include transport/cache bytes and tokens, first-pass and final schema validity,
answer coverage, accuracy, Brier, ECE, citations, condition-wise coverage, remove and
reverse decision changes, paired responsiveness, confidence changes, majority
accuracy/agreement, harmful false consensus, and causal-risk AUROC/AUPRC/AURC/Risk@80
when defined.

All intervals use 1,000 question-cluster bootstrap replicates with seed `20260831`.
One-class metrics remain null.  Formal instrumentation passes only with:

- exactly 25 questions per label and the frozen dataset digest;
- HTTP response or valid cache for at least 98% of 750 calls;
- strict valid decisions for at least 95% of calls;
- complete paired triplets for at least 95% of 250 question-agent pairs;
- packet validation for every accepted citation.

These gates do not require provenance superiority.  Low intervention response, weak
risk ranking, high abstention, or no false-consensus cases remain valid negative or
underpowered results.  V2 is an LLM instrumentation/external-validity pilot, not S&P 500
predictability or investment evidence.
