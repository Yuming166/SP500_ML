# Pilot-LLM V3 preregistration: forced answers before selective routing

## Status and prior pilot failures

This protocol is frozen on 2026-08-31 before any V3 model call.  It preserves two prior
instrumentation failures rather than editing them after observation:

- V1 stopped because generated StrategyQA claims could change the polarity of the
  original question while retaining the original label;
- V2 used the original question and fixed label alignment, but its single smoke agent
  abstained on all six original/remove/reverse calls, leaving consensus undefined.

V3 retains V2's question-only task and changes the decision-layer contract: an agent
must answer yes or no.  Only the downstream selective router may abstain.  V3 also uses
the new salt `pilot-llm-v3-2026-08-31` and a separately frozen balanced manifest.  Any
substantive change after V3 outputs requires a new version.

## Frozen model, retry, and transfer controls

- endpoint: `http://10.63.0.88:31519/v1/chat/completions`;
- model: `Qwen3.5-4B`;
- temperature: `0.0`;
- maximum completion tokens: `160`;
- timeout: 60 seconds;
- one initial request and at most one fixed JSON-repair or transport retry;
- SHA-256 content-addressed response cache;
- no model/data download and no hidden chain-of-thought request, storage, or scoring.

The only permitted pre-formal V3 smoke is two questions, one agent, and the three paired
conditions: six calls.  Smoke outputs remain separate from formal results.

## Frozen data and sample

The source is the local StrategyQA Prompt JSONL at
`/storage/gaoym/argumentative-llms/Datasets/StrategyQA/Prompt/data.jsonl`, with required
SHA-256:

```text
be345e0a08ea87fb5d1642c076bfb8bb186efd1cc39f9c07a812a26eef606760
```

Only `qid`, original `question`, binary `valid`, and parsed `facts` enter V3.  The
generated `claim` is excluded from prompts and evaluation.  Rows are stratified by
`valid`, ranked within stratum by
`SHA256("pilot-llm-v3-2026-08-31\n" + qid)`, and the first 25 per label are frozen.
The manifest records qid, label, fact count, and selection digest.  Source or manifest
drift is a hard failure.

## Frozen paired evidence conditions

Facts are environment-owned items `E01`, `E02`, and so on.  IDs are held constant across
conditions:

- `original`: all facts verbatim;
- `remove`: an empty packet with no allowed citation IDs;
- `reverse`: every fact becomes
  `Task-local counterfactual: it is false that: <original fact>`.

The model must answer even if the packet is empty or contradictory, using its best
judgment and general knowledge when necessary.  Reverse calls do not have a
counterfactual correctness target.  Generic negation is a strong and sometimes
non-minimal intervention; this limitation remains explicit in every report.  Calls are
stateless and ordered question, agent, original/remove/reverse.

## Frozen agents and response contract

The five fixed personas remain `literal_evidence`, `skeptical_auditor`,
`consistency_checker`, `counterfactual_reasoner`, and `minimal_judge`.  Persona text
cannot instruct abstention.  All agents receive the same question and evidence access.

Each response must be exactly:

```json
{
  "agent_id": "literal_evidence",
  "answer": "yes",
  "confidence": 0.8,
  "cited_evidence_ids": ["E01"]
}
```

Answer must be yes/no.  Confidence must be finite and in `[0,1]`.  Citations must be
unique and belong to the current packet; an empty citation list is allowed.  Unknown
fields, abstain/null answers, extra prose outside an optional JSON fence, agent-ID drift,
and invented evidence are rejected.  A repaired request only repeats the JSON contract
and cannot add task information.

## Frozen intervention and consensus definitions

For one complete question-agent triplet:

- remove change: remove answer differs from original answer;
- reverse change: reverse answer differs from original answer;
- paired responsive: either change occurs;
- evidence-inert: neither change occurs.

Confidence changes are reported separately and cannot change the binary label.  For a
complete five-agent question, original consensus is majority yes/no, agreement is the
majority fraction, consensus error compares to `valid`, and harmful false consensus is
an error with agreement at least `0.8`.  Causal-effect risk is the fraction of agents
that are evidence-inert.  The fixed V3 intervention contribution is
`min(1, 0.60 * causal-effect risk)`.  No StrategyQA source-quality, staleness, or
temporal component is invented.

The selective router is evaluated after agent consensus.  V3 does not ask the LLM to
make the routing/abstention decision.

## Frozen metrics and instrumentation gates

Reports include first-pass/final schema validity, retries, cache use, request/response
bytes, tokens, latency, original individual and majority accuracy, Brier, ECE,
confidence, citations, remove/reverse answer flips and confidence changes, paired
responsiveness, evidence inertia, agreement, harmful false consensus, and causal-risk
AUROC/AUPRC/AURC/Risk@80 when outcome classes permit.

All 95% intervals use 1,000 question-cluster bootstrap replicates with seed `20260831`.
Undefined one-class metrics remain null.  The formal instrument passes only if:

- the dataset digest and balanced 25/25 manifest match;
- at least 98% of 750 calls have an HTTP response or valid cache;
- at least 95% produce a strict yes/no decision after the fixed retry;
- at least 95% of 250 question-agent triplets are complete;
- every accepted citation is packet-validated.

These are operational gates, not superiority criteria.  Low answer-flip sensitivity,
weak risk ranking, calibration failure, or no harmful false consensus is a valid
negative or underpowered result.  V3 does not establish S&P 500 predictability,
investment value, universal LLM faithfulness, or cross-model generalization.
