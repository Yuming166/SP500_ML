# Pilot-LLM V1 preregistration: paired evidence interventions

## Status and separation from prior experiments

This protocol is frozen on 2026-08-31 before any Pilot-LLM outcome is inspected.  The
single `Reply with exactly: OK` API connectivity check made before this document was
written is not a task example and is excluded from every result.  Pilot-LLM V1 does not
alter Synthetic V1--V4 or Historical Replay V0--V2.

If the task data, evidence transformation, prompts, agent definitions, parser, retry
policy, sample, or decision rules below need a substantive change after model outputs
are observed, the changed experiment must be named Pilot-LLM V2.  Failed and mixed V1
results remain reportable.

## Research purpose

Pilot-LLM V1 is an external-validity and instrumentation experiment.  It asks whether
real Qwen agents that cite a shared evidence packet change their observable answer or
abstain when that packet is removed or reversed.  It does not test S&P 500
predictability, investment performance, fine-tuning, or reinforcement learning.

The pilot is not powered as a confirmatory method-comparison study.  Its purposes are:

1. validate the API, cache, strict response contract, and paired-call joins;
2. measure schema validity, citation validity, answer calibration, and intervention
   sensitivity;
3. determine whether false consensus and evidence inertia occur often enough to justify
   the predeclared 200-question extension;
4. preserve negative evidence if the model cites task evidence but ignores the paired
   intervention.

## Frozen model endpoint and decoding

- OpenAI-compatible endpoint:
  `http://10.63.0.88:31519/v1/chat/completions`;
- model: `Qwen3.5-4B`;
- temperature: `0.0`;
- maximum completion tokens: `160`;
- one initial request and at most one schema-repair retry;
- request timeout: 60 seconds;
- no model download and no hidden chain-of-thought request, storage, or evaluation.

The retry appends only a fixed instruction saying that the previous response violated
the JSON contract.  It cannot add task hints or change the evidence.  Transport and
response metadata are logged without credentials.  Every request is content-addressed
by SHA-256 and cached, so an identical request is not transmitted twice.

## Frozen task data and sample

The source is the local StrategyQA Prompt split:

```text
/storage/gaoym/argumentative-llms/Datasets/StrategyQA/Prompt/data.jsonl
```

The frozen file SHA-256 is:

```text
be345e0a08ea87fb5d1642c076bfb8bb186efd1cc39f9c07a812a26eef606760
```

The file has 200 rows.  Pilot selection is deterministic and outcome-blind:

1. reject duplicate or empty `qid` values and rows without facts;
2. split rows by the existing binary `valid` label;
3. rank each label stratum by
   `SHA256("pilot-llm-v1-2026-08-31\n" + qid)`;
4. retain the first 25 `true` and first 25 `false` rows;
5. order the retained rows by the same digest and write only qids, labels, fact counts,
   and hashes to the frozen manifest.

This makes the 50-question sample label-balanced without inspecting Qwen outputs.  The
manifest and source-file digest together define the exact sample; raw dataset text is
not copied into the repository.

## Frozen evidence conditions

Each StrategyQA fact becomes an environment-owned evidence item `E01`, `E02`, and so
on.  The same IDs are retained across paired conditions.

- **original**: expose every fact verbatim as task-local evidence;
- **remove**: expose an empty packet and no allowed evidence IDs;
- **reverse**: replace every fact with
  `Task-local counterfactual: it is false that: <original fact>`.

Reverse outputs have no counterfactual correctness label.  They are used only to
measure observable answer/abstention and confidence changes relative to the original
call.  The transformation is deliberately mechanical and must be manually audited on
the frozen manifest before the 50-question run.  Pilot conclusions must report that
generic negation may not produce a logically minimal counterfactual for every item.

Call order is question, agent, then `original`, `remove`, `reverse`.  Cache keys make
the order operationally reproducible.  Each call is stateless; no earlier model answer
is included in a later prompt.

## Frozen agents and response contract

Five deterministic prompt personas receive the same claim and evidence condition:

1. `literal_evidence`;
2. `skeptical_auditor`;
3. `consistency_checker`;
4. `counterfactual_reasoner`;
5. `minimal_judge`.

Personas change only the fixed decision style.  They receive identical evidence access
and cannot see the gold label.  They return exactly one JSON object:

```json
{
  "agent_id": "literal_evidence",
  "decision": "answer",
  "answer": "yes",
  "confidence": 0.80,
  "cited_evidence_ids": ["E01"]
}
```

Allowed decisions are `answer` and `abstain`.  An answer must be `yes` or `no`;
abstention requires `answer: null` and `confidence: 0.0`.  Confidence must be finite and
in `[0, 1]`.  Citation IDs must be unique and belong to the current environment packet;
an empty list is allowed.  Unknown fields, prose outside an optional Markdown JSON
fence, invented evidence IDs, agent-ID drift, and schema/type errors are rejected.

## Frozen paired-effect definitions

For one question-agent pair, a condition is **decision-changing** if the paired output
abstains when the original answered, answers when the original abstained, or gives the
opposite yes/no answer.  It is **paired responsive** if either removal or reversal is
decision-changing.  It is **evidence-inert** if all three calls parse successfully and
neither intervention is decision-changing.

Confidence changes are reported separately and do not change the binary responsiveness
definition.  This prevents a post-outcome confidence threshold from determining the
main intervention label.

For each complete five-agent question:

- original consensus is the majority of yes/no answers;
- agreement is majority votes divided by all five agents, so abstentions reduce it;
- consensus error compares original consensus with the existing StrategyQA label;
- harmful false consensus means consensus error with agreement at least `0.8`;
- causal-effect risk is the fraction of complete agent triplets that are evidence-inert;
- the fixed V3 intervention contribution is `min(1, 0.60 * causal-effect risk)`.

No source-quality, staleness, or temporal-violation term is invented for StrategyQA.
Consequently, Pilot V1 evaluates the paired causal-effect component and must not call
the resulting score a complete market provenance score.

## Frozen metrics and uncertainty

The automatically generated report includes:

- transport/parse completion, first-pass parse success, retry use, and schema failure;
- response bytes, prompt/completion tokens when returned, latency, and cache hits;
- original answer coverage, individual accuracy, confidence, Brier score, and ECE;
- citation rate and mean number of cited packet items;
- remove/reverse answer coverage, answer-change rates, and confidence changes;
- paired responsiveness and evidence-inertia rates;
- majority coverage, accuracy, agreement, false-consensus rate, and harmful
  false-consensus rate;
- causal-effect-risk AUROC, AUPRC, AURC, and Risk@80 for original consensus error when
  both outcome classes exist.

All uncertainty intervals use 1,000 bootstrap replicates clustered by question with
seed `20260831`.  No row-level bootstrap is allowed.  Undefined one-class metrics are
written as `null`, not coerced to a favorable number.

## Instrumentation gates and interpretation

The 50-question pilot is operationally valid only if:

- the manifest contains 25 examples per label and matches the frozen source digest;
- at least 98% of the 750 requested calls obtain an HTTP response after the fixed retry;
- at least 95% yield a strictly valid decision after the fixed retry;
- at least 95% of the 250 question-agent triplets are complete;
- every accepted citation is verified against the condition-specific packet.

These are instrumentation gates, not evidence that provenance routing is superior.  A
low paired-response rate, weak risk ranking, or no false-consensus examples is a valid
negative or underpowered result and cannot be repaired by changing V1 prompts,
thresholds, personas, or examples after inspection.

Before the formal 50-question run, only the following smoke test is allowed: at most two
manifest questions, one frozen agent, and all three conditions (six calls maximum).
Smoke artifacts must remain under `results/pilot_llm_v1/smoke/` and cannot enter the
formal report.

## Predeclared next step

If instrumentation gates pass, freeze a separate confirmatory protocol before expanding
to approximately 200 questions and 3,000 calls.  That extension may compare Majority,
Confidence, Quality only, fixed V3, learned V4, and a Hybrid router that preserves the
fixed structural prior while learning calibration and a deployment threshold.  S&P 500
temporal replay is connected only after this paired-call contract is validated.
