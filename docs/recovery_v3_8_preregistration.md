# Recovery V3.8 preregistration: zero-shot cross-model ELAR

Protocol version: `recovery-v3.8-qwen-to-ling-elar-2026-09-02`

Status: **frozen by `recovery_v3_8 prepare` before any task-bearing Ling
formal call**. Transport checks may query `/v1/models`; the only permitted
pre-formal task call is the schema smoke on two development-only examples.

## 1. Question and non-retroactivity

V3.8 asks whether the already frozen Qwen-trained V3.7.1 entailment-ledger
action router transfers without target-model fitting to a different model
family:

> Can a router trained entirely from Qwen3.5-4B development behavior use
> Ling-3.0-tiny actions, atomic certificates, and ledgers to acquire a
> corrective provenance root and improve final answer accuracy?

V3.7.1 remains immutable. V3.8 does not alter its selection, training records,
feature model, policy thresholds, prompts, seeds, parser, metrics, gates, or
reported result. Any V3.8 change after task-bearing target-model output requires
a new version and must preserve V3.8 artifacts as an abort or result.

## 2. Frozen source router and target model

- Source model: `Qwen3.5-4B`.
- Frozen source router: `results/recovery_v3_7_1/router/manifest.json`.
- Source-router policy: ledger confidence at least `0.8`, lexical coverage at
  least `0.0`, and at most one unsupported term.
- Target model: `Ling-3.0-tiny`, served from the local
  `Ling-3.0-tiny-int4` checkpoint.
- Runtime: vLLM `0.28.0`, Torch `2.13.0`, Transformers `5.16.1`, and
  compressed-tensors `0.17.0` on GPU 4 (`NVIDIA GeForce RTX 4090`),
  8,192-token context, compressed-tensors INT4, offline checkpoint loading,
  and `--generation-config vllm`.
- Target endpoint: `http://127.0.0.1:31520/v1/chat/completions`.
- Decoding: temperature `0.0`, unchanged V3.6.2/V3.7 seeds, one unchanged
  schema-repair attempt, and `chat_template_kwargs={"enable_thinking": false}`.
- Action maximum completion tokens: inherited from the Qwen action runner.
- Atomic-certificate maximum completion tokens: `512`.
- Entailment-ledger maximum completion tokens: `768`.

The frozen manifest hashes the server script and all small checkpoint/runtime
artifacts and records every weight shard's name and byte size. Endpoint
inventory and every response must identify the served model as
`Ling-3.0-tiny`; mismatches abort rather than silently entering the analysis.

No Ling label, correctness, action outcome, annotation role, source identity,
or retrieval score may fit, calibrate, select, or gate the router. Ling is a
strictly held-out target model.

## 3. Frozen evaluation universe

V3.8 inherits the exact 400-example V3.7.1 FEVER-train formal selection and its
order. It contains 200 Supported and 200 Refuted claims and 1,200 globally
unique page roots, with zero claim or root overlap with the source-router
development folds. The same roots were previously evaluated with Qwen, enabling
a paired model comparison, but they never entered router fitting.

Every model-dependent observable is regenerated with Ling:

1. five anchor-only persona decisions;
2. complete `candidate_0`, `candidate_1`, and `both` recovery actions;
3. two single-root atomic certificates;
4. an exact-quote entailment ledger and hostile challenge for every
   proof-eligible action;
5. a pre-outcome route artifact written before correctness is constructed.

It is forbidden to run only the Qwen-routed or Qwen-wrong subset. All 400 fixed
examples execute before the target-model route result is evaluated.

## 4. Schema smoke and fail-closed behavior

The smoke uses exactly two V3.7.1 development examples, not formal examples.
It tests endpoint identity, action schema, certificate schema, ledger schema,
cache separation, and exact-quote validation. Smoke accuracy and action benefit
are not computed and cannot change the protocol.

Any action bundle that remains invalid after the inherited single repair aborts
formal completion. Certificate or ledger failures remain fail-closed to KEEP,
as in V3.7.1. A formatting-only amendment after smoke must receive a new version
before any formal call; semantic prompt or threshold changes are forbidden.

## 5. Primary estimand and gates

The primary estimand is Ling final-answer native-label macro accuracy gain of
the frozen zero-shot ELAR policy over Ling KEEP on the 400-example formal set.
The question bootstrap uses the inherited seed `20261102`, 10,000 replicates,
and a percentile 95% interval.

V3.8 passes only if every inherited V3.7.1 gate passes on Ling:

1. macro-gain 95% CI lower bound is above zero;
2. damage among high-consensus-correct KEEP cases is at most 5%;
3. gain is nonnegative in both native labels;
4. at least ten routed repairs cite the acquired annotated root; and
5. net fixes exceed KEEP and every root-budget-matched baseline.

The available-action oracle is diagnostic only. Atomic proof-only, fixed
candidate, both-candidate, retrieval-score, and deterministic hash policies are
reported under the same realized root budget and cannot replace the primary.

## 6. Cross-model analyses

The report includes target action/certificate/ledger first-pass validity,
proof eligibility, fail-closed counts, route budget, fixes, harms, annotation
support, label-stratified gains, and the paired Qwen-versus-Ling metric table.
No target-model subgroup can replace the primary result.

A pass supports only zero-shot transfer from one Qwen deployment to one Ling
checkpoint in a controlled, static Wikipedia-root environment. It does not
establish universal model invariance, publisher independence, live-search
robustness, natural error prevalence, or transfer to translation-only models.

## 7. Registered continuation

After V3.8 is complete, Fin-R1 may be evaluated in a separately frozen V3.9
replication. Hy-MT2 is translation-specialized and is excluded from the primary
cross-model recovery claim.
