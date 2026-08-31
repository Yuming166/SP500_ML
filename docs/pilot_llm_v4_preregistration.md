# Pilot-LLM V4 preregistration: TruthfulQA, partitioned evidence, substitute intervention

**Date frozen:** 2026-08-31
**Status:** Frozen before any V4 model call. Any substantive change after V4 outputs
requires a new version.

## 1. Why a fourth pilot is needed

V3 (`docs/pilot_llm_v3_preregistration.md`) produced a negative primary result:
**causal-risk AUROC = 0.4125** ([report](results/pilot_llm_v3/formal/report.md)).
Post-hoc diagnostic ([`analysis/v3_diagnostic_report.md`](analysis/v3_diagnostic_report.md))
identified three protocol limits, not model limits:

1. **Qwen3.5-4B's parametric prior dominates StrategyQA.** Removing or mechanically
   negating evidence triggers a "fall back to prior" response, not a "use wrong evidence"
   response. Inert agents (those that ignore interventions) are *more* accurate than
   responders, inverting the preregistered risk score.
2. **V3 sends every agent the same evidence packet**, so shared-citation clustering
   is the baseline and cannot signal anything.
3. **Causal risk was defined as a single over-restrictive signal** (inert-only). The
   diagnostic showed that the union (inert OR confidence-stable) recovers a
   direction-consistent +0.144 AUROC improvement (D definition, AUROC = 0.556 vs.
   V3's 0.4125).

V4-LLM addresses all three limits without re-touching V3 outputs, seeds, or claims.

## 1.1 Scope and positioning of V4 (the central framing)

V4 is **not** a router-strength benchmark. The paper's headline contribution is
provenance-aware selective routing under correlated evidence; that claim is
premature until a more basic question is settled.

V4's pre-registered scope is the foundational one:

> **Question A:** Does Qwen3.5-4B's binary action actually respond to paired
> evidence interventions when (a) the evidence is partitioned across agents and
> (b) the intervention is a confusable wrong substitution, not just removal or
> mechanical negation?

> **Question B:** When 5 agents see overlapping (but not identical) subsets of the
> same evidence pool, does shared-citation clustering produce a detectable
> signal for false consensus — i.e., is `shared_citation_signal` non-trivially
> correlated with `harmful_fc`?

Both questions must show a non-trivial answer (not necessarily strong; not
necessarily positive; but not silent either) for V4 to be claimed as a
**ground-truth foundation** for the subsequent Qwen financial-agent experiments.
If Question A is silent, the financial-agent experiment has no footing. If
Question B is silent, the shared-citation methodology cannot be claimed.

A negative V4 outcome on either question is a valid scientific result, reported
as such. V4 is explicitly designed to **fail informatively** — to surface whether
the methodology is currently capable of answering these basic questions, not to
showcase a strong router. The downstream financial-experiment paper assumes V4's
questions have been answered; if V4 says they have not, the financial paper's
contribution narrows accordingly.

## 2. Status of prior pilots (preserved, not edited)

| Version | Outcome | Stop reason |
|---|---|---|
| Pilot-LLM V1 | Stopped at smoke | Generated StrategyQA claim can flip polarity while label is attached to original question |
| Pilot-LLM V2 | Stopped at smoke | Agent-level abstention erased false-consensus observations |
| Pilot-LLM V3 | Formal run completed (750/750) | Protocol-limited negative result (see §1); not retried |
| **Pilot-LLM V4** | **This document** | — |

## 3. Frozen model, retry, and transfer controls

- endpoint: `http://10.63.0.88:31519/v1/chat/completions`;
- model: `Qwen3.5-4B`;
- temperature: `0.0`;
- maximum completion tokens: `160`;
- timeout: 60 seconds;
- one initial request and at most one fixed JSON-repair or transport retry;
- SHA-256 content-addressed response cache;
- no model/data download and no hidden chain-of-thought request, storage, or scoring.

## 4. Domain change: StrategyQA → TruthfulQA

V4-LLM uses the local TruthfulQA JSONL at
`/storage/gaoym/argumentative-llms/Datasets/TruthfulQA/Prompt/data.jsonl`.
This avoids any remote data download and is bandwidth-neutral.

Required source SHA-256:

```text
TO BE COMPUTED AT FREEZE TIME — placeholder sha256=v4-truthfulqa-2026-08-31
```

Each TruthfulQA row is a single-fact misconception probe with a binary label
(`valid`). V4-LLM constructs **composite questions** by grouping three
TruthfulQA rows on a manual topical cluster (e.g., "geography", "health",
"common sayings"). For each cluster:

- 3 rows → 3 `claim` strings → 3 evidence items `E01`, `E02`, `E03`;
- 1 **synthesis question** is constructed by joining the three
  `original_question_answer` strings into a single multi-fact question;
- the gold label is the **majority of the three rows' `valid` values**;
  ties (1-1-1) are excluded by stratification.

The composite-question manifest is balanced: 25 positive / 25 negative.

Stratification rule: rank rows within `valid` stratum by
`SHA256("pilot-llm-v4-2026-08-31\n" + qid)`; take the first 25 per stratum × 3 rows
per composite question = **50 composite questions / 150 evidence rows / 3 fact labels
each**.

This decision is **frozen** before V4 calls.

## 5. Frozen partitioned evidence packets (V4's central design change)

For one composite question with evidence items `{E01, E02, E03}`, the **5 agents see
different 2-of-3 subsets**:

```text
agent 0: {E01, E02}
agent 1: {E01, E03}
agent 2: {E02, E03}
agent 3: {E01, E02}
agent 4: {E01, E03}
```

The partitioning is **deterministic from the manifest** (no random shuffling per
call). Every agent sees the **same question text** and the **same agent persona
text** as in V3 (`literal_evidence`, `skeptical_auditor`, `consistency_checker`,
`counterfactual_reasoner`, `minimal_judge`).

This recovers the within-question citation variance that V3 lacked and enables the
shared-citation false-consensus detector (Adjustment 6 in the V3 diagnostic).

**Partition robustness (secondary, mandatory).** V4 also pre-registers a
**permutation robustness** check: 200 random permutations of the 5-agent ↔
2-of-3-subset assignment, recomputing `D_OR` AUROC for each. Reported as median
and 5th/95th percentile of permuted AUROCs alongside the deterministic AUROC.
Full procedure in §9.4.

## 6. Frozen paired evidence conditions (4, not V3's 3)

For one (question, agent) pair, V4 runs four conditions:

| Condition | Packet | Intervention |
|---|---|---|
| `original` | agent's assigned 2-of-3 subset, verbatim | none |
| `remove`  | empty packet (no allowed citation IDs) | all evidence removed |
| `reverse` | agent's subset with each item wrapped as `Task-local counterfactual: it is false that: <item>` | per-item mechanical negation |
| `substitute` | agent's subset with **one randomly chosen item replaced** by a topic-cluster confusable wrong claim (the substitute claim is drawn from a different row in the same topic cluster and is recorded in the manifest) | confusable wrong substitution |

The `substitute` condition is the central new intervention: it forces the agent to
either (a) use the substituted (wrong) evidence or (b) explicitly recognise and
override it. Neither is possible from parametric prior alone.

**Substitute matching rule (frozen).** A substitute claim must satisfy all four:

1. **Same topic cluster** as the item it replaces (manual cluster assignment
   recorded in the manifest);
2. **Length comparable** to the original item (within ±30% token count);
3. **Same named-entity slots** as the original (e.g., swap country names but
   keep the predicate structure);
4. **Opposite gold truth**: the substitute's underlying TruthfulQA `valid` value
   is the opposite of the original's `valid` value (if the original is `valid=1`,
   the substitute is `valid=0`, and vice versa).

This rules out degenerate cases such as swapping "All Europeans speak French"
with a totally unrelated sentence, which would be trivially detectable by topic
overlap alone. The substitute manifest is reproducible from the salt and is
written to disk before any V4 calls.

Total calls per formal run:

```text
50 questions × 5 agents × 4 conditions = 1,000 calls
```

Estimated transfer: ≈ 2.2 MB (≈ 2.2 KB/call observed in V3).

## 7. Frozen agents and response contract

Identical to V3's frozen persona set and JSON contract (`docs/pilot_llm_v3_preregistration.md`
§ Frozen agents). Answer must be `yes`/`no`. Confidence in `[0, 1]`. Citations must be
unique and belong to the **agent's currently assigned packet**. Unknown fields,
abstain/null answers, agent-ID drift, or invented evidence IDs are rejected.

## 8. Frozen intervention and consensus definitions

Per-question (5 agents × 4 conditions = 20 records):

- `remove_flip`, `reverse_flip`, `substitute_flip`: 1 if answer differs from `original`;
- `inert_no_flip`: 1 if remove, reverse, AND substitute all leave the answer unchanged;
- `max_conf_drop`: max over the three interventions of `original_conf − intervention_conf`;
- `conf_stable`: 1 if `|conf_drop| < 0.05` on **all three** interventions.

Per-question outcomes (target for AUROC):

- `consensus`: majority of `original` answers across the 5 agents;
- `agreement`: majority fraction;
- `correct`: 1 if consensus equals the gold label;
- `harmful_fc`: 1 if `correct == 0 AND agreement >= 0.8`;
- `shared_citation_cluster`: count of evidence IDs cited by ≥ 2 agents.

## 9. **Frozen primary risk score** (the single substantive change from §1)

### 9.0 Why three endpoints, not one

The OR-combined primary score from the V3 diagnostic conflates two distinct
failure modes:

- an agent that **ignores all three paired interventions** (answer-inert);
- an agent that **holds its confidence within ±0.05** on all three interventions
  (confidence-stable).

A reviewer will read `D = inert OR conf_stable` and ask: "is the signal driven by
answer-inertia, by confidence-stability, or by both?" V4 pre-registers all three
simultaneously so the answer is in the report regardless of which mode Qwen3.5-4B
exhibits.

### 9.1 Frozen endpoints (all three reported together)

For each (question, agent) the four conditions (`original`, `remove`, `reverse`,
`substitute`) yield three binary signals:

```text
agent.inert_no_flip     = 1[answer_remove == answer_original
                            ∧ answer_reverse == answer_original
                            ∧ answer_substitute == answer_original]
agent.conf_stable       = 1[|conf_drop_remove|  < 0.05
                            ∧ |conf_drop_reverse| < 0.05
                            ∧ |conf_drop_substitute| < 0.05]
```

Per-question risk scores (over the 5 agents):

```text
D_inert(qid) = (1/5) * Σ_agent agent.inert_no_flip
D_conf(qid)  = (1/5) * Σ_agent agent.conf_stable
D_OR(qid)    = (1/5) * Σ_agent [agent.inert_no_flip ∨ agent.conf_stable]    ← PRIMARY
```

`D_OR` is the primary; `D_inert` and `D_conf` are **co-registered secondary
endpoints reported side-by-side**. V4's methodology claim is supported only if the
**three AUROCs are all reported**, not if only `D_OR` is reported.

### 9.2 Frozen primary hypothesis

`D_OR` AUROC > 0.5 for `harmful_fc`, with the 95% **question-cluster** bootstrap CI
lower bound above 0.5. This is the minimum bar. Passing it alone is **not**
sufficient to claim a positive methodology result.

### 9.3 Pre-registered secondary hypotheses (mandatory)

AUROC > 0.5 is a weak bar (an AUROC of 0.53 could appear significant under
call-level bootstrap, since V4 produces 1,000 calls — but the statistical unit is
the **question**, not the call). V4 pre-registers three additional secondary
hypotheses that must **all hold** alongside §9.2 for V4 to claim a robust
methodology result:

| # | Secondary hypothesis | Why it is required |
|---|---|---|
| S1 | **AUPRC(`D_OR`) > AUPRC(`D_majority`)** on the same 50 questions, where `D_majority = 1 − agreement` | Class imbalance (9/50 harmful_fc) makes AUPRC the honest metric; beats disagreement-based ranking |
| S2 | **`Risk@80%Coverage`(`D_OR`) < `Risk@80%Coverage`(`D_majority`)** | This is the metric the selective router actually consumes |
| S3 | **Calibration**: Brier(`D_OR`) and ECE(`D_OR`) both < 0.30 after Platt scaling fit on leave-one-question-out | Prerequisite for handing `D_OR` to any downstream policy layer |

Pass criteria: §9.2 passes **and** at least two of {S1, S2, S3} pass. If only
§9.2 passes, V4's contribution is reported as **direction-consistent but
under-powered**, not as a methodology success.

### 9.4 Partition robustness (secondary, mandatory)

The deterministic partitioning table in §5 is the primary design. To rule out
that V4's result is an artefact of one specific partition choice, V4 also reports
a **permutation robustness** secondary analysis: 200 random permutations of the
5-agent ↔ 2-of-3-subset assignment, recomputing `D_OR` AUROC for each. V4 reports
the median and 5th/95th percentile of the permuted AUROCs. If the permuted
AUROCs' median is below the deterministic AUROC by more than 0.05, this is
**not** a failure of the methodology — it is a finding about the partition
dependence of evidence-ignoring behaviour, and is reported as such.

## 10. Frozen shared-citation detector (Adjustment 6, now testable)

For each question, define:

```text
shared_citation_signal(qid) = (# agents that cite ≥ 1 evidence ID also cited by ≥ 1 other agent) / 5
```

This signal is registered as the **secondary detector**. It was non-functional on
V3 data (AUROC = 0.500) and is testable on V4 data because of the partitioned
packets in §5.

## 11. Frozen metrics and instrumentation gates

Reports include the V3 audit set **plus** substitute-condition breakdown:
first-pass/final schema validity, retries, cache use, request/response bytes,
tokens, latency, original individual and majority accuracy, Brier, ECE,
confidence, citations, **per-condition answer flips and confidence changes**,
paired responsiveness, evidence inertia, agreement, harmful false consensus,
shared-citation signal, causal-risk AUROC/AUPRC/AURC/Risk@80 when outcome classes
permit.

### 11.0 Statistical unit (mandatory)

**The statistical unit of inference is the question (n = 50), not the call
(n = 1,000).** V3's preregistration mentioned "1,000 question-cluster bootstrap
replicates" — V4 makes this explicit and forbids call-level bootstrap. Every
95% interval reported in V4 must use one of:

- **Question-level non**`: sample 50 questions with replacement; aggregate all
  20 calls per question into the question's summary statistics before
  recomputing the metric. Seed `20260901`. 1,000 replicates.
- **Cluster-by-topic non**`: same procedure but stratified by the manifest's
  topic cluster, to avoid one topic dominating a bootstrap sample.

**Call-level bootstrap is forbidden.** Call-level bootstrap on 1,000 calls would
inflate effective n by 20× and produce CIs that are roughly √20 ≈ 4.5× narrower
than the honest question-level CI. V3's CI was effectively of this (incorrect)
form; V4 explicitly corrects it.

### 11.1 Mandatory reporting set (per §9)

For each of `D_inert`, `D_conf`, `D_OR`:
- AUROC with 95% question-cluster bootstrap CI (seed `20260901`),
- AUPRC with 95% CI,
- `Risk@80%Coverage` with 95% CI,
- Brier score and ECE after Platt scaling (LOO fit) with 95% CI.

Plus:
- `D_majority` baseline (1 − agreement) for AUROC, AUPRC, Risk@80%Coverage.
- Per-condition flip rates (`remove_flip`, `reverse_flip`, `substitute_flip`)
  on the 50 questions, broken down by `correct == 0` vs `correct == 1`.
- Harmful false consensus prevalence and per-condition true-positive rates.
- Permutation robustness median + 5th/95th percentile over 200 permutations
  (§9.4).

The formal run passes only if:

- the dataset digest and balanced 25/25 manifest match;
- at least 98% of 1,000 calls have an HTTP response or valid cache;
- at least 95% produce a strict yes/no decision after the fixed retry;
- at least 95% of (50 × 5 = 250) question-agent quadruplets are complete (all four
  conditions returned);
- every accepted citation is packet-validated (in particular, **substitute
  citations are not in the original packet and are not in another agent's packet**).

These are operational gates, not superiority criteria.

## 12. Smoke and pre-formal checks

The only permitted pre-formal V4 smoke is **two composite questions, one agent
(`literal_evidence`), the four conditions**: 8 calls. Smoke outputs remain
separate from formal results.

The pre-formal audit must confirm:

- SHA-256 of the source JSONL matches the frozen digest;
- the balanced manifest of 50 composite questions is reproducible from the salt;
- the partitioning table in §5 reproduces byte-for-byte;
- the substitute manifest is reproducible from the salt;
- all 8 smoke calls pass instrumentation gates.

## 13. Interpretation boundary

These are still controlled, paired-intervention results on a single model
(Qwen3.5-4B) and a single non-financial domain (TruthfulQA composites). They do
not establish LLM faithfulness in general, S&P 500 predictability, investment
performance, or cross-model generalization. A negative V4 outcome (AUROC ≤ 0.5)
is a valid scientific result and will be reported as such.

## 14. Registered deviations (added 2026-08-31, before any V4 formal call)

The following deviations were locked in during pre-formal implementation.
None of them changes the preregistered hypotheses (§9.2 / §9.3 / §10).
All are recorded in the manifest and reported alongside the formal results.

| # | Item | Preregistered (§6 / §9.4) | Deviation | Why |
|---|---|---|---|---|
| D1 | Length window for substitute | ±30 % of source claim token count | ±50 %, with **nearest-length fallback** when no in-window candidate exists in the same cluster | TruthfulQA's short claims and 10-cluster partition put 5/200 items at ratio 0.14–4.29 with zero in-window candidates. A silent empty-substitute would degenerate the `substitute` condition into `original`; a fail-fast would lose 5 items. Nearest-length fallback keeps all 200 items in the manifest with the smallest possible length distortion. |
| D2 | Named-entity slot matching | Exact entity-slot match | Same-cluster topical similarity (first content word) | Out of scope for the protocol; an NER-based slot matcher is deferred to a v5 preregistration if the cluster approximation turns out to be too coarse. |
| D3 | Partition-permutation robustness (§9.4) | 200 random permutations of the 5-agent ↔ 2-of-3-subset assignment | **Leave-one-agent-out (LOAO)** median + [p05, p95] AUROC across 5 variants | V4's single-call-per-agent design cannot re-assign subsets without additional LLM calls. LOAO is the closest honest proxy from the per-agent signals already retained in each row. Reported alongside the deterministic AUROC with a deviation note in `summary.json` and `report.md`. |
| D4 | Substitute manifest yield | Implicit ("written to disk before any V4 calls") | **Hard fail-fast** if any source item has zero opposite-label same-cluster candidates | Silent skip was the previous behaviour; it produced a manifest with 195/200 substitute entries and a hidden exclusion of 5 items from composite sampling. Fail-fast surfaces the deviation at `prepare` time instead of mid-formal-run. |

## 15. Operational additions (added 2026-08-31, not protocol changes)

- `audit` subcommand: runs the pre-formal checks (digest, balance, ties, yield) and exits non-zero if any gate fails.
- `all` subcommand: `prepare → audit → smoke → formal` in one process; resumable; `--yes` removes the single confirmation prompt.
- `--no-resume` on `smoke` / `run` to ignore any `records.partial.jsonl` left from an interrupted run.
- `progress.json` written every call with `{completed, total, rate, eta, last_cqid, last_agent, last_condition, last_success}` for live polling.
- Bash driver: `scripts/run_pilot_llm_v4.sh [--yes|--skip-smoke|--skip-formal]` runs `prepare → audit → smoke` in the foreground and the formal run in the background; `scripts/wait_pilot_llm_v4.sh` blocks on it and prints the final `D_OR__harmful_fc` summary when done.