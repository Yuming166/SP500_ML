# Pilot-LLM V5 preregistration: FEVER cross-domain replication + co-primary shared_weighted

**Date frozen:** 2026-08-31
**Status:** Frozen before any V5 model call. Any substantive change after V5 outputs
requires a new version.

## 1. Why a fifth pilot is needed

V4 (`docs/pilot_llm_v4_preregistration.md`) recovered a positive primary result on
TruthfulQA composites: **`D_OR` AUROC = 0.676 [0.515, 0.821]** ([report](results/pilot_llm_v4/formal/report.md)),
plus §11.4's discovery that **`shared_weighted` AUROC = 0.785 [0.665, 0.897]** is
the strongest signal in the V4 packet. V4's §12 scorecard confirms the methodology
on a single domain (TruthfulQA) and a single model (Qwen3.5-4B).

V5 addresses the next two layers of external validity that an ACL reviewer will
demand, without re-touching V4 outputs, seeds, or claims:

1. **Cross-domain replication.** V4's signal lives on TruthfulQA composites
   (binary misconception probes, English factual claims). The strongest
   falsification of "this is a real provenance signal" would be a different
   domain with a different surface form. V5 picks **FEVER** — Wikipedia-claim
   fact verification with a publicly specified label space — because (a) it is
   the most-cited binary-label verification benchmark, (b) its claim surface is
   Wikipedia-style declarative (structurally distinct from TruthfulQA's
   misconception probes), and (c) it ships with explicit evidence sentences,
   which we need for the substituted-intervention design (see §6).
2. **Pre-registered co-primary detector.** §11.4 of `work_log_4.md` showed that
   the diagnostic `shared_weighted` detector outperforms V4's preregistered
   `shared_agents` detector by +0.188 AUROC. Reporting `shared_weighted` as a
   *post-hoc* addition in V5 would invite a "garden of forking paths"
   objection. V5 **promotes `shared_weighted` to a co-primary endpoint alongside
   `D_OR`**, with Bonferroni-style verdict logic spelled out in §9.2. This
   protects the integrity of the V5 finding whether `shared_weighted` ends up
   higher or lower than V4's 0.785 on FEVER.

V5 inherits V4's frozen evidence-partitioning (§5) and four-condition protocol
(§6) without modification, except for the substitute-condition matching rule
which moves to an LLM-rewritten evidence sentence (§6.3, deviation `D1_v5`).

## 2. Status of prior pilots (preserved, not edited)

| Version | Outcome | Stop reason |
|---|---|---|
| Pilot-LLM V1 | Stopped at smoke | Generated StrategyQA claim can flip polarity while label is attached to original question |
| Pilot-LLM V2 | Stopped at smoke | Agent-level abstention erased false-consensus observations |
| Pilot-LLM V3 | Formal run completed (750/750) | Protocol-limited negative result (AUROC = 0.4125); not retried |
| Pilot-LLM V4 | Formal run completed (1,000/1,000) | §9.2 + 2/3 secondaries passed; `D_OR` 0.676; `shared_weighted` 0.785 (discovered post-hoc) |
| **Pilot-LLM V5** | **This document** | — |

## 3. Frozen model, retry, and transfer controls

Inherited verbatim from V4 §3:

- endpoint: `http://10.63.0.88:31519/v1/chat/completions`;
- model: `Qwen3.5-4B`;
- temperature: `0.0`;
- maximum completion tokens: `160`;
- timeout: 60 seconds;
- one initial request and at most one fixed JSON-repair or transport retry;
- SHA-256 content-addressed response cache;
- no model/data download and no hidden chain-of-thought request, storage, or scoring.

**V5-only addition:** the substitute-condition LLM rewrite (see §6.3) uses the
**same** endpoint with `temperature = 0.0` and `maximum completion tokens = 200`
(one rewrite per source claim, written to the substitute manifest before any
formal run).

## 4. Domain change: TruthfulQA composites → FEVER (binary-only)

### 4.1 FEVER label space and binary mapping

FEVER ships three labels: `SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO` (NEI).
V5 uses the **binary mapping** (decision registered with the user on
2026-08-31):

```text
V5 keeps only rows with label ∈ {SUPPORTS, REFUTES}; NEI rows are excluded
entirely from the manifest.
```

Rationale: NEI is a "no-evidence" label whose harm mode ("agents over-commit to
an unsupported claim") is qualitatively different from SUPPORTS/REFUTES errors
("agents over-commit to a wrong label"). Mixing them would dilute the
`harmful_fc` definition; excluding them gives a clean, well-defined binary
outcome target that matches the V4 protocol's premise.

After exclusion:

```text
gold_label(qid) = SUPPORTS  → 0   (treated as "negative" for harmful_fc)
gold_label(qid) = REFUTES   → 1   (treated as "positive" for harmful_fc)
```

The choice of REFUTES = 1 makes `harmful_fc` (agents majority-vote the
*incorrect* label with agreement ≥ 0.8) operational as
`majority == SUPPORTS AND gold == REFUTES`. This keeps the definition
mechanically identical to V4 §8.

### 4.2 FEVER source data (acquisition strategy: open question)

As of freeze date, FEVER is **not** present on the jump host:

```text
/storage/gaoym/sp500-forecastability-lab/data/fever/   ← does not exist
```

V5 records three acceptable acquisition strategies; the chosen one is locked
in §14 `D6_v5` at freeze:

**Decision registered 2026-08-31:** **Option A** — FEVER gold-evidence validation split
from the `copenlu/fever_gold_evidence` HuggingFace mirror, sourced from the
original FEVER 1.0 release (Thorne et al., 2018). The HF mirror was selected over
the official FEVER 2.0 S3 bucket because the jump-host whitelist allows
`hf-mirror.com` but blocks `*.amazonaws.com` and `raw.githubusercontent.com`
(see acquisition log below). The HF mirror ships a `valid.jsonl` with the
exact `(claim, label, evidence)` structure V5 needs (§4.1, §4.2), pre-annotated
with gold evidence sentences — saving the entire "relabel from NEI-merged"
pass that FEVER 2.0's `shared_task_dev.jsonl` would have required.

Source landed at:

```text
/storage/gaoym/sp500-forecastability-lab/data/fever/fever-validation.jsonl
```

Source SHA-256 (frozen 2026-08-31, computed at freeze time):

```text
5da0ccc0ccf77f974611de13f8aac6f78c6bba6293912835099eb6029baa85d9  fever-validation.jsonl
```

Dataset statistics (validated at freeze):

| Metric | Value |
|---|---|
| File size | 6,471,611 bytes (6.2 MB) |
| Rows | 15,935 |
| SUPPORTS | 4,638 |
| REFUTES | 4,887 |
| NOT ENOUGH INFO | 6,410 (excluded per §4.1) |
| Field set per row | `claim`, `label`, `evidence[entity, sent_idx, sentence]`, `id`, `verifiable`, `original_id` |
| V5 manifest budget (binary-only, balanced 25/25) | max(4638, 4887) / 3 = 1,546 candidate clusters ≥ SUPPORTS×3, 1,629 candidate clusters ≥ REFUTES×3 |

Acquisition log (single-shot download, 2026-08-31):

1. `mkdir -p /storage/gaoym/sp500-forecastability-lab/data/fever`
2. `curl -fL --progress-bar -o fever-validation.jsonl https://hf-mirror.com/datasets/copenlu/fever_gold_evidence/resolve/main/valid.jsonl`
3. `sha256sum fever-validation.jsonl > SHA256`
4. Label-distribution and field-name sanity checks passed (see "Dataset
   statistics" table above).

The SHA-256 line written in step 3 is recorded both above and in §14 `D6_v5`.

Each row of FEVER validation is a `(claim, gold_label, evidence_sentence_id)`
triple with no decomposition. V5 does **not** use FEVER's multi-hop evidence
chains; it uses the first annotated evidence sentence as the single evidence
item per claim.

### 4.3 Manifest construction (50 composite questions, balanced, binarized)

Each V5 composite question groups three FEVER rows on a **claim-cluster** tag
(provided by FEVER's wiki-page grouping: claims mentioning the same Wikipedia
title cluster). For each cluster of size ≥ 3 within the binary-only FEVER
validation set:

- 3 rows → 3 `claim` strings + 3 evidence sentences → 3 evidence items `E01`,
  `E02`, `E03`;
- the **synthesis question** is constructed by concatenating the three claims
  into a single triple-fact verification prompt:
  > "Verify the following three claims about `<cluster_title>`: (1) ...
  > (2) ... (3) ... For each, respond yes if supported and no if refuted."
- the gold label for the composite question is the **unanimous majority** of
  the three rows' `gold_label` (SUPPORTS = 0, REFUTES = 1) — **only 3-0 or 0-3
  composites are kept**; 2-1 splits are dropped to avoid post-hoc tie-breaking
  (deviation `D2_v5`, see §14).

The composite-question manifest is balanced: 25 positive (gold = 1) / 25
negative (gold = 0).

Stratification rule: rank rows within `gold_label` stratum by
`SHA256("pilot-llm-v5-2026-08-31\n" + qid)`; take the first 25 per stratum ×
3 rows per composite question = **50 composite questions / 150 evidence rows
/ 3 fact labels each**.

This decision is **frozen** before V5 calls.

## 5. Frozen partitioned evidence packets (inherited from V4 §5)

Identical to V4 §5 with one modification: each evidence item is now a single
**FEVER evidence sentence** rather than a TruthfulQA `original_question_answer`.

For one composite question with evidence items `{E01, E02, E03}`, the **5 agents
see different 2-of-3 subsets**:

```text
agent 0: {E01, E02}
agent 1: {E01, E03}
agent 2: {E02, E03}
agent 3: {E01, E02}
agent 4: {E01, E03}
```

The partitioning is **deterministic from the manifest** (no random shuffling
per call). Every agent sees the **same question text** and the **same agent
persona text** as in V4 (`literal_evidence`, `skeptical_auditor`,
`consistency_checker`, `counterfactual_reasoner`, `minimal_judge`).

**Partition robustness (secondary, mandatory, inherited from V4 §9.4).** LOAO
median + [p05, p95] AUROC across 5 variants is reported alongside the
deterministic AUROC, with the same deviation note format as V4 (`D3_v4`).

## 6. Frozen paired evidence conditions (4, with substitute rewritten)

For one (question, agent) pair, V5 runs four conditions:

| Condition | Packet | Intervention |
|---|---|---|
| `original` | agent's assigned 2-of-3 subset, verbatim | none |
| `remove`  | empty packet (no allowed citation IDs) | all evidence removed |
| `reverse` | agent's subset with each item wrapped as `Task-local counterfactual: it is false that: <item>` | per-item mechanical negation |
| `substitute` | agent's subset with **one randomly chosen item replaced** by an LLM-rewritten evidence sentence (see §6.3) | confusable wrong substitution |

The `substitute` condition is the central cross-domain design. On TruthfulQA
(V4 §6), the substitute was a topic-cluster confusable wrong claim drawn from
another row. On FEVER, the natural analogue is to take the **same evidence
sentence** but rewrite it to support the **opposite** gold label — a "negative
paraphrase" of the original evidence. This is harder for the agent to detect
than a same-cluster swap (the surface form is closer to the original) and
matches V4's intent more faithfully than a topically-related-but-unrelated
swap would.

### 6.3 Substitute generation rule (frozen, deviation `D1_v5`)

For each source evidence item `E` with gold label `L` (SUPPORTS or REFUTES),
the substitute is generated by calling the same `Qwen3.5-4B` endpoint with the
following prompt:

```text
You will see a single sentence that was used as evidence for a claim that
the following statement is {L}: "{original_claim}".

Rewrite the evidence sentence so that it would support the OPPOSITE
verdict (i.e., evidence that the statement is {opposite_of_L}), while
keeping the sentence about the same topic and entity.

Constraints:
- keep the sentence length within ±50% of the original token count
- do not introduce new entities not present in the original
- do not include meta-language like "rewritten" or "opposite"
- respond with a single sentence, no bullet points

Original evidence sentence:
{source_evidence_sentence}
```

The rewrite is generated **once per source evidence sentence** (not per
agent, not per condition), written to the substitute manifest before any
formal run, and reused for all 5 agents that need it. This is the
nearest-equivalent of V4's "substitute manifest written to disk before any
V4 calls" principle (§6 of V4 preregistration).

**Hard fail-fast** rules for substitute generation:

1. If the LLM rewrite response fails JSON/schema validation after the fixed
   retry, the substitute manifest entry for that item is **marked unusable**
   and the composite question that depends on it is **dropped** from the
   manifest (preserving balance via the §4.3 salt).
2. If more than 10% of source items are marked unusable after the
   substitution-generation pass, V5 is **stopped** before any formal run —
   this threshold guards against a generation failure mode.

The substitute generation pass itself consumes **150 LLM calls** (one per
source evidence sentence) plus retries. This is reported separately in the
audit (`substitute_generation_stats`) and does **not** count against the
1,000-call formal run budget.

### 6.4 Total call budget

```text
Manifest construction (offline):          0 calls
Substitute generation pass:               ≤ 150 calls + retries (separate budget)
Formal run:                               50 × 5 × 4 = 1,000 calls
Estimated transfer (formal):              ≈ 2.5 MB
Estimated transfer (substitute-gen):      ≈ 0.3 MB
```

## 7. Frozen agents and response contract

Identical to V4 §7: agents must answer `yes`/`no`. Confidence in `[0, 1]`.
Citations must be unique and belong to the **agent's currently assigned packet**
(after intervention). Unknown fields, abstain/null answers, agent-ID drift, or
invented evidence IDs are rejected.

For the FEVER verification prompt, the agent's `yes`/`no` answer corresponds to
"the claim is supported" / "the claim is refuted". The agent answers **per
claim** (three answers per packet), but V5 records only the **majority of the
three per-claim answers** as the question-level answer. If the per-claim
answers are 1-1-1 (split), the question-level answer is recorded as `null` and
the row is rejected by `validate_records` (same rule as V4 §7 for ties).

## 8. Frozen intervention and consensus definitions

Inherited verbatim from V4 §8, with one clarification for FEVER:

- `consensus`: majority of **original-condition** per-question answers across
  the 5 agents (each agent contributes one question-level answer per §7);
- `agreement`: majority fraction;
- `correct`: 1 if `consensus == gold_label`;
- `harmful_fc`: 1 if `correct == 0 AND agreement >= 0.8`;
- `shared_citation_cluster`: count of evidence IDs cited by ≥ 2 agents.

## 9. **Frozen primary risk scores** (D_OR + shared_weighted, co-primary)

### 9.0 Why two co-primary endpoints, not one

V4 §11.4 demonstrated that `shared_weighted` (a detector not preregistered in
V4) reaches AUROC = 0.785 [0.665, 0.897] on V4 data — a stronger signal than
the preregistered primary `D_OR`. V5 promotes `shared_weighted` to a
**co-primary** alongside `D_OR` so that:

- if both signals hold on FEVER, the cross-domain claim is robust;
- if one signal holds and the other regresses, V5 reports the asymmetry
  explicitly (no post-hoc cherry-picking);
- the preregistration explicitly forbids the analyst from declaring a winner
  between the two; both are reported with CIs and both must clear §9.2.

### 9.1 Frozen co-primary endpoints

**Endpoint 1 — `D_OR(qid)`** (inherited from V4 §9.1):

```text
D_OR(qid) = (1/5) * Σ_agent [agent.inert_no_flip ∨ agent.conf_stable]
```

with the same `inert_no_flip` and `conf_stable` definitions as V4 §9.1.

**Endpoint 2 — `shared_weighted(qid)`** (promoted from V4 §11.4 S4):

```text
let:
  frac_shared(qid)     = (# agents that cite ≥ 1 evidence ID also cited by
                          ≥ 1 other agent) / 5
  correct_consensus(qid) = 1[consensus(qid) == gold_label(qid)]

shared_weighted(qid) = frac_shared(qid) * (1 - correct_consensus(qid))
                    + 0.5 * frac_shared(qid) * correct_consensus(qid)
```

i.e., shared-citation proportion, weighted by whether the consensus was wrong
(full weight) or right (half weight).

### 9.2 Frozen co-primary hypothesis

Both `D_OR` and `shared_weighted` AUROC > 0.5 for `harmful_fc`, with each
metric's 95% **question-cluster** bootstrap CI lower bound above 0.5. **§9.2
passes only if both endpoints clear the bar.** If only one clears, V5 reports
the result as "direction-consistent on one co-primary" and discusses the
asymmetry.

V5 reports 95% CIs (not Bonferroni-corrected CIs) on each metric individually;
the Bonferroni-style correction is in the **verdict logic** (both must pass),
not in the CI width. This matches the user's pre-registration decision on
2026-08-31: "95% CI + 检验说明". Rationale: CIs are summary statistics; the
verdict bar is a separate inference decision. ACL reviewers will read the
"both clear" verdict directly without re-deriving the Bonferroni math.

### 9.3 Pre-registered secondary hypotheses (mandatory, inherited from V4 §9.3)

| # | Secondary hypothesis | Why it is required |
|---|---|---|
| S1 | **AUPRC(`D_OR`) > AUPRC(`D_majority`)** on the same 50 questions | Class imbalance (FEVER binarized prevalence ≈ 50%) makes AUPRC the honest metric; beats disagreement-based ranking |
| S2 | **`Risk@80%Coverage`(`D_OR`) does not exceed prevalence baseline by more than 0.05** | Reformulated from V4's S2 — the D_majority-vs-D_OR asymmetry (§2.1 of `work_log_4.md`) means the V4 S2 wording is mechanically broken; the corrected S2 reports whether D_OR rank-orders within the harmful majority rather than trivially separating the rare class |
| S3 | **Calibration**: Brier(`D_OR`) and ECE(`D_OR`) both < 0.30 after Platt scaling fit on leave-one-question-out | Prerequisite for handing `D_OR` to any downstream policy layer |
| S4 | **Cross-domain replication**: `D_OR` AUROC on FEVER ≥ 0.5 AND `shared_weighted` AUROC on FEVER ≥ 0.5 | Soft pre-registration of the headline V5 claim; if both co-primaries fall below 0.5 on FEVER, V5 reports the methodology as failing to generalize and §9.2 fails on that basis |

Pass criteria: §9.2 passes (both co-primaries clear) **and** at least two of
{S1, S2, S3} pass. If §9.2 passes and S4 fails, V5 reports the cross-domain
finding as "direction-consistent but below the preregistered bar on FEVER".

### 9.4 Partition robustness (secondary, mandatory, inherited from V4 §9.4)

LOAO median + [p05, p95] AUROC across 5 variants, reported for both
co-primaries. If the LOAO median for either co-primary is below the
deterministic AUROC by more than 0.05, this is **not** a methodology failure
— it is reported as a finding about partition dependence.

## 10. Frozen shared-citation detectors (Adjustment 6 + S4, both reported)

For each question:

- `shared_agents(qid)` (V4 preregistered): # agents that cite ≥ 1 evidence ID
  also cited by ≥ 1 other agent, divided by 5;
- `shared_weighted(qid)` (V5 promoted co-primary, see §9.1);
- `shared_count_total(qid)`: total # of distinct evidence IDs cited by ≥ 2
  agents (raw count, not normalized);
- `shared_id_count(qid)`: # of distinct evidence IDs cited at all across the
  5 agents' original-condition answers (denominator for shared fractions).

All four are reported in `report.md` with AUROC and 95% CI; only
`shared_weighted` carries §9.2 co-primary status.

## 11. Frozen metrics and instrumentation gates

### 11.0 Statistical unit (mandatory, inherited from V4 §11.0)

**The statistical unit of inference is the question (n = 50), not the call
(n ≈ 1,000).** Every 95% interval reported in V5 uses question-level bootstrap
(seed `20260902`, 1,000 replicates, stratified by FEVER cluster). Call-level
bootstrap is forbidden.

### 11.1 Mandatory reporting set (per §9)

For each of `D_OR` and `shared_weighted`:
- AUROC with 95% question-cluster bootstrap CI,
- AUPRC with 95% CI,
- `Risk@80%Coverage` with 95% CI,
- Brier score and ECE after Platt scaling (LOO fit) with 95% CI.

For `D_inert`, `D_conf` (secondary, for traceability with V4):
- AUROC with 95% CI.

For each of the four shared-citation detectors (V4 §11.4 + §10 of this doc):
- AUROC with 95% CI on `harmful_fc`.

Plus:
- `D_majority` baseline (1 − agreement) for AUROC, AUPRC, Risk@80%Coverage.
- Per-condition flip rates (`remove_flip`, `reverse_flip`, `substitute_flip`)
  on the 50 questions, broken down by `correct == 0` vs `correct == 1`.
- Harmful false consensus prevalence and per-condition true-positive rates.
- LOAO robustness median + 5th/95th percentile over 5 variants (both
  co-primaries).
- Substitute-generation pass statistics: number of items rewritten, number
  marked unusable, number of composites dropped as a result, JSON-validation
  failure rate.

The formal run passes only if:
- the FEVER dataset digest matches the frozen SHA-256;
- the balanced manifest of 50 composite questions is reproducible from the salt;
- the partitioning table in §5 reproduces byte-for-byte;
- the substitute manifest is reproducible from the salt;
- the substitute-generation pass left < 10% items unusable;
- at least 98% of 1,000 calls have an HTTP response or valid cache;
- at least 95% produce a strict yes/no decision after the fixed retry;
- at least 95% of (50 × 5 = 250) question-agent quadruplets are complete (all
  four conditions returned);
- every accepted citation is packet-validated (in particular, **substitute
  citations are not in the original packet and are not in another agent's
  packet**).

These are operational gates, not superiority criteria.

## 12. Smoke and pre-formal checks

The only permitted pre-formal V5 smoke is **two composite questions, one agent
(`literal_evidence`), the four conditions**: 8 calls. Smoke outputs remain
separate from formal results.

The pre-formal audit must confirm:
- SHA-256 of the FEVER JSONL matches the frozen digest;
- the binarized balanced manifest of 50 composite questions is reproducible
  from the salt;
- the partitioning table in §5 reproduces byte-for-byte;
- the substitute manifest is reproducible from the salt and the LLM rewrites
  pass length-window + JSON validation;
- substitute-generation pass yield ≥ 90% (fail-fast gate from §6.3);
- all 8 smoke calls pass instrumentation gates.

## 13. Interpretation boundary

These are still controlled, paired-intervention results on a single model
(Qwen3.5-4B) and two non-financial domains (TruthfulQA composites from V4;
FEVER from V5). They do not establish LLM faithfulness in general, S&P 500
predictability, investment performance, or cross-model generalization. A
negative V5 outcome (one or both co-primaries AUROC ≤ 0.5 on FEVER) is a valid
scientific result and will be reported as such.

V5 does **not** establish cross-model generalization — this is registered as
the next open question for V6 (see §16).

## 14. Registered deviations (added 2026-08-31, before any V5 formal call)

The following deviations were locked in during pre-formal V5 design. None of
them changes the preregistered hypotheses (§9.2 / §9.3 / §10). All are recorded
in the substitute manifest and reported alongside the formal results.

| # | Item | Preregistered analogue | Deviation | Why |
|---|---|---|---|---|
| `D1_v5` | Substitute-condition source | V4 §6: same-cluster confusable wrong claim, length ±30%, exact named-entity slot | **LLM-rewritten negative paraphrase** of the original evidence sentence, length ±50%, no new entities, one rewrite per source item written to manifest | FEVER's evidence sentences are Wikipedia-style declarative statements; the natural confusable-wrong analogue is a "negative paraphrase" of the same evidence rather than a same-cluster swap (which on FEVER would often be a totally different Wikipedia article's evidence sentence and trivially detectable). The LLM rewrite also removes V4's `D2_v4` deviation (manual cluster-similarity matching) by using the original sentence itself as the rewrite substrate. |
| `D2_v5` | Composite tie-breaking | V4: ties (1-1-1) excluded by stratification | V5: **all 3-way splits excluded** (composites with 2-1 splits are also dropped, since 2-1 splits are themselves ambiguous for majority-vote consensus) | FEVER's gold labels are coarse (SUPPORTS/REFUTES), and a 2-1 split within a composite is the FEVER analogue of a V4 1-1-1 tie. Dropping all non-unanimous composites keeps the gold-label definition sharp and avoids post-hoc tie-breaking that V4's §11 `validate_manifest` was guarding against. |
| `D3_v5` | Shared-citation detector set | V4 §11.4: 4 detectors (shared_agents, shared_count_total, shared_id_count, shared_weighted), shared_weighted reported post-hoc | V5: **shared_weighted promoted to co-primary** alongside D_OR; verdict logic in §9.2 explicitly requires both to clear the bar | Pre-registration defense against "garden of forking paths" objection to V4's strongest signal. The promotion is documented here in advance so that V5's reported headline is not retroactively selected after seeing the data. |
| `D4_v5` | §9.3 S2 wording | V4 §9.3 S2: `Risk@80(D_OR) < Risk@80(D_majority)` | V5: **`Risk@80(D_OR)` does not exceed the prevalence baseline by more than 0.05** | V4's `work_log_4.md §2.1` showed that the V4 S2 wording is mechanically broken: D_majority and D_OR pull in opposite directions on harmful_fc (which is a majority-class outcome, not a minority-class outcome). The V5 reformulation reports whether D_OR ranks informatively within the harmful majority rather than trivially separating a rare class. |
| `D5_v5` | Substitute-generation budget | V4: substitute drawn from manifest, 0 extra LLM calls | V5: **up to 150 extra LLM calls** for the substitute-generation pass (separate budget; reported in audit, not in formal-run transfer total) | FEVER's evidence sentences cannot be substituted by an offline manifest lookup (no "same-cluster confusable wrong evidence" structure exists in FEVER); the LLM rewrite is the only mechanism that keeps the substitute condition structurally comparable to V4's. |
| `D6_v5` | FEVER source acquisition | V4: local TruthfulQA JSONL pre-existing | V5: **acquire FEVER 1.0 gold-evidence `valid.jsonl` (6.2 MB, 15,935 rows) from `https://hf-mirror.com/datasets/copenlu/fever_gold_evidence/resolve/main/valid.jsonl`, hash `5da0ccc0...baa85d9` and lock at freeze** | FEVER is not currently on the jump host. Acquisition source was downgraded from FEVER 2.0 official S3 to `copenlu/fever_gold_evidence` HF mirror because the jump-host whitelist allows `hf-mirror.com` but blocks `*.amazonaws.com` and `raw.githubusercontent.com`. The HF mirror ships the exact `(claim, label, evidence)` structure V5 needs, with gold evidence sentences pre-annotated, and the binary SUPPORTS/REFUTES labels preserved (NEI still present but excluded per §4.1). One-time write (~6 MB), bandwidth-friendly. |

## 15. Operational additions (added 2026-08-31, not protocol changes)

- `audit` subcommand: runs pre-formal checks (FEVER digest, binarized balance,
  partition reproducibility, substitute manifest reproducibility, substitute
  generation yield) and exits non-zero if any gate fails.
- `all` subcommand: `prepare → substitute-generation → audit → smoke → formal`
  in one process; resumable; `--yes` removes the single confirmation prompt.
- `--no-resume` on `smoke` / `run` to ignore any `records.partial.jsonl` left
  from an interrupted run.
- `progress.json` written every call with `{completed, total, rate, eta,
  last_cqid, last_agent, last_condition, last_success, phase}` where `phase ∈
  {substitute_generation, formal}` for live polling.
- Bash driver: `scripts/run_pilot_llm_v5.sh [--yes|--skip-smoke|--skip-formal]`
  runs `prepare → substitute-generation → audit → smoke` in the foreground and
  the formal run in the background; `scripts/wait_pilot_llm_v5.sh` blocks on
  it and prints the final co-primary summary when done.
- Pre-formal audit artifact: `results/pilot_llm_v5/audit/dryrun_2026-08-31.json`
  is the offline manifest-construction dry-run output (computed before any V5
  call). The `audit` subcommand re-runs the same gates (`scripts/dryrun_v5_manifest.py`)
  and confirms the JSON matches byte-for-byte before allowing the formal run to
  proceed.

## 16. Open question for V6 (not in scope of V5)

V5 does **not** address cross-model generalization. The strongest current
threat to the paper's headline claim is "this only works on Qwen3.5-4B". V6 is
provisionally scoped as: same V5 protocol (FEVER, same manifest), swap the
endpoint to a second model (e.g., Llama-3.1-8B or Claude Haiku 4.5),
report both co-primaries. V6 preregistration is **not** drafted in this
document — it is registered here as the next open question so that the V5
report can signpost it cleanly to reviewers.
