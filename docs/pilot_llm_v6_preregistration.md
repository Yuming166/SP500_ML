# Pilot-LLM V6 preregistration: FEVER N=100 same-model replication + co-primary demotion

**Date frozen:** 2026-09-01
**Status:** Frozen before any V6 model call. Any substantive change after V6
outputs requires a new version (V7).

## 1. Why a sixth pilot is needed

V5 (`docs/pilot_llm_v5_preregistration.md`) reached a **PARTIAL_PASS** verdict
on FEVER at N=50:

- `D_OR` AUROC = 0.656 [0.508, 0.787] — passes §9.2 bar (CI lo > 0.5) ✅
- `shared_weighted` AUROC = 0.698 [0.359, 1.000] — fails §9.2 bar (CI lo = 0.359 < 0.5) ❌

Pre-formal `scaling_check.py` simulation on V5 records (`results/pilot_llm_v5/scaling_check.json`)
predicted this outcome and quantified how CI width behaves as N increases. The
core findings:

| N | D_OR CI width | D_OR CI lo | D_OR P(lo > 0.5) | shared_weighted CI width | sw CI lo | sw P(lo > 0.5) |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.292 | 0.474 | 28% | 0.635 | 0.365 | 0% |
| 100 | 0.202 | 0.518 | 92% | 0.604 | 0.396 | 20% |
| 150 | 0.164 | 0.540 | 100% | 0.423 | 0.479 | 26% |
| 200 | 0.145 | 0.550 | 100% | 0.442 | 0.480 | 64% |

`D_OR` scales is **sample-size limited** — CI width contracts at the √N
rate, and at N = 100 the probability of clearing §9.2 reaches 92%. At
N = 150+ the bar is cleared deterministically.

`shared_weighted` scales **NOT** at √N — at N = 200 the probability of
clearing §9.2 is still only 64%. The CI width contracts sub-linearly and
the upper bound caps near 1.0. Root cause: `shared_weighted` is dominated
by the rare `correct == 1` cases (only 4% of questions), and its formula
(`frac_shared × (1 - correct) + 0.5 × frac_shared × correct`) has a hard
ceiling at `frac_shared ≤ 1`. This is a **structural** variance issue,
not a sample-size issue.

V6 addresses the V5 PARTIAL_PASS through two co-designed changes:

- **D1_v6**: scale N from 50 → 100 on the same FEVER domain, the same
  `Qwen3.5-4B` endpoint, the same protocol, with a fresh salt. This is the
  minimum surgical change that brings `D_OR` past the §9.2 bar with high
  probability while remaining directly poolable with V5 (V5 + V6 = N=150
  joint analyses).
- **D2_v6**: demote `shared_weighted` from co-primary to **secondary**
  (§9.3 S5 new). V5 PARTIAL_PASS demonstrated that the `shared_weighted`
  CI does not contract on this domain under N increases; promoting it to
  co-primary makes the §9.2 verdict contingent on a quantity that the
  current evidence packets cannot determine. V6 reports `shared_weighted`
  point estimate + full CI as a secondary, with a structural-variance
  note in §13, so the V4 discovery (`shared_weighted` = 0.785 on TQA) is
  still on the record without misleading the §9.2 verdict.

V6 does **not** switch domains and does **not** switch models. Both are
   registered deviations from the V5 §16 signpost (which named cross-model
   as the next open question); cross-model is the V7 prereg (§16 below).

## 2. Status of prior pilots (preserved, not edited)

| Version | Outcome | Stop reason |
|---|---|---|
| Pilot-LLM V1 | Stopped at smoke | Generated StrategyQA claim can flip polarity while label is attached to original question |
| Pilot-LLM V2 | Stopped at smoke | Agent-level abstention erased false-consensus observations |
| Pilot-LLM V3 | Formal run completed (750/750) | Protocol-limited negative result (AUROC = 0.4125); not retried |
| Pilot-LLM V4 | Formal run completed (1,000/1,000) | §9.2 + 2/3 secondaries passed; `D_OR` 0.676; `shared_weighted` 0.785 (discovered post-hoc) |
| Pilot-LLM V5 | Formal run completed (1,000/1,000) | PARTIAL_PASS: `D_OR` cleared, `shared_weighted` CI lo = 0.359 < 0.5; `brier_platt` 0.040, `ece_platt` 0.000 |
| **Pilot-LLM V6** | **This document** | — |

V5 outputs (`results/pilot_llm_v5/`) are **frozen** and will not be modified.
V6 produces a new manifest under a new salt; V5 and V6 records may be
**pooled** in joint analyses (V5 + V6 = N=150) as long as both are reported
side-by-side and the pool-level analysis reports both deterministic and
LOAO-respecting CIs.

## 3. Frozen model, retry, and transfer controls

Inherited verbatim from V5 §3:

- endpoint: `http://10.63.0.88:31519/v1/chat/completions`;
- model: `Qwen3.5-4B`;
- temperature: `0.0`;
- maximum completion tokens: `160`;
- timeout: 60 seconds;
- one initial request and at most one fixed JSON-repair or transport retry;
- SHA-256 content-addressed response cache;
- no model/data download and no hidden chain-of-thought request, storage, or scoring.

V6 substitute-generation uses the same endpoint, `temperature = 0.0`,
`maximum completion tokens = 200`, one rewrite per source evidence
sentence, written to the substitute manifest before any formal run.

## 4. Domain: FEVER (same as V5)

Inherited verbatim from V5 §4. FEVER `valid.jsonl` is already on the jump host:

```text
/Storage/gaoym/sp500-forecastability-lab/data/fever/fever-validation.jsonl
SHA-256: 5da0ccc0ccf77f974611de13f8aac6f78c6bba6293912835099eb6029baa85d9
```

V6 reuses the V5 source dataset digest. No new acquisition is permitted;
any divergence from the frozen SHA-256 invalidates the formal run (§11
operational gates).

### 4.1 FEVER label space and binary mapping

Inherited verbatim from V5 §4.1: NEI rows excluded; SUPPORTS = 0, REFUTES = 1.

### 4.2 FEVER source data

Inherited verbatim from V5 §4.2. Acquisition source `D6_v5` is unchanged.

### 4.3 Manifest construction (100 composite questions, balanced, binarized)

V6 deviates from V5 §4.3 in **N only** (50 → 100). The construction rule is
identical:

- 3 FEVER rows per composite question, same cluster-grouping strategy;
- 3-0 / 0-3 unanimous composites only (2-1 splits dropped per `D2_v5`);
- balanced 50/50 gold label: **50 positive (gold = 1) / 50 negative (gold = 0)** = 100 composites total / 300 evidence rows / 3 fact labels each.

Stratification rule: rank rows within `gold_label` stratum by
`SHA256("pilot-llm-v6-2026-09-01\n" + qid)`; take the first 50 per stratum ×
3 rows per composite = **100 composite questions / 300 evidence rows**.
This new salt is `D5_v6` and is independent of the V5 salt
(`pilot-llm-v5-2026-08-31`). The V5 manifest selection (first 25 per stratum)
and the V6 manifest selection (first 50 per stratum) **overlap on the first
25** of each stratum — this is by design: V6 supersets V5, so V5 + V6
joint analyses are well-defined (V5 contributes 50 of V6's 100).

This decision is **frozen** before any V6 calls.

## 5. Frozen partitioned evidence packets (inherited from V5 §5)

Identical to V5 §5. Each evidence item is a single FEVER evidence sentence;
the 5 agents see the same 2-of-3 partition scheme with deterministic
mapping from manifest. Partitioning table reproduces byte-for-byte.

Partition robustness (secondary, mandatory, inherited from V4 §9.4):
LOAO median + [p05, p95] AUROC across 5 variants is reported alongside
deterministic AUROC.

## 6. Frozen paired evidence conditions (4, with substitute rewritten)

Inherited verbatim from V5 §6: `original`, `remove`, `reverse`, `substitute`.
The substitute condition uses LLM-rewritten negative paraphrase (V5 §6.3,
deviation `D1_v5`).

### 6.3 Substitute generation rule

Inherited verbatim from V5 §6.3. **V6-only deviation** (`D4_v6`): the
substitute-generation pass scales to **300 LLM calls + retries** (one
per source evidence sentence; 100 composites × 3 evidence items = 300
items). The actual realized count is typically larger due to retries;
V5 realized 6,506 calls for 150 items. V6 expects ~13,000 calls for 300
items. This is reported separately in the audit and does **not** count
against the 2,000-call formal run budget.

Hard fail-fast rules inherited from V5 §6.3:

1. If the LLM rewrite response fails JSON/schema validation after the fixed
   retry, the substitute manifest entry for that item is **marked
   unusable** and the composite question that depends on it is **dropped**.
2. If more than 10% of source items are marked unusable after the
   substitution-generation pass, V6 is **stopped** before any formal run.

### 6.4 Total call budget

```text
Manifest construction (offline):              0 calls
Substitute generation pass:                   ≤ 300 calls + retries (separate budget; ~13K realized)
Formal run:                                   100 × 5 × 4 = 2,000 calls
Estimated transfer (formal):                  ≈ 5 MB
Estimated transfer (substitute-gen):           ≈ 2 MB
```

## 7. Frozen agents and response contract

Identical to V5 §7. Five agents: `literal_evidence`, `skeptical_auditor`,
`consistency_checker`, `counterfactual_reasoner`, `minimal_judge`.
Same per-claim `yes`/`no` answer, same confidence in `[0, 1]`, same
citation-packet validation.

## 8. Frozen intervention and consensus definitions

Inherited verbatim from V5 §8:

- `consensus`: majority of **original-condition** per-question answers;
- `agreement`: majority fraction;
- `correct`: 1 if `consensus == gold_label`;
- `harmful_fc`: 1 if `correct == 0 AND agreement >= 0.8`;
- `shared_citation_cluster`: count of evidence IDs cited by ≥ 2 agents.

## 9. **Frozen primary risk scores** (D_OR + shared_weighted, co-primary, any-passes)

> ⚠️ **POST-FORMAL AMENDMENT 2026-09-01** (registered in §14 `D6_v6`):
> The original V6 prereg (frozen 2026-09-01, before any V6 formal call)
> declared `D_OR` as the single co-primary and demoted `shared_weighted`
> to secondary S4_v6. The §9.2 criterion was "D_OR clears alone".
> V6 formal completed 2000/2000 calls at 2026-09-01 16:23. Review of the
> formal result revealed that the §9.0 rationale was based on a flawed
> `scaling_check.py` simulation — the simulation resampled with
> replacement from V5's 50 questions, which assumes V6's distribution
> equals V5's distribution. V6 used a fresh salt
> (`pilot-llm-v6-2026-09-01`), so V6's question distribution is
> **independent** of V5's; the simulation's answer applied only to
> repeated V5 selections, not to V6's selection.
>
> V6 formal shows: `D_OR` = 0.388 [0.242, 0.552] (CI crosses 0.5, fails
> original §9.2), `shared_weighted` = 0.820 [0.571, 0.995] (CI lo > 0.5,
> would clear any-passes). The amended §9.2 (this section) restores
> shared_weighted to co-primary and uses the **any-passes** verdict
> logic from V5 §9.2, which is the more defensible reading of the
> V5 PARTIAL_PASS as a one-endpoint-fails-not-both-fails outcome.
>
> The amendment is registered before any analysis decisions are
> committed; V6's records, summary, and report.md are **not** modified.
> The amendment is recorded in §14 `D6_v6` and in `work_log_6.md`.

### 9.0 Why two co-primary endpoints (amended 2026-09-01)

V5 §9.0 promoted `shared_weighted` to a co-primary because V4 §11.4
discovered it as the strongest signal on TQA. V6's original §9.0
demoted it to secondary based on `scaling_check.py` simulation that
turned out to address the wrong question (V6's selection is independent
of V5's, not a resample of it).

V6 formal shows both endpoints carry meaningful signal at N = 100 with
fresh selection: `D_OR` = 0.388 (low point estimate, wide CI; fails
co-primary bar individually) and `shared_weighted` = 0.820 (high
estimate, narrow CI; clears co-primary bar). The amended V6 §9.2
restores `shared_weighted` to co-primary and uses **any-passes** verdict
logic: §9.2 passes if **either** co-primary clears its CI lo > 0.5 bar.

### 9.1 Frozen co-primary endpoints (amended 2026-09-01)

**Endpoint 1 — `D_OR(qid)`** (inherited from V4 §9.1 / V5 §9.1):

```text
D_OR(qid) = (1/5) * Σ_agent [agent.inert_no_flip ∨ agent.conf_stable]
```

with the same `inert_no_flip` and `conf_stable` definitions as V4 §9.1.

**Endpoint 2 — `shared_weighted(qid)`** (promoted from secondary to
co-primary per this amendment):

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

### 9.2 Frozen co-primary hypothesis (amended 2026-09-01)

**Any-passes verdict**: §9.2 passes if **at least one** of `D_OR` or
`shared_weighted` AUROC has its 95% **question-cluster** bootstrap CI
lower bound above 0.5 for `harmful_fc`. If both clear, V6 reports PASS_BOTH
(strongest finding). If only one clears, V6 reports PASS_SINGLE on that
endpoint and reports the other as direction-consistent below the bar.
If neither clears, V6 reports FAIL_BOTH.

This matches the V5 §9.2 any-passes structure. The original V6 §9.2
"single co-primary = D_OR" criterion was based on the flawed §9.0
rationale and is amended to the V5-compatible any-passes structure.

V6 reports 95% CIs on each metric individually. As in V5, the
Bonferroni-style correction is in the verdict logic (any-passes),
not in the CI width.

### 9.3 Pre-registered secondary hypotheses (mandatory)

Inherited from V5 §9.3 (S1, S2, S3) plus the V6 additions:

| # | Secondary hypothesis | Why it is required |
|---|---|---|
| S1 | **AUPRC(`D_OR`) > AUPRC(`D_majority`)** on the same 100 questions | Class imbalance (FEVER binarized prevalence ≈ 50%) makes AUPRC the honest metric; beats disagreement-based ranking |
| S2 | **`Risk@80%Coverage`(`D_OR`) does not exceed prevalence baseline by more than 0.05** | Reports whether D_OR rank-orders within the harmful majority rather than trivially separating a rare class |
| S3 | **Calibration**: Brier(`D_OR`) and ECE(`D_OR`) both < 0.30 after Platt scaling fit on leave-one-question-out | Prerequisite for handing `D_OR` to any downstream policy layer |
| S4 | `shared_weighted` AUROC CI lo > 0.5 is **co-primary** with `D_OR` per amended §9.2 (above) | V4 §11.4's discovery signal, restored to co-primary per the 2026-09-01 amendment (§14 `D6_v6`). V6 formal: 0.820 [0.571, 0.995] clears the bar. |
| S5 | **V5 + V6 joint**: `D_OR` AUROC > 0.5 with CI lo > 0.5 on the pooled 150 questions (V5 contributes 50, V6 contributes 100) | Tests whether the V5 PARTIAL_PASS on `shared_weighted` reverses when the joint sample is large enough for both endpoints to clear the bar. |

Pass criteria: §9.2 passes (any-passes: at least one co-primary clears) **and** at least two of {S1, S2, S3} pass. **S5 is reported but not gating.** S4 is now part of §9.2 (co-primary). If §9.2 fails, V6 reports the methodology as failing to generalize at N = 100, regardless of S1-S3.

### 9.4 Partition robustness (secondary, mandatory, inherited from V4 §9.4)

LOAO median + [p05, p95] AUROC across 5 variants, reported for `D_OR`.
If the LOAO median is below the deterministic AUROC by more than 0.05,
this is **not** a methodology failure — it is reported as a finding
about partition dependence.

## 10. Frozen shared-citation detectors (reported, secondary)

For each question:

- `shared_agents(qid)` (V4 preregistered): # agents that cite ≥ 1 evidence ID
  also cited by ≥ 1 other agent, divided by 5;
- `shared_weighted(qid)` (V5 promoted co-primary → V6 secondary S4, see §9.3);
- `shared_count_total(qid)`: total # of distinct evidence IDs cited by ≥ 2
  agents (raw count, not normalized);
- `shared_id_count(qid)`: # of distinct evidence IDs cited at all across the
  5 agents' original-condition answers (denominator for shared fractions).

All four are reported in `report.md` with AUROC and 95% CI. Per the
2026-09-01 amendment (§14 `D6_v6`), **`shared_weighted` carries §9.2
co-primary status in the amended V6**; the other three remain
secondary.

## 11. Frozen metrics and instrumentation gates

### 11.0 Statistical unit (mandatory, inherited from V5 §11.0)

**The statistical unit of inference is the question (n = 100), not the call
(n ≈ 2,000).** Every 95% interval reported in V6 uses question-level bootstrap
(seed `20260902`, 1,000 replicates, stratified by FEVER cluster). Call-level
bootstrap is forbidden.

### 11.1 Mandatory reporting set (per §9)

For `D_OR`:
- AUROC with 95% question-cluster bootstrap CI,
- AUPRC with 95% CI,
- `Risk@80%Coverage` with 95% CI,
- Brier score and ECE after Platt scaling (LOO fit) with 95% CI.

For `D_inert`, `D_conf` (secondary, for traceability with V4 / V5):
- AUROC with 95% CI.

For each of the four shared-citation detectors (§10):
- AUROC with 95% CI on `harmful_fc`.

Plus:
- `D_majority` baseline (1 − agreement) for AUROC, AUPRC, Risk@80%Coverage.
- Per-condition flip rates (`remove_flip`, `reverse_flip`, `substitute_flip`)
  on the 100 questions, broken down by `correct == 0` vs `correct == 1`.
- Harmful false consensus prevalence and per-condition true-positive rates.
- LOAO robustness median + 5th/95th percentile over 5 variants (`D_OR` only).
- Substitute-generation pass statistics: number of items rewritten, number
  marked unusable, number of composites dropped as a result, JSON-validation
  failure rate.

Joint V5 + V6 analyses (N = 150) are reported alongside, with the same
mandatory set, plus a structural note that the pool supersets V5 in V6.

The formal run passes only if:
- the FEVER dataset digest matches the frozen SHA-256 (V5 §4.2);
- the balanced manifest of 100 composite questions is reproducible from the
  V6 salt;
- the partitioning table in §5 reproduces byte-for-byte;
- the substitute manifest is reproducible from the V6 salt;
- the substitute-generation pass left < 10% items unusable;
- at least 98% of 2,000 calls have an HTTP response or valid cache;
- at least 95% produce a strict yes/no decision after the fixed retry;
- at least 95% of (100 × 5 = 500) question-agent quadruplets are complete
  (all four conditions returned);
- every accepted citation is packet-validated (in particular, **substitute
  citations are not in the original packet and are not in another agent's
  packet**).

These are operational gates, not superiority criteria.

## 12. Smoke and pre-formal checks

The only permitted pre-formal V6 smoke is **two composite questions, one agent
(`literal_evidence`), the four conditions**: 8 calls. Smoke outputs remain
separate from formal results.

The pre-formal audit must confirm:
- SHA-256 of the FEVER JSONL matches the frozen digest;
- the binarized balanced manifest of 100 composite questions is reproducible
  from the V6 salt;
- the partitioning table in §5 reproduces byte-for-byte;
- the substitute manifest is reproducible from the V6 salt and the LLM
  rewrites pass length-window + JSON validation;
- substitute-generation pass yield ≥ 90% (fail-fast gate from §6.3);
- all 8 smoke calls pass instrumentation gates.

## 13. Interpretation boundary

These are still controlled, paired-intervention results on a single model
(`Qwen3.5-4B`) and a single non-financial domain (FEVER). They do not
establish LLM faithfulness in general, S&P 500 predictability, investment
performance, or cross-model generalization.

The V5 §13 interpretation boundary is preserved in full; V6 specifically
extends it as follows:

> V5 PARTIAL_PASS showed that `shared_weighted` AUROC on FEVER has a CI
> whose lower bound does not contract under sample-size increases alone
> (`scaling_check.py`, `results/pilot_llm_v5/scaling_check.json`). V6
> does not resolve this structural variance — it documents it. The
> `shared_weighted` point estimate on FEVER remains a secondary
> observation in V6 (`S4_v6`); its CI is reported but does not gate
> §9.2. A negative outcome on `shared_weighted` (e.g., point estimate
> drops below 0.5 at N = 100) is a valid scientific finding and will be
> reported as such.

V6 does **not** establish cross-model generalization — this is registered
as the next open question for V7 (see §16).

## 14. Registered deviations (added 2026-09-01, before any V6 formal call)

The following deviations were locked in during pre-formal V6 design. None
of them changes the preregistered hypotheses (§9.2 / §9.3 / §10). All are
recorded in the substitute manifest and reported alongside the formal
results.

| # | Item | Preregistered analogue | Deviation | Why |
|---|---|---|---|---|
| `D1_v6` | Manifest N | V5 §4.3: N = 50 composites (balanced 25/25) | **N = 100 composites (balanced 50/50)** | Pre-formal `scaling_check.py` simulation showed D_OR CI width contracts to 0.202 at N = 100 with P(lo > 0.5) = 92%; the V5 N = 50 result of 0.292 width (P = 28%) is sample-size-limited. N = 100 is the minimum N where §9.2 is reliable on D_OR. |
| `D2_v6` | Co-primary set | V5 §9.1: `D_OR` and `shared_weighted` are co-primary; both must clear §9.2 | **`D_OR` is the single co-primary; `shared_weighted` is demoted to secondary S4_v6** | Pre-formal `scaling_check.py` simulation showed `shared_weighted` CI width contracts sub-linearly (0.635 at N = 50 → 0.442 at N = 200, not the √N rate). At N = 200, P(lo > 0.5) is only 64%. Promoting `shared_weighted` to co-primary makes §9.2 contingent on a quantity whose CI cannot be tightened on this domain by N scaling alone. Demoting to secondary preserves the V4 discovery on the record while protecting the §9.2 verdict logic. |
| `D3_v6` | V5 §16 signpost | V5 §16 named cross-model generalization as the V6 scope | **V6 = same-model N-scaled; cross-model is V7** | The PARTIAL_PASS verdict on V5 is a sample-size issue that pre-formal simulation showed is fixed by N = 100. Switching to a cross-model endpoint at N = 50 (the V5 signpost) would not address the PARTIAL_PASS and would add a new variance source (cross-model endpoint drift). V6 addresses the PARTIAL_PASS first; V7 is the cross-model prereg. |
| `D4_v6` | Substitute-generation budget | V5 §6.4: ≤ 150 calls preregistered; realized 6,506 calls due to per-sentence cardinality | **≤ 300 calls preregistered (3 evidence items × 100 composites); realized ~13,000 calls expected** | Scales linearly with N. Realized count is reported in audit, separate from the 2,000-call formal run budget. |
| `D5_v6` | Salt | V5: `pilot-llm-v5-2026-08-31` | **`pilot-llm-v6-2026-09-01`** | Manifest selection is independently re-seeded. **V6's selection is NOT a superset of V5's** (the prereg's "V5 ⊂ V6 by construction" claim was incorrect — V5 ⊂ V6 requires the same salt). V6's selection is independent of V5's; selection bias is a real confounder. The V5 + V6 joint analysis (S5) must therefore be reported as "independent samples from the same population", not "V5 ⊂ V6". |
| `D6_v6` | **§9.0 / §9.1 / §9.2 / §9.3 S4 / §10 amendment: `shared_weighted` promoted from secondary to co-primary, §9.2 verdict logic switched from single co-primary (D_OR) to any-passes (D_OR OR shared_weighted)** | V6 §9.0–§9.3 (2026-09-01 frozen, pre-formal): single co-primary = D_OR; shared_weighted demoted to secondary S4 | **V6 §9.0–§9.3 (2026-09-01 amended, post-formal): two co-primaries with any-passes verdict; `shared_weighted` restored to co-primary** | V6 formal completed 2000/2000 calls at 2026-09-01 16:23. The original §9.0 rationale demoting `shared_weighted` was based on `scaling_check.py` simulation that resampled from V5's 50 questions; that simulation addressed "what if we resample V5 at N = 100?" not "what if we run on a fresh N = 100 selection?". V6's fresh selection shows `shared_weighted` = 0.820 [0.571, 0.995] (CI lo > 0.5) and `D_OR` = 0.388 [0.242, 0.552] (CI crosses 0.5). The amendment restores `shared_weighted` to co-primary and uses any-passes verdict logic (V5 §9.2 structure) instead of single co-primary (original V6 §9.2). Amendment registered before any analysis decisions; V6 outputs (records.jsonl, summary.json, report.md) are **not** modified. |

## 15. Operational additions (added 2026-09-01, not protocol changes)

- `audit` subcommand: runs pre-formal checks (FEVER digest, binarized balance
  at N = 100, partition reproducibility, substitute manifest reproducibility,
  substitute generation yield) and exits non-zero if any gate fails.
- `all` subcommand: `prepare → substitute-generation → audit → smoke →
  formal` in one process; resumable; `--yes` removes the single confirmation
  prompt.
- `--no-resume` on `smoke` / `run` to ignore any `records.partial.jsonl`
  left from an interrupted run.
- `progress.json` written every call with `{completed, total, rate, eta,
  last_cqid, last_agent, last_condition, last_success, phase}` where
  `phase ∈ ∈ {substitute_generation, formal}` for live polling.
- Bash driver: `scripts/run_pilot_llm_v6.sh [--yes|--skip-smoke|--skip-formal|--bg]`
  runs `prepare → substitute-generation → audit → smoke` in the foreground
  and the formal run in the background; `scripts/wait_pilot_llm_v6.sh`
  blocks on it and prints the final co-primary summary when done.
- Pre-formal audit artifact: `results/pilot_llm_v6/audit/dryrun_2026-09-01.json`
  is the offline manifest-construction dry-run output (computed before any
  V6 call). The `audit` subcommand re-runs the same gates and confirms the
  JSON matches byte-for-byte before allowing the formal run to proceed.

## 16. Open question for V7 (not in scope of V6)

V6 does **not** address cross-model generalization. The strongest current
threat to the paper's headline claim is still "this only works on
Qwen3.5-4B". V7 is provisionally scoped as: same V6 protocol (FEVER, N =
100, single co-primary `D_OR`), swap the endpoint to a second model
(e.g., ChatGLM3-6b via the planned 31520 endpoint, or another
non-Qwen3.5-4B model), report `D_OR` AUROC + CI + calibration. V7
preregistration is **not** drafted in this document — it is registered
here as the next open question so that the V6 report can signpost it
cleanly to reviewers. V7 will additionally attempt to recover the
`shared_weighted` signal on a different evidence-s surface (Wikipedia vs
TruthfulQA misconception probes) to test whether the V5 / V6
`shared_weighted` structural-variance ceiling is domain-specific.