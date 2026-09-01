# Pilot-LLM V9 preregistration: Anti-herding subset selection + agent-level router stress test

**Date frozen:** 2026-09-01
**Status:** Frozen BEFORE any V9 model call. All design decisions below are pre-registered and were made WITHOUT inspecting V9 records. Any substantive change after V9 outputs requires a new version (V10).

## 1. Why V9

V8 (cross-model, Ling-3.0-tiny-int4) was preregistered but not executed — the SGLang install path exhausted the available bandwidth budget without bringing Ling online. The bandwidth cost of re-attempting V8 is bounded but non-trivial, so V9 is a higher-leverage use of remaining capacity: it tests a paper-central methodological question using existing infrastructure.

The central question for V9 is:

> **Does the agent-level router's failure mode (V4-V7 / `analysis/agent_router_comparison.md`) — that simple majority vote dominates on saturated-prevalence domains — come from saturation, or from the selection rule?**

The V4-V7 router analysis showed:
- V4 (TQA, N=50, low prevalence 22%): routers competitive with majority vote
- V5/V6/V7 (FEVER, N=50–100, prevalence 92–96%): majority vote crushes every router (0.92–1.00 vs 0.4–0.5)

Two hypotheses are consistent with the observation:
1. **Prevalence hypothesis**: routers cannot beat majority when almost every question is "harmful consensus" (95%) because all agents are uniformly fragile.
2. **Selection hypothesis**: V5/V6/V7's selection rule (top-K per stratum) preferentially picked herding-prone clusters.

V9 separates these hypotheses via a **selection rule change** that reduces herding without changing the model or protocol. The change is motivated by **structural cluster features** (FEVER metadata), not by LLM outputs. The design is pre-registered before any V9 records are collected.

## 2. The "anti-herding" selection rule (pre-registered, frozen)

**Rule V9 §4: cluster-level evidence-overlap filter.**

For each candidate cluster (entity-grouped FEVER rows):

- Compute pairwise cosine similarity of the 3 evidence sentences' surface forms (TF-IDF, lowercase, no stopwords, bigrams).
- Define `intra_cluster_overlap = max pairwise cosine`.
- Keep the cluster iff `intra_cluster_overlap ∈ [0.10, 0.60]`.

Rationale (independent of any V9 records):
- Clusters with overlap ≥ 0.60 have evidence sentences that "say the same thing" in different words. Five agents with persona diversity will all converge on the same evidence-derived answer → herding.
- Clusters with overlap ≤ 0.10 have evidence sentences about disjoint topics, making the "consensus evidence" an artifact of the manifest rather than a real signal.
- The middle band [0.10, 0.60] preserves clusters where evidence is related but has distinguishable content — exactly the regime where agent-persona differentiation is informative.

**This filter is computed from FEVER `evidence_sentence` strings only. No LLM output is consulted.**

After filtering, apply V5's existing selection rule (§4.3 of V5 preregistration): top-50 per label × 3 evidence items = 100 composites, stratified via SHA256("pilot-llm-v9-2026-09-01\n" + qid).

Pre-registered targets:
- **Target prevalence** (predicted from the filter's effect on FEVER binary labels, not from any V9 records): 50–75% harmful_fc. V5/V7 was 93–96%; the filter is designed to drop herding-prone (uniformly-harmful) clusters.
- **N = 100**, balanced 50/50 SUPPORTS/REFUTES within the filtered set.

If the filter removes enough clusters that the manifest cannot reach 50/50, the run is **stopped before any LLM call** with the deviation logged. The filter is **not** relaxed.

## 3. Frozen model, retry, and transfer controls

Inherited verbatim from V5 §3 / V7 §3:

- endpoint: `http://10.63.0.88:31519/v1/chat/completions`;
- model: `Qwen3.5-4B`;
- temperature: `0.0`;
- maximum completion tokens: `160`;
- timeout: 60 s;
- one initial request + at most one JSON-repair or transport retry;
- SHA-256 content-addressed response cache;
- no model/data download and no hidden chain-of-thought.

V9 substitute-generation uses the same V5 substitute manifest via the
content-addressed cache (the substitute prompt and seed are
content-identical to V5). V9 reuses V5's substitute rewrites; no new
LLM rewrite is issued.

## 4. Frozen partitioned evidence packets (inherited from V5 §5)

Identical to V5/V6/V7. 5 agents × 2-of-3 subsets of {E01, E02, E03}.

## 5. Frozen paired evidence conditions (4, same as V5/V6/V7)

Inherited verbatim from V5 §6: `original`, `remove`, `reverse`, `substitute`.

## 6. Frozen agents and response contract (inherited from V5 §7)

Identical to V5/V6/V7. Five agents: `literal_evidence`,
`skeptical_auditor`, `consistency_checker`,
`counterfactual_reasoner`, `minimal_judge`.

## 7. Frozen intervention and consensus definitions (inherited from V5 §8)

Identical to V5/V6/V7. `harmful_fc = correct == 0 AND agreement >= 0.8`.

## 8. Frozen primary risk scores (D_OR + shared_weighted, co-primary, any-passes)

Inherited verbatim from V7 §9.

### 8.1 Frozen co-primary endpoints

Same as V7 §9.1 (D_OR + shared_weighted).

### 8.2 Frozen co-primary hypothesis

Same as V7 §9.2 (any-passes: at least one co-primary CI lo > 0.5).

### 8.3 Pre-registered secondary hypotheses

| # | Hypothesis | Why it is required |
|---|---|---|
| S1 | **AUPRC(`D_OR`) > AUPRC(`D_majority`)** | Inherited from V5 |
| S2 | **`Risk@80%Coverage`(`D_OR`) does not exceed prevalence baseline by more than 0.05** | Inherited from V5 |
| S3 | **Calibration**: Brier(`D_OR`) and ECE(`D_OR`) < 0.30 after Platt LOO | Inherited from V5 |
| S4 | `shared_weighted` AUROC CI lo > 0.5 | Co-primary; reported not gating (V7 §9.3) |
| S5 | **Anti-herding filter effect**: V9's `harmful_fc` prevalence ≤ 0.75 | Tests whether the §2 filter actually reduces herding on this domain (against the V5/V7 baseline of 0.93–0.96) |
| S6 | **V9 + V7 joint**: `D_OR` AUROC > 0.5 with CI lo > 0.5 on the pooled 193 questions (V7 92 + V9 100; V9 ⊃ V5 ⊂ V7 ⊂ V9 ⊃ V5, but joint is defined as V7 ∪ V9 ∩ same salt family — both use V5 salt) | Tests whether the V7 signal is preserved on a less-herding-prone selection |

### 8.4 Partition robustness

Same as V7 §9.4 (LOAO median + [p05, p95] AUROC).

## 9. Pre-registered agent-level router variants (frozen)

The V4-V7 router analysis (`analysis/agent_router_comparison.md`)
showed that simple majority vote dominates saturated-prevalence
domains. V9's router variants are **pre-registered** to test whether
the anti-herding filter creates headroom for routers to win.

All three variants use **only V9 records** for their internal weights
(no leakage from V4/V5/V6/V7). The variants are **frozen** in this
section — no parameter tuning based on V9 outcomes is permitted.

### R1_top_auroc_v9 (frozen)
For each question, pick the answer from the agent with the highest
**V9 per-agent AUROC of fragility → harmful_fc** (computed on V9 records
only). Tie-break: choose the agent whose fragility on this question is
lowest. Score: 1 - chosen agent's fragility.

### R2_weighted_v9 (frozen)
Weighted majority vote. Per-agent weight = clip(V9 per-agent AUROC, [0.5, 1.0]).
Score: weighted yes-vote / total weight.

### R3_min_fragility_v9 (frozen)
For each question, pick the answer from the agent with the lowest fragility on
that question. Score: 1 - chosen agent's fragility.

### Baselines (frozen)
- **BL_majority**: simple unweighted majority. Score: vote agreement (n/5).
- **BL_D_OR**: per-question D_OR score (1 - mean(inert OR conf_stable)).

### Reporting policy (frozen, locked 2026-09-01)
- **All 5 methods × 1 version (V9) = 5 cells**, all reported.
- **No method hidden**, **no parameter tuned after V9 runs**, **no cherry-picking**.
- If all three R* routers lose to BL_majority on V9, this is the
  finding. It will be reported as: "Even after anti-herding subset
  selection, simple majority vote remains the strongest per-question
  aggregation method on FEVER — the V4-V7 router failure is not
  selection-driven but architecture-driven (i.e., agent personas
  cannot differentially rescue wrong consensus once it forms)."

## 10. Frozen metrics and instrumentation gates

### 10.0 Statistical unit

Same as V7 §11.0. Question-level bootstrap, seed `20260902`,
1,000 replicates.

### 10.1 Mandatory reporting set

For each of `D_OR` and `shared_weighted`:
- AUROC, AUPRC, Risk@80%, Brier+ECE (Platt LOO), each with 95% CI.

For each of the four shared-citation detectors: AUROC + CI.

Plus: `D_majority` baseline, per-condition flip rates, harmful_fc
prevalence, LOAO, substitute-generation pass stats.

**The central V9 deliverable**: a 5-method × 5-version AUROC table
that includes V9 alongside V4/V5/V6/V7, with all routers using V9-only
internal weights. If V9's routers still lose, the table still reports
all rows. This is the paper's "router efficacy is bounded" section.

The formal run passes only if:
- the FEVER dataset digest matches the frozen SHA-256;
- the V9 manifest of 100 composites is reproducible from V9's salt;
- the anti-herding filter (§2) is reproducible from the FEVER dataset;
- the substitute manifest is reproducible;
- substitute-generation pass left < 10% items unusable;
- at least 98% of 2,000 calls have an HTTP response or valid cache;
- at least 95% produce a strict yes/no decision after the fixed retry;
- at least 95% of (100 × 5 = 500) question-agent quadruplets are complete;
- every accepted citation is packet-validated.

These are operational gates, not superiority criteria.

## 11. Smoke and pre-formal checks

Pre-formal audit must confirm:
- SHA-256 of the FEVER JSONL matches the frozen digest;
- the V9 manifest of 100 composites is reproducible from V9's salt;
- the §2 anti-herding filter applied to the dataset yields ≥ 50
  SUPPORTS and ≥ 50 REFUTES available for balanced manifest
  construction; if not, run is aborted before any LLM call;
- the partitioning table reproduces byte-for-byte;
- the substitute manifest is reproducible;
- substitute-generation pass yield ≥ 90%;
- all 8 smoke calls pass instrumentation gates.

## 12. Interpretation boundary

V9 tests whether the agent-router failure mode (V4-V7) is
selection-driven (herding-prone clusters) or prevalence-driven
(saturated FEVER). If V9's routers still lose, the answer is:
"agent-level routing cannot rescue wrong consensus on FEVER — even
after structural anti-herding filtering, simple majority vote wins."
This is a clean, publishable methodology boundary statement.

V9 does **not** establish S&P 500 predictability, investment
performance, LLM faithfulness in general, or cross-model
generalization.

## 13. Registered deviations (added 2026-09-01, before any V9 formal call)

| # | Item | Preregistered analogue | Deviation | Why |
|---|---|---|---|---|
| `D1_v9` | Manifest selection | V5 §4.3 / V7 §4.3: top-K per stratum, no cluster filter | **§2 cluster-overlap filter** `[0.10, 0.60]` applied before top-K selection | Pre-registered structural test of "anti-herding" hypothesis. Filter is computed from FEVER evidence strings only — no LLM involvement. The filter is locked at this exact threshold band; no post-hoc relaxation is permitted. |
| `D2_v9` | Substitute generation | V5 §6.4: regenerate on V5 salt; V7 §6.3: reuse V5's | **V9 reuses V5's substitute manifest verbatim** (no regeneration) | Substitute prompts and seed are content-identical to V5; V9's manifest selects a different cluster subset but each cluster's substitute sentence is unchanged. Re-running would be wasteful. |
| `D3_v9` | Co-primary verdict | V7 §9.2: any-passes (D_OR OR shared_weighted) | **Inherited unchanged** | The amendment-style any-passes verdict logic from V7 carries forward. V9 does not introduce a new verdict structure. |
| `D4_v9` | Salt | V5: `pilot-llm-v5-2026-08-31`; V6: `pilot-llm-v6-2026-09-01`; V7: `pilot-llm-v5-2026-08-31` (V5 ⊂ V7) | **V9: `pilot-llm-v9-2026-09-01`** (fresh salt for fresh selection, after anti-herding filter) | The anti-herding filter changes the candidate set. A fresh salt ensures the selection is not a re-run of V5/V7's selection with the filter bolted on. |
| `D5_v9` | Router variants | V4-V7 analysis: 3 routers (R1/R2/R3) computed from V7 records | **R1/R2/R3 redesigned with V9-only weights** | Router weights must come from V9 records only (no leakage). The variant structure (top-AUROC / weighted / min-fragility) is preserved because V4-V7 showed those are the relevant design dimensions. |

## 14. Operational additions

- `audit` subcommand: pre-formal checks (FEVER digest, anti-herding filter
  yield, balanced manifest, substitute yield). Aborts if any pre-formal gate fails.
- `all` subcommand: `prepare → substitute-generation → audit → smoke → formal`.
- `progress.json` written every call.
- Bash driver: `scripts/run_pilot_llm_v9.sh [--bg|--skip-smoke|--skip-formal]`.

## 15. Open question for V10 (not in scope of V9)

V9 does **not** address cross-model signal transfer (V8's question). V8
remains as the next open question. V10 is provisionally scoped as V9's
protocol on a second cross-model target, *if* the cross-model target
is available without prohibitive bandwidth. V10 preregistration is
not drafted in this document.

V9's preregistration is the FINAL open methodological question the
paper needs to close before submission. If V9's routers fail, the
paper documents the boundary; if V9's routers win, the paper gets a
positive agent-router contribution. Both outcomes are pre-registered
as acceptable.