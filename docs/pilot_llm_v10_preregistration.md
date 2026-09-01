# Pilot-LLM V10 preregistration: BoolQ (Wikipedia yes/no) — diverse-consensus replication

**Date frozen:** 2026-09-01
**Status:** Frozen BEFORE any V10 model call. All design decisions
below are pre-registered and were made WITHOUT inspecting V10 records.

## 1. Why V10

V9 §7 signposted V10 as the next open question. The V4-V9 trajectory
established:

| Domain | Prevalence | Agent consensus pattern | Router viable? |
|---|---:|---|---|
| TQA (V4) | 22% | heterogeneous (32% 3-2 splits) | ✅ R2 +0.43 vs BL_majority |
| FEVER (V5/V6/V7) | 92% | saturated (89% 5/5 unanimous) | ❌ R2 −0.50 vs BL_majority |

V9's anti-herding filter rejected 98.4% of FEVER clusters as
over-redundant, ruling out the "selection-driven" hypothesis. The
V4-V7 router failure is **structural**: FEVER's evidence design forces
5/5 agent consensus on wrong answers, leaving no answer diversity for
routers to exploit.

V10 is a **domain pivot**: use a structurally-different yes/no dataset
to test whether the **R2 (AUROC-weighted vote) router design** — which
won on TQA — generalizes to other diverse-consensus domains.

**BoolQ** is the natural V10 target:
- 9,427 train + 3,270 validation questions (HF mirror `google/boolq`)
- Yes/no format (identical protocol to TQA)
- Wikipedia passages as evidence (different source from TQA
  misconceptions)
- Answer distribution: 62% True / 38% False (similar to TQA's
  50/50 in the manifest) — **vastly more balanced than FEVER's 92%**
- ~5 MB total (HF mirror, downloaded 2026-09-01)
- 1.4% of FEVER's bandwidth cost

## 2. Status of prior pilots (preserved, not edited)

| Version | Outcome | Stop reason |
|---|---|---|
| Pilot-LLM V1 | Stopped at smoke | Generated StrategyQA claim can flip polarity while label is attached to original question |
| Pilot-LLM V2 | Stopped at smoke | Agent-level abstention erased false-consensus observations |
| Pilot-LLM V3 | Formal run completed (750/750) | Protocol-limited negative result (AUROC = 0.4125); not retried |
| Pilot-LLM V4 | Formal run completed (1,000/1,000) | TQA, D_OR = 0.676; shared_weighted = 0.785 |
| Pilot-LLM V5 | Formal run completed (1,000/1,000) | FEVER PARTIAL_PASS: D_OR=0.656, shared_weighted=0.698 (CI wide) |
| Pilot-LLM V6 | Formal run completed (2,000/2,000) | FEVER fresh-salt: D_OR=0.388; shared_weighted=0.820 |
| Pilot-LLM V7 | Formal run completed (2,000/2,000) | FEVER V5-salt: D_OR=0.621, shared_weighted=0.816 |
| Pilot-LLM V8 | Not executed | Ling cross-model: SGLang install exhausted bandwidth without bringing Ling online |
| Pilot-LLM V9 | Pre-formal audit only | Anti-herding filter rejected 98.4% of FEVER clusters; §11 pre-registered abort |
| **Pilot-LLM V10** | **This document** | — |

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

V10 inherits V5's substitute manifest mechanism (`D2_v10`). The
substitute rewrite prompt and seed (20_260_903) are unchanged. Each
V10 composite's substitute is an LLM rewrite of one of its 3
passages, generated once before the formal run.

## 4. Domain: BoolQ (HF mirror `google/boolq`)

V10 uses BoolQ train.parquet (9,427 questions, 3.6 MB) and
validation.parquet (3,270 questions, 1.3 MB). Each question has:
- `question` (string, yes/no answerable)
- `passage` (string, Wikipedia passage, ~100-300 words)
- `answer` (boolean)

**Composite-question construction (§4.3 pre-registered)**:

1. Group BoolQ questions by **Wikipedia page title** (extracted from
   passage text via the heuristic "the first 5 words of the passage
   in title case"). Each cluster must have ≥ 3 BoolQ questions on
   the same page to be eligible.
2. From each eligible cluster, take 3 questions (sorted by
   `SHA256("pilot-llm-v10-2026-09-01\n" || qid)`).
3. **3 evidence items** = the 3 passages from these questions.
4. **Composite question text** = concatenation of the 3 BoolQ
   questions (each with its `passage` as supporting evidence). Agent
   sees a 2-of-3 subset of passages (V5's partition table).
5. **Gold label** = unanimous 3-0 of BoolQ's boolean answers
   (converted to yes/no).
6. **Substitute condition**: one of the 3 passages is replaced by
   the LLM-rewritten negative paraphrase (V5 §6.3 mechanism).

**Manifest construction** (frozen):
- Salt: `pilot-llm-v10-2026-09-01` (new salt, fresh selection)
- N = 100 balanced 50/50 (50 True + 50 False), 3 questions per
  composite → 300 passages + 100 composite question text
- Stratification: top-50 per label (True/False) by SHA256 salt rank

## 5. Frozen paired evidence conditions (4, same as V5/V6/V7)

Inherited verbatim from V5 §6: `original`, `remove`, `reverse`,
`substitute`. The "evidence" for the BoolQ composite is the **passage**
field of each constituent question.

## 6. Frozen agents and response contract (inherited from V5 §7)

Identical to V5/V6/V7. Five agents: `literal_evidence`,
`skeptical_auditor`, `consistency_checker`,
`counterfactual_reasoner`, `minimal_judge`. Per-claim `yes`/`no`
answer, confidence in `[0, 1]`, citation-packet validation.

For BoolQ, the "packet" is the **passage ID** of each constituent
question. The "gold" answer is the boolean `answer` field of the
constituent question.

## 7. Frozen intervention and consensus definitions (inherited from V5 §8)

Inherited verbatim from V5 §8. `harmful_fc = correct == 0 AND
agreement >= 0.8`.

## 8. Frozen primary risk scores (D_OR + shared_weighted, co-primary, any-passes)

Inherited verbatim from V7 §9. Both D_OR and shared_weighted are
co-primary. §9.2 passes if at least one CI lo > 0.5.

### 8.3 Pre-registered secondary hypotheses (V10-specific)

| # | Hypothesis | Why it is required |
|---|---|---|
| S1 | **AUPRC(`D_OR`) > AUPRC(`D_majority`)** | Inherited from V5 |
| S2 | **`Risk@80%Coverage`(`D_OR`) does not exceed prevalence baseline by more than 0.05** | Inherited from V5 |
| S3 | **Calibration**: Brier(`D_OR`) and ECE(`D_OR`) < 0.30 after Platt LOO | Inherited from V5 |
| S4 | `shared_weighted` AUROC CI lo > 0.5 | Co-primary; reported not gating (V7 §9.3) |
| S5 | **Prevalence target** (V10 §4): `harmful_fc` prevalence ∈ [0.30, 0.70] | BoolQ is 62% True / 38% False at the raw label level. After agent consensus under Qwen3.5-4B, expected consensus-wrong rate is in this band (vs V5/V7's 92% on FEVER). If observed prevalence > 0.70, the §11 abort contingency fires. |
| S6 | **R2 (AUROC-weighted vote) generalizes from TQA**: R2 AUROC > BL_majority AUROC on V10 (vs V4's +0.43 advantage, vs V7's −0.50 disadvantage) | Central V10 question: does the V4 finding of R2 > BL_majority generalize to other diverse-consensus domains, or is it TQA-specific? |
| S7 | **V4 + V10 joint** (V4 + V10 use the same model Qwen3.5-4B but different datasets): R2 AUROC > 0.5 with CI lo > 0.5 on the pooled 150 questions (V4 N=50 + V10 N=100) | Tests whether the R2 signal is **model-robust + dataset-robust** (Qwen on TQA + BoolQ) |

## 9. Pre-registered agent-level router variants (frozen)

V10 uses the **same 3 routers + 2 baselines** as V4-V7 controlled-
heterogeneity (frozen in `analysis/agent_router_comparison.md`):

- **R1_top_auroc**: pick the answer from the agent with the highest
  V10 per-agent AUROC_fragility (computed on V10 records only, no V4
  leakage).
- **R2_weighted**: weighted majority vote with per-agent weight =
  clip(V10 per-agent AUROC, [0.5, 1.0]).
- **R3_min_frag**: pick the agent with the lowest fragility on this
  question.
- **BL_majority**: unweighted majority. Score = vote agreement.
- **BL_D_OR**: per-question D_OR score (1 - mean(inert OR conf_stable)).

### Reporting policy (frozen, locked 2026-09-01)
- **All 5 methods reported for V10**, no cherry-picking.
- V10 added to the V4-V7 cross-version router table (5 methods × 5
  versions = 25 cells).
- If R2 wins on V10 too, the **TQA + BoolQ pair confirms** R2's
  domain-robust efficacy. V8 (cross-model) becomes the next open
  question.
- If R2 loses on V10, the TQA finding is **dataset-specific**.
  V10 is a **boundary dataset** in either case.

## 10. Frozen metrics and instrumentation gates

### 10.0 Statistical unit

Same as V7 §11.0. Question-level bootstrap, seed `20260902`,
1,000 replicates.

### 10.1 Mandatory reporting set

For each of D_OR and shared_weighted: AUROC, AUPRC, Risk@80%,
Brier+ECE (Platt LOO), each with 95% CI.

For each of the four shared-citation detectors: AUROC + CI.

Plus: D_majority baseline, per-condition flip rates, harmful_fc
prevalence (S5), LOAO, substitute-generation pass stats.

The formal run passes only if:
- the BoolQ dataset digest matches the frozen SHA-256 (computed at
  freeze);
- the V10 manifest of 100 composites is reproducible from V10's salt;
- the substitute manifest is reproducible;
- substitute-generation pass left < 10% items unusable;
- at least 98% of 2,000 calls have an HTTP response or valid cache;
- at least 95% produce a strict yes/no decision after the fixed retry;
- at least 95% of (100 × 5 = 500) question-agent quadruplets are complete;
- every accepted citation is packet-validated.

## 11. Pre-registered contingencies

| Trigger | Response |
|---|---|
| S5 violated: `harmful_fc` prevalence > 0.70 | **§11 abort**: V10 follows V9's pattern — structural saturation observed on V10, no LLM call executed. Document as a V10-boundary finding. |
| Substitute generation yield < 90% | §6.3 fail-fast: 10% threshold inherited from V5. |
| vLLM endpoint unreachable at formal-run time | **§11 abort**: V8's path-dependence repeats. Document. |

## 12. Interpretation boundary

V10 tests the **R2 router's domain-robustness**. The 5-method × 5-version
table (V4/V5/V6/V7/V10) is the paper's main methodology-boundary
section. V10 does not address:
- S&P 500 predictability, investment performance
- LLM faithfulness in general
- Cross-model generalization (V8 is the next open question)

## 13. Registered deviations (added 2026-09-01, before any V10 formal call)

| # | Item | Preregistered analogue | Deviation | Why |
|---|---|---|---|---|
| `D1_v10` | Manifest construction (dataset) | V5/V7: FEVER triples; V9: anti-herding filter | **V10: BoolQ triples grouped by Wikipedia page title** | V10 pivots to BoolQ to test diverse-consensus regime. Wikipedia title grouping is the natural cluster identity for BoolQ (each passage is a Wikipedia page). |
| `D2_v10` | Salt | V5/V7: `pilot-llm-v5-2026-08-31`; V9: `pilot-llm-v9-2026-09-01` | **V10: `pilot-llm-v10-2026-09-01`** (fresh salt, fresh selection) | V10's selection is independent of V5/V7/V9 — same-family domains would not test the diverse-consensus hypothesis. |
| `D3_v10` | Co-primary verdict logic | V7 §9.2 any-passes (D_OR OR shared_weighted) | **Inherited unchanged** | V10 does not introduce a new verdict structure. |
| `D4_v10` | Router variants | V7 / V8: 3 routers + 2 baselines | **Inherited unchanged** (R1/R2/R3 = V9's; BL_majority, BL_D_OR) | The router design question is independent of dataset. V10 reuses V9's frozen router variants. |
| `D5_v10` | Wikipedia page title extraction | (n/a in V5-V9) | **Heuristic: first 5 words of passage, title-cased** | BoolQ's parquet does not include the page title explicitly. The 5-word heuristic is the closest stable proxy. Documented as a methodological shortcut. |

## 14. Operational additions

- `audit` subcommand: pre-formal checks (BoolQ digest, balanced
  manifest, Wikipedia clustering yield, substitute yield). Aborts if
  S5 violated.
- `all` subcommand: `prepare → substitute-generation → audit → smoke →
  formal`.
- `progress.json` written every call.
- Bash driver: `scripts/run_pilot_llm_v10.sh [--bg|--skip-smoke|--skip-formal]`.

## 15. Open question for V11 (not in scope of V10)

V8 (Ling cross-model) remains the only open question for the
methodology-boundary section. V11 is provisionally scoped as a
follow-up to V10: use a third diverse-consensus domain (HotpotQA
multi-hop, MMLU, or NaturalQuestions) to confirm the V4 + V10
two-domain R2 finding.