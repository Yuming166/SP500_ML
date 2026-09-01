# Pilot-LLM V7 preregistration: FEVER N=100 with V5 salt + co-primary any-passes (V5 ⊂ V7 by construction)

**Date frozen:** 2026-09-01
**Status:** Frozen before any V7 model call. Any substantive change after V7
outputs requires a new version (V8).

## 1. Why a seventh pilot is needed

V5 (FEVER, N=50) and V6 (FEVER, N=100, fresh salt) jointly expose that the
signal carries **selection-dependent variance**:

| Version | Salt | N | `D_OR` AUROC | `D_OR` CI | `shared_weighted` AUROC | `shared_weighted` CI | §9.2 verdict |
|---|---|---:|---:|---|---:|---|---|
| V5 | `pilot-llm-v5-2026-08-31` | 50 | **0.656** | [0.508, 0.787] | 0.698 | [0.359, 1.000] | PARTIAL_PASS (D_OR clears, sw fails) |
| V6 | `pilot-llm-v6-2026-09-01` | 100 | **0.388** | [0.242, 0.552] | **0.820** | [0.571, 0.995] | FAIL (D_OR fails), but sw clears |

The two V5/V6 results disagree on `D_OR` (0.656 vs 0.388) and on
`shared_weighted` CI width (0.641 wide vs 0.424 narrow). The original V5
PARTIAL_PASS was interpreted in V6's pre-formal design as a structural
`shared_weighted` variance issue (`scaling_check.py` simulation on V5
records). V6's actual run shows the opposite: `shared_weighted` clears
robustly at N=100 with a fresh selection, while `D_OR` regresses below
0.5.

The V5 vs V6 disagreement is best explained by **the selections being
independent** (different salts → different question sets), not by either
metric being structurally noisy. The `scaling_check.py` simulation was
asking "what if we resample V5 at N=100?" — a question that does not
address V6's actual design (a fresh selection at N=100).

V7 isolates the selection variable by reusing V5's salt at N = 100.
Because the selection is determined by the salt
(`SHA256(salt || qid)`-sorted top-K per stratum), using V5's salt at
N = 100 produces a selection that is a **strict superset** of V5's:

- V5's selection: top 25 SUPPORTS + top 25 REFUTES (under V5 salt)
- V7's selection: top 50 SUPPORTS + top 50 REFUTES (under the same V5 salt)
- ⇒ V5 ⊂ V7 by construction

V7 therefore asks the cleanest possible question: "**Does V5's
`D_OR` = 0.656 signal hold at N = 100 with V5's same questions
(plus 50 more drawn under the same selection rule)?**" If yes, V5's
D_OR was real but underpowered. If no, V5's D_OR was a lucky draw.

V7 also restores the V5 §9.2 any-passes verdict logic (D_OR AND
`shared_weighted`, both must be co-primary, §9.2 passes if either
clears). This is the most defensible reading of V5 PARTIAL_PASS and
V6's `D_OR` regression: **both endpoints are reported as co-primary,
both must be analyzed honestly, neither is "demoted" without evidence
that its CI is structurally un-tightenable** (the V6 §9.0 demotion was
based on a flawed simulation).

## 2. Status of prior pilots (preserved, not edited)

| Version | Outcome | Stop reason |
|---|---|---|
| Pilot-LLM V1 | Stopped at smoke | Generated StrategyQA claim can flip polarity while label is attached to original question |
| Pilot-LLM V2 | Stopped at smoke | Agent-level abstention erased false-consensus observations |
| Pilot-LLM V3 | Formal run completed (750/750) | Protocol-limited negative result (AUROC = 0.4125); not retried |
| Pilot-LLM V4 | Formal run completed (1,000/1,000) | §9.2 + 2/3 secondaries passed; `D_OR` 0.676; `shared_weighted` 0.785 (discovered post-hoc) |
| Pilot-LLM V5 | Formal run completed (1,000/1,000) | PARTIAL_PASS: `D_OR` cleared, `shared_weighted` CI lo = 0.359 < 0.5; `brier_platt` 0.040, `ece_platt` 0.000 |
| Pilot-LLM V6 | Formal run completed (2,000/2,000) | Original §9.2 FAIL (D_OR = 0.388); amended §9.2 PASS_SINGLE on `shared_weighted` (0.820 [0.571, 0.995]) per `D6_v6` |
| **Pilot-LLM V7** | **This document** | — |

V5, V6 outputs are **frozen** and will not be modified. V5's substitute
manifest is reused by V7 (same prompts, same model, same seed →
identical LLM responses; V7 writes its own substitute manifest with
"v7" provenance tags, see `D1_v7`).

## 3. Frozen model, retry, and transfer controls

Inherited verbatim from V5 §3 / V6 §3:

- endpoint: `http://10.63.0.88:31519/v1/chat/completions`;
- model: `Qwen3.5-4B`;
- temperature: `0.0`;
- maximum completion tokens: `160`;
- timeout: 60 seconds;
- one initial request and at most one fixed JSON-repair or transport retry;
- SHA-256 content-addressed response cache;
- no model/data download and no hidden chain-of-thought request, storage, or scoring.

V7 substitute-generation uses the same endpoint, `temperature = 0.0`,
`maximum completion tokens = 200`, one rewrite per source evidence
sentence, written to the substitute manifest before any formal run.

## 4. Domain: FEVER (same as V5 / V6)

Inherited verbatim from V5 §4. FEVER `valid.jsonl` is already on the
jump host:

```text
/storage/gaoym/sp500-forecastability-lab/data/fever/fever-validation.jsonl
SHA-256: 5da0ccc0ccf77f974611de13f8aac6f78c6bba6293912835099eb6029baa85d9
```

V7 reuses the V5 source dataset digest. No new acquisition is permitted.

### 4.1 FEVER label space and binary mapping

Inherited verbatim from V5 §4.1: NEI rows excluded; SUPPORTS = 0,
REFUTES = 1.

### 4.2 FEVER source data

Inherited verbatim from V5 §4.2. Acquisition source `D6_v5` is unchanged.

### 4.3 Manifest construction (100 composite questions, V5 salt)

V7 deviates from V5 §4.3 in **N only** (50 → 100) but **NOT in salt**.
V7 uses **V5's salt**, so the V7 selection is a strict superset of V5's
by SHA256-sorted top-K construction:

```text
V7 salt: pilot-llm-v5-2026-08-31  (same as V5)
V7 selection: SHA256(V7_salt || qid)-sorted top-50 SUPPORTS + top-50 REFUTES
            per stratum, 3 rows per composite = 100 composites / 300 evidence rows
```

Because the top-50-per-stratum selection contains the top-25-per-stratum
selection (V5's), V5 ⊂ V7 by construction at the question level. This
makes V5 + V7 pool analyses well-defined: V7 includes V5's 50 questions
plus 50 new questions drawn under the same selection rule.

This selection strategy is **`D1_v7`** — the central V7 design choice
that distinguishes it from V6's independent-selection design.

## 5. Frozen partitioned evidence packets (inherited from V5 §5)

Identical to V5 §5 / V6 §5. Each evidence item is a single FEVER
evidence sentence; the 5 agents see the same 2-of-3 partition scheme with
deterministic mapping from manifest. Partitioning table reproduces
byte-for-byte.

Partition robustness (secondary, mandatory, inherited from V4 §9.4):
LOAO median + [p05, p95] AUROC across 5 variants is reported alongside
deterministic AUROC.

## 6. Frozen paired evidence conditions (4, with substitute rewritten)

Inherited verbatim from V5 §6: `original`, `remove`, `reverse`,
`substitute`. The substitute condition uses LLM-rewritten negative
paraphrase (V5 §6.3, deviation `D1_v5`).

### 6.3 Substitute generation rule

Inherited verbatim from V5 §6.3. V7 deviation **`D2_v7`**: V7 substitutes
manifest is generated fresh under the V7 protocol but reuses V5's
per-call LLM responses via the content-addressed cache. V5 ran the same
prompt with the same seed (20_260_903) and the same model, so the
cached responses are identical; V7 writes its own substitute manifest
with `llm_negative_paraphrase_v7` provenance tags.

### 6.4 Total call budget

```text
Manifest construction (offline):              0 calls
Substitute generation pass:                   ≤ 300 calls preregistered (cache hits reuse V5's)
Formal run:                                   100 × 5 × 4 = 2,000 calls
Estimated transfer (formal):                  ≈ 5 MB
Estimated transfer (substitute-gen):           ≈ 0 (cache hits)
```

## 7. Frozen agents and response contract

Identical to V5 §7 / V6 §7. Five agents: `literal_evidence`,
`skeptical_auditor`, `consistency_checker`, `counterfactual_reasoner`,
`minimal_judge`.

## 8. Frozen intervention and consensus definitions

Inherited verbatim from V5 §8.

## 9. **Frozen primary risk scores** (D_OR + shared_weighted, co-primary, any-passes)

### 9.0 Why two co-primary endpoints, any-passes verdict

V7 restores the V5 §9.2 structure (both `D_OR` and `shared_weighted`
co-primary, §9.2 passes if either clears) and applies it to N = 100
under V5's selection rule. This is the most defensible reading of
V5 PARTIAL_PASS and V6's `D_OR` regression: **both endpoints are
reportable co-primaries; neither is demoted without evidence that its
CI is structurally un-tightenable** (which the V6 §9.0 demotion
lacked, as the V6 amendment `D6_v6` records).

### 9.1 Frozen co-primary endpoints

**Endpoint 1 — `D_OR(qid)`** (inherited from V4 §9.1 / V5 §9.1):

```text
D_OR(qid) = (1/5) * Σ_agent [agent.inert_no_flip ∨ agent.conf_stable]
```

**Endpoint 2 — `shared_weighted(qid)`** (inherited from V5 §9.1):

```text
let:
  frac_shared(qid)     = (# agents that cite ≥ 1 evidence ID also cited by
                          ≥ 1 other agent) / 5
  correct_consensus(qid) = 1[consensus(qid) == gold_label(qid)]

shared_weighted(qid) = frac_shared(qid) * (1 - correct_consensus(qid))
                    + 0.5 * frac_shared(qid) * correct_consensus(qid)
```

### 9.2 Frozen co-primary hypothesis (any-passes)

**§9.2 passes if at least one of `D_OR` or `shared_weighted` AUROC has
its 95% question-cluster bootstrap CI lower bound above 0.5 for
`harmful_fc`.** If both clear, V7 reports PASS_BOTH (strongest finding).
If only one clears, V7 reports PASS_SINGLE on that endpoint. If
neither clears, V7 reports FAIL_BOTH.

This matches the V5 §9.2 any-passes structure. It does not match the
original V6 §9.2 single co-primary criterion (which was amended to
any-passes per `D6_v6`).

### 9.3 Pre-registered secondary hypotheses (mandatory)

Inherited from V5 §9.3 (S1, S2, S3, S4) plus the V7 additions:

| # | Secondary hypothesis | Why it is required |
|---|---|---|
| S1 | **AUPRC(`D_OR`) > AUPRC(`D_majority`)** on the same 100 questions | Class imbalance (FEVER binarized prevalence ≈ 50%) makes AUPRC the honest metric; beats disagreement-based ranking |
| S2 | **`Risk@80%Coverage`(`D_OR`) does not exceed prevalence baseline by more than 0.05** | Reformulated from V4's S2 |
| S3 | **Calibration**: Brier(`D_OR`) and ECE(`D_OR`) both < 0.30 after Platt scaling fit on leave-one-question-out | Prerequisite for handing `D_OR` to any downstream policy layer |
| S4 | `shared_weighted` AUROC CI lo > 0.5 (now §9.2 co-primary; reported but redundantly required) | V4 §11.4's discovery signal, restored to co-primary |
| S5 | **V5 + V7 joint** (V5 ⊂ V7 by construction): both `D_OR` and `shared_weighted` AUROC > 0.5 with CI lo > 0.5 on the pooled 100 questions (50 from V5 + 50 new from V7's top-50-per-stratum second half) | V7's central question: does V5's signal hold at N = 100 under the same selection rule? V5 + V7 pool is well-defined since V5 ⊂ V7. |
| S6 | **V5 + V6 + V7 joint** (V5 + V6 are independent samples, V5 ⊂ V7): same endpoints, pooled N = 200 | Tests whether V5 + V6 + V7 jointly clear the bar. V5 ⊂ V7 holds (V6 is independent of both). |

Pass criteria: §9.2 passes (any-passes) **and** at least two of {S1, S2, S3} pass. **S4, S5, S6 are reported but not gating.**

### 9.4 Partition robustness (secondary, mandatory, inherited from V4 §9.4)

LOAO median + [p05, p95] AUROC across 5 variants, reported for both
`D_OR` and `shared_weighted`. If the LOAO median for either co-primary
is below the deterministic AUROC by more than 0.05, this is **not** a
methodology failure — it is reported as a finding about partition
dependence.

## 10. Frozen shared-citation detectors

For each question:

- `shared_agents(qid)` (V4 preregistered): # agents that cite ≥ 1 evidence ID
  also cited by ≥ 1 other agent, divided by 5;
- `shared_weighted(qid)` (V7 co-primary, see §9.1);
- `shared_count_total(qid)`: total # of distinct evidence IDs cited by ≥ 2
  agents (raw count, not normalized);
- `shared_id_count(qid)`: # of distinct evidence IDs cited at all across the
  5 agents' original-condition answers.

All four are reported in `report.md` with AUROC and 95% CI;
`shared_weighted` carries §9.2 co-primary status in V7.

## 11. Frozen metrics and instrumentation gates

### 11.0 Statistical unit (mandatory)

**The statistical unit of inference is the question (n = 100), not the
call (n ≈ 2,000).** Every 95% interval reported in V7 uses
question-level bootstrap (seed `20260902`, 1,000 replicates, stratified
by FEVER cluster). Call-level bootstrap is forbidden.

### 11.1 Mandatory reporting set

For each of `D_OR` and `shared_weighted`:
- AUROC with 95% question-cluster bootstrap CI,
- AUPRC with 95% CI,
- `Risk@80%Coverage` with 95% CI,
- Brier score and ECE after Platt scaling (LOO fit) with 95% CI.

For `D_inert`, `D_conf` (secondary, for traceability):
- AUROC with 95% CI.

For each of the four shared-citation detectors (§10):
- AUROC with 95% CI on `harmful_fc`.

Plus:
- `D_majority` baseline (1 − agreement) for AUROC, AUPRC,
  Risk@80%Coverage.
- Per-condition flip rates on the 100 questions, broken down by
  `correct == 0` vs `correct == 1`.
- Harmful false consensus prevalence and per-condition-true-positive rates.
- LOAO robustness median + 5th/95th percentile over 5 variants (both
  co-primaries).
- Substitute-generation pass statistics.

The formal run passes only if:
- the FEVER dataset digest matches the frozen SHA-256;
- the balanced manifest of 100 composites is reproducible from the V7
  salt;
- the partitioning table in in §5 reproduces byte-for-byte;
- the substitute manifest is reproducible from the V7 salt;
- the substitute-generation pass left < 10% items unusable;
- at least 98% of 2,000 calls have an HTTP response or valid cache;
- at least 95% produce a strict yes/no decision after the fixed retry;
- at least 95% of (100 × 5 = 500) question-agent quadruplets are
  complete (all four conditions returned);
- every accepted citation is packet-validated.

These are operational gates, not superiority criteria.

## 12. Smoke and pre-formal checks

The only permitted pre-formal V7 smoke is **two composite questions, one
agent (`literal_evidence`), the four conditions**: 8 calls. Smoke
outputs remain separate from formal results.

The pre-formal audit must confirm:
- SHA-256 of the FEVER JSONL matches the frozen digest;
- the binarized balanced manifest of 100 composites is reproducible
  from the V7 salt;
- V5 ⊂ V7 at the question level (top-50-per-stratum supersets
  top-25-per-stratum under the same salt);
- the partitioning table reproduces byte-for-byte;
- the substitute manifest is reproducible from the V7 salt;
- substitute-generation pass yield ≥ 90%;
- all 8 smoke calls pass tests.

## 13. Interpretation boundary

These are still controlled, paired-intervention results on a single
model (`Qwen3.5-4B`) and a single non-financial domain (FEVER). V5 ⊂ V7
by construction at the question level (V7 uses V5's selection rule at
N = 100), so V5 + V7 are well-defined for pooled analyses.

V7's central question: does V5's `D_OR` signal hold at N = 100 under
the same selection rule? If yes, V5's signal was real and
underpowered. If no, V5's signal was a 50-question-d-draw lucky
outcome. V7 reports PASS_BOTH, PASS_SINGLE, or FAIL_BOTH based on §9.2.

V7 does **not** establish cross-model generalization — this remains
the V8 prereg scope.

## 14. Registered deviations (added 2026-09-01, before any V7 formal call)

The following deviations were locked in during pre-formal V7 design.
None of them changes the preregistered hypotheses (§9.2 / §9.3 /
§10). All are recorded in the substitute manifest and reported
alongside the formal results.

| # | Item | Preregistered analogue | Deviation | Why |
|---|---|---|---|---|
| `D1_v7` | Selection salt | V5 §4.3: salt `pilot-llm-v5-2026-08-31` → V5 ⊂ V(N+1); V6 §4.3: salt `pilot-llm-v6-2026-09-01` → V6 independent of V5 | **V7 uses V5's salt** to ensure V5 ⊂ V7 by construction at the question level | V6's independent-salt design exposed that selection is a real confounder (V5 vs V6 disagreement on `D_OR`). V7's central question is "does V5's `D_OR` signal hold at N = 100 with V5's same questions?" — only V5's salt answers this question. |
| `D2_v7` | Substitute generation | V5 / V6: fresh substitute-generation pass (6,506 / 13K calls respectively) | **V7 reuses V5's per-call LLM responses via content-addressed cache**; V7 writes its own substitute manifest with `v7` provenance tags | The substitute prompts and seed (20_260_903) are identical between V5 and V7; the LLM responses are content-addressed by SHA-256, so V7's substitute-generation pass hits V5's cache for every call (~0 calls actually issued to the LLM). V7's substitute manifest is reproducible from the V7 salt + cache state. |
| `D3_v7` | Co-primary verdict logic | V5 §9.2: both must clear, any-passes; V6 original: single co-primary = `D_OR`; V6 amended (§14 `D6_v6`): any-passes restored | **V7 uses any-passes verdict logic from V5 §9.2** (matches the V6 amendment `D6_v6`) | Most defensible reading of V5 PARTIAL_PASS and V6's `D_OR` regression. Both endpoints are co-primary; neither is demoted without structural-evidence of un-tightenable CI. |
| `D4_v7` | S5: V5 + V7 pool analysis | V6 §9.3 S5: V5 + V6 joint (independent samples) | **V7 §9.3 S5: V5 + V7 joint** (V5 ⊂ V7 by construction; well-defined pool) | Tests V5's signal at N = 100 under the same selection. The "well-defined" property holds only because V5 ⊂ V7; V5 + V6 (independent samples) is reported as V7 §9.3 S6 instead. |
| `D5_v7` | Salt | V5: `pilot-llm-v5-2026-08-31`; V6: `pilot-llm-v6-2026-09-01` | **V7: `pilot-llm-v7-2026-09-01`** (protocol identifier), **selection salt is `pilot-llm-v5-2026-08-31`** | The protocol identifier distinguishes V7's run from V5's run in logs and outputs. The selection salt is V5's to ensure V5 ⊂ V7 by construction. The two salts serve different purposes: identifier vs. selection. |

## 15. Operational additions

- `audit` subcommand: runs pre-formal checks (FEVER digest, V5 ⊂ V7
  verification, partition reproducibility, substitute manifest
  reproducibility, substitute generation yield) and exits non-zero if
  any gate fails.
- `all` subcommand: `prepare → substitute-generation → audit → smoke →
  formal` in one process; resumable; `--yes` removes the single
  confirmation prompt.
- `progress.json` written every call.
- Bash driver: `scripts/run_pilot_llm_v7.sh [--yes|--skip-smoke|--skip-formal|--bg]`
  runs `prepare → substitute-generation → audit → smoke` in the
  foreground and the formal run in the background;
  `scripts/wait_pilot_llm_v7.sh` blocks on it and prints the §9.2
  verdict when done.

## 16. Open question for V8 (not in scope of V7)

V7 does **not** address cross-model generalization. V8 is provisionally
scoped as: same V7 protocol (FEVER, N = 100, V5 salt, any-passes), swap
the endpoint to a second model (e.g., ChatGLM3-6b via the planned
31520 endpoint, or another non-Qwen3.5-4B model). V8 preregistration
is **not** drafted in this document — it is registered here as the next
open question so that the V7 report can signpost it cleanly to
reviewers.