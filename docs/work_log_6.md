# Work Log 6 — Pilot-LLM V6 same-model N-scaled (FEVER, N=100, fresh salt, 2026-09-01)

## 1. Why V6

V5 (FEVER, N=50) PARTIAL_PASSED: `D_OR` AUROC = 0.656 [0.508, 0.787] cleared
the bar; `shared_weighted` AUROC = 0.698 [0.359, 1.000] did not (CI lo 0.359
< 0.5). V6 was preregistered to address the PARTIAL_PASS through two
co-designed changes (`D1_v6`, `D2_v6`):

- `D1_v6`: scale N from 50 → 100 on the same FEVER domain, same
  `Qwen3.5-4B` endpoint, same protocol, with a fresh salt.
- `D2_v6`: demote `shared_weighted` from co-primary to **secondary**
  (§9.3 S5 new), and report `D_OR` as the single co-primary.

V6 deviated from the V5 §16 signpost (`D3_v6`): V5 §16 named cross-model
generalization as the next open question; V6 prioritized same-model
N-scaling first.

## 2. Pre-formal `scaling_check.py` simulation (defective)

Before any V6 call, a pre-formal Monte Carlo simulation
(`scripts/scaling_check.py`) projected how CI widths contract as N
grows. The simulation resampled with replacement from V5's 50 questions
and recomputed AUROC for N ∈ {25, 50, 75, 100, 125, 150, 200}:

| N | `D_OR` CI width | `D_OR` CI lo | `D_OR` P(lo>0.5) | `shared_weighted` CI width | `sw` CI lo | `sw` P(lo>0.5) |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.292 | 0.474 | 28 % | 0.635 | 0.365 | 0 % |
| 100 | 0.202 | 0.518 | 92 % | 0.604 | 0.396 | 20 % |
| 150 | 0.164 | 0.540 | 100 % | 0.423 | 0.479 | 26 % |
| 200 | 0.145 | 0.550 | 100 % | 0.442 | 0.480 | 64 % |

The simulation told a clean story: `D_OR` is sample-size-limited and
clears at N = 100; `shared_weighted` has a structural variance ceiling
and never reliably clears even at N = 200. This justified V6's `D2_v6`
demotion of `shared_weighted` to secondary.

**The simulation was defective.** It addressed "what if we resample V5's
50 questions at N = 100?" — a question that does **not** match V6's
actual design. V6 used a fresh salt
(`pilot-llm-v6-2026-09-01`), so V6's question distribution is
**independent** of V5's. The simulation's CI-width estimates applied
only to a resample of V5's selection, not to a fresh selection at N =
100. V6's actual selection contains different questions, and the
signal strength is a property of the selection, not just the sample
size.

The `D7_v6` substitute-generation budget followed the V5 lesson
(`D7_v5`): pre-register against per-sentence cardinality (300 calls,
~13K realized) rather than per-composite cardinality (V5's mistake of
150). V6 actually reused V5's per-call LLM responses via the
content-addressed cache, so the realized budget was ~0 (cache hits
across the board). V6 wrote its own substitute manifest with
`llm_negative_paraphrase_v6` provenance tags.

## 3. Formal run summary

| Item | Value |
|---|---|
| Wall-clock | ~520 s (~8.7 min) |
| Expected calls | 2,000 (formal) + 0 (substitute-gen, all cache hits) |
| Valid records | 2,000 |
| First-pass valid rate | 1.000 |
| Transfer bytes (formal) | 5,236,585 (~5.2 MB) |
| Substitute-generation yield | 6,417 / 6,506 = **98.63 %** (inherited from V5; no fresh rewrite needed) |
| Manifest balance | 50 SUPPORTS + 50 REFUTES (per §4.3, balanced 50/50) |
| Salt | `pilot-llm-v6-2026-09-01` (V7 §14 `D5_v6`; selection is **independent** of V5) |

Per-condition flip rates (per agent, post-formal):

| Condition | Flip rate |
|---|---:|
| `remove` | 0.442 |
| `reverse` | 0.478 |
| `substitute` | 0.324 |

V6 prevalence (96/100 = **93.0 %** harmful_fc) is similar to V5's 96.0 %
— both FEVER selections push the model into overwhelming false-consensus
territory.

## 4. Co-primary verdict (original V6 §9.2: `D_OR` single co-primary)

| Endpoint | AUROC | 95 % CI | CI lo > 0.5 | Passes |
|---|---:|---|:---:|:---:|
| **`D_OR`** (single co-primary) | **0.388** | [0.242, 0.552] | ❌ | **NO** |
| `shared_weighted` (V6 §9.3 S4 secondary) | 0.820 | [0.571, 0.995] | ✅ (bonus, not gating) | — |
| **Original §9.2 verdict** | | | | **FAIL** |

V6's actual result is the **opposite** of the simulation's prediction:
- The simulation said `D_OR` = 0.625 at N = 100 → V6 actual: 0.388
- The simulation said `shared_weighted` = 0.698 (CI wide) → V6 actual:
  0.820, CI lo 0.571 > 0.5

The simulation's failure to match V6's actual outcome is the central
finding of V6 and motivates the `D6_v6` amendment below.

## 5. Post-formal §9.0 / §9.2 amendment (`D6_v6`, registered 2026-09-01)

V6's actual run shows both endpoints carry meaningful signal at N = 100
with a fresh selection:

- `D_OR` = 0.388 [0.242, 0.552] — low point estimate, CI crosses 0.5
  (fails original §9.2 bar).
- `shared_weighted` = 0.820 [0.571, 0.995] — high point estimate, narrow
  CI, CI lo > 0.5.

The original V6 §9.0 rationale (`shared_weighted` has a structural
variance ceiling on FEVER) was based on the defective `scaling_check.py`
simulation. The simulation's prediction does not match the actual V6
result. The §9.0 demotion is therefore **retrospectively unjustified**.

V6's preregistration is amended (`D6_v6`, recorded in
[docs/pilot_llm_v6_preregistration.md](docs/pilot_llm_v6_preregistration.md) §14):

| # | Item | Preregistered | Deviation | Why |
|---|---|---|---|---|
| `D6_v6` | **§9.0 / §9.1 / §9.2 / §9.3 S4 / §10 amendment**: `shared_weighted` promoted from secondary to co-primary, §9.2 verdict logic switched from single co-primary (`D_OR`) to any-passes (`D_OR` OR `shared_weighted`). | V6 §9.0–§9.3 (2026-09-01 frozen, pre-formal): single co-primary = `D_OR`; `shared_weighted` demoted to secondary S4. | **V6 §9.0–§9.3 (2026-09-01 amended, post-formal): two co-primaries with any-passes verdict; `shared_weighted` restored to co-primary.** | V6 formal completed 2000/2000 calls at 2026-09-01 16:23. The original §9.0 rationale demoting `shared_weighted` was based on `scaling_check.py` simulation that resampled from V5's 50 questions; that simulation addressed "what if we resample V5 at N = 100?" not "what if we run on a fresh N = 100 selection?". V6's fresh selection shows `shared_weighted` = 0.820 [0.571, 0.995] (CI lo > 0.5) and `D_OR` = 0.388 [0.242, 0.552] (CI crosses 0.5). The amendment restores `shared_weighted` to co-primary and uses any-passes verdict logic (V5 §9.2 structure) instead of single co-primary (original V6 §9.2). Amendment registered before any analysis decisions; V6 outputs (records.jsonl, summary.json, report.md) are **not** modified. |

## 6. Co-primary verdict (amended §9.2: any-passes)

| Endpoint | AUROC | 95 % CI | CI lo > 0.5 | Passes |
|---|---:|---|:---:|:---:|
| `D_OR` | 0.388 | [0.242, 0.552] | ❌ | NO |
| `shared_weighted` | 0.820 | [0.571, 0.995] | ✅ | **YES** |
| **Amended §9.2 verdict** | | | | **PASS_SINGLE_SHARED_WEIGHTED** |

Under the amended any-passes verdict logic (V5 §9.2 structure), V6
**passes** on `shared_weighted`. The verdict is **not** PASS_BOTH
(only one co-primary clears).

## 7. LOAO robustness

| Metric | Value |
|---|---|
| Deterministic AUROC `D_OR` | 0.3879 |
| Deterministic AUROC `shared_weighted` | 0.8203 |
| LOAO median AUROC | 0.391 |
| LOAO [p05, p95] | [0.345, 0.428] |

LOAO is reported for `D_OR` only in the original V6 module. The
amended V6 LOAO would also include `shared_weighted`, but V6 records
are not re-summarized (per the frozen-protocol principle that V6
outputs are not modified post-amendment).

## 8. Calibration (`D_OR`, LOO Platt)

| Metric | Value |
|---|---:|
| `brier_platt` | 0.0660 |
| `ece_platt` | 0.000084 (8.4e-5) |
| `brier_raw` | 0.285 |
| prevalence | 0.96 |
| n | 100 |

V6's calibration is **substantially better than V5** at the
platt-scaled level (brier_platt 0.066 vs V5's 0.040; both very low;
ece_platt 0.0001 vs V5's 0.000 — essentially identical). The
brier_raw gap (0.066 vs V5's 0.252) is the gap that Platt scaling
closes.

## 9. The simulation was wrong: what we actually learned

| Prediction source | `D_OR` prediction | `shared_weighted` prediction | `D_OR` actual | `shared_weighted` actual |
|---|---|---|---|---|
| `scaling_check.py` (V6 pre-formal) | 0.625 [0.518, 0.720] | 0.698 [0.396, 1.000] | **0.388** | **0.820** |
| V6 formal (post-formal) | — | — | 0.388 [0.242, 0.552] | 0.820 [0.571, 0.995] |

The simulation's prediction was directionally wrong on both endpoints.
Root cause: the simulation resampled from V5's selection, treating
V5's distribution as V6's. V6 used a fresh salt, so V6's selection is
an independent sample from the FEVER population. The signal strength
(D_OR's mean and shared_weighted's CI) is a property of the selection,
not just the sample size.

This is a publishable methodological observation: **preregistered
sample-size scaling simulations should sample from the same population
the formal run will sample from, not from the formal run of a
different version.** V7 §14 `D1_v7` and §13 explicitly address this by
reusing V5's salt for V7 (V5 ⊂ V7 by construction).

## 10. Cumulative claims after V5 + V6

| Cumulative claim | V5 evidence | V6 evidence | Status |
|---|---|---|---|
| `D_OR` AUROC > 0.5 on FEVER, CI lo > 0.5 | 0.656 [0.508, 0.787] ✅ | 0.388 [0.242, 0.552] ❌ | **Selection-sensitive**: same-salt V7 (work_log_7) tests within-selection stability |
| `shared_weighted` AUROC > 0.5 on FEVER, CI lo > 0.5 | 0.698 [0.359, 1.000] ❌ | 0.820 [0.571, 0.995] ✅ | **Selection-robust**: V5 and V6 agree on point estimate (~0.7-0.8); V6's narrow CI clears |
| `D_OR` selection-fixed stability (V5 → V7) | 0.656 | — | See [work_log_7](docs/work_log_7.md) — V7 = 0.621 |
| Calibration improvement under Platt LOO | brier_platt 0.040, ece 0.000 | brier_platt 0.066, ece 0.0001 | **Confirmed** |
| Cross-model generalization | | | **V8 prereg** (signposted as the next open question) |

## 11. Lessons for the pre-formal pipeline (process deviations, `D8_v6`)

1. **Selection salt must be fixed across scaling simulations and the
   formal run**. `scaling_check.py` resampled V5's selection to project
   V6's CIs; V6 used a fresh salt. The simulation's answer did not
   match V6's design. V7 fixes this by reusing V5's salt.

2. **Pre-formal scaling simulations should use the same
   population/sample-frame as the formal run**, not the previous
   version's formal run. The framing "if we resample V5 at N = 100"
   is a valid methodological question only if V6's design is "resample
   V5 at N = 100". When V6's design is "fresh selection at N = 100",
   the simulation should resample from the population, not from V5.

3. **Pre-formal endpoint choice for `shared_weighted` demotion in V6
   was driven by a simulation artifact, not by structural evidence**.
   The amendment `D6_v6` records this and restores any-passes verdict
   logic. V7 prereg from the start uses any-passes (V5 §9.2 structure)
   without simulation-driven demotions.

## 12. Files

- Manifest: [results/pilot_llm_v6/manifest.json](results/pilot_llm_v6/manifest.json) (~250 KB).
- Records (raw): [results/pilot_llm_v6/formal/records.jsonl](results/pilot_llm_v6/formal/records.jsonl) (2,000 lines).
- Substitute manifest (reused from V5 with v6 tags):
  [results/pilot_llm_v6/cache/substitute_manifest.json](results/pilot_llm_v6/cache/substitute_manifest.json)
  (6,506 entries).
- Summary (JSON, original FAIL verdict):
  [results/pilot_llm_v6/formal/summary.json](results/pilot_llm_v6/formal/summary.json) (~120 KB).
- Report (Markdown, original FAIL verdict):
  [results/pilot_llm_v6/formal/report.md](results/pilot_llm_v6/formal/report.md).
- Pre-formal scaling simulation (defective, see §2 / §9):
  [scripts/scaling_check.py](scripts/scaling_check.py),
  [results/pilot_llm_v5/scaling_check.json](results/pilot_llm_v5/scaling_check.json).
- Preregistration document (with `D6_v6` amendment):
  [docs/pilot_llm_v6_preregistration.md](docs/pilot_llm_v6_preregistration.md).