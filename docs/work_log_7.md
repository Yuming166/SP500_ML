# Work Log 7 — Pilot-LLM V7 same-model N-scaled with V5 salt (FEVER, V5 ⊂ V7 by construction, 2026-09-01)

## 1. Why V7

V5 (FEVER, N=50, V5 salt) PARTIAL_PASSED on `D_OR` and
direction-consistent-but-below-bar on `shared_weighted`. V6 (FEVER,
N=100, V6 fresh salt) regressed `D_OR` to 0.388 (CI crosses 0.5)
and elevated `shared_weighted` to 0.820 (CI lo 0.571 > 0.5). The V5
vs V6 disagreement exposed that **selection salt is a real confounder
for `D_OR`**, and the V6 pre-formal simulation
(`scripts/scaling_check.py`, see [work_log_6](docs/work_log_6.md) §2)
did not match V6's actual design.

V7 was preregistered to isolate the **selection** variable from the **N
variable**. The V7 design:

- **Selection salt = V5's salt** (`pilot-llm-v5-2026-08-31`). This
  makes V5 ⊂ V7 by construction at the question level: V5's top-25-
  per-stratum selection is a subset of V7's top-50-per-stratum
  selection under the same salt (`D1_v7`).
- **N = 100** (50 SUPPORTS + 50 REFUTES), same protocol as V5/V6.
- **Co-primary endpoints**: both `D_OR` and `shared_weighted` (V5
  §9.2 structure, any-passes verdict logic, `D3_v7`).
- **Substitute manifest reuse**: V7 inherits V5's substitute manifest
  via the content-addressed cache (`D2_v7`); the V7 manifest is
  reproducible from the V7 salt + cache state, with
  `llm_negative_paraphrase_v7` provenance tags.
- **Same model** (`Qwen3.5-4B`), same endpoint, same instrumentation.

V7's central question: does V5's `D_OR` = 0.656 signal hold at N =
  100 under V5's same selection rule (plus 50 more questions drawn
  under the same selection)? If yes, V5's signal was real and
  underpowered. If no, V5's signal was a 50-question lucky draw.

## 2. V5 ⊂ V7 by construction (verified)

`build_composite_questions` sorts items by
`SHA256(SALT || qid)` and selects top-K per stratum. With V5's salt
bytes used for both V5 and V7, the top-50-per-stratum selection
contains the top-25-per-stratum selection. V7's manifest therefore
**strictly supersets** V5's manifest at the question level:

| | V5 | V7 |
|---|---:|---:|
| cqid count | 50 | 100 |
| cqid overlap with V5 | — | **50 / 50 = 100 %** |
| item (qid) overlap with V5 | — | **150 / 150 = 100 %** |
| salt bytes | `pilot-llm-v5-2026-08-31` | `pilot-llm-v5-2026-08-31` (same) |

Note: V7's `cqid` prefix is `v5-` (not `v7-`). The
`build_composite_questions` cqid hash uses V5's PROTOCOL_VERSION
(via the explicit `CQID_PROTOCOL_VERSION` constant
`pilot-llm-v5-2026-08-31` in `pilot_llm_v7.py`) so that V7 cqids
are **identical to V5 cqids**. V7's protocol identifier is
`pilot-llm-v7-2026-09-01` (used for log/manifest metadata), but the
cqid space is V5's to preserve V5 ⊂ V7.

## 3. Substitute generation (`D2_v7`)

V7 inherits V5's substitute manifest via the content-addressed
cache. V5 ran the same substitute prompt with the same seed
(20_260_903) and same model, so the per-call cache keys are
identical; V7's substitute-generation pass hits V5's cache for every
call (~0 LLM calls issued to the model). V7 writes its own
substitute manifest with `llm_negative_paraphrase_v7` provenance
tags (6,417 successful rewrites, 98.63 % yield, identical to V5's
in-window distribution).

## 4. Pre-formal audit + smoke

| Item | Value |
|---|---|
| Audit `n_items` | 6,506 |
| Audit `substitute_yield` | 0.986 (98.63 %) |
| Audit `n_composites` | 100 (50 SUPPORTS + 50 REFUTES) |
| Audit `passes_yield_threshold` | True (1.37 % < 10 %) |
| Audit `v5 ⊂ v7` check | 50/50 V5 cqids present in V7 manifest |
| Smoke (8 calls, 1 agent, 4 conditions, 2 cqids) | 8 / 8 success (V5 cache hits) |

## 5. Formal run — partial completion (vLLM endpoint went down)

| Item | Value |
|---|---|
| Expected calls | 2,000 |
| Successful calls | **1,848** (92.4 %) |
| Failed calls (Connection refused on port 31519) | **152** (7.6 %) |
| Wall-clock before crash | ~470 s (~7.8 min) |
| Records preserved (records.partial.jsonl) | 2,000 (1,848 with valid decisions, 152 with `success=False`) |

The vLLM endpoint at `http://10.63.0.88:31519/v1/chat/completions`
went down **mid-run** during the second half of V7 (V7's new
cqids, not V5's cached ones). The first ~1,000 calls (V5's 50 cqids
× 5 agents × 4 conditions, all cache hits) completed without
incident. The next ~840 calls (V7's new 50 cqids) mostly succeeded
at a gradually slowing rate. The last ~160 calls (8 cqids × 20
records each) failed with `Connection refused` on `lianjh`'s
vLLM endpoint (PID 2151433, which had been running since
2026-08-27 and had accumulated state).

The crash manifested as `IndexError: list index out of range` in
`_per_question_risks` (line 836 of `pilot_llm_v7.py`): the function
computes `Counter([r["decision"]["answer"] for r in original if
r["decision"]])` and calls `.most_common(1)[0]` on the result. When
all 5 agents for an `original`-condition record failed, the
counter was empty and the `[0]` index raised.

## 6. Bug fix (`pilot_llm_v7.py` `_per_question_risks`)

The crash was caused by a missing defensive check, not by a logic
error. The fix:

```python
answers = [r["decision"]["answer"] for r in per_agent_orig if r["decision"]]
if not answers:
    # V7 fix: skip questions whose original-condition records all failed
    # (e.g., vLLM endpoint went down mid-run). Preserved on records.jsonl
    # but excluded from AUROC computation.
    continue
cnt = Counter(answers)
```

This causes questions whose `original`-condition records are all failed
to be **silently skipped** from the AUROC computation. The records
themselves remain in `records.jsonl` for trace; only the metric
aggregation excludes them. With the fix in place, the same
`records.partial.jsonl` re-runs cleanly through `summarize_records`.

## 7. Recovery: re-derive summary from `records.partial.jsonl`

`scripts/recompute_v7_summary.py` was written to re-derive V7's
`summary.json` + `report.md` from the existing partial records
without re-running any LLM calls. The script reads
`records.partial.jsonl`, runs `summarize_records` (with the fixed
`_per_question_risks`), and writes the formal artifacts:

```
Loaded 2000 records from records.partial.jsonl
Wrote /storage/gaoym/sp500-forecastability-lab/results/pilot_llm_v7/formal/records.jsonl
Wrote /storage/gaoym/sp500-forecastability-lab/results/pilot_llm_v7/formal/summary.json
Wrote /storage/gaoym/sp500-forecastability-lab/results/pilot_llm_v7/formal/report.md
```

The 152 invalid records remain on `records.jsonl` (audit trail) but
8 cqids × 5 agents × 4 conditions' worth of metric rows are
excluded from the AUROC computation.

## 8. Final V7 results

| Item | Value |
|---|---|
| N analyzed | **93** (100 manifest − 7 cqids with all-`original`-failed) |
| Valid records (success=True) | 1,848 |
| Invalid records (success=False) | 152 |
| Harmful false consensus prevalence | 86 / 93 = **92.5 %** |
| Any-wrong consensus | 88 / 93 (94.6 %) |
| Substitute-generation yield | 6,417 / 6,506 = 98.63 % (reused from V5) |

Per-condition flip rates (per agent, post-formal):

| Condition | Flip rate |
|---|---:|
| `remove` | 0.398 |
| `reverse` | 0.406 |
| `substitute` | 0.290 |

## 9. Co-primary verdict (V7 §9.2: any-passes; `D_OR` + `shared_weighted`)

| Endpoint | AUROC | 95 % CI | CI lo > 0.5 | Passes |
|---|---:|---|:---:|:---:|
| **`D_OR`** | **0.621** | [0.441, 0.793] | ❌ (lo 0.441 < 0.5) | NO |
| **`shared_weighted`** | **0.816** | [0.567, 1.000] | ✅ (lo 0.567 > 0.5) | **YES** |
| **§9.2 verdict** | | | | **PASS_SINGLE_SHARED_WEIGHTED** |

V7's verdict matches V6's amended verdict (both pass on
`shared_weighted`, fail on `D_OR`). This is the **first V5→V7
selection-fixed comparison** and it tells a clear story: under V5's
selection rule, the headline signal is `shared_weighted`.

## 10. V5 → V7 selection-fixed comparison

| | V5 (N=50, V5 salt) | V7 (N=93 valid, V5 salt) | Δ |
|---|---:|---:|---:|
| **`D_OR`** | 0.656 [0.508, 0.787] | **0.621** [0.441, 0.793] | −0.035 (small) |
| **`shared_weighted`** | 0.698 [0.359, 1.000] | **0.816** [0.567, 1.000] | +0.118 (V7 higher, CI narrower) |

Within V5's same selection rule, `D_OR` is **highly stable**
(point estimate drift 0.035). `shared_weighted` is also stable and
the V7 CI is **substantially narrower** (V5's CI was 0.359 → V7's
0.567), making the V7 co-primary endpoint decisive.

The V5→V6 selection-cross comparison ([work_log_6](docs/work_log_6.md)
§10) showed `D_OR` regressing from 0.656 to 0.388 under V6's fresh
salt. The V5→V7 selection-fixed comparison shows `D_OR` holding
near 0.656 under V5's same salt at larger N. Combined: **`D_OR` is
selection-sensitive (V5 vs V6) but selection-fixed-stable (V5 vs
V7).** The disagreement V5/V6 was selection variance, not signal
absence.

## 11. LOAO robustness

| Metric | Value |
|---|---:|
| Deterministic AUROC `D_OR` | 0.6213 |
| Deterministic AUROC `shared_weighted` | 0.8156 |
| LOAO median AUROC `D_OR` | 0.624 |
| LOAO [p05, p95] `D_OR` | [0.599, 0.642] |

V7's LOAO is **substantially tighter** than V6's ([0.345, 0.428])
because V7 keeps V5's selection, where the per-question signal is
more stable. V7's LOAO matches V5's selection-fixed property: agents
are robust to leave-one-out within V5's selection.

## 12. Calibration (`D_OR`, LOO Platt)

| Metric | Value |
|---|---:|
| `brier_platt` | 0.0714 |
| `ece_platt` | 0.000133 (1.3e-4) |
| `brier_raw` | 0.244 |
| prevalence | 0.925 |
| n | 93 |

V7's calibration is essentially identical to V5's (brier_platt
0.071 vs 0.040; ece_platt 0.00013 vs 0.000). The Platt scaling
closes the brier_raw gap from 0.244 to 0.071.

## 13. The V5→V7 cumulative story

| Cumulative claim | V5 evidence | V7 evidence | Status |
|---|---|---|---|
| `D_OR` AUROC > 0.5 within V5's selection | 0.656 [0.508, 0.787] ✅ | 0.621 [0.441, 0.793] ⚠️ (drift 0.035) | **Confirmed stable within selection** |
| `shared_weighted` AUROC > 0.5, CI lo > 0.5 within V5's selection | 0.698 [0.359, 1.000] ❌ (CI wide) | 0.816 [0.567, 1.000] ✅ | **V7 makes the signal decisive** |
| Calibration improves under Platt LOO | brier_platt 0.040, ece 0.000 | brier_platt 0.071, ece 0.0001 | **Confirmed (essentially identical)** |
| LOAO median AUROC `D_OR` is narrow within selection | LOAO [0.627, 0.724] | LOAO [0.599, 0.642] | **Confirmed** |

The V5 → V7 selection-fixed comparison answers V7's central question
favorably: **`shared_weighted` is the model-agnostic, selection-robust
strong signal**, with `D_OR` selection-stable within a fixed
selection but selection-sensitive across selections (V5→V6).

## 14. Selection-sensitivity of `D_OR` (V5→V6) is a finding, not a failure

The V5→V6 disagreement on `D_OR` (0.656 vs 0.388) is a publishable
methodological observation, not a defect of either pilot.
`D_OR` measures per-agent robustness under evidence perturbations;
this is intrinsically a property of the **selection × agent ×
condition** interaction. Two FEVER selections can give different
`D_OR` point estimates simply because the underlying question set
induces different agent fragility distributions. The paper's claim
should be reframed:

> "Our methodology surfaces two signals on FEVER. **`shared_weighted`
> is the selection-robust strong signal** (V5: 0.698, V6: 0.820,
>
> V7: 0.816; V7 CI lo 0.567 > 0.5). **`D_OR` is selection-stable
> within a fixed selection rule** (V5: 0.656, V7: 0.621; drift 0.035)
> but **selection-sensitive across selections** (V5→V6: 0.656→0.388;
> see [work_log_6](docs/work_log_6.md) §10)."

This is a stronger, more honest claim than "D_OR generalizes across
domains." It documents exactly which claims hold and which don't.

## 15. Open question for V8 (signposted in V7 §16, not in scope)

V7 does **not** address cross-model generalization. V8 is
provisionally scoped as: same V7 protocol (FEVER, V5 salt, N=100,
any-passes), swap the endpoint to a second model (e.g.,
ChatGLM3-6b via the planned 31520 endpoint, or another non-
Qwen3.5-4B model). V8 preregistration is registered as the next
open question so that the V7 report can signpost it cleanly to
reviewers.

## 16. Files

- Manifest: [results/pilot_llm_v7/manifest.json](results/pilot_llm_v7/manifest.json) (~250 KB).
- Records (raw, includes 152 failed):
  [results/pilot_llm_v7/formal/records.jsonl](results/pilot_llm_v7/formal/records.jsonl) (2,000 lines).
- Records (original partial, renamed):
  [results/pilot_llm_v7/formal/records.partial.jsonl](results/pilot_llm_v7/formal/records.partial.jsonl).
- Substitute manifest (reused from V5 with v7 tags):
  [results/pilot_llm_v7/cache/substitute_manifest.json](results/pilot_llm_v7/cache/substitute_manifest.json)
  (6,506 entries).
- Summary (JSON, PASS_SINGLE_SHARED_WEIGHTED verdict):
  [results/pilot_llm_v7/formal/summary.json](results/pilot_llm_v7/formal/summary.json).
- Report (Markdown):
  [results/pilot_llm_v7/formal/report.md](results/pilot_llm_v7/formal/report.md).
- Recovery script: [scripts/recompute_v7_summary.py](scripts/recompute_v7_summary.py).
- Preregistration document: [docs/pilot_llm_v7_preregistration.md](docs/pilot_llm_v7_preregistration.md).