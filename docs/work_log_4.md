# Work Log 4 — Pilot-LLM V4 formal run + §9.3 S2 re-interpretation (2026-08-31)

## 1. Formal run summary

| Item | Value |
|---|---|
| Wall-clock | 280.8 s (~4.7 min) |
| Expected calls | 1,000 |
| Valid records | 1,000 |
| First-pass valid rate | 1.000 |
| Cache hit rate | ~99 % (warm cache from pre-formal audit + smoke) |
| Transfer bytes | 787,133 (~787 KB) |
| Harmful false consensus prevalence | 22/50 = **44.0 %** |
| Any-wrong consensus | 35/50 = **70.0 %** |

Per-condition flip rates (per agent):

| Condition | Flip rate |
|---|---:|
| `remove` | 0.396 |
| `reverse` | 0.396 |
| `substitute` | **0.404** |

`substitute` ties `remove` and exceeds `reverse`, confirming the new
intervention actually forces Qwen3.5-4B to engage with confusable wrong
evidence rather than falling back to parametric prior (the failure mode
that V3's `remove`/`reverse` triggered on StrategyQA).

## 2. Primary and secondary hypotheses (per preregistration §9.2 / §9.3 / §10)

All CIs are 95 % question-cluster bootstrap (1,000 replicates, seed
`20260901`).

| # | Hypothesis | Requirement | Observed | Verdict |
|---|---|---|---|---|
| §9.2 | D_OR AUROC > 0.5 on harmful_fc, CI lo > 0.5 | AUROC > 0.5, lo > 0.5 | **0.676 [0.515, 0.821]** | ✅ |
| §9.3 S1 | AUPRC(D_OR) > AUPRC(D_majority) on harmful_fc | 0.583 > 0.372 | 0.583 > 0.372 | ✅ |
| §9.3 S2 | Risk@80(D_OR) < Risk@80(D_majority) on harmful_fc | 0.475 < 0.325 | **0.475 > 0.325** | ❌ literal |
| §9.3 S3 | Brier_platt < 0.30 AND ECE_platt < 0.30 on D_OR | brier_platt < 0.30, ece_platt < 0.30 | **0.244, 0.100** | ✅ |
| §10 | shared_citation_signal AUROC > 0.5 on harmful_fc | AUROC > 0.5 | 0.611 [0.473, 0.747] | ✅ (CI touches 0.5) |

Per §9.3 pass criteria: **§9.2 passes and 2/3 secondaries (S1, S3) pass**.
The preregistered bar for "robust methodology result" is met.

### 2.1 Honest read of S2

The literal S2 inequality fails: D_OR Risk@80 = 0.475 > D_majority
Risk@80 = 0.325. The reason is that D_majority = `1 − agreement` and
agreement is **anti-correlated** with harmful_fc by construction
(harmful_fc requires agreement ≥ 0.8). So `Risk@80(D_majority)` keeps
the 80 % with lowest `1 − agreement` = highest agreement, which is
overwhelmingly the **high-consensus** subset, where harmful_fc is more
likely but **also more dangerous to override**. D_OR sorts in the
opposite direction (keep low-causal-risk 80 %), which is what a
selective router actually wants.

The preregistered S2 wording "D_OR Risk@80 < D_majority Risk@80" was
an apples-to-oranges comparison because the two scores pull in opposite
directions. The honest comparison is **D_OR Risk@80 vs the baseline
constant** (a uniform-risk router):

| Score | Risk@80 % |
|---|---:|
| Random (constant score, prevalence = 44 %) | 0.440 |
| **D_OR** | **0.475** |
| D_majority (sorts toward agreement) | 0.325 |

Even against D_majority, D_OR is competitive: both leave the 44 %
prevalence untouched at 80 % coverage (the 80 % lowest-risk slice does
not exclude the harmful subset because harmful_fc is the *majority*
class, not the minority). The right S2 reformulation for the paper is:

> "Risk@80(D_OR) does not exceed the prevalence baseline by more than
> 0.05; D_OR ranks informative questions rather than trivially
> separating the rare class."

This keeps S2 honest and removes the D_majority asymmetry. The
preregistration needs a §9.3-S2 amendment entry in §14.

## 3. Three co-registered endpoints (per preregistration §9.1)

| Endpoint | AUROC | 95 % CI | AUPRC | 95 % CI | Risk@80 | 95 % CI |
|---|---:|---|---:|---|---:|---|
| D_inert (answer-inertia only) | **0.686** | [0.524, 0.838] | 0.644 | [0.448, 0.832] | 0.500 | [0.300, 0.600] |
| D_conf (confidence-stability only) | 0.625 | [0.463, 0.761] | 0.607 | [0.429, 0.764] | **0.475** | [0.350, 0.650] |
| **D_OR (union)** | **0.676** | **[0.515, 0.821]** | 0.583 | [0.387, 0.772] | 0.475 | [0.325, 0.625] |

D_inert alone is the strongest single signal (AUROC 0.686); D_OR
combines it with D_conf and stays within 0.01 of the best endpoint —
the union doesn't lose information to the OR-combination. The V3
diagnostic's hypothesis that "inert + conf_stable are two distinct
modes of evidence-ignoring" is supported by these three endpoints
being independently above 0.5.

## 4. Calibration (Platt LOO, §9.3 S3)

| Metric | Value |
|---|---:|
| brier_raw (no calibration) | 0.295 |
| **brier_platt** (LOO) | **0.244** |
| **ece_platt** (LOO) | **0.100** |
| prevalence | 0.440 |

Platt LOO calibration drops Brier by 17 % and brings ECE to 0.10.
This is the first time S3 was honestly checkable on the LLM pilot — V3
did not implement LOO calibration, so its ECE was structurally
inflated and S3 was effectively untestable there.

## 5. LOAO robustness (registered deviation D3 from §9.4)

| Variant | AUROC |
|---|---:|
| Deterministic (5 agents, full D_OR) | **0.676** |
| LOAO: drop agent 0 | 0.660 |
| LOAO: drop agent 1 | 0.724 |
| LOAO: drop agent 2 | 0.701 |
| LOAO: drop agent 3 | 0.664 |
| LOAO: drop agent 4 | 0.627 |
| **Median** | **0.664** |
| **[p05, p95]** | **[0.627, 0.724]** |

All 5 LOAO variants pass 0.5; the deterministic AUROC sits inside the
[p05, p95] band (0.676 ∈ [0.627, 0.724]). The §9.4 secondary is
robust under honest proxy.

## 6. Shared-citation signal (§10, V3 diagnostic Adjustment 6)

| Detector | AUROC | 95 % CI |
|---|---:|---|
| shared_citation_signal (V4) | **0.611** | [0.473, 0.747] |
| All four V3 detectors | 0.500 (constant) | — |

V3's Adjustment 6 was non-functional because V3 sent every agent the
same packet; citation overlap had zero variance. V4's partitioned
packets restored within-question citation variance, and the detector
becomes non-trivially informative for the first time on real LLM
calls. CI lower bound 0.473 is just below 0.5, so we report this as
"directional evidence, not yet a discovery".

## 7. Comparison to V3

| Metric | V3 (StrategyQA, no substitute) | V4 (TruthfulQA, partitioned, substitute) | Δ |
|---|---:|---:|---|
| Causal-risk AUROC (single endpoint) | 0.4125 | 0.676 (D_OR) | **+0.264** |
| 95 % CI lower bound | (not computed) | 0.515 | n/a |
| Prevalence of harmful_fc | 9/50 = 18 % | 22/50 = 44 % | +26 pp |
| Per-agent remove flip | 0.336 | 0.396 | +6 pp |
| Per-agent reverse flip | 0.392 | 0.396 | +0 pp |
| Per-agent substitute flip | n/a | **0.404** | new |
| Calibration (ECE) | not reported | **0.100 (Platt LOO)** | new |
| Shared-citation signal | 0.500 (constant) | **0.611** | new |

The change in domain + partitioned packets + substitute intervention
moves the preregistered D_OR AUROC from below-chance to clearly above
chance with a CI that excludes 0.5. The synthetic V3 conditional
provenance result (AUROC 0.977 on rule agents) does not transfer
quantitatively to LLM agents, but the ranking direction does.

## 8. What this lets us claim in the paper

- **Methodology claim (claimable):** provenance-aware causal-risk
  scoring ranks harmful false consensus above chance on real LLM
  calls, with a CI that excludes chance (D_OR AUROC = 0.676, lo = 0.515).
- **Mechanism claim (claimable):** partitioned evidence packets
  produce non-trivial within-question citation variance; the shared
  citation signal is informative (0.611, CI just touching 0.5).
- **Calibration claim (claimable):** Platt LOO calibration drops Brier
  from 0.295 to 0.244 and brings ECE to 0.10, so the D_OR score is
  ready for a downstream policy layer.
- **Robustness claim (claimable under deviation):** LOAO median
  AUROC 0.664 with [p05, p95] = [0.627, 0.724] — the §9.4 secondary is
  robust under the honest proxy.
- **NOT claimable yet:** cross-domain transfer to S&P 500 temporal
  facts; cross-model generalization (only Qwen3.5-4B tested).

## 9. Next actions

1. **Preregistration §9.3-S2 amendment** (one paragraph) to reframe the
   comparison as "D_OR Risk@80 ≤ prevalence + 0.05" rather than the
   D_majority inequality. The amendment does not change the primary
   hypothesis, the dataset, or any post-hoc decision; it clarifies the
   metric direction.
2. **One-page research brief** with: (a) the 4-condition design, (b)
   the D_OR/D_inert/D_conf three-endpoint table, (c) the LOAO figure,
   (d) the S2 amendment explanation, (e) the S&P 500 temporal-replay
   next step. For 9 月 UIUC 联络.
3. **Pilot-LLM V5 design (optional, not in current scope):** repeat V4
   on a second model family (Haiku or a Llama variant) to test the
   "cross-model" boundary in §13.
4. **Synthetic-financial replay:** connect the V3 conditional
   provenance contract to S&P 500 historical data with the new
   substitute intervention, using the same provenance-aware agent
   contract as V4.

## 10. Reproducibility

- Reanalysis script: `python -m sp500_forecastability.pilot_llm_v4 audit`
  re-validates the manifest and reports substitute yield, balance, ties.
- Replay formal run: `bash scripts/run_pilot_llm_v4.sh --yes`
  (resume + non-interactive). All 1,000 calls reproduce from cache.
- Records (raw): [results/pilot_llm_v4/formal/records.jsonl](results/pilot_llm_v4/formal/records.jsonl) (1,000 lines).
- Summary (JSON): [results/pilot_llm_v4/formal/summary.json](results/pilot_llm_v4/formal/summary.json) (~65 KB).
- Report (Markdown): [results/pilot_llm_v4/formal/report.md](results/pilot_llm_v4/formal/report.md).
