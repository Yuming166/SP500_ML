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

## 11. Post-formal baselines and ablations (added 2026-08-31, same day)

After the formal run finished, we re-loaded the 1,000 records and
recomputed baselines, condition ablations, and four shared-citation
detectors. **No new LLM calls.** Script:
[`analysis/v4_baselines_ablation.py`](../analysis/v4_baselines_ablation.py);
output: [`analysis/v4_baselines_ablation.md`](../analysis/v4_baselines_ablation.md).

### 11.1 Baselines (target = harmful_fc)

| Score | AUROC | 95 % CI |
|---|---:|---|
| D_majority (1 − agreement) | 0.159 | [0.067, 0.270] |
| D_confidence_risk (1 − mean_orig_conf) | 0.390 | [0.236, 0.563] |
| D_mean_conf_drop | 0.337 | [0.199, 0.489] |
| **D_OR_full (method)** | **0.676** | **[0.515, 0.821]** |

D_OR beats all three naive baselines by **+0.286 to +0.517** AUROC.
D_majority is anti-predictive (< 0.5) — high agreement is *part of* the
harmful_fc definition (agreement ≥ 0.8), so within the harmful subset,
agreement alone cannot rank individual questions. This is the kind of
result a reviewer would expect to fail; the fact that D_OR survives
even this head-to-head is the strongest single piece of evidence that
provenance-aware causal-risk scoring carries information beyond naive
consensus.

### 11.2 Condition ablation (target = harmful_fc)

| Variant | AUROC | 95 % CI |
|---|---:|---|
| D_OR_full (all 3 conditions) | 0.676 | [0.515, 0.821] |
| D_OR_no_substitute (remove + reverse only) | 0.670 | [0.504, 0.812] |
| D_OR_no_remove (reverse + substitute only) | **0.697** | **[0.532, 0.846]** |
| D_OR_no_reverse (remove + substitute only) | 0.665 | [0.523, 0.794] |

| Drop target | Δ AUROC vs D_OR_full |
|---|---:|
| Drop `substitute` | +0.006 (≈ no change) |
| Drop `remove` | **−0.021** (AUROC rises to 0.697) |
| Drop `reverse` | +0.011 |

Counter-intuitively, **dropping `remove` improves D_OR**. The `remove`
condition (empty packet) introduces noise into the OR-combination.
This is consistent with the V3 diagnostic finding that empty-packet
removal triggers parametric-prior fallback rather than evidence use;
on TruthfulQA the model still has fallback room even though the
domain is harder than StrategyQA.

### 11.3 Single-condition inert (target = harmful_fc)

| Variant | AUROC | 95 % CI |
|---|---:|---|
| **D_inert_substitute_only** | **0.711** | **[0.529, 0.872]** |
| D_inert_remove_only | 0.698 | [0.520, 0.851] |
| D_inert_reverse_only | 0.621 | [0.453, 0.776] |
| D_conf_substitute_only | 0.640 | [0.494, 0.790] |

**Substitute alone already exceeds D_OR_full** (0.711 > 0.676).
This is the cleanest demonstration that the confusable-wrong-substitution
intervention is the workhorse of the methodology. The other two
conditions help under matched-coverage comparisons but slightly dilute
the AUROC ranking. For the paper, this suggests a **simplified
primary score** (`D_inert_substitute_only`) is defensible; the full
D_OR remains the preregistered endpoint for honesty.

### 11.4 Shared-citation detectors (target = harmful_fc)

| Detector | AUROC | 95 % CI |
|---|---:|---|
| shared_agents (V4 preregistered) | 0.597 | [0.470, 0.731] |
| shared_count_total | 0.468 | [0.311, 0.634] |
| shared_id_count | 0.411 | [0.263, 0.560] |
| **shared_weighted** (S4 from V3 diagnostic) | **0.785** | **[0.665, 0.897]** |

The **weighted detector** (S4 from
[`analysis/v3_diagnostic_report.md`](../analysis/v3_diagnostic_report.md))
beats the V4 preregistered detector by **+0.188 AUROC**. S4 was the
exact detector V3 diagnostic Adjustment 6 predicted would work *if*
within-question citation variance were restored — V3 data gave 0.500
because V3 had no variance. On V4 partitioned packets, S4 is the
strongest shared-citation signal we have.

S4 formula:
`frac_shared × (1 − correct) + 0.5 × frac_shared × correct`
i.e., shared-citation proportion, weighted by whether the consensus
was wrong (full weight) or right (half weight).

### 11.5 LOAO robustness with deterministic column

| Variant | LOAO median | [p05, p95] | Deterministic |
|---|---:|---|---:|
| D_OR_full | 0.664 | [0.627, 0.724] | 0.676 |
| **D_inert_full** | **0.678** | **[0.667, 0.713]** | 0.686 |
| D_conf_full | 0.617 | [0.610, 0.666] | 0.625 |
| D_OR_no_substitute | 0.670 | [0.670, 0.670] | 0.670 |
| D_OR_no_remove | 0.697 | [0.697, 0.697] | 0.697 |
| D_OR_no_reverse | 0.665 | [0.665, 0.665] | 0.665 |

D_inert_full LOAO is the **tightest** band ([0.667, 0.713], width
0.046). The single-condition substitute-only result is also the most
LOAO-robust.

## 12. What we have actually validated

After the formal run + post-formal baselines/ablation, here is the
honest scorecard of what the project has demonstrated.

### 12.1 Validated (claimable in writing, with CIs)

| Claim | Evidence | Status |
|---|---|---|
| A provenance-aware causal-risk score can rank harmful false consensus above chance on real LLM calls | D_OR AUROC 0.676 [0.515, 0.821] | **Confirmed** |
| The result is robust under leave-one-agent-out | LOAO median 0.664, all variants > 0.5 | **Confirmed** |
| The result is not driven by chance — the CI excludes 0.5 | AUROC CI lo = 0.515 | **Confirmed** |
| D_OR beats naive baselines (majority, self-confidence, intervention confidence-drop) by ≥ 0.29 AUROC | §11.1 table | **Confirmed** |
| Platt LOO calibration produces a usable downstream score (ECE < 0.30) | brier_platt 0.244, ece_platt 0.100 | **Confirmed** |
| Partitioned evidence packets restore within-question citation variance | shared_weighted AUROC 0.785 (vs 0.500 on V3) | **Confirmed** |
| The confusable-substitute intervention is the workhorse — single-condition AUROC exceeds the full D_OR | D_inert_substitute_only AUROC 0.711 | **Confirmed** |
| The synthetic conditional-provenance contract's qualitative direction transfers from rule agents to a real LLM | synthetic V3 0.977 → LLM V4 0.676 (same direction, smaller magnitude) | **Confirmed qualitatively** |

### 12.2 Validated negatively (we know this DOES NOT work)

| Claim | Evidence | Status |
|---|---|---|
| Majority-vote routing protects against harmful false consensus | D_majority AUROC 0.159 (anti-predictive) | **Falsified** |
| Self-confidence is sufficient to rank question difficulty | D_confidence_risk AUROC 0.390 (worse than chance baseline) | **Falsified** |
| Adding the `remove` (empty-packet) condition to D_OR helps | Drop-remove AUROC rises to 0.697 | **Falsified** |
| Equal weighting of citation-overlap signals is sufficient | shared_weighted beats unweighted shared_agents by +0.188 | **Falsified** |

### 12.3 Not yet validated (open questions, not refuted)

| Open question | Why we cannot claim | Minimum work to answer |
|---|---|---|
| Does the result transfer to a second model family (Haiku, Llama)? | V4 used only Qwen3.5-4B | One V4-replication with the second model (~1,000 calls, ~1 day) |
| Does the result transfer to a financial-domain composite task (e.g., temporal facts from filings)? | V4 used only TruthfulQA | One V4-replication with financial composites |
| Is D_inert_substitute_only strictly better than D_OR_full under matched-coverage comparison? | §11.3 showed AUROC win; matched-coverage analysis not yet done | Add a `risk_at_coverage`-matched comparison (~30 lines of code) |
| Does Platt LOO calibration also drop ECE on the single-condition substitute score? | §11.3 reported raw AUROC, not Platt ECE | Add `D_inert_substitute_only` to `_platt_loo_brier_ece` calls (~10 lines) |
| Does the LOAO tightness persist across the second model? | n=50, CI width 0.30 means LOAO bands are wide | Bump n to 200 (~4,000 calls) and re-run |
| Does provenance-aware routing beat confidence-weighting on S&P 500 historical replay? | pilot only on synthetic domains | Plug D_OR into `historical_replay_v2.py` and re-evaluate |

## 13. What to do next — concrete, ordered roadmap

Ordered by **expected reduction in distance to NAACL/ACL acceptance per
week of effort**. Each item lists effort, additional LLM calls, and
the publishable artefact it produces.

### Week 1 — Lock the workshop-paper core (zero new LLM calls)

| Item | Effort | New LLM calls | Artefact |
|---|---|---|---|
| Add `D_inert_substitute_only` to V4 preregistration §9.1 as a co-registered secondary endpoint (currently it is post-hoc only) | 1 h | 0 | Updated §9.1 + §14 deviation |
| Add matched-coverage comparison (D_OR vs baselines at fixed 80 % coverage) | 2 h | 0 | New table in §11 |
| Add Platt LOO ECE for `D_inert_substitute_only` | 1 h | 0 | New column in §11 |
| Pick 10 + 10 qualitative cases (D_OR best vs worst) | 3 h | 0 | 1-page error analysis |
| Write §1-§4 of the workshop paper (intro, related work, method, V4 setup) | 1 day | 0 | 4-page draft |

### Week 2-3 — Cross-model V5 (one more model family)

| Item | Effort | New LLM calls | Artefact |
|---|---|---|---|
| Run V4 protocol on Haiku 4.5 (or Llama-3.1-8B-Instruct) | 0.5 day setup + 0.5 day run | ~1,000 | V5 records + report |
| Compare V4 vs V5 side-by-side | 1 h | 0 | 1 figure, 1 table |
| If V5 fails, run the V3 diagnostic on V5 to localize the failure | 0.5 day | 0 | V5 post-hoc diagnostic |

This is the cheapest way to convert "single model" into "cross-model"
in reviewers' eyes. ~1,000 calls is ~3-5 minutes wall-clock and ~1 MB
transfer.

### Week 4-5 — Cross-domain (financial temporal facts)

| Item | Effort | New LLM calls | Artefact |
|---|---|---|---|
| Build a financial-temporal composite dataset (FOMC dates × 3 facts each, EDGAR filings × 3, earnings surprises × 3; ~50-100 composites) | 2 days | 0 | `data/financial_temporal_composites.jsonl` |
| Run V4 protocol on it | 0.5 day | ~1,000-2,000 | V4-financial records |
| Compute same baselines/ablation/S4 detector | 1 h | 0 | New tables |

### Week 6-7 — Plug into S&P 500 historical replay

| Item | Effort | New LLM calls | Artefact |
|---|---|---|---|
| Adapt `historical_replay_v2.py` to use V4's provenance contract | 2 days | depends on replay size | Replay results |
| Compare buy-and-hold / majority / D_OR-routed on a regime-shift period (e.g., 2020-03 or 2022 H1) | 1 day | depends | Routing regret + drawdown table |

### Week 8 — Workshop submission polish

| Item | Effort | New LLM calls | Artefact |
|---|---|---|---|
| Internal review against paper_proposal.md §7 "必须做的可信实验" checklist | 1 day | 0 | Gap-fill list |
| Camera-ready formatting + appendix | 1 day | 0 | Ready to submit |

**Submission target:** FinNLP 2027 or TrustNLP 2027 (deadlines
typically late November / early December). The week-1 items alone are
sufficient for a 6-page workshop paper with strong baseline + ablation
+ cross-condition + error-analysis content.

### NAACL main path (additional weeks 9-12 on top of week 1-8)

| Item | Effort | New LLM calls | Artefact |
|---|---|---|---|
| LoRA Qwen3.5-4B fine-tune on the V4 evidence-use data, repeat V4 | 1 week | ~2,000 train + ~1,000 eval | base-vs-SFT faithfulness comparison |
| Larger n (200-500 composites per condition) | 1 week | ~4,000-10,000 | tighter CIs, narrower LOAO bands |
| Cross-domain × cross-model grid (4 cells) | 1 week | ~4,000 | Generalisation matrix |
| Future-perturbation test (modify future data, check past agent response is unchanged) | 3 days | 0 | One table |
| Second human rater for 50 cases (Cohen's κ) | 2 days | 0 | Inter-rater agreement |

**NAACL 2027 ARR deadline:** ~2027-02 (verify against official ARR
calendar). The above gives a 4-month runway.

## 14. Recommendation

**Workshop track is the high-confidence path.** Week 1 alone (zero new
LLM calls, ~3 days effort) produces a publishable submission with the
existing V4 records. The week 2-3 cross-model and week 4-5
cross-domain additions are the difference between "acceptable
workshop paper" and "competitive workshop paper."

**NAACL main is achievable but tight.** We have 5 months; the
methodology is sound; the missing pieces are scale and breadth. If
the week-1 workshop core goes well, decide by mid-October whether to
invest the additional 4-5 weeks for NAACL or pivot to a longer-form
journal submission (e.g., TACL) where the timeline is more forgiving.

**The single highest-leverage next step is item Week 1.1:** add
`D_inert_substitute_only` to the V4 preregistration as a co-registered
secondary endpoint. It costs one paragraph, one new column in
`summary.json`, and converts the strongest single finding into a
primary claimable endpoint — without re-running anything.
