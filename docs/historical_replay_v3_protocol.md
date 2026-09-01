# Historical replay V3: cross-domain reliability routing

**Date frozen:** 2026-09-01
**Status:** Frozen BEFORE any V3 model fit. All design decisions below
were made WITHOUT inspecting V3 outcomes. Any substantive change after
V3 outputs requires a new version (V4).

## 1. Why V3

V0/V1/V2 (`results/historical_replay_v{0,1,2}.md`) established a
4-router ablation on S&P 500 historical data: **majority**,
**confidence**, **recent_performance**, **provenance**. V0/V1/V2 use
only intra-market signals (logistic-regression agents fit on the
expanding training window). The Pilot-LLM work (V1-V10) developed a
parallel methodology: per-agent reliability from **out-of-domain
yes/no classifiers** (TQA, FEVER, BoolQ).

V3's central question: **does the LLM-derived per-agent reliability
signal transfer to the S&P 500 market data?**

V3 is a **cross-domain reliability test**, NOT a parameter-tuning
exercise. The pre-registered design fixes all weights, all thresholds,
all reporting choices BEFORE running.

## 2. Status of prior historical replays (preserved, not edited)

| Version | Outcome | Notes |
|---|---|---|
| V0 | Done (results/historical_replay_v0.md) | Rule-agent, fixed-price expanding walk-forward; 4 routers |
| V1 | Done (results/historical_replay_v1.md) | Source grouping: 3 market + 3 ETF agents; soft_root_cap added |
| V2 | Done (results/historical_replay_v2.md) | Source expansion: CBOE + ICI features added; 4 routers |
| **V3** | **This document** | Cross-domain reliability routing |

## 3. Frozen data contract (inherited from V2 §asof)

Inherited verbatim from V2:
- 1247 as-of feature rows (2021-05-10 to 2026-04-27)
- 6 agents (3 market + 3 ETF-flow), 5 source roots
- Expanding walk-forward: 504-day train, 5-day gap, 126-day test
- 5-day return labels
- Same 4 baseline routers: majority, confidence, recent_performance, provenance

## 4. Frozen V3 routers (pre-registered)

5 routers total. The 2 NEW routers (R_v10, R_brier) are pre-registered
with FROZEN weights. The 3 BASELINE routers (majority, confidence,
provenance) are inherited from V2 unchanged.

### 4.1 R_v10 (NEW, frozen) — Cross-domain reliability from V10

For each test row, the agent's reliability weight = **the V10 per-agent
AUROC_fragility score from `analysis/individual_agent_reliability.json`**,
clipped to [0.5, 1.0] to prevent any agent from being zeroed out:

```
R_v10:  R_v10(a, q) = clip(V10_per_agent_AUROC[a], 0.5, 1.0)
        weighted_vote("yes" | "no", weights={R_v10(a, q) : a in agents})
```

Frozen source: V10 per-agent AUROC = {literal_evidence: 0.423,
skeptical_auditor: 0.493, consistency_checker: 0.439,
counterfactual_reasoner: 0.427, minimal_judge: 0.468}.

This is the V10-derived per-agent reliability, applied WITHOUT
retraining on market data. The signal is the "LLM says this agent is
more reliable" pattern.

### 4.2 R_brier (NEW, frozen) — Market-OOF per-agent Brier

For each test row, the agent's reliability weight = **1 / (Brier + ε)**
on the agent's OOF predictions inside the training block. Brier and ε
are pre-registered constants:

```
R_brier:  Brier[a, q] = 1 - (1/n_OOF) Σ (y_i - p_i(a))^2
          ε = 0.01  (pre-registered; prevents div-by-zero)
          weight[a, q] = 1 / (Brier[a, q] + ε)
          weighted_vote("yes" | "no", weights={weight[a, q] : a})
```

This is the market-internal per-agent reliability, applied AFTER
training. The signal is "this agent's recent OOF Brier is low, so its
current prediction is reliable."

### 4.3 R_equal (NEW, frozen) — Equal-weight control

For each test row, the agent's reliability weight = 1.0 (uniform):

```
R_equal:  weighted_vote with all weights = 1.0  (≡ simple majority, control)
```

This is a sanity check: if R_v10 and R_brier both beat R_equal, the
weighting is doing real work, not just adding noise.

### 4.4 R_majority, R_confidence, R_provenance — V2 baselines

Inherited verbatim from V2:
- **R_majority**: equal hard vote across agents
- **R_confidence**: majority action, with agreeing-agent probability
  confidence
- **R_provenance**: probabilities averaged within each source root,
  then roots weighted by inverse root-level recent Brier loss

## 5. Pre-registered hypotheses

| # | Hypothesis | Why it is required |
|---|---|---|
| H1 | `R_v10` mean 5d return > `R_majority` mean 5d return | Tests whether LLM-derived per-agent reliability transfers to market data without retraining |
| H2 | `R_brier` mean 5d return > `R_majority` mean 5d return | Tests whether market-internal per-agent reliability is useful |
| H3 | `R_v10` ≈ `R_brier` mean 5d return (within ±0.001 absolute) | Tests whether the two reliability sources give comparable signals |
| H4 | `R_v10` and `R_brier` both > `R_equal` mean 5d return | Tests whether weighting does real work (vs random) |

**Pre-registered contingency (locked)**: if R_v10 loses to R_majority
or to R_equal, this is a **valid negative result** that the LLM-derived
reliability does not transfer. The paper documents this as a
methodology boundary ("LLM per-agent reliability is not a universal
routing signal; it is V0-V10 specific").

## 6. Frozen thresholds and reporting

Inherited from V2:
- 75th percentile of train-only nested-OOF risk for abstention
- Every-fifth calendar decision for portfolio (5-day return non-overlap)
- No transaction costs

V3 reports:
- All 5 routers in one table (R_majority, R_confidence, R_provenance,
  R_v10, R_brier, R_equal)
- Coverage, routed error, false rejection, mean 5d return, max drawdown
  for each
- 5d returns are non-overlapping

## 7. Pre-registered deviations (D1_v3 - D5_v3)

| # | Item | Preregistered | Deviation | Why |
|---|---|---|---|---|
| `D1_v3` | New routers (R_v10, R_brier, R_equal) | V0/V1/V2: 4 routers | **+ 3 routers (V10-derived, market-Brier, equal-weight control)** | V3 tests cross-domain reliability. 3 new routers added; 2 baselines (R_majority, R_provenance) inherited from V2. |
| `D2_v3` | R_v10 weights | (n/a in V0/V1/V2) | **FROZEN V10 per-agent AUROC** (literal_evidence=0.423, skeptical_auditor=0.493, consistency_checker=0.439, counterfactual_reasoner=0.427, minimal_judge=0.468) | Weights come from V10 records, **not retrained on market data**. Tests the "LLM says this agent is reliable" claim directly. |
| `D3_v3` | R_brier weights | (n/a in V0/V1/V2) | **1 / (Brier + 0.01) on train-only OOF predictions** | Pre-registered ε=0.01 to prevent div-by-zero. Brier and weight formulas are locked. |
| `D4_v3` | R_equal weights | (n/a in V0/V1/V2) | **All 1.0 (uniform)** | Pre-registered control: if R_v10 and R_brier both > R_equal, weighting is doing real work. |
| `D5_v3` | Reporting | V0/V1/V2: 4 routers | **6 routers (5d return) in one table** (D1 + R_majority) | All 6 routers reported; no cherry-picking. |

## 8. Interpretation boundary

V3 reports whether per-agent reliability from a yes/no classifier
(V10) transfers to market-data routing. This is a **transfer test**,
not a claim that the historical data supplies audited publication
timestamps. No transaction costs; no intraday release-time audit.

## 9. Open question for V4 (not in scope of V3)

V4's central question would be: does any router (R_majority, R_v10,
R_brier, etc.) survive transaction costs and a held-out 2026 H2
backtest? V3 is pre-formal; V4 is post-formal-with-costs. V4
preregistration is not drafted in this document.