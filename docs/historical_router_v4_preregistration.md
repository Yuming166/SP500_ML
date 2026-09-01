# Historical Router V4: cross-domain causal-provenance router

**Protocol version:** `historical-router-v4-2026-09-02`
**Frozen:** 2026-09-02, before fitting the V4 router or inspecting any V4 output
**Status:** retrospective development experiment; not a prospective market claim

## 1. ACL-facing research question

Can a selective router learn a domain-invariant **intervention signature** from
real LLM-agent paired interventions, adapt it using only matured training-window
market outcomes, and improve reliability under temporal and source-regime shift?

The proposed contribution is not an LLM that predicts prices. It is a
cross-domain reliability mechanism for teams of agents:

```text
agent decisions + evidence/source graph + paired interventions
                         |
                         v
     domain-invariant causal-provenance signature
                         |
                         v
     monotone, group-robust risk router + abstention
```

Finance is an external sequential validation environment. The linguistic source
domain is the frozen real-Qwen BoolQ V12.1 run. The router never consumes hidden
chain-of-thought, raw question text, or raw market outcomes at decision time.

## 2. Why this is a new V4

Historical V3 used global LLM-agent AUROC values as static weights for market
agents. It did not improve on majority routing. V4 changes the mechanism rather
than retuning V3: it transfers a relational intervention representation and
then performs train-window-only conditional adaptation.

V3 documentation also says `6 agents / 5 roots`, whereas the frozen executable
agent dictionary contains `11 agents / 7 roots`. V4 uses and reports the actual
`11 / 7` contract. V3 artifacts remain unchanged.

## 3. Frozen data and split contract

### 3.1 Source domain

- Records: `results/pilot_llm_v12_1/formal/records.jsonl`.
- Use only questions with all five agents, all four conditions, and original
  agreement at least `0.8`.
- Label: whether the original high-consensus answer is wrong.
- Source features: `D_inert`, `flip_inertia`, and `frac_shared`.
- These data train a non-negative logistic source prior. They are not a new
  held-out result; V12.1 already reported their outcomes.

### 3.2 Market domain

- Inherit the V2/V3 as-of data builder without changing feature availability.
- Target: next five-trading-day S&P 500 direction.
- Agents: the exact 11 source-specialized logistic agents in
  `historical_replay_v3.AGENTS`.
- Roots: the exact seven root IDs referenced by those agents.
- Outer evaluation: expanding 504-row train, five-row label gap, 126-row test,
  step 126.
- Inner agent predictions: expanding 252-row train, five-row label gap,
  63-row test, step 63.
- Within each outer train, the earliest 70% of inner-OOF timestamps fit the
  router; the latest 30% calibrate all abstention thresholds.
- Agent/root quality at a timestamp uses only labels matured at least five rows
  earlier. No centered window, back-fill, random split, or test-period tuning.

The current file contains 1,247 as-of rows and six outer test blocks. Because
earlier V0-V3 analyses exposed aggregate outcomes on the same date range, V4 is
explicitly developmental. A later date extension or separately frozen index is
required for prospective confirmation.

## 4. Domain-invariant intervention signature

The three transferred coordinates are:

1. `intervention_inertia`: decisions and confidence remain stable after paired
   evidence/source interventions;
2. `flip_inertia`: fraction of interventions that do not flip the action;
3. `source_concentration`: agreeing agents reuse the same evidence/root.

For the LLM source these map exactly to V12.1 `D_inert`, `flip_inertia`, and
`frac_shared`. For the market target, interventions are outcome-free graph
operations performed on root-level probabilities: remove one root, reverse one
root, or replace one root by neutral probability `0.5`.

Market-only risk coordinates are:

- `consensus_risk`;
- `action_confidence_risk`;
- `root_disagreement`;
- `quality_risk`, computed from matured train-only Brier histories.

All seven coordinates are normalized to `[0, 1]` and constrained to have
non-negative risk coefficients. Hence increasing an explicitly declared risk
coordinate cannot lower predicted failure risk.

## 5. CPR-Router

The **Causal-Provenance Residual Router (CPR-Router)** is a constrained logistic
risk model. Its first three coefficients are anchored to the real-LLM source
prior while target-only residual coordinates and two method offsets are learned
inside each market outer-training window.

For risk features `x`, method indicators `m`, and source coefficients
`beta_src`:

```text
risk(x, m) = sigmoid(b0 + beta_nonnegative^T x + delta_method^T m)
```

The frozen target objective is:

```text
softmax_group_loss(tau=0.10)
+ 0.01 * L2(parameters excluding intercept)
+ 0.10 * ||beta_common - beta_src||^2
+ 0.10 * paired_stress_ranking_loss(margin=0.02)
```

Groups combine train-only VIX tertile and chronological third. The smooth
worst-group loss is `tau * log(mean(exp(group_loss / tau)))`. The paired stress
term increases the three transferred risk coordinates by `0.20` (clipped to
one) and penalizes a stressed risk that fails to exceed the clean risk by the
frozen margin. No V4 constant may be changed after V4 output exists.

## 6. Routing and calibration

Candidate actions are produced by three frozen base voters:

1. `majority`;
2. `recent_performance`;
3. `provenance` root-deduplicated vote.

CPR-Router scores the error risk of all three and chooses the minimum-risk
candidate. Ties use the fixed order `provenance`, `recent_performance`,
`majority`. The router then answers only if its risk is at or below the
train-window calibration threshold.

Every method uses its own latest-30%-OOF calibration scores. The threshold is
the finite-sample `0.80` conformal quantile
`ceil((n + 1) * 0.80) - 1`, clipped to the observed index range. This is an
empirical rolling calibration device: because market observations are
non-exchangeable, V4 does not claim an exact IID conformal guarantee.

## 7. Frozen comparisons

Report all of:

- `majority`;
- `confidence`;
- `recent_performance`;
- `provenance`;
- `cpr_router`.

The inherited methods are recomputed on the same outer predictions and use the
same 80% calibration protocol. The primary named comparator is `confidence`,
which had the lowest routed error in historical V3. All methods remain visible.

## 8. Metrics and interpretation

Primary developmental endpoint:

```text
Delta_error = routed_error(cpr_router) - routed_error(confidence)
```

Report a 95% moving-block bootstrap interval using contiguous 21-row blocks,
1,000 replicates, seed `20260902`. The directional target is an interval upper
bound below zero, subject to realized CPR coverage in `[0.70, 0.90]`. Because
the date range was used by earlier experiments, this target is diagnostic and
cannot create a confirmatory `PASS` claim.

Also report:

- coverage, routed error, false rejection, selected Brier, and AURC;
- worst VIX-regime routed error;
- non-overlapping five-day long/cash returns, maximum drawdown, and turnover;
- net return sensitivity at 5 and 10 basis points per position change;
- source-prior and per-fold target coefficients;
- monotonicity and paired-stress audit counts.

Financial return metrics are secondary. V4 cannot establish alpha, causal
market impact, investment utility, or universal LLM-agent reliability.

## 9. Frozen ablations

All are evaluated without hiding the full model:

1. no LLM source anchor (`anchor = 0`);
2. no group robustness (mean logistic loss);
3. no paired-stress ranking term;
4. fixed V12.1 structural score without target learning;
5. inherited confidence and provenance baselines.

The ablations localize whether any gain comes from cross-domain transfer,
worst-group learning, paired intervention structure, or merely a new
parameterized classifier.

## 10. ACL claim boundary

Eligible wording after this experiment is limited to a method and feasibility
claim: a domain-invariant intervention signature can be trained on real LLM
agent records and evaluated under leakage-aware financial temporal replay.
Cross-model LLM validation and a prospective market window remain required for
a strong ACL generalization claim. Component methods such as monotonic models,
group robustness, and conformal calibration are not individually claimed as
new; the proposed novelty is their causal-provenance transfer contract,
paired-intervention representation, and cross-domain evaluation.
