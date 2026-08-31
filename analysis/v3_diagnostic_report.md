# Pilot-LLM V3 post-hoc diagnostic: re-framing the negative causal-risk AUROC

**Date:** 2026-08-31  
**Scope:** Reanalyze the 750 Pilot-LLM V3 records (no new LLM calls) to test whether
alternative causal-risk definitions and shared-citation detectors flip the sign of the
preregistered causal-risk AUROC (which was 0.4125, below chance).

This report is **post-hoc and explicitly labeled as such**. It does **not** modify the
frozen V3 preregistration or its primary hypothesis. It exists to (a) test whether the
AUROC < 0.5 finding is the result of the *metric definition* rather than a true
absence of signal, and (b) decide whether the next pilot (V4-LLM) should re-run on a
harder domain or with a richer intervention set.

## 1. Why the original V3 causal-risk AUROC is below chance

The V3 preregistration defines causal-effect risk as the fraction of agents that are
**evidence-inert** (no answer flip on either `remove` or `reverse`). The diagnostic
splits the 250 complete agent-triplets into two groups:

| Group | Count | Share | Original-answer accuracy |
|---|---:|---:|---:|
| **Evidence-INERT** (no flip on either intervention) | 123 | 49.2% | **82.1%** |
| **RESPONDER** (flipped on ≥1 intervention)        | 127 | 50.8% | 76.4% |

The gap is **+5.7 percentage points in favor of inert agents**. Because Qwen3.5-4B has
strong parametric knowledge of StrategyQA's general-knowledge questions, removing or
mechanically negating the evidence does not force the model to use it — it falls back
to its prior, which is independently accurate. **The intervention design tests "what
if the evidence is missing", not "what if the evidence is wrong but believable".** As
a result, inert agents are systematically the model's most accurate agents, and the
preregistered risk score is inversely correlated with correctness.

This is **a protocol finding**, not a finding about Qwen3.5-4B's reasoning in general.

## 2. Adjustment 5: alternative causal-risk definitions

Tested 8 alternatives on the same 50 questions. Target outcome is **harmful false
consensus** (consensus wrong AND agreement ≥ 0.8). Prevalence: 9/50 = 18.0%.

| Risk definition | AUROC (harmful_fc) | 95% bootstrap CI (q-level, 2000 reps) |
|---|---:|:---:|
| A — original V3 (inert only) | 0.393 | [0.228, 0.570] |
| B — confidence-stable only | 0.512 | — |
| **C — inert AND confidence-stable** | 0.390 | — |
| **D — inert OR confidence-stable** | **0.556** | **[0.381, 0.692]** |
| E — 50/50 composite of A and B | 0.416 | — |
| F — 1 − max intervention confidence drop | 0.523 | — |
| G — 1 − clipped max intervention confidence drop | 0.523 | — |
| H — raw count of inert agents | 0.393 | — |

**D (inert OR confidence-stable) is the only definition that pushes the point
estimate clearly into the predictive region (0.556)**, and its CI is centered above the
V3 baseline's CI. Definitions B, F, and G are also above 0.5 but with smaller gaps.

**Interpretation.** The signal that V3 was *trying* to measure — "agents that neither
update their answer nor their confidence under paired evidence interventions" — is
real but weak. The original V3 formulation collapsed two distinct phenomena
(answer-inertia and confidence-stability) into one over-restrictive definition. The
OR-combination (D) recovers the union, which has the strongest predictive validity
for harmful false consensus.

**Limits.**
- n = 50 questions, only 9 positives. The 95% CI on D is wide ([0.381, 0.692]) and
  still includes 0.5. **The result is directionally consistent but not
  statistically strong enough to claim a discovery on its own.**
- The metric change is post-hoc. To claim the result as primary evidence, V4-LLM
  must lock D (or a registered variant) **before** running new calls.

## 3. Adjustment 6: shared-citation false-consensus detector

Tested four shared-citation detectors against harmful false consensus:

| Detector | AUROC (harmful_fc) |
|---|---:|
| S1 — max shared citation count | 0.500 |
| S2 — number of evidence-id shared | 0.500 |
| S3 — fraction of agents sharing ≥1 citation | 0.500 |
| S4 — `frac_shared × (1 − correct) + 0.5 × frac_shared × correct` | 0.500 |

**All four detectors return AUROC = 0.500**, which means the score is essentially
constant across the 50 questions. The cause is structural: **V3 sends every agent
the same evidence packet**, so citation overlap is the baseline, not an anomaly.
The synthetic harness distinguished shared vs. independent evidence by *partitioning*
the catalog across agents; Pilot-LLM V3 does not.

**Implication.** Adjustment 6 cannot work on V3 data because V3 has no within-question
citation variance by design. To make shared-citation false consensus detectable in a
real-LLM pilot, **V4-LLM must give each agent a partitioned evidence packet**
(restoring the synthetic V1–V4 design). This is a small but important protocol
change and should be registered before the new run.

## 4. Decision matrix for Pilot-LLM V4

The diagnostic supports the following design choices for the next pilot:

| Component | V3 (frozen) | V4-LLM recommendation | Reason |
|---|---|---|---|
| Question domain | StrategyQA (general knowledge) | **Harder domain** (HotpotQA multi-hop, financial temporal facts, or obscure-entity QA) | Qwen3.5-4B's parametric prior dominates StrategyQA |
| Evidence packet | Identical for all 5 agents | **Partitioned across agents** (overlapping subsets) | Required for shared-citation signal (Adjustment 6) |
| Interventions | remove + reverse | **+ substitute with confusable wrong evidence** | remove/reverse trigger prior-fallback; substitute forces evidence use |
| Causal-risk definition | inert-only | **inert OR confidence-stable (D)**, preregistered | Direction-consistent signal; pre-register before running |
| Outcome target | harmful false consensus | **same** | Stable; 18% prevalence in V3 is workable |

The minimum incremental experiment is one extra condition per agent (`substitute`),
holding everything else fixed: **+250 calls × 5 agents = +1,250 calls ≈ 2.5 MB
transfer**. Combined with the harder domain and partitioned packet, the design
recovers the structural signal the synthetic V1–V4 series already demonstrated.

## 5. Honest assessment and recommended path

- **The original V3 primary result (AUROC 0.4125) is real but under-powered and
  protocol-limited.** It is not strong enough to support either a positive or
  negative claim about Qwen3.5-4B's evidence-use behavior on real text.
- **Adjustment 5 (D) gives a directionally consistent +0.144 AUROC improvement**
  with no new data, suggesting the *idea* of provenance-based routing is intact but
  needs a finer-grained risk definition.
- **Adjustment 6 cannot rescue V3** — it requires partitioned evidence packets,
  which V3 did not implement.
- **The single highest-leverage next step** is to freeze a V4-LLM preregistration
  that (a) uses a harder domain, (b) partitions the evidence packet across agents,
  (c) adds the substitute intervention, and (d) pre-registers D as the primary
  risk score. Re-running on 50 questions costs ≈ 2.5 MB and ≈ 1,250 extra calls
  on top of the existing 750.

## 6. Reproducibility

- Reanalysis script: `analysis/v3_diagnostic.py` (reads only `results/pilot_llm_v3/formal/records.jsonl`).
- Raw outputs: `analysis/v3_diagnostic_results.json`.
- No LLM calls were made for this diagnostic.
- All AUROCs use Mann-2 over the 50-question set; CIs use 2,000 question-level
  bootstrap replicates with seed 42.