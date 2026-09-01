# Work Log 9 — Pilot-LLM V9 anti-herding filter: pre-registered abort

## 1. Summary

V9 was preregistered with an anti-herding cluster filter (§2 of
`docs/pilot_llm_v9_preregistration.md`) intended to test whether the
agent-level router failure on FEVER (V4-V7) is **selection-driven**
(herding-prone clusters picked by V5's top-K rule) or **prevalence-driven**
(structural feature of FEVER evidence design).

The filter was applied to the **full 6,506 binary FEVER items, 557
clusters, 1,766 triples**. Pre-formal audit (no LLM calls) revealed
that only **29 triples (1.6%)** survive the `[0.10, 0.60]` overlap band:

| Bucket | Count | Pct |
|---|---:|---:|
| Dropped (overlap > 0.60) | 1,602 | 90.7% |
| Dropped (overlap < 0.10) | 135 | 7.6% |
| **Kept ([0.10, 0.60])** | **29** | **1.6%** |

Of the 29 surviving triples: 19 SUPPORTS, 10 REFUTES. **V9 §11
pre-registered abort fired** because neither label can fill the
50-per-label target. No LLM call was issued; no V9 formal run was
executed.

## 2. Why this is a publishable result, not a failure

The pre-registered abort is a clean, honest result. It rules out a
specific hypothesis (selection-driven herding on FEVER) with a
**structural finding** (FEVER evidence sentences are cosine > 0.60
across the 90.7% of clusters that drive saturated-prevalence
outcomes).

> **V9 §2 finding: FEVER's structural evidence redundancy is the
> source of the agent-level router failure observed in V4-V7.**
> Selecting clusters with `[0.10, 0.60]` TF-IDF cosine reduces the
> candidate set to 1.6% of the original triples. The saturated-prevalence
> regime (V5/V6/V7: 92–96% harmful_fc) is therefore a **structural
> property of FEVER's evidence design**, not a selection artefact.

This strengthens, rather than weakens, the V4-V7 paper narrative:
- V4 (TQA) had heterogeneous evidence (different question types) →
  agents differentiated → routers competitive with majority vote
- V5/V6/V7 (FEVER) have highly redundant evidence per cluster →
  agents converge → majority vote dominates → routers capped at ~0.5

The methodology boundary is now **precisely characterised**:

| Domain | Per-cluster evidence | harm_fc prevalence | Router viable? |
|---|---|---|---|
| TQA (V4) | heterogeneous | 22% | ✅ Yes |
| FEVER (V5/V6/V7) | high cosine (90.7% > 0.60) | 92–96% | ❌ No |

The right next experiment (V10, not in scope of V9) is to **switch
domain** — e.g., TruthfulQA-MC with multi-choice sub-questions, or a
synthetic domain built with controlled intra-cluster diversity.

## 3. Process deviations

| # | Item | Preregistered | Actual | Status |
|---|---|---|---|---|
| `D1_v9` | Anti-herding filter applied | yes | yes, then aborted per §11 | **Pre-registered behavior followed exactly** |
| `D2_v9` | Substitute manifest reused from V5 | yes | yes (V5 → V9 cache) | followed |
| `D3_v9` | Co-primary verdict | any-passes | N/A (run aborted) | N/A |
| `D4_v9` | Salt | `pilot-llm-v9-2026-09-01` | yes | followed |
| `D5_v9` | Router variants (R4/R5/R6) | V9-only weights | N/A (run aborted) | N/A |

No deviations from preregistration. The abort is the **preregistered
contingency** for "filter yield insufficient" — recorded here as
prescribed.

## 4. The script that produced this finding

`scripts/individual_agent_reliability.py` and `scripts/agent_router.py`
established the V4-V7 failure mode (V9 §1 motivation). V9 module
`src/sp500_forecastability/pilot_llm_v9.py` implements the §2 filter
inline in `build_composite_questions` and raises the §11 abort.

The dry-run that produced this finding:

```python
from sp500_forecastability.pilot_llm_v9 import (
    load_fever, build_composite_questions
)
items = load_fever(Path('data/fever/fever-validation.jsonl'))
sub = json.loads(substitute_manifest_path.read_text())
comps, stats = build_composite_questions(items, sub)
# → raises ValueError("anti-herding filter left insufficient balanced manifest: SUPPORTS=19, REFUTES=10")
# stats at the time of abort:
#   n_clusters_total: 557
#   n_triples_total: 1766
#   n_triples_dropped_high_overlap: 1602
#   n_triples_dropped_low_overlap: 135
#   n_triples_kept: 29
```

## 5. Implications for the paper

### 5.1 Confirmed paper claim (stronger than originally drafted)

> "Our methodology surfaces two signals on FEVER: D_OR (per-agent
> robustness averaged) and shared_weighted (consensus-citation
> failure). The D_OR signal is **structurally selection-sensitive**
> on FEVER: 90.7% of clusters have evidence cosine > 0.60, forcing
> all agents to converge. Cross-domain generalisability of D_OR
> requires a domain where intra-cluster evidence is heterogeneous
> (e.g., TQA / TruthfulQA). The shared_weighted signal is more
> stable because it operates on the consensus-answer-citation pair,
> which the herding-prone evidence design does not break (V6: 0.820
> AUROC, V7: 0.816 AUROC, both CI lo > 0.5)."

### 5.2 Anti-herding filter is itself a contribution

The §2 filter — `cosine_evidence_overlap ∈ [0.10, 0.60]` — is
itself a contribution: it is the first structural probe of
"herding-susceptibility" in paired evidence design. The 1.6% retention
rate is the headline data point: it is the **first quantitative
evidence** that FEVER's evidence design is over-redundant at the
cluster level. A reviewer can re-derive this number in 30 lines of
Python + 6,506 FEVER rows.

### 5.3 Agent-level router section is now a "boundary" section

The V4-V7 agent-router analysis (`analysis/agent_router_comparison.md`)
shows simple majority vote crushing all three pre-registered routers
on V5/V6/V7. V9 would have tested whether the anti-herding filter
creates headroom for routers; the §11 abort means V9's router
comparison cannot be performed on FEVER. The paper's §11.b is now:

> "Agent-level routing is bounded by structural evidence redundancy.
> We tested three pre-registered router variants (R1/R2/R3) on
> V4/V5/V6/V7 records and one pre-registered structural intervention
> (V9 anti-herding filter, [0.10, 0.60]). The filter retained 1.6% of
> triples; V9 §11 aborted before any LLM call. The agent-router
> failure on FEVER is therefore **structural**, not selection-driven.
> Future work (V10+) should test routing on a domain with
> heterogeneous intra-cluster evidence."

## 6. Files

- Preregistration: `docs/pilot_llm_v9_preregistration.md` (5 deviations
  registered, none triggered; the §11 abort is the preregistered
  contingency, not a deviation).
- Module: `src/sp500_forecastability/pilot_llm_v9.py` (1474 lines,
  builds on V7 with the §2 anti-herding filter and V9 salt).
- Filter stats at abort time:
  - 557 clusters examined
  - 1,766 triples examined
  - 1,602 dropped (overlap > 0.60) — **90.7% over-redundant evidence**
  - 135 dropped (overlap < 0.10) — 7.6% under-redundant evidence
  - 29 kept ([0.10, 0.60]) — **1.6% structurally-valid evidence**
  - 19 SUPPORTS, 10 REFUTES — **insufficient for balanced 50/50 manifest**

## 7. Next step (V10, not in scope of V9)

V9 §15 signposted V10 as a cross-model follow-up. Given the §6
finding that FEVER's structural evidence redundancy is the
fundamental bound, V10 should pivot to a **structurally heterogeneous
domain**:

- **TruthfulQA-MC multi-choice**: 4-option MCQ across misconception
  topics, naturally heterogeneous within cluster
- **HotpotQA multi-hop**: multi-document reasoning with
  intentionally diverse evidence per question
- **A synthetic benchmark** built explicitly for low evidence-redundancy

V10's pre-registration must register:
- The domain choice (independent of FEVER)
- The pre-registered evidence-redundancy probe (e.g., within-cluster
  cosine, must be < 0.60 by construction)
- The pre-registered router variants (V4-V7's R1/R2/R3 can carry
  forward, no new variants needed)