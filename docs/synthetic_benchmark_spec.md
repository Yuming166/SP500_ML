# Synthetic False-Consensus Benchmark Specification

## 1. Scientific question

> Can provenance predict the risk that a multi-agent consensus is wrong before the outcome is observed?

The benchmark separates two concepts that must not be conflated:

- **correlated consensus**: agents agree while sharing one or more leaf evidence sources, stale evidence, or an invalid temporal dependency;
- **harmful false consensus**: correlated consensus whose majority action is later wrong.

The detector sees only pre-outcome observations. The later outcome is retained solely for offline labels and evaluation.

## 2. Episode contract

Each episode follows this causal order:

```text
latent outcome
    -> source observations
    -> evidence graph and agent-specific packets
    -> agent claims and actions
    -> consensus / pre-outcome risk score
    -> realized outcome (offline evaluation only)
```

The pre-outcome view contains only agent decisions, the environment-owned `ProvenanceGraph`, evidence packet assignments, the decision timestamp and historical source-quality estimates. It deliberately excludes `outcome_action`, `harmful_false_consensus` and all post-decision information. Source quality is an as-of historical estimate, not an oracle label for the current episode.

## 3. Initial controlled scenarios

| Scenario | Source structure | Evidence state | Expected role |
| --- | --- | --- | --- |
| `independent_clean` | each agent has a distinct leaf source | current and correct | independent correct consensus |
| `shared_clean` | agents see different derived features with one common leaf source | current and correct | correlated but harmless consensus |
| `shared_corruption` | agents share one common leaf source | current but directionally corrupted | harmful false consensus |
| `stale_evidence` | agents share one common leaf source | validly available but old and regime-misaligned | harmful false consensus |
| `partial_corruption` | two agents share a bad source; one agent has an independent source | current, mixed correct/incorrect actions | 2:1 harmful correlated consensus |

`future_leakage` is a contract-violation stress test rather than a routable episode: the generator creates the invalid payload, and the agent parser must reject `available_at > decision_time` before any router sees it. Evidence removal and reversal are exposed as deterministic rule-agent interventions; they validate causal wiring, while later LLM experiments must replace them with paired model calls.

Each corruption experiment is paired with an `independent_clean` episode generated from the same seed and latent outcome. The pair changes only evidence provenance/content, not the task outcome.

## 4. Minimal methods

The first code version exposes only transparent baselines:

1. majority action and agreement strength;
2. mean self-reported confidence;
3. provenance risk score based on maximum leaf-source concentration, historical source-quality risk, stale-evidence fraction and temporal violations;
4. selective router that abstains when the provenance risk crosses a fixed threshold.

This score is intentionally not claimed as the paper method. It is a wiring and falsifiability harness: `shared_clean` measures the coverage cost of abstaining from all correlated consensus, while `shared_corruption`, `partial_corruption` and `stale_evidence` measure the possible safety gain. The explicit source-quality estimate lets the harness test whether a router can retain correlated but historically reliable consensus without observing the current outcome.

## 5. Evaluation protocol

For each scenario, seed and corruption strength, report:

- correlated-consensus detection AUROC/AUPRC after the generator contains nontrivial partial-sharing cases;
- harmful false-consensus AUROC/AUPRC using the realized outcome only as an offline label;
- high-confidence error rate at matched coverage;
- risk-coverage curve and AURC;
- independent-correct-consensus rejection rate;
- clean-to-corrupted robustness drop.

The original generator has explicit source IDs, so source-overlap detection is expected to be easy. It remains a contract-wiring check rather than a publishable performance claim.

## 6. Parameterized difficulty suite

The parameterized generator provides the next benchmark layer without changing the original readable control cases:

- **agent count**: any odd number of agents at least three;
- **corruption strength**: controls the fraction of agents connected to a corrupt shared source; low strengths can create correlated but non-harmful consensus, while higher strengths can flip the majority;
- **source visibility**: `full` exposes true source identities, `aliased` preserves source linkage under opaque names, and `hidden` gives each agent a distinct visible alias even when their latent source is shared;
- **renamed transformations**: evidence and transform IDs can be opaque, preventing a detector from exploiting names such as `trend` or `vix`;
- **noisy source quality**: the router sees a reproducible historical estimate clipped to `[0, 1]`; latent current quality is retained only for offline analysis;
- **mechanism-held-out split**: corruption mechanisms are disjoint between train and test, rather than randomly splitting near-duplicate episodes; clean control scenarios may appear in both splits so coverage and unnecessary abstention remain measurable.

The `hidden` setting is intentionally adversarial: a source-overlap heuristic should degrade because the observed graph no longer reveals the shared root. This quantifies how much the method relies on provenance completeness and motivates future provenance reconstruction or semantic source-linking methods.

## 7. Stop/go criteria

Before adding Qwen, S&P 500 replay or RL, pre-register the scenario distribution, random seeds, threshold selection rule and primary metric. Continue only if a provenance-aware selective policy reduces harmful high-confidence errors at matched coverage without rejecting nearly all independent correct consensus. If not, inspect whether the failure is due to missing provenance, insufficient source-quality information, or the underlying hypothesis being false.
