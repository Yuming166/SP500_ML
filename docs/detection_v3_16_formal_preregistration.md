# Detection V3.16 formal preregistration: label-symmetric cross-family transfer

Date: 2026-09-03 (Asia/Shanghai)

Protocol: `detection-v3.16-formal-vitaminc-qwen-ling-2026-09-03`

Status: **frozen after Qwen development and Ling transfer-pilot qualification,
before any of the 500 formal items is called**.

## Frozen inputs

- Selection manifest SHA-256:
  `8a8024c88fca948fd5086464b0ee9a2cc4bc6c84431419ffd1d4061d654eeb75`
- Risk manifest SHA-256:
  `4c7f6f833849ac1fc25f88980b8c4d483ba2a30f540a09b29aea210e6f6aa215`
- Ling pilot summary SHA-256:
  `e1d9e448fc9db5a075269430607ae826c96da3b0e59d73e5e9e26ace85feefe9`
- Frozen risk:
  `0.3 * reverse_inertia + 0.7 * intervention_disagreement`.
- High-consensus threshold: 0.8.
- Selective coverage: 0.8.

The Ling pilot qualified with overall/macro/worst-label AUROC
`0.709/0.737/0.710` and Risk@80 error `0.328 -> 0.283`. These are development
statistics only.

## Formal population

The formal set contains 250 previously uncalled Wikipedia page roots and 250
natural contrastive pairs. Each pair yields one SUPPORTS and one REFUTES item,
for exactly 500 balanced items. Target pages and distractor pages are globally
unique and disjoint from every smoke/development/formal target or distractor
root registered in the selection manifest.

The runtime receives an outcome-free public manifest with opaque item IDs. Gold
labels are stored in a separate outcome ledger. Model calls, pre-outcome risk
coordinates, and retained IDs are frozen before the evaluator may read that
ledger.

## Models and calls

- Qwen3.5-4B, locally served at its frozen endpoint;
- Ling-3.0-tiny, locally served from the frozen int4 checkpoint;
- five agents and four conditions per item;
- exactly 500 x 5 x 4 = 10,000 calls per model;
- common prompt-only JSON interface, strict parser, 256-token budget, fixed
  model-specific seeds, and fresh model-specific formal caches.

Formal execution is all-or-nothing. Partial metrics, optional stopping, example
replacement, parser tuning, or model-specific score changes are prohibited.

## Fail-closed transport

Each model must achieve final validity at least 0.98 and first-pass validity at
least 0.95. An incomplete intervention bundle remains in the population and is
assigned risk 1.0. Original consensus uses valid original answers with the
fixed five-agent denominator; fewer than four agreeing agents cannot enter the
high-consensus subset.

## Primary metrics and gates

Inference resamples natural contrastive pairs, keeping the SUPPORTS/REFUTES
members together. Five thousand bootstrap replicates use seed `20261616`.

Each model must independently satisfy all of:

1. final and first-pass transport gates above;
2. at least 400 high-consensus items;
3. at least 20 high-consensus errors in each native label;
4. overall error-detection AUROC 95% pair-bootstrap CI lower bound > 0.5;
5. macro-label AUROC 95% CI lower bound > 0.5;
6. worst-label AUROC 95% CI lower bound > 0.5; and
7. Risk@80 absolute error-reduction 95% CI lower bound > 0.

The cross-family headline passes only if both Qwen and Ling pass all seven
model-level gates. Pooled data cannot rescue a failed model or label.

## Same-prediction baselines

All risk methods rank the same original-condition consensus errors:

- vote disagreement (`1 - agreement`);
- low mean consensus confidence;
- reverse inertia alone;
- intervention disagreement alone;
- deterministic hash-random ordering.

For each baseline, report overall/macro/worst-label AUROC, AURC, and Risk@80 at
the identical retained count. Baseline dominance is secondary and cannot
replace the primary gates.

## Secondary analyses

- common-item Qwen--Ling risk-rank correlation;
- pairwise polarity consistency under natural reverse;
- per-agent and per-condition flip rates;
- error taxonomy for the highest-risk and lowest-risk tails;
- transport failures by model/agent/condition; and
- component ablations without alternative formal verdicts.

## Claim boundary

A pass supports label-symmetric selective detection transfer from Qwen to Ling
under natural contrastive Wikipedia evidence. It does not establish universal
factuality, independent agents, arbitrary-domain transfer, answer repair, live
retrieval robustness, or financial performance.
