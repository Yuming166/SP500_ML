# Recovery V3.4.1 preregistration: publisher-blocked witness jury

Protocol version: `recovery-v3.4.1-pbwj-climate-fever-2026-09-02`

Status: **frozen before fitting and before any formal CLIMATE-FEVER Qwen call**.

## Pre-formal amendment

V3.4 stopped before fitting or model calls because its structural audit found
363 distinct Wikipedia page roots rather than the preregistered minimum of 400.
V3.4.1 changes only that feasibility gate to 350. This preserves a requirement
covering more than 96% of the observed roots. The selection, method, thresholds,
endpoint, analyses, primary success gates, and every other structural gate are
unchanged. V3.4 artifacts and hashes remain preserved.

V3.3 and its AVeriTeC-dev result also remain immutable. Its test is exposed and
is source development material only. V3.4.1 is evaluated once on the previously
unused public CLIMATE-FEVER dataset; external outcomes cannot alter the router,
selection, thresholds, sample, gates, or analysis.

## External selection

The official public `climate-fever.jsonl` has SHA-256
`8a4b9032d861be482ffb49dddfd283ffa6089e654f1e968040011882c5eb6e0b`.
Eligibility requires a `SUPPORTS` or `REFUTES` claim and at least two distinct
Wikipedia article titles whose nonempty evidence has the same label as the
claim. All 483 eligible items are retained: 361 Supported and 122 Refuted.

One matching article is the anchor, a second is the hidden annotated root, and
a retrieval-score-matched article packet from another claim is the distractor.
Candidate position is deterministically balanced. The selection must have:

- exact counts and unique claims;
- zero normalized-claim overlap with local AVeriTeC and FEVER material;
- three distinct roots per item;
- candidate-0 annotated fraction in `[0.49, 0.51]`;
- oriented candidate-role retrieval AUC at most 0.85;
- at least 100 examples per label and 450 examples overall; and
- at least 350 distinct page roots.

These roots are Wikipedia article pages, not independent publishers. This tests
cross-dataset evidence-page-root transfer, not publisher independence.

## Method: PBWJ

Publisher-Blocked Witness Jury extends CEW without altering prior artifacts.
Five word/character TF-IDF logistic stance witnesses use the union of the
exposed AVeriTeC partitions. Source claims are grouped by fact-check publisher,
the groups are greedily balanced into five folds, and member `k` trains while
leaving fold `k` out. Each member sees one retrieval-matched irrelevant packet
per source claim as a hard negative.

At inference, models receive only claim and packet text. Initial consensus `no`
uses `supports` as the counter-consensus stance; `yes` uses `refutes`. Each
member votes for the candidate with higher counter-consensus probability. For
the modal candidate:

```text
pessimistic_score = mean(member candidate probabilities) - sample_sd(...)
pessimistic_delta = mean(candidate - anchor) - sample_sd(candidate - anchor)
```

These are conservative dispersion penalties, not statistical confidence bounds
or distribution-free guarantees. PBWJ acquires one root exactly when:

```text
initial five-agent agreement >= 0.8
jury agreement >= 0.8 for consensus no
jury agreement == 1.0 for consensus yes
pessimistic_score >= 0.4
pessimistic_delta >= -0.2
```

Score and delta thresholds are inherited unchanged from V3.3. The stricter
`yes -> refutes` quorum is a source-development amendment motivated by the V3.3
harm audit and is frozen before this external test. External native labels,
gold answers, annotation roles, action outcomes, and source identities are
forbidden route features.

## Execution and evaluation

The frozen service is `Qwen3.5-4B` at
`http://10.63.0.82:31518/v1/chat/completions`. Each example receives five
anchor-only baseline calls and three counterfactual recovery calls. A two-item
smoke test checks transport and parsing only. Formal routes and jury diagnostics
are serialized and hashed before gold-derived outcomes are constructed.

Prespecified ablations remove the dispersion veto and then all uncertainty
vetoes. Nonlearned comparators are retrieval-score, hash-random, and each fixed
action, reported both unlimited and at PBWJ's realized root budget. An available-
action oracle is diagnostic only.

PBWJ passes only if every frozen primary gate holds on all 483 examples:

1. 10,000-replicate label-stratified paired-bootstrap 95% lower bound for
   native-label-macro gain is above zero;
2. damage among initially correct high-consensus cases is at most 5%;
3. net gain is nonnegative for both native labels;
4. at least ten repairs cite the acquired annotated-root evidence; and
5. net fixes exceed KEEP and every root-budget-matched nonlearned comparator.

The bootstrap seed is `20260943`. Any failed gate is a negative result; no
threshold or sample will be revised after external outcome inspection.

## Claim boundary

A pass supports prospective cross-dataset transfer of uncertainty-gated,
single-root repair for one Qwen deployment. It does not establish cross-model
transfer, live retrieval robustness, publisher independence, causal independence
between Wikipedia pages, or a per-query safety guarantee.
