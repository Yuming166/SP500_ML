# Recovery V3.6.2 preregistration: fail-closed Atomic-PACE

Protocol: `recovery-v3.6.2-fail-closed-atomic-pace-ex-fever-2026-09-02`

Status: **frozen before every V3.6.2 formal model call and before all test
actions, routes, gold outcomes, and aggregate metrics**.

## Pre-outcome amendment

V3.6.1 passed development unlock and its two-item transport smoke. Its formal
certificate pass then produced 998 valid candidate rows and two truncated rows
belonging to one test item. The initial and fixed repair responses for both
candidates stopped at exactly 512 completion tokens. The run aborted before
formal action generation or outcome access and is retained separately.

V3.6.2 does not repair, replace, or evaluate those 500 items. It freezes a new
test set containing 400 previously uncalled EX-FEVER examples. Every new test
example comes from a different evidence-root connected component; all V3.6.1
claims and page roots and all CLIMATE-FEVER roots are excluded. The split has
200 Supported and 200 Refuted examples, zero development/test claim or page-root
overlap, 800 distinct test page roots, and exactly balanced candidate order.
The frozen selection SHA-256 is
`0a8d1f9a4b39aeb37993fd740483deb266b4448cf462f16f5f94fcc5369ba5c9`.

## Frozen hypothesis and method

The hypothesis is unchanged: among high-consensus decisions, an action-specific
atomic certificate grounded in a new provenance root, combined with separately
estimated repair and harm probabilities, can improve native-label-macro
accuracy without damaging initially correct decisions.

The Qwen certificate prompt, exact-span parser, packet-local evidence checker,
512-token budget, temperature, seeds, one repair attempt, feature schema,
ensemble, and policy are unchanged from V3.6.1. A candidate is proof-eligible
only when a new-root-grounded atomic relation opposes the current consensus.
After both frozen certificate attempts fail, the candidate is deterministically
represented as `insufficient` and is ineligible for routing. This fail-closed
state cannot create a route and is reported as a certificate failure.

V3.6.2 reuses the exact frozen V3.6.1 development records and serialized router;
there is no refit or threshold search. The frozen policy is:

```text
anchor agreement >= 0.80
p_fix >= 0.50
p_harm <= 0.15
p_fix - p_harm >= 0.20
one grounded counter-consensus candidate root at most
```

The original three out-of-fold development folds had macro-label gains of
0.040, 0.065, and 0.110; 44 net fixes, zero harms, and nonnegative gains in both
native-label groups. These are development diagnostics, not formal evidence.

## Frozen execution and analysis

The endpoint is `Qwen3.5-4B` at
`http://10.63.0.82:31518/v1/chat/completions`. Certificate seeds are 20261031
and 20261041; action seeds are 20260991, 20261001, 20261011, and 20261021.
Calls use temperature zero. A two-item smoke may validate only transport and
schema. Full route assignments and `p_fix`/`p_harm` values must be hashed and
serialized before gold-derived outcomes are constructed.

Primary is fail-closed Atomic-PACE versus KEEP. Prespecified ablations are the
same learned policy without the proof gate and proof-only routing. Retrieval,
deterministic hash-random, candidate-0, candidate-1, and both-action policies
are reported both unlimited and truncated to the Atomic-PACE root budget. The
available-action oracle remains diagnostic.

The endpoint is native-label-macro accuracy gain. The paired bootstrap is
stratified by native label, uses 10,000 replicates and seed 20260986. The primary
claim passes only if all gates hold:

1. macro-gain 95% interval lower bound is above zero;
2. damage among initially correct high-consensus items is at most 5%;
3. net gain is nonnegative for both Supported and Refuted;
4. at least ten repairs cite acquired annotated-root evidence; and
5. net fixes exceed KEEP and every root-budget-matched nonlearned comparator.

Any failed gate is a retained negative result. Test outcomes cannot change the
split, fail-closed rule, prompt, parser, features, router, thresholds, metrics,
comparators, or gates.

## Frozen hashes and claim boundary

Implementation SHA-256:

- `recovery_v3_6_2.py`:
  `b80eeb19d1b290ba485ef1db154d105e53de9df857956ac25f74d53b9c599292`;
- `pilot_llm_v1.py`:
  `e1976878d8f6df12a00e384121ec86604999a81e541f8920c61233d214c3b3c5`;
- `recovery_v2.py`:
  `3a458b5a9f03326ab989cf74239d1104fee3df0cdcadbf45c7d7c9b930202c35`;
- `recovery_v3.py`:
  `cf12e563e282fa93fc4406a89459f6b4bc5db271e5bcb265e2d19b20541b7804`;
- `recovery_v3_4.py`:
  `c0213b226f91881eb5083f1b4a062578eabaf5686a87416360ec489edfe6c61b`;
- `recovery_v3_6_1.py`:
  `3a477b173b51adb0c442ef3c9d864501491f32b5054741aa3c6871fcc15cd947`.

The novelty claim is the composition of machine-checkable, provenance-local
atomic witnesses with intervention-trained fix/harm routing and fail-closed
execution. A pass supports prospective component-disjoint repair for one Qwen
deployment on EX-FEVER. It does not establish cross-model transfer, publisher
independence, live-retrieval robustness, or a formal safety guarantee.
