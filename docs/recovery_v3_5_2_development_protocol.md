# Recovery V3.5.2 development protocol: PACE

Protocol candidate: `recovery-v3.5.2-pace-ex-fever-2026-09-02`

Status: **development amendment; frozen before resuming/completing action
collection and fitting, and before any formal EX-FEVER test call**.

## Pre-fit schema amendments

V3.5 completed its 1,200 development certificate calls but stopped before
action collection or fitting because one certificate (0.083%) cited valid new-
root IDs together with an anchor ID in `new_evidence_ids`; the fixed retry made
the same error. V3.5 records, caches, implementation hash, and preregistration
are retained.

V3.5.1 changes only certificate parsing. It deterministically retains unique
IDs that belong to the proposed new-root packet and records all other IDs as
`dropped_evidence_ids`. If no local ID remains, the certificate cannot pass the
grounding gate. The prompt, data, evidence, features, models, threshold grid,
gates, seeds, and formal test are unchanged. This amendment used schema
validity only, before full development outcomes or any formal test call.

V3.5.1 then stopped after 105/600 complete development action bundles and
before fitting or formal calls. One `both` response cited `C01`, which is not a
packet ID and cannot be uniquely mapped to `C001` or `C101`; its fixed retry
repeated the output. V3.5.2 applies the same exact-match projection to all
decision citations: it retains unique packet-local IDs and records all other or
duplicate strings as `dropped_evidence_ids`. It never fuzzy-remaps an ID and
does not alter the answer or confidence. Therefore this amendment cannot create
an annotation-supported repair and was frozen before resuming development.

## Goal and separation from prior work

V3.5.2 leaves every V1--V3.5.1 artifact and conclusion unchanged. It develops a
new router for the harder problem exposed by V3.4.1: deciding which additional
evidence root is likely to repair an initially wrong consensus without damaging
an initially correct one.

The method is PACE (Provenance-grounded Action Certificates for Error repair).
For each candidate root, Qwen first produces a structured, packet-local
certificate describing whether the anchor plus that root supports, refutes, is
insufficient for, or conflicts on the claim. A learned router then estimates
two action-specific quantities from paired interventions:

```text
p_fix  = P(action becomes correct | baseline is wrong, pre-action features)
p_harm = P(action becomes wrong   | baseline is correct, pre-action features)
```

The policy acquires at most one root. It requires a grounded certificate in the
direction opposite the high-confidence initial consensus and then applies
independently selected fix, harm, and utility thresholds. This is an action-
value router, rather than another static error detector or relevance ranker.

## Data boundary

The source is the official EX-FEVER development split and its accompanying
Wikipedia database. Exactly 600 balanced examples are development data. The
prospective test contains 500 different balanced examples. Test items are drawn
only from singleton evidence-page components; all 1,000 annotated test page
roots are unique across test items. Development and test have zero normalized
claim overlap and zero page-root overlap. Test roots also do not overlap the
CLIMATE-FEVER evidence roots used in V3.4.1.

Candidate order is exactly balanced. Raw retrieval similarity has an oriented
candidate-role AUROC of 0.823 on development and 0.888 on test, so retrieval
score, title overlap, source identity, root identity, and annotation role are
explicitly excluded from router features. The raw AUROC remains a disclosed
selection diagnostic; PACE must rely on the semantic certificate.

## Permitted development

The first 20 deterministically ordered development examples form a plumbing and
prompt pilot. Pilot records are written to a separate immutable directory. The
pilot may be inspected for transport, parse validity, certificate grounding,
and development-only action behavior. Any semantic prompt or schema amendment
requires a new protocol identifier and new cache/output paths before proceeding.

After the prompt is accepted, all 600 development examples receive:

- two certificate calls, one per candidate root;
- five anchor-only baseline calls; and
- three paired recovery calls for candidate 0, candidate 1, and both roots.

Three-fold example-level out-of-fold predictions select a policy only from the
prespecified threshold grid. Equal-weight logistic and random-forest models are
used for each of `p_fix` and `p_harm`. Formal fitting is locked unless every
fold has a nontrivial policy with positive net fixes, no more than 5% damage on
initially correct high-consensus cases, nonnegative gain in both native-label
groups, and at least five annotation-supported repairs.

## Prospective lock

Before any test call, a formal preregistration must freeze the accepted prompt,
selection hashes, model family, threshold grid, development policy, endpoint,
seeds, comparisons, metrics, and success gates. The serialized router and its
manifest are then hashed. Test inference may use only baseline decisions and
certificates; feature construction is tested to work without gold labels or
recovery outcomes. Routes are serialized before outcome analysis.

No development or formal sample is filtered according to whether Qwen happens
to answer correctly. A failed development unlock condition or failed formal
gate is retained and reported as a negative result.

## Pilot decision (2026-09-02)

The frozen first-20 pilot contained 12 Supported and 8 Refuted examples. All 40
certificate calls and all 160 action calls parsed on their first attempt. The
anchor-only baseline was correct on 13/20 examples and all 20 had high
consensus. The certificate-only counter-consensus gate proposed ten single-root
actions; all ten selected the held-out annotated root, six repaired an error,
and one damaged a correct baseline (net +5). This is a development diagnostic,
not a paper result or confidence interval.

The pilot record hashes are:

- certificates: `7e4f7527b022dbb489d0f46854d7a0a3893304fed5e48535dce9334e56b8fc73`;
- actions: `ebfee58b6ab5134fbcf93ffd44886e568dc41cf603eda823d300f6ac04b85da3`.

Because parsing, grounding, and action direction behaved as intended, the
certificate prompt and schema are accepted without amendment. No pilot example
is removed from the prespecified 600-example development set.
