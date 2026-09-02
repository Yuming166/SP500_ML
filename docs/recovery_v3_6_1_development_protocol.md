# Recovery V3.6.1 development protocol: proof-carrying Atomic-PACE

Protocol candidate:
`recovery-v3.6.1-atomic-pace-ex-fever-development-2026-09-02`

Status: **development only; formal EX-FEVER test remains untouched**.

## Pre-fit execution amendment

V3.6 introduced an atomic proof certificate after V3.5.2 showed that scalar
certificates found the right evidence root but missed isolated false attributes
inside long conjunctions. In the fixed first-40 pilot, 21/80 certificates
completed and validated, while 59/80 hit the inherited 160-token completion
limit mid-JSON. V3.6.1 changes only atomic certificate `max_tokens` to 512.
Ordinary baseline and recovery calls stay at 160. The prompt, parser, examples,
actions, features, models, grids, seeds, and formal gate remain unchanged.

## Method

Qwen splits a claim into 2--8 exact-span material details, checks each against
the anchor and one proposed new root, and attaches only packet-local evidence
IDs. A deterministic checker derives the candidate relation:

```text
any grounded contradicted atom -> refutes
complete coverage and all atoms grounded-supported -> supports
otherwise -> insufficient
```

The new root itself must carry evidence for the derived relation. Nonlocal or
duplicate IDs are dropped and disclosed; invalid spans are not repaired. PACE
then learns action-specific `p_fix` and `p_harm` from the already frozen paired
development interventions. The design is proof-carrying because route
eligibility is tied to a locally checkable claim-span/evidence witness, rather
than an opaque relevance or truth score.

## Development boundary

The same deterministic first 40 of 600 development examples are rerun with the
larger output budget. Their schema validity, native-label relation, root
selection, fixes, and harms may be inspected. No formal item may be called. If
the prompt is accepted, all 600 development examples receive two atomic
certificates; the reused action responses are revalidated under the new
protocol. A formal preregistration and exact implementation hash are required
before fitting or test calls. Failure of the unchanged three-fold OOF unlock
conditions keeps the formal test locked.

## Pilot decision (2026-09-02)

All 80 certificates completed and passed on their first 512-token attempt. The
pilot contained 26 Supported and 14 Refuted examples; the frozen baseline was
correct on 25/40. Certificates contained 3.61 retained atoms on average, with
11 nonexact spans conservatively dropped and no nonlocal evidence IDs.

Across the 28 candidate certificates attached to Refuted examples, 26 derived
`refutes` and two derived `insufficient`. The direct counter-consensus proof gate
routed ten examples, chose the held-out annotated root nine times, repaired
eight errors, and caused zero harms (macro-label gain +15.38 percentage points;
Supported +30.77, Refuted +0.00). These numbers are prompt-development
diagnostics, not prospective evidence.

The record SHA-256 is
`4bad54209f5a0c38852cb9f688ad6e4ed33e21bf3c920b42a01e8e3c04ce164c`.
The prompt, parser, and 512-token budget are accepted without further change.
