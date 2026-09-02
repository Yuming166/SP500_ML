# Recovery V3.6 development protocol: proof-carrying Atomic-PACE

Protocol candidate:
`recovery-v3.6-atomic-pace-ex-fever-development-2026-09-02`

Status: **development only; formal EX-FEVER test remains untouched**.

## Motivation and novelty target

V3.5.2 showed that the candidate-root problem and the action-safety problem are
separable. Its scalar certificate selected the hidden annotated root on 263/266
gated actions, yet missed a single false atom in many Refuted conjunctions. For
example, it treated a claim as supported when the packets agreed on the actor
and film but explicitly disagreed on `Canadian` versus `American`, or on 2005
versus 2007.

V3.6 keeps the paired `p_fix`/`p_harm` action learner but replaces the scalar
semantic opinion with a proof-carrying state-transition certificate. Qwen must
split the claim into material exact-span atoms and attach packet-local evidence
to `supported`, `contradicted`, or `unresolved` checks. A deterministic checker,
not Qwen, derives the relation:

```text
any grounded contradicted atom -> refutes
complete coverage and all atoms grounded-supported -> supports
otherwise -> insufficient
```

The candidate root must carry evidence for the derived relation. Invalid spans,
nonlocal IDs, and duplicate IDs are removed and disclosed; fuzzy remapping is
forbidden. This makes every route carry a locally auditable witness for why the
new provenance root should change the consensus. The research contribution is
the combination of proof obligations, action-specific paired outcomes, and an
explicit learned harm model—not atomic fact checking alone.

## Development boundary

V3.6 uses exactly the same 600 development and locked 500 formal items as the
V3.5 family. It may reuse the already frozen development action responses,
because claims, packets, action prompts, Qwen model, and seeds are identical;
only the new certificate calls differ. No formal V3.5 or V3.6 test call exists.

The first 40 deterministically ordered development examples form the Atomic-
PACE prompt pilot. The pilot may be inspected for schema validity, exact-span
retention, derived relation by native development label, candidate-root
selection, fixes, and harms. The main questions are whether explicit attribute
checking reduces Refuted false-support certificates without destroying useful
Supported certificates. Any prompt/schema change requires a new protocol
identifier and preserved pilot artifacts.

If accepted, all 600 development examples receive two atomic certificate
calls. Three-fold OOF policy selection uses the same prespecified model family,
threshold grids, and strict per-fold unlock conditions as V3.5.2. A formal
preregistration and implementation hash must be written before fitting or any
test call. If no fold-robust policy exists, formal execution remains locked.
