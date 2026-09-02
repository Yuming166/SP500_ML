# Recovery V3.3 preregistration: contrastive evidence witnessing

Protocol version: `recovery-v3.3-cew-averitec-2026-09-02`

Status: **frozen before any AVeriTeC-dev prospective-test call**.

## Motivation and amendment boundary

V3.2 remains immutable and ended before test because its independent safety
shield selected the preregistered KEEP fallback. Development diagnostics found
that its action-value policy often selected `both`: this repaired Supported
claims but could inject a misleading root into already-correct Refuted cases.
No prospective-test response or outcome has been generated or inspected.

V3.3 treats every AVeriTeC-train item as method-development data. The former
policy-selection and calibration partitions become two equally binding
development folds; neither is presented as a prospective result. The same
untouched official AVeriTeC dev split remains the single prospective test.

## Method: CEW-Router

**Contrastive Evidence Witnessing (CEW)** separates error suspicion from
repair-path selection. It reuses the V3.2 stance encoder, trained only on the
model-train partition, and never uses a native label, annotation role, source
name, gold answer, or action outcome at inference.

For initial consensus `no`, the counter-consensus stance is `supports`; for
initial consensus `yes`, it is `refutes`. For each candidate root CEW computes
its counter-consensus probability. It selects the single candidate with the
larger probability and routes only if:

```text
candidate counter-consensus probability >= p
candidate probability - anchor probability >= delta
initial agreement >= 0.8
```

Thus the same semantic object decides both **whether to repair** and **which
provenance root to acquire**. It is not a detector followed by a fixed repair,
and it never routes to `both` in the primary policy.

## Frozen development selection

The only grid is:

```text
p     in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
delta in [-0.2, 0.0, 0.1, 0.2, 0.3, 0.4]
```

A pair is feasible only if it separately achieves, on both development folds:

- damage at most 5% among initially correct high-consensus cases;
- nonnegative net gain in both native labels; and
- at least ten routed items.

Among feasible pairs, selection lexicographically maximizes worst-fold net
fixes, total net fixes, lower damage, fewer routes, then higher `p` and
`delta`. If none is feasible, V3.3 falls back to KEEP and the prospective test
is not called.

The stance model, selected pair, all input hashes, feature boundary, and model
hash are serialized before any prospective-test call.

## Prospective evaluation

All non-oracle routes are computed from frozen evidence packets and the five
baseline decisions, then written to a hashed `preoutcome_routes.json` before
gold-derived outcomes or the oracle are constructed.

The primary test gates are unchanged from V3.2:

1. native-label-macro gain has a 2,000-replicate paired-bootstrap 95% lower
   bound above zero;
2. damage among initially correct high-consensus cases is at most 5%;
3. net gain is nonnegative in both native labels;
4. at least ten repairs cite evidence in the acquired annotated root; and
5. net fixes exceed KEEP, retrieval-score routing, hash-random acquisition,
   and each fixed action capped at CEW's realized root budget.

Secondary results report unlimited baselines, the no-threshold semantic
witness, the frozen V3.2 KEEP fallback, oracle action selection, source-domain
seen/unseen groups, and annotation-role selection accuracy. The test cannot be
used to revise V3.3.

## Claim boundary

A pass would establish prospective, budget-matched evidence-path routing for
one Qwen deployment on AVeriTeC. It would not prove source-publisher
independence, cross-model transfer, live-search robustness, or a
distribution-free per-query safety guarantee.
