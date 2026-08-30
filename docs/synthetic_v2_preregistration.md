# Synthetic V2 preregistration: decoupled behavior baselines

## Why V2 exists

V1 exposed a generator-design confound: its self-reported confidence was a
deterministic transformation of the same source-quality estimate used by the
recent-performance baseline. That makes their pooled performance partly
tautological and does not test whether provenance adds information beyond agent
behavior. V1 results remain unchanged under `results/synthetic_v1/`.

V2 is a new, explicitly post-V1 stress suite. The following choices are frozen
before V2 results are generated; it must not be represented as a preregistration
of the already observed V1 result.

## Fixed protocol

V2 retains V1's five scenarios, four agent counts, four source-quality noise
levels, four corruption strengths, aliased provenance, opaque transformations,
leave-one-corruption-mechanism-out partitions, metric definitions, 75th-percentile
training-only threshold rule, and 1,000 base-seed-cluster bootstrap intervals.

Its fixed base seeds are `1103, 1201, 1301, 1409, 1511, 1601, 1709, 1801, 1901,
2003`. Mechanism/configuration episodes are generated from independent stable
derivations of those seeds.

## The V2 intervention

The benchmark separates three as-of signals that V1 conflated:

1. **Agent confidence** is mostly an independently generated, noisy and
   potentially miscalibrated behavioral report (`15%` coupling to source
   quality plus fixed confidence noise).
2. **Recent performance** is each agent's independently sampled historical
   track record. It does not reveal whether several agents currently rely on
   one source.
3. **Provenance** still receives the environment-held evidence graph and the
   source-integrity audit already used by `provenance_risk`: leaf-source
   concentration, as-of source quality, staleness, and temporal validity.

This is not an oracle intervention: no deployable score reads the current
outcome or `harmful_false_consensus` label. It models the intended distinction
between behavior-only routing and a runtime that can audit the provenance of
the evidence packets it assigned.

## Hypothesis and decision rule

The primary V2 hypothesis is that provenance has lower pooled held-out AURC
and lower high-confidence error than both Confidence and Recent performance,
without higher false rejection. This supports only the controlled setting
where a source-integrity audit is available. The same metrics must still be
reported mechanism-wise; a pooled win alone is insufficient for a broad claim.

If V2 does not support the hypothesis, report the negative result and revise
the mechanism rather than changing seeds, thresholds, or metrics after seeing
the outcome.
