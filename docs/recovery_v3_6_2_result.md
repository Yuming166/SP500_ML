# Recovery V3.6.2 result: fail-closed Atomic-PACE

Protocol: `recovery-v3.6.2-fail-closed-atomic-pace-ex-fever-2026-09-02`

Verdict: **NO_VERIFIED_ATOMIC_PACE_DOMINANCE (four of five primary gates
passed)**.

## Integrity and execution

The formal split contains 400 previously uncalled EX-FEVER examples, balanced
200 Supported and 200 Refuted, with one item per evidence-root component. It has
zero claim or page-root overlap with V3.6.1, zero development/test root overlap,
and 800 distinct test page roots. The exact V3.6.1 router and thresholds were
reused without refitting.

All 800 candidate certificate rows validated; 798 were valid on the first call
and two on the fixed repair. Fail-closed routing was therefore available but not
triggered. All 3,200 action rows validated on the first call. The pre-outcome
route and prediction snapshot was written before outcome construction.

## Frozen primary result

| metric | KEEP | Atomic-PACE |
| --- | ---: | ---: |
| accuracy | 65.50% | 73.50% |
| native-label-macro gain | 0.00pp | **+8.00pp** |
| paired 95% CI for macro gain | [0.00, 0.00]pp | **[+5.50, +10.75]pp** |
| fixes / harms | 0 / 0 | **33 / 1** |
| high-consensus-correct damage | 0.00% | **0.40%** |
| acquired roots | 0 | 35 |
| annotation-supported repairs | 0 | **33** |

Atomic-PACE chose the held-out annotated root on 34 of 35 routes (97.14%). Its
35 routes comprised 18 candidate-0 and 17 candidate-1 actions. Native-label
results were:

- Supported: 42.00% to 58.50%, a **+16.50pp** gain;
- Refuted: 89.00% to 88.50%, a **-0.50pp** gain.

The macro-gain CI, damage, annotation-supported repair, and matched-comparator
gates passed. The primary verdict failed solely because the Refuted group lost
one net example, violating the frozen requirement that both native-label gains
be nonnegative.

## Prespecified secondary result

The proof-only certificate ablation was stronger than the learned risk gate:

- accuracy 77.25%, native-label-macro gain **+11.75pp**;
- paired 95% CI **[+8.50, +15.25]pp**;
- 53 fixes, 6 harms, and 2.42% high-consensus-correct damage;
- Supported +23.00pp and Refuted +0.50pp;
- 53 annotation-supported repairs with 91 acquired roots.

It satisfies numerical analogues of all five primary gates, but it remains a
prespecified secondary/ablation result and is not promoted to the primary
verdict. At the same 35-root budget, nonlearned baselines achieved only 0--3 net
fixes versus Atomic-PACE's 32.

The policy without the certificate gate made 41 routes but produced the same
33 fixes and one harm as Atomic-PACE. Thus the learned `p_fix`/`p_harm` model was
not the source of the main gain: the atomic proof gate carried most of the
effective signal, while the learned harm cap filtered out 20 repairs achieved
by proof-only routing.

## Post-hoc failure analysis

The only primary harm was the Refuted claim that *The Hobbit* was by an
American writer. The router selected the unannotated `Middle-earth` candidate.
Its certificate marked the American-nationality atom supported with confidence
1.0 even though the cited candidate text was only `Tolkien's fantasy.` The
frozen router predicted `p_fix=0.9898` and `p_harm=0.1311`, so the action passed
the 0.50/0.15/0.20 thresholds and changed a unanimous correct `no` consensus to
`yes`.

This is a semantic-grounding false positive, not a provenance-ID failure. The
current deterministic checker proves that atomic claim spans and evidence IDs
are local and structurally valid, but it does not independently prove textual
entailment. The result therefore supports provenance-local atomic repair while
also identifying the exact missing safety layer.

## Paper interpretation and next hypothesis

The honest primary claim is that fail-closed Atomic-PACE produced a significant
+8.00pp macro gain with one harm and beat every budget-matched nonlearned
baseline, but narrowly missed its strict per-label safety gate. A prespecified
proof-only policy produced a larger +11.75pp macro gain and positive gains in
both labels.

A future protocol should be developed on development data only and require an
entailment-carrying atom: each supported or contradicted status must cite an
exact evidence quote/span, followed by a deterministic compatibility check and
an independently calibrated semantic verifier. This would target the observed
failure mechanism without altering V3.6.2.

## Artifact hashes

- selection: `0a8d1f9a4b39aeb37993fd740483deb266b4448cf462f16f5f94fcc5369ba5c9`;
- router manifest: `5622b8dc8ff416bd27297258d2a9e9ad0702432655907c4cabc24c4a6992c990`;
- certificates: `0585eec3b47757e4e3072b88d1fcdb3e23910a7bf5f13e4bfa8a5b40a62c9cb0`;
- actions: `6d880224d27c02dc8ca2430370cd607c5ae28a659526c541c73be4e15d02e6ba`;
- pre-outcome routes: `5e1546029a3b421f8016f3cb7568bb75eca7558344474e104bfec156f406913b`;
- summary: `b9503ed59efafb861e072d0ed7db710af56026e2374d71961ea4dc75ed44d570`.
