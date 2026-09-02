# Recovery V2.2 result: provenance-action routing

Protocol: `recovery-v2.2-page-root-2026-09-02`

Formal verdict: **NO_VERIFIED_NET_RESCUE**.

This is not a null result. The conservative router produced a statistically
positive, low-damage improvement over keeping the initial consensus, but it
failed two deliberately stronger preregistered gates.

## Primary result

| Policy | Accuracy | Fixes | Harms | Net fixes | 95% paired net-gain CI | Added roots/example |
|---|---:|---:|---:|---:|---:|---:|
| KEEP | 76.52% | 0 | 0 | 0 | [0.00, 0.00] pp | 0.000 |
| Learned conservative | 78.70% | 6 | 1 | +5 | [+0.43, +4.78] pp | 0.091 |
| Learned unrestricted | 84.78% | 25 | 6 | +19 | [+3.91, +13.04] pp | 0.630 |
| Retrieval-score routing | 82.17% | 23 | 10 | +13 | [+0.87, +10.43] pp | 0.965 |
| Fixed candidate 0 | 80.87% | 23 | 13 | +10 | [-0.87, +9.57] pp | 0.965 |
| Fixed candidate 1 | 77.83% | 11 | 8 | +3 | [-2.61, +4.78] pp | 0.965 |
| Fixed both | 86.09% | 31 | 9 | +22 | [+4.35, +15.22] pp | 1.930 |
| Available-action oracle | 91.30% | 34 | 0 | +34 | [+10.43, +19.57] pp | 0.161 |

The oracle is diagnostic and reads action outcomes; it is not deployable.

## Frozen gates

| Gate | Result | Pass |
|---|---|---:|
| Paired 95% CI lower bound above zero | +0.43 pp | Yes |
| Net fixes above every fixed action | +5 versus fixed-both +22 | No |
| Damage rate at most 5% | 0.57% | Yes |
| Non-negative gain in both labels | SUPPORTS +4.88 pp; REFUTES -0.93 pp | No |
| At least five annotation-supported repairs | 6 | Yes |

## Interpretation

The conservative policy demonstrates that pre-outcome provenance and consensus
features can identify a small subset worth intervening on: it selected a
non-KEEP action for 13/230 examples, repaired six errors, and caused one harm.
This is stronger than Recovery V1's learned result and establishes a real
routing signal under page-root-disjoint evaluation.

It does not establish the full paper claim as preregistered. A simple fixed
two-root acquisition policy recovered substantially more errors, although at
far greater evidence cost and with a 5.14% damage rate. The unrestricted router
also recovered more than the conservative policy, showing that conservative
calibration trades much of the available gain for safety.

The principal generalization failure is label-asymmetric. Initial REFUTES
accuracy was already 97.20%, leaving little repair headroom; interventions
mostly introduced harm there. SUPPORTS started at 58.54%, and the conservative
router improved it to 63.41%. This asymmetry is a result to explain, not a basis
for retuning V2.2.

## Claim boundary

The evidence roots are distinct Wikipedia pages, not independent publishers.
The generator and adjudicator use the same Qwen model family. The result does
not establish cross-model transfer, live retrieval robustness, or provenance
independence at the organization level. A subsequent protocol may study those
questions, but V2.2 remains frozen unchanged.

## Operational amendment

The same named Qwen deployment was moved from `10.63.0.88:31519` to
`10.63.0.82:31518` during dev collection. The relocation was documented before
dev resumed and before test calls; the new service reported the same model ID
and named weight path. All test calls used the relocated endpoint. This should
be disclosed as an operational limitation even though no scientific field or
existing outcome changed.
