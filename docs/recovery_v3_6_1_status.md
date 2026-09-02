# Recovery V3.6.1 status: development unlock, formal certificate abort

Protocol: `recovery-v3.6.1-atomic-pace-ex-fever-development-2026-09-02`

Status: **retained pre-outcome formal abort; no test action or outcome was accessed**.

## Development result

All 1,200 development certificates and 4,800 development action records passed
validation. The frozen three-fold out-of-fold policy selected
`p_fix >= 0.50`, `p_harm <= 0.15`, and `p_fix - p_harm >= 0.20` from 40
feasible configurations. The held-out fold results were:

| fold | routes | fixes minus harms | harms | macro-label gain | Supported gain | Refuted gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 9 | 8 | 0 | +0.040 | +0.080 | +0.000 |
| 1 | 13 | 13 | 0 | +0.065 | +0.130 | +0.000 |
| 2 | 22 | 22 | 0 | +0.110 | +0.220 | +0.000 |

The router manifest was frozen before any formal test call. Its implementation
SHA-256 is
`3a477b173b51adb0c442ef3c9d864501491f32b5054741aa3c6871fcc15cd947`.

## Formal execution abort

The two-item transport/schema smoke passed. Formal certificate collection then
covered all 500 test bundles (1,000 candidate rows), but one bundle failed the
pre-outcome completeness gate. Both candidates for example
`2f521d8a944d5a1fe4c47df825935265` exhausted the initial call and the single
fixed repair at exactly 512 completion tokens. All four responses were
truncated before a complete JSON object. The resulting partial file contains
998 valid and two invalid candidate rows; no completed formal certificate file
was promoted.

The run stopped before formal action generation. No test gold, repair outcome,
route metric, or aggregate test result was inspected. The partial records and
caches are retained under `results/recovery_v3_6_1/test/certificates/`.

## Next protocol boundary

V3.6.2 may reuse the frozen development router but must use an entirely new,
previously uncalled, page-root-disjoint formal test split. It will preregister a
fail-closed rule: after the same fixed certificate attempts are exhausted, the
candidate has no proof and is ineligible for routing. This is an operational
safety rule, not an outcome-conditioned repair.
