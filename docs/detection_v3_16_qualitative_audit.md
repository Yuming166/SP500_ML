# Detection V3.16 Qualitative Audit

Date: 2026-09-05 (Asia/Shanghai)

Status: **post-hoc qualitative analysis only**. This document does not change
V3.16 records, routes, weights, bootstrap intervals, gates, or the registered
joint verdict.

## Selection rule

The examples are selected from the Qwen V3.16 formal high-consensus subset
after the formal outcome ledger was opened:

1. choose the error with maximum frozen `R_sym`; and
2. choose the correct consensus with minimum frozen `R_sym`.

Ties are resolved by the deterministic analysis opaque ID. The examples are
illustrative, not estimates of tail frequency and not new evidence for the
primary endpoint. Agent-vector order is `literal, skeptic, consistency,
counterfactual, minimal`.

## Selected Qwen tails

### High-risk error

- Formal item ID: `d6b70cd9d1de2081f8eb`
- Analysis opaque ID: `f2a489b56d91b0bf4452`
- Pair ID: `bae474a16206b430`
- Claim: “Malcolm Young died before November 19, 2017.”
- Gold label: `SUPPORTS`; original consensus: `REFUTES` at agreement `0.80`.
- Mean consensus confidence: `1.00`; frozen `R_sym`: `0.80`.
- Components: reverse inertia `0.80`; intervention disagreement `0.80`.
- Original evidence excerpt: “Malcolm Young ... on the 18th of November 2017.”
- Reverse evidence changes only the date to “the 19th of November 2017.”
- Observable answers: original `S/R/R/R/R`; remove `R/R/R/R/R`; reverse
  `R/R/R/R/R`; substitute `R/R/R/R/R`.

This is a concrete false-consensus pattern: the original evidence supports the
claim, but four of five agents produce a unanimous-looking `REFUTES` consensus
with confidence 1.0. Most agents remain unchanged under the reverse condition,
which makes the item high-risk under the frozen audit. The example does not
prove that the agents ignored evidence internally; it only illustrates the
registered observable association.

### Low-risk correct

- Formal item ID: `abf769048439e64a6554`
- Analysis opaque ID: `00a50d7da33ec942c9ff`
- Pair ID: `c30d02fb24ebff8e`
- Claim: “In Infinity Blade, Drain replenishes the character's hit points.”
- Gold label and original consensus: `SUPPORTS`, agreement `1.00`.
- Mean consensus confidence: `0.98`; frozen `R_sym`: `0.00`.
- Original evidence excerpt: “with the exception of Drain, which restores the
  character's hit points.”
- Reverse evidence replaces `Drain` with `Life`.
- Observable answers: original `S/S/S/S/S`; remove `R/R/R/R/R`; reverse
  `R/R/R/R/R`; substitute `R/R/R/R/R`.

Here all agents change their observable answer when the assigned evidence is
removed or reversed. The low risk reflects response behavior under the frozen
protocol, not a guarantee that the answer is correct in future settings.

## Label-asymmetry table

“Errors” means consensus error within the high-consensus subset. The BoolQ and
V3.15.2 rows are included to show the diagnostic sequence; V3.16 is the
label-symmetric follow-up. These are not pooled into one statistical test.

| Protocol/model | Native label | High-consensus N | Errors | AUROC |
|---|---|---:|---:|---:|
| BoolQ V12.1 / Qwen | yes | 210 | 55 | 0.834 |
| BoolQ V12.1 / Qwen | no | 90 | 11 | 0.213 |
| V3.15.2 / Ling | yes | 224 | 25 | 0.957 |
| V3.15.2 / Ling | no | 86 | 35 | 0.120 |
| V3.16 / Qwen | SUPPORTS | 242 | 37 | 0.792 |
| V3.16 / Qwen | REFUTES | 240 | 50 | 0.830 |
| V3.16 / Ling | SUPPORTS | 238 | 17 | 0.635 |
| V3.16 / Ling | REFUTES | 226 | 96 | 0.897 |

The sequence is consistent with at least three possible sources of asymmetry:
answer priors, asymmetric evidence construction, and intervention polarity.
The existing data do not identify their relative contributions. V3.16 controls
the most direct design issue—native-label balance and pairwise contrastive
construction—but its Ling SUPPORTS event count still falls below the frozen
adequacy threshold. We therefore report reduced asymmetry without claiming a
label-invariant mechanism.
