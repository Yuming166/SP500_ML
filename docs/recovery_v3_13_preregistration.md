# Recovery V3.13 preregistration: provenance-gated counter-consensus cascade

Protocol version: `recovery-v3.13-counter-consensus-cascade-2026-09-03`

Status: **to be frozen by a content-addressed protocol manifest before any
V3.13 study-root call**.

## 1. Motivation and declared development use

V3.12 was a valid formal negative result: its two accepted routes both fixed
the Hy-MT2-1.8B consensus, but requiring the small target to independently
change its answer limited coverage. V3.13 preserves V3.12 unchanged and, only
after that verdict, declares its 112 examples as calibration data.

The fixed development grid is relation margin `{0.13, 0.15, 0.18, 0.20}`.
For each margin, a route is accepted when the frozen provenance head selects a
root, the frozen relation head disagrees with the target consensus, and Qwen
agrees with the relation label at confidence at least `0.80` while citing the
selected root. The declared selection rule is: among thresholds with zero
calibration harms and at least five fixes, choose the one with most accepted
routes. This selects relation margin `0.15` (13 fixes, 0 harms), versus 15/1 at
0.13, 7/0 at 0.18, and 3/0 at 0.20.

This is therefore a calibrated, not zero-shot, Hy-MT2-1.8B experiment.

## 2. Frozen method

The primary method is a provenance-gated counter-consensus cascade:

1. five Hy-MT2-1.8B anchor-only agents form a baseline consensus;
2. the frozen Qwen3-Embedding-0.6B provenance scorer selects one of two roots;
3. the frozen logistic relation head predicts Supported or Refuted from that
   root;
4. a provisional route requires target agreement at least `0.80`, provenance
   margin at least `0.30`, relation margin at least `0.15`, and relation/target
   disagreement;
5. only provisional routes are sent to Qwen3.5-4B; and
6. the consensus is overridden only when Qwen agrees with the relation label,
   has confidence at least `0.80`, and cites evidence from the selected root.

On accepted routes Qwen is explicitly the answer source. The contribution is
selective evidence-path routing and counter-consensus verification, not a
claim that the small target generated the repaired answer itself.

## 3. Formal data and independence

The formal set has 80 FEVER-validation examples, balanced 40/40. Every claim,
annotated root, anchor root, and distractor root is disjoint from V3.7.1,
V3.11, and V3.12. Selection may inspect labels only to enforce the fixed
40/40 balance; it cannot inspect any target, teacher, or router output.

Annotated roots are unique and disjoint from auxiliary roots. Auxiliary root
reuse is at most seven, candidate position is exactly balanced, and the
lexical retrieval gap is at most `0.10`. Retrieval score is excluded from the
primary router. Its role AUC and above/below direction counts are reported as
diagnostics rather than success gates because the exhausted untouched
validation pool cannot support the earlier direction quotas.

## 4. Immutable execution order

1. Build and audit the fresh selection.
2. Freeze the V3.12-derived calibration table and V3.13 router manifest.
3. Encode formal packets in inference-only mode.
4. Freeze implementation, data, model, prompts, thresholds, scripts, and
   success gates in the protocol manifest.
5. Verify endpoints and schema-smoke only old V3.12 roots.
6. Generate all target records on the 80 formal examples.
7. Freeze provisional routes before any formal Qwen call.
8. Query Qwen only for frozen provisional routes.
9. Freeze final routes, override labels, and same-budget baselines.
10. Access outcomes once for evaluation.

## 5. Primary gates

Native-label-stratified bootstrap uses seed `20261305` with 10,000 replicates.
All gates must pass:

- macro-label gain CI lower bound above zero;
- zero observed harms;
- both native-label gains nonnegative;
- at least five annotation-supported repairs;
- net fixes above KEEP and every registered same-budget action baseline;
- provenance-path accuracy at least 90%; and
- fewer formal Qwen calls than formal target examples.

## 6. Claim boundary

A pass supports a calibrated sparse cross-model repair cascade on fresh roots.
It does not support zero-shot transfer to Hy-MT2-1.8B, universal transfer,
or literature-level priority. The result must be reported alongside the V3.11
and V3.12 negatives and with Qwen's answer-source role stated plainly.
