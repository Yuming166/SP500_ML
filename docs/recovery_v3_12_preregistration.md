# Recovery V3.12 preregistration: selective cross-model co-sign repair

Protocol version: `recovery-v3.12-selective-cosign-hy18-2026-09-03`

Status: **frozen by the content-addressed protocol manifest before any
Hy-MT2-1.8B study-root call**.

## 1. Motivation and method

V3.11 produced 26 repairs and one harm. The harm was not a provenance error:
the annotated root was selected, but the relation head and Hy-MT2-7B both
misread a subtle Refuted claim as Supported. V3.12 preserves every V3.11
artifact and adds a selective, independent co-sign stage.

The frozen pipeline is:

1. the V3.11 embedding provenance head selects one candidate root;
2. the unchanged V3.11 relation head predicts Supported or Refuted;
3. a provisional route requires provenance margin `0.30`, relation margin
   `0.13`, target confidence `0.80`, a change from target consensus, and
   target/relation agreement;
4. only provisional routes are sent to Qwen3.5-4B; and
5. execution occurs only if Qwen independently gives the same answer with
   confidence at least `0.80` and cites an evidence ID from the selected root.

The final answer remains the target model's own candidate action. Qwen is a
sparse safety co-signer, not the answer source. The provisional route file is
hashed before any teacher calls, so teacher-query selection cannot depend on
teacher responses or outcomes.

The relation margin and co-sign rule are selected using old Qwen, Ling,
SUFE-Fin-R1, and V3.11 Hy-MT2-7B records. Hy-MT2-1.8B is absent from all
fitting, threshold selection, prompt adaptation, and calibration.

## 2. Formal data

The formal target has 112 FEVER validation examples, balanced 56/56. Claims
and every packet root are disjoint from the complete V3.7.1 selection and the
V3.11 formal selection. The 140 annotated roots are unique and disjoint from
auxiliary roots. Auxiliary roots may repeat at most seven times because the
strictly untouched validation pool contains only 515 roots. Candidate position
and higher/lower retrieval-score direction are balanced within label, and
retrieval-score role AUC must not exceed 0.65.

Formal embedding inputs contain only IDs, split, provenance scores, and
relation vectors. Neither primary-head inference nor teacher-query selection
may access gold labels, annotation roles, or target outcomes.

## 3. Immutable execution order

1. Build and audit the formal selection.
2. Materialize Qwen co-sign decisions for V3.11 development routes.
3. Freeze the safety-router manifest and development diagnostics.
4. Encode formal packets in inference-only mode.
5. Freeze the complete V3.12 protocol and model fingerprints.
6. Start and schema-smoke Hy-MT2-1.8B on old development roots.
7. Generate five target baselines and three target actions for all 112 roots.
8. Freeze provisional target routes.
9. Query Qwen only for the frozen provisional routes.
10. Freeze final co-signed routes and matched baselines.
11. Access outcomes once for evaluation.

## 4. Gates

Native-label-stratified bootstrap uses seed `20261104` with 10,000 replicates.
All gates must pass:

- macro label gain CI lower bound above zero;
- zero observed harms;
- both native-label gains nonnegative;
- at least five annotation-supported repairs;
- net fixes above KEEP and every same-budget registered baseline;
- provenance-path accuracy at least 90%; and
- fewer Qwen teacher calls than formal target examples.

The final gate verifies selective rather than full teacher escalation. The
available-action oracle remains diagnostic only.

## 5. Claim boundary

A pass supports sparse cross-model co-sign transfer to a never-calibrated
smaller Hy checkpoint under both checkpoint and FEVER train-to-validation
shift. It does not establish universal transfer or literature-level priority.
Any novelty claim must be qualified by a separate current literature review.
