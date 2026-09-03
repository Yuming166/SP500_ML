# Detection V3.15 preregistration: Ling replication of BoolQ V12.1

Protocol: `detection-v3.15-ling-boolq-v12.1-2026-09-03`

Status: **to be content-addressed before any Ling BoolQ call**.

## Goal

V3.15 is a cross-model-family replication of the paper's detection experiment,
not another repair-router experiment. It applies the frozen Qwen V12.1
intervention protocol and risk score to `Ling-3.0-tiny-int4`, whose architecture
is `BailingMoeV3ForCausalLM` rather than Qwen.

Ling has prior FEVER recovery/transport records, so it is not globally untested.
It has no BoolQ V12.1 output and has never been used to select the paper's risk
weights, questions, evidence packets, intervention text, coverage, or metrics.

## Exact inherited surface

V3.15 uses all 358 V12.1 questions, their order, five personas, 2-of-3 evidence
partitions, and all four conditions (`original`, `remove`, `reverse`,
`substitute`). The 1,074 frozen V12.1 substitute strings are reused byte for
byte. No question or label subgroup may be removed after Ling output.

The target makes 7,160 logical calls. This design uses the full sample rather
than choosing a favorable dataset or balanced subset after the Qwen result. It
isolates model-family shift while preserving statistical power and per-question
comparability.

## Frozen score and routing point

The score remains exactly:

`R_PI = 0.1 * D_inert + 0.3 * flip_inertia + 0.6 * frac_shared`.

High consensus is agreement at least `0.80`. Risk@80 retains the 80% lowest-risk
high-consensus questions with `(R_PI, cqid)` tie-breaking. Target records omit
labels and gold answers. Risk rows and retained IDs are serialized and hashed
before evaluation loads BoolQ labels.

## Ling transport boundary

The local int4 checkpoint is served by vLLM 0.28.0 on GPU 4 at port 31520 with
thinking disabled, temperature 0, 160 completion tokens, one fixed repair, and
content-addressed caches. The parser permits only the five transformations
frozen from outcome-blind Ling V3.8.3 FEVER transport failures: yes/no
case-folding, unambiguous evidence-id aliasing, empty or singleton citation
string normalization, and finite percentage confidence normalization.

No new parser behavior is permitted after protocol freeze. The four-question
smoke has 80 calls and requires 80 final valid rows and at least 76 first-pass
valid rows. Formal validity must be 100%, with at least 95% first-pass validity.

## Metrics and verdicts

All confidence intervals use 10,000 bootstrap replicates and seed `20261503`.

Aggregate replication passes only if all hold:

1. at least 80 high-consensus questions and at least 10 correct/errors;
2. aggregate AUROC 95% CI lower bound is above 0.5;
3. frozen Risk@80 error-reduction CI lower bound is above zero;
4. formal final validity is 100%; and
5. formal first-pass validity is at least 95%.

Label robustness is a separate stronger verdict. It additionally requires:

1. both yes/no groups have at least 40 high-consensus questions and at least 10
   correct/errors;
2. label-macro AUROC CI lower bound above 0.5; and
3. worst-label AUROC at least 0.5.

The report includes aggregate AUROC, AUPRC, Risk@80, yes/no AUROC, label-macro
AUROC, worst-label AUROC, intervention flip rates, parse modes, and Qwen--Ling
R_PI Spearman correlation on all common items.

An aggregate-only pass must not be called label-invariant. Failure of either
verdict remains a formal negative; no V3.15 weight, prompt, parser, item, or
threshold may be tuned after outcome access.

## Claim boundary

A full pass supports the frozen intervention-based detector across Qwen and Ling
on the same BoolQ evidence regime. It does not establish cross-dataset,
multilingual, financial, or universal transfer. The five personas remain
prompted instances of one target checkpoint rather than independent models.

