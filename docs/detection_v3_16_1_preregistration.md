# Detection V3.16.1 development transport amendment

Date: 2026-09-03 (Asia/Shanghai)

Parent protocol: `detection-v3.16-vitaminc-symmetric-development-2026-09-03`

Status: **frozen after the V3.16 Qwen smoke failure and before any V3.16.1
call**. The parent selection manifest and all sample, intervention, metric, and
split rules remain byte-for-byte unchanged.

## Observed transport failure

The first Qwen transport smoke produced 153/160 final-valid and 150/160
first-pass-valid rows. All seven terminal failures were confined to one
unrelated `substitute` packet: Qwen returned a valid answer and confidence but
left `cited_evidence_ids` empty rather than citing irrelevant text. Original,
remove, and natural-reverse calls had no terminal failure.

## Frozen amendment

V3.16.1 permits `substitute` to return either the exact visible ID or an empty
citation list. Original and reverse still require the exact visible ID, while
remove still requires an empty list.

No answer, confidence, sample, condition, prompt evidence, metric, risk
coordinate, or outcome field changes. V3.16.1 uses a fresh cache and repeats
the complete 160-call smoke.

The first failed smoke remains preserved under
`results/detection_v3_16_development/qwen/`. V3.16.1 writes only under
`results/detection_v3_16_development/calls_1/`.
