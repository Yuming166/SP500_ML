# Detection V3.15.2 formal transport amendment

Protocol: `detection-v3.15.2-ling-boolq-v12.1-2026-09-03`

Status: **to be frozen after the V3.15.1 formal transport abort and before any
V3.15.2 call**.

V3.15.1 completed all 7,160 formal logical positions but failed closed before
outcome analysis because 20 rows remained invalid after the fixed retry. It had
7,140 final-valid and 7,133 first-pass-valid rows. All 20 terminal failures were
in the `remove` condition, whose environment packet is empty: 13 cited the
placeholder `evidence_1` and seven cited `evidence_packet`. Target records contain
no labels or gold answers; no AUROC, Risk@80, subgroup, or correctness metric was
computed.

V3.15.2 adds exactly one environment-enforced normalization: when and only when
the allowed evidence-ID set is empty, serialize `cited_evidence_ids=[]` and drop
an `evidence_ids` alias if present. This cannot alter the answer, confidence,
intervention, or risk score. Citations for every nonempty packet remain subject
to the full strict membership check.

All 7,160 calls are rerun with a fresh cache. No V3.15.1 response is reused.
Question set, model, prompt, fixed agent ID amendment, substitutes, score,
coverage, bootstrap, metrics, and both verdict layers remain unchanged. The
protocol manifest content-addresses all four V3.15.1 formal partials and their
outcome-blind transport audit before any V3.15.2 call.

