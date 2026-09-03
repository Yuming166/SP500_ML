# Detection V3.15.1 preformal conformance amendment

Protocol: `detection-v3.15.1-ling-boolq-v12.1-2026-09-03`

Status: **to be frozen after the V3.15 smoke abort and before any V3.15.1
Ling call**.

V3.15 stopped at its four-question, 80-call smoke and made no formal call. It
produced 29 finally valid rows and 51 terminal failures; all 51 terminal errors,
and all 131 failed attempts, were exactly `missing decision fields: ['agent_id']`.
No correctness, label, consensus-risk, AUROC, or routing metric was inspected.

V3.15.1 adds exactly one semantic-preserving adapter: when and only when the
top-level `agent_id` field is absent, insert the environment-known
`expected_agent_id` already fixed by the request tuple and prompt. A present but
wrong ID remains invalid. Answer, confidence, citations, evidence membership,
JSON shape, retry count, prompt, model, questions, substitutes, `R_PI`, coverage,
bootstrap, and all gates remain unchanged.

V3.15 smoke responses and caches are not reused. V3.15.1 runs a fresh smoke and
fresh formal cache. Its protocol manifest content-addresses the V3.15 protocol,
all four V3.15 smoke partials, and their outcome-free error audit before any new
call.

The original V3.15 preregistration remains the complete scientific protocol;
this document changes transport conformance only.

