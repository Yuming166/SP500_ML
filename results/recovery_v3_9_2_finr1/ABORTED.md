# Recovery V3.9.2 pre-evaluation structured-output abort

Status: `ABORTED_FORMAL_ACTION_JSON_GRAMMAR_BEFORE_OUTCOME_EVALUATION`

The frozen V3.9.2 manifest has SHA-256
`aa751e8488cf67ceee8033e4d8bd37a04b3192dc473e3bcd3d0c2d0c0ae0e401`.
The two-example smoke completed 16/16 actions and 4/4 certificates; its one
ledger candidate failed closed on the unchanged ledger schema.

The formal action run was stopped after 22 complete bundles (176 rows), before
formal certificates, ledgers, route selection, or outcome evaluation. Eighteen
rows remained invalid after repair because Fin-R1 emitted unquoted evidence
identifiers inside otherwise JSON-shaped arrays, for example `[A02, C000]`.
There were 36 failed attempts, all ending as
`ValueError: response must be exactly one JSON object`.

No accuracy, gain, label-stratified result, route, or action outcome was
computed. Partial records and caches are preserved and forbidden as V3.10
inference inputs.

V3.10 is a separately frozen transport protocol that requests vLLM-native
JSON-schema constrained generation for all three artifact kinds. The generated
objects still pass the unchanged strict semantic validators.
