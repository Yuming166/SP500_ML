# Recovery V3.9.1 development-smoke abort

Status: `ABORTED_DEVELOPMENT_CERTIFICATE_ENVELOPE_BEFORE_FORMAL`

The frozen V3.9.1 manifest has SHA-256
`718fe74d5a6de9d5a266ef5773f82721393a0deec40fda89ddeb959461b803d8`.
No V3.9.1 formal target call was made. Its fixed two-example smoke completed
all 16 action rows successfully (11 strict JSON and five leading-JSON plus
trailing-text rows). Two of four certificate rows were successful; the other
two returned schema-valid certificate JSON followed by brace-free explanatory
prose on both attempts. Offline, outcome-blind replay confirmed that all four
failed responses pass the unchanged strict certificate validator after only
removing the trailing envelope text.

No correctness, gain, route, ledger, action outcome, or formal record was
computed. The smoke records and cache are preserved and forbidden as V3.9.2
inference inputs.

V3.9.2 applies the already qualified one-leading-JSON envelope rule uniformly
to actions, certificates, and ledgers. Certificate and ledger semantic
validators remain unchanged.
