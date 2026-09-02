# Recovery V3.8.3 analysis-only amendment

Amendment version: `recovery-v3.8.3-analysis-none-ledger-2026-09-03`

Status: **frozen before the first completed outcome evaluation**.

The frozen V3.8.3 target run completed actions, certificates, and ledgers. Its
first evaluation attempt stopped before writing pre-outcome routes and before
the outcome loop because the registered diagnostic `atomic_proof_only` policy
dereferenced a fail-closed ledger whose value was `None`.

This amendment changes only that diagnostic's missing-ledger representation:
when `require_ledger=False`, a ledger row whose `ledger` value is `None` is
copied with an empty ledger dictionary. This matches the existing behavior for
a wholly absent ledger row and gives its diagnostic ranking fields their
already coded zero defaults. When `require_ledger=True`—including the primary
ELAR policy—the original inputs and function are used without alteration.

No model is called. No response, certificate, ledger, router threshold,
policy, metric, bootstrap setting, or pass gate changes. The analysis manifest
content-addresses the frozen protocol and all three formal target-record files
before outcome evaluation.
