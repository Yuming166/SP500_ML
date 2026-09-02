# Recovery V3.8.2 development-smoke abort

Status: `ABORTED_DEVELOPMENT_SCHEMA_SMOKE_BEFORE_FORMAL`

The frozen V3.8.2 manifest has SHA-256
`095a15163e9867f9d20978e518f2cb25f4fc7a1bd6fc7842b6cb6c5d37f287ae`.
No V3.8.2 formal target call was made. On the same two fixed development-only
smoke examples, 15/16 action rows were terminally valid. The remaining row
returned `confidence: 95` on both attempts, representing a percentage rather
than the required `[0, 1]` fraction. The terminal error was
`ValueError: confidence must be finite and in [0, 1]`.

No correctness, gain, label-stratified result, formal route, certificate,
ledger, or action outcome was computed. The smoke records and cache are
preserved and forbidden as V3.8.3 inference inputs.

V3.8.3 is a separately frozen, closed transport-conformance protocol. It adds
one standard semantic-preserving rule: a finite numeric confidence strictly
above 1 and at most 100 is divided by 100. All nonnumeric, Boolean, nonfinite,
and out-of-range values remain invalid. All formal scientific choices remain
those frozen in V3.8.
