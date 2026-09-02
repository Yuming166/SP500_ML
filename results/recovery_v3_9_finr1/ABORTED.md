# Recovery V3.9 development-smoke abort

Status: `ABORTED_DEVELOPMENT_SCHEMA_SMOKE_BEFORE_FORMAL`

The frozen V3.9 manifest has SHA-256
`691d0c1b550bdbf0b64d9c782993ffd9352c9dc549f5724f42c4336daa6f6ca9`.
No V3.9 formal target call was made. The two fixed development examples
produced 16 cached baseline attempts before the bundle runner stopped because
it could not assemble five valid decisions. Four responses were strict JSON;
12 began with a complete, schema-valid JSON decision and appended explanatory
prose. The frozen strict parser rejected the latter as not exactly one JSON
object.

No correctness, gain, route, certificate, ledger, action outcome, or formal
record was computed. The smoke cache is preserved and forbidden as V3.9.1
inference input.

V3.9.1 is a separately frozen transport-only amendment. It accepts one
complete JSON object at the very start of a response followed only by non-JSON
prose containing no braces. It records this parse mode and then applies the
unchanged V3.8.3 conformance and strict schema validator. Multiple objects,
leading prose, braces in the suffix, and incomplete JSON remain invalid.
