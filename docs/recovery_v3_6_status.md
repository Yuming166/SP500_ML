# Recovery V3.6 status

Status: **development pilot stopped; no fitting or formal test calls**.

The fixed first-40 pilot requested 80 proof-carrying atomic certificates. The
legacy client limited every completion to 160 tokens. Twenty-one certificates
completed and passed the exact-span/packet-ID parser; 59 were truncated in the
middle of an otherwise structured JSON object on both attempts. Thus V3.6
failed an execution-size gate rather than a semantic gate.

The partial records and caches are preserved. V3.6.1 changes only the maximum
completion length for atomic certificates from 160 to 512 tokens. Baseline and
recovery action calls remain at 160 tokens, and no formal example has been
called.
