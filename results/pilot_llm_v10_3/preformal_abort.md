# V10.3 pre-formal instrumentation abort

**Recorded:** 2026-09-01
**Protocol:** `pilot-llm-v10.3-2026-09-01`

V10.3 inherited the same frozen 100-question BoolQ selection and ran its
exact-source-token substitute generator. It produced 295 usable initial
rewrites. Five fixed one-time repair calls were attempted; none entered the
unchanged 0.5--1.5x token window. The workflow therefore stopped before audit,
smoke, and formal evaluation; it contains no evaluation-agent or router result.

The observed failure class was five one-line, short-by-one-or-two-token
responses. That aggregate auxiliary instrumentation observation motivates the
separately preregistered V10.4 deterministic length normalization. No V10.3
rewrite text, cache response, or substitute manifest is used by V10.4.
