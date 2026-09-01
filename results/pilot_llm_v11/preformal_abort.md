# V11 pre-formal instrumentation abort

**Recorded:** 2026-09-01
**Protocol:** `pilot-llm-v11-2026-09-01`

V11 froze 200 official BoolQ validation questions and 600 evidence units before
model calls. Its auxiliary substitute stage produced 590 directly valid
rewrites, 7 valid deterministic short normalizations, and 3 unusable length
cases. Per protocol, V11 stopped before audit, smoke, or formal evaluation.

The three failures were all one-line responses: one contained 20 tokens for a
6--18 window; two contained 16 and 19 tokens for lower bounds 26 and 25, and
remained short after the single fixed suffix. No validation agent answer,
confidence, consensus, risk score, metric, or outcome was generated or
inspected. V11.1 inherits the exact selection and the 597 valid auxiliary
rewrites and applies its frozen length-repair rule only to these three cases.
