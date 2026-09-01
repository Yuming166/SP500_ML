# V10.4 smoke-stage protocol deviation

**Recorded before resuming the formal run:** 2026-09-01

The preregistration and driver comments described a 40-logical-call smoke
stage (2 questions × 5 agents × 4 conditions). The inherited execution helper
actually constrains smoke mode to its first agent, so V10.4 executed and
validated 8 calls (2 questions × 1 agent × 4 conditions), all successful.

This is an implementation/documentation mismatch inherited from the V10.1
runner, not a response-dependent choice. The formal mode remains separately
hard-gated at 100 questions × 5 agents × 4 conditions = 2,000 logical tuples;
no selected item, prompt, substitute, formal metric, threshold, or reporting
rule was changed in response to the smaller smoke. The final report must state
that smoke coverage was 8 rather than the preregistered 40.
