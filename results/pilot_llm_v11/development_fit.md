# V11 frozen development fit

**Development source:** V10.4 BoolQ train formal records
**Frozen before V11 validation model calls:** 2026-09-01

V10.4 supplied 100 complete development questions, including 74 with original
agreement at least 0.8 and 12 wrong high-consensus outcomes. The invalid
outcome-dependent `shared_weighted` field was excluded.

All 66 convex combinations on the 0.1-spaced simplex over `D_inert`,
`flip_inertia`, and `frac_shared` were evaluated on the high-consensus
development subset. The highest development AUROC was 0.7338709677 at weights
0.1, 0.3, and 0.6 respectively. The exact frozen V11 score is:

```text
R_PI = 0.1 * D_inert + 0.3 * flip_inertia + 0.6 * frac_shared
```

For context only, development high-consensus AUROCs were 0.715726 for
`D_inert`, 0.718414 for `flip_inertia`, and 0.613575 for `frac_shared`. V11 uses
only the held-out validation result for its confirmatory verdict.
