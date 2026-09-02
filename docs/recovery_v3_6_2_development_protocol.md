# Recovery V3.6.2 execution note

V3.6.2 is a pre-outcome operational continuation of Atomic-PACE. V3.6.1 is
retained as an abort because one 512-token certificate bundle remained invalid
after its fixed repair. No V3.6.1 test action or outcome was accessed.

The statistical model is not redesigned or retrained. V3.6.2 reuses the exact
V3.6.1 development records, serialized ensemble, and thresholds. Its two changes
are fixed before new formal calls:

1. a certificate that remains invalid after two attempts fails closed to KEEP;
2. formal evaluation moves to 400 uncalled examples sampled one per untouched
   evidence-root connected component, balanced 200/200 by native label.

Selection audit gates require zero overlap with V3.6.1 claims and page roots,
zero development/test overlap, zero CLIMATE-FEVER page-root overlap, unique gold
roots across test items, balanced candidate position, and exclusion of retrieval
and title shortcuts from router features.

The fail-closed state is represented by all-zero support/refute strength,
`insufficient` relation, incomplete coverage, and no new evidence IDs. It can be
passed to the frozen model for a complete feature matrix, but the proof gate
always makes the corresponding action ineligible. Invalid response records and
attempt metadata remain in the formal artifact.
