# Recovery V3.13 analysis-only amendment

The V3.13 target actions, provisional routes, Qwen decisions, final routes,
override answers, same-budget policy assignments, and protocol manifest were
all frozen before evaluation. Their hashes are unchanged.

The first invocation of `recovery_v3_13 evaluate` stopped before writing a
summary. The primary metric had been computed internally, but no value was
printed. The failure occurred when the code passed the action-record JSONL
list to the inherited V3.11 matched-policy metric, whose documented interface
requires a mapping from `example_id` to record bundle:

`TypeError: list indices must be integers or slices, not str`

The analysis-only module `recovery_v3_13_analysis.py` verifies the original
frozen protocol-manifest hash, leaves the frozen V3.13 module and all routes
untouched, and inserts exactly one deterministic operation at that interface:

`base._record_groups(action_rows)`

It then calls the original frozen evaluator. No threshold, policy, route,
answer, outcome rule, bootstrap setting, or gate is changed. The rerun must be
reported as an analysis amendment rather than a pristine one-shot evaluation.
