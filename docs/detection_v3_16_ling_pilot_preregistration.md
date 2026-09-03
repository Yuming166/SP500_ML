# Detection V3.16.1 Ling transfer-pilot preregistration

Date: 2026-09-03 (Asia/Shanghai)

Protocol: `detection-v3.16.1-ling-transfer-pilot-2026-09-03`

Status: **frozen after Qwen development and before any Ling V3.16 call**.

## Frozen source result

- Risk manifest SHA-256:
  `f85673c5dfa9d68d0bdc4e37f37f7f21bffefd3d2d80814adb5397e2c5367a24`
- Qwen development summary SHA-256:
  `70aaa2e7c771520b67a6261ff41094ba7180d9d17a1e791ace9cd39de17d3914`
- Selected score:
  `0.3 * reverse_confident_nonresponse + 0.7 * intervention_disagreement`.
- Remove inertia, substitute inertia, and unqualified reverse inertia have zero
  weight.

The Qwen nested pair-OOF AUROC was 0.905; SUPPORTS/REFUTES AUROCs were
0.928/0.885, and Risk@80 error was 0.083 versus 0.167 unfiltered. These are
development numbers, not paper results.

## Frozen Ling pilot

- Target: `Ling-3.0-tiny` at the separately verified local endpoint.
- Inputs: exactly the same 30 development pairs / 60 items used by Qwen.
- Calls: 60 items x 5 agents x 4 conditions = 1,200, after a separate
  160-call transport smoke on the four smoke pairs.
- Prompts, response schema, conditions, agents, seeds, parser, and selection
  manifest are inherited unchanged from V3.16.1.
- Ling outcomes may not alter weights, gates, examples, or parser.

The pilot qualifies only if all registered gates in the frozen risk manifest
pass: final validity at least 0.98, first-pass validity at least 0.95, at least
20 high-consensus items and at least four errors per label, overall and
macro-label AUROC above 0.55, worst-label AUROC above 0.50, and nonnegative
Risk@80 error reduction.

A pass authorizes writing a separate formal preregistration; it does not itself
authorize any of the 500 formal candidate items. A failure stops this score and
does not authorize Ling-specific tuning.
