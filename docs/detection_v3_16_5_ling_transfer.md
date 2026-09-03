# Detection V3.16.5 Ling transfer binding

Date: 2026-09-03 (Asia/Shanghai)

Status: **frozen after calls_4 Qwen development and before calls_4 Ling
development calls**.

The common prompt-only interface passed both model smokes. Qwen development
then selected the following intervention-only score:

`0.3 * reverse_inertia + 0.7 * intervention_disagreement`.

- calls_4 risk manifest SHA-256:
  `4c7f6f833849ac1fc25f88980b8c4d483ba2a30f540a09b29aea210e6f6aa215`
- calls_4 Qwen development summary SHA-256:
  `95943da3ab4ec231f253f0df1a79f2b8eabb309450d145660634167922922c3c`
- Qwen nested pair-OOF overall/macro/worst-label AUROC:
  `0.887 / 0.898 / 0.806`.
- Qwen nested Risk@80 error: `0.167 -> 0.104`.

Ling receives the same 30 pairs, 60 items, five agents, four conditions,
prompt-only schema, strict parser, 256-token budget, and stable-seed rule.
Neither Ling outputs nor outcomes may change the score or go/no-go gates. Any
incomplete intervention bundle fails closed at risk 1.0 under the registered
V3.16.2 convention.

This binding authorizes only the 1,200-call Ling development transfer pilot.
The 500 formal candidate items remain unauthorized even if every pilot gate
passes; a separate formal preregistration is required.
