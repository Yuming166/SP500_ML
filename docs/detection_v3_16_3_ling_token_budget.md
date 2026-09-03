# Detection V3.16.3 Ling completion-budget amendment

Date: 2026-09-03 (Asia/Shanghai)

Status: **frozen after the V3.16.2 Ling transport abort and before any V3.16.3
call**.

V3.16.2 completed all 1,200 registered development calls but failed its
transport gate: 1,107/1,200 were final-valid and 1,045/1,200 were first-pass
valid. No pre-outcome risk rows were frozen and no Ling development outcome was
joined.

The outcome-blind transport audit found 93 terminal failures:

- 91 were incomplete JSON objects; 89 ended at exactly the 128-token ceiling;
- two returned a complete decision but violated the evidence-ID contract; and
- failures occurred across original, reverse, and substitute conditions.

V3.16.3 changes only the Ling maximum completion budget from 128 to 256 tokens.
It retains the same 30 pairs, 60 items, five agents, four conditions, prompts,
strict schema, citations, seeds, Qwen-selected risk weights, and pilot gates.
It uses a fresh `calls_2` cache and reruns all 1,200 calls. Qwen development is
not rerun and the 500 formal candidate items remain unauthorized.
