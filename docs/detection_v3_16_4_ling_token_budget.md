# Detection V3.16.4 Ling completion-budget amendment

Date: 2026-09-03 (Asia/Shanghai)

Status: **frozen after the V3.16.3 256-token smoke and before any V3.16.4
call**.

The fresh V3.16.3 Ling smoke produced 156/160 final-valid and 154/160
first-pass-valid rows. All four terminal failures were incomplete JSON objects
at exactly the 256-token completion ceiling. The visible JSON contained an
answer and valid evidence ID but ended before the confidence field closed.

V3.16.4 changes only the Ling maximum completion budget from 256 to 512 tokens.
The 30 development pairs, 60 items, five agents, four conditions, prompts,
strict schema, citation contract, seeds, Qwen-selected risk weights, and pilot
gates are unchanged. It uses a fresh `calls_3` cache. The 500 formal candidate
items remain unauthorized.
