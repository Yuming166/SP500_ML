# Recovery V2.2 execution status

Last audited: 2026-09-02 16:46 CST

Recovery V2.2 is complete. The formal preregistered verdict is
`NO_VERIFIED_NET_RESCUE`: three of five primary gates passed.

## Execution

- Frozen FEVER universe: 1,149 exactly-two-page-root claims.
- Connected-component train/dev/test split: 689/230/230.
- Cross-split page-root overlap, including retrieval candidates: zero.
- Train: 5,512/5,512 calls successful; 1,192,689 tokens.
- Dev: 1,840/1,840 calls successful; 408,492 tokens.
- Test: 1,840/1,840 calls successful; 416,472 tokens.
- Router manifest was written at 16:42:00 CST; untouched test records were
  completed later at 16:45:46 CST.

## Formal result

The learned conservative router changed only 13 of 230 decisions and acquired
21 additional roots. Accuracy increased from 76.52% to 78.70%: six errors were
repaired and one correct answer was harmed, for +5 net fixes (+2.17 percentage
points; 95% paired bootstrap CI +0.43 to +4.78 points). Six repairs cited the
held-out annotated root. Its damage rate among initially correct
high-consensus examples was 0.57%.

It nevertheless failed the composite success criterion because fixed `both`
achieved +22 net fixes, and the learned policy's REFUTES subgroup net gain was
-0.93 points. The other three gates passed: positive CI lower bound, damage at
most 5%, and at least five annotation-supported repairs.

## Runtime relocation

Before dev resume and before any test call, the operator moved the same named
`Qwen3.5-4B` deployment from `10.63.0.88:31519` to `10.63.0.82:31518`. The new
endpoint exposed only `Qwen3.5-4B`, reported model root
`/storage/lianjh/modelzoos/Qwen/Qwen3.5-4B`, and passed a deterministic chat
smoke. The relocation is retained as an operational amendment.

Train records were produced at the original endpoint. Of the dev calls, 904
retained records came from the original endpoint and 936 came from the
relocated endpoint. All 1,840 test calls came from the relocated endpoint. No
existing outcome was rewritten and no router or threshold was refit after test.

## Integrity anchors

```text
selection_manifest.json 56168f1088c0aa0a8081c0d4cf45d79fca09017e2741d2c971d805cc31c1566c
train/records.jsonl     eaeb72e15863d99c32df50c32265be79f151131892d043e1e7d2c78921272f8a
dev/records.jsonl       01a6aa47b0af3a77009e62559a9e2ae581e9a0c1af841aa2c83ac6caebbcb0b8
router/manifest.json    bdb54c15591de2304b665fb90bac833a4246cedc464ada6e364e0294ddf0bd9d
router/router.joblib    591bcec76c1ea81367e174fac1219a6bc492b555cc857cec207471c5d9710e1a
test/records.jsonl      eea7bd6e7dd17ffc7d323776914a5ab49ed29b8469c69fbe654834833d26f88d
analysis/summary.json   4c4c88ec36114cd3017901dd61f152e001592e376dc73f9f4f2273f7c1c57da6
```
