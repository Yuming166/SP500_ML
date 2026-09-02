# Recovery V3.3 execution status

Last audited: 2026-09-02 17:45 CST

Recovery V3.3 is complete. Formal verdict: `NO_VERIFIED_CEW_DOMINANCE`; four of
five frozen primary gates passed.

## Execution

- Development action matrices reused from frozen V3.2: train 6,512 records,
  policy-dev 1,392, calibration 1,400; all successful.
- CEW manifest frozen at 17:39:41 CST.
- Prospective test completed later at 17:44:15 CST: 1,888/1,888 records
  successful, with 589,912 tokens.
- All calls used `Qwen3.5-4B` at
  `http://10.63.0.82:31518/v1/chat/completions`.
- The test route snapshot was written before outcome metrics.

## Formal result

CEW routed 60/236 items and raised accuracy from 80.93% to 84.75%: 14 fixes,
five harms, and +9 net fixes. It beat the best same-budget baseline (+6), had
2.66% high-consensus damage, improved both labels, and produced 11
annotation-supported repairs. Its macro-gain CI was [-0.69, +11.03] points, so
the strict positive-lower-bound gate failed.

## Integrity anchors

```text
router/manifest.json             83424c1fec6ad87d498e38d1def4b43dad8adc8c96dc4d55ec2169c81ac74153
router/stance_router.joblib      a90adce1d98d6f37a4bd80b938694f9eedb7422f36f119e4fd867b841b4cb40b
test/records.jsonl               951ee7b168d08a37a8c66623e645c775a9fed2c00804c83fecee6d38232153fc
evaluation/preoutcome_routes.json f2b7f662984587bbaf8b38d0e55fe8b5ba40438ef70628b59364070f935121ee
evaluation/summary.json          9095bd5d44b5e5a0446876fc790ece68b0d9daa3ba4ec63a6ead43d1399726cd
```
