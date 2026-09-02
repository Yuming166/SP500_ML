# Recovery V3.2 execution status

Last audited: 2026-09-02 17:34 CST

Formal status: **PRETEST_ABORT_NO_NONTRIVIAL_SAFE_POLICY**.

All development calls completed successfully: model train 814/814 (6,512
records), policy selection 174/174 (1,392 records), and safety calibration
175/175 (1,400 records). The train support gates passed with 798 high-consensus
items, 184 high-consensus errors, 114 repairable errors, and 74 cases exposing
possible intervention harm.

The policy-selection split chose thresholds 0.30/0.20 and harm cap 0.20. It
routed 18 items for +7 net fixes, no observed harm, and nonnegative gains in
both native labels. On the separate safety-calibration split, every frozen
monotone offset retained at least one harm in the Refuted group. The
preregistered fail-safe therefore produced a KEEP-only final policy.

No AVeriTeC-dev prospective-test model call was made. Running the test would
have been uninformative because a KEEP-only policy cannot meet the frozen
minimum-repair or strict-dominance gates. V3.2 and its router artifacts remain
unchanged; any method revision must use a new protocol version.

## Integrity anchors

```text
selection_manifest.json f7a91ac0f53f4ffa5ed42cc4d3b51c672381ccefa7577275b92ab18f45940e21
train/records.jsonl     db25bcc007bad0829371efb89cd418da21878cbb7a631ac08eca56bf033aa3f5
policy_dev/records.jsonl c2944d47dcda106ffbf5a349bb6eeac83c4e19c1ecb24c75a6d4cd93c9f27d25
calibration/records.jsonl c62986cfa7618fe7b7d9f36d73338caf0f318a5a20e6d792857dfed6ec245f53
router/manifest.json    7846df0c1c37bbf5f324f81ef969f66c1b61e102a767e810854b433a60f62ec6
router/router.joblib    8f1a90de84cdb05773ce030b6736a567902645f1e2b63c5b329b87eb5747c7c8
```
