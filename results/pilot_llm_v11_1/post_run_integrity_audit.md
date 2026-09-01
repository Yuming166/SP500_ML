# Pilot-LLM V11.1 post-run integrity and validity audit

Date: 2026-09-01 (Asia/Shanghai)

## Frozen decision

- Protocol: `pilot-llm-v11.1-2026-09-01`.
- Confirmatory endpoint: `AUROC(R_PI, consensus_wrong | original_agreement >= 0.8)`.
- Frozen risk score: `R_PI = 0.1 * D_inert + 0.3 * flip_inertia + 0.6 * frac_shared`.
- Pass rule: at least 80 high-consensus questions, at least 10 positive and 10 negative outcomes, and the lower endpoint of the frozen 1,000-replicate question bootstrap 95% AUROC interval strictly above 0.5.
- V11.1 inherited the unchanged 200-question V11 selection. V11 produced no validation-agent outcomes before its auxiliary rewrite fail-fast abort; V11.1 repaired exactly the three preregistered unusable rewrites before any formal outcomes existed.

## Execution integrity

- The formal process exited normally after 4,000/4,000 successful records.
- There are 4,000 unique `(cqid, agent_index, condition)` keys.
- Each of the four conditions has 1,000 records; each of the five agents has 800 records.
- All 4,000 records carry protocol version `pilot-llm-v11.1-2026-09-01` and `success=true`.
- The 200-question selection remains balanced: 100 `yes`, 100 `no`.
- All 200 questions yielded complete risk rows; 180 met the frozen high-consensus threshold.
- First-pass validity was 1.0. Observed intervention flip rates were 0.405 for removal, 0.278 for reversal, and 0.286 for substitution.

## Independent metric and leakage audit

- High-consensus outcomes: N=180, wrong=30, correct=150. Both frozen count gates pass.
- A separate pairwise, tie-aware AUROC implementation reproduced the reported point estimate exactly: 0.6054444444444445.
- A separate implementation of the frozen question bootstrap with seed 20260902 reproduced the reported 95% interval exactly: [0.4945054945054945, 0.721028971028971].
- Recomputed `R_PI` had maximum absolute formula error 0.0 across all 200 questions.
- Poisoning the candidate rows with altered `any_wrong`, `correct`, `gold_binary`, `harmful_fc`, and `label` fields did not change any risk score.
- The invalid V10.4 `shared_weighted` field is absent from the V11.1 summary and risk contract.

## Confirmatory result

**FAIL.** The score ranks high-consensus errors above correct consensus in the favorable direction (AUROC 0.6054), but the frozen 95% interval includes 0.5. The lower endpoint is 0.4945, so the preregistered criterion is not met. This is suggestive directional evidence, not confirmatory support for the hypothesis.

Secondary results do not override that decision:

- `R_PI` for harmful false consensus over all 200 questions: AUROC 0.6200, 95% CI [0.4980, 0.7299].
- Frozen 80%-coverage router on high-consensus questions: error 0.1667 to 0.1319, a 0.0347 absolute reduction, bootstrap interval [0.0000, 0.0694].
- Individual high-consensus feature AUROCs were 0.5784 (`D_inert`), 0.5887 (`flip_inertia`), and 0.5524 (`frac_shared`); all intervals include 0.5.

No endpoint, threshold, sample, feature weight, or pass rule was changed after formal outcomes were available. Any follow-up must be registered as a new experiment and must preserve this V11.1 failure.

## Artifact hashes

- V11 preregistration: `43563651d6e8dc6df8a3c64aebaaf2069e3fa9a6f49a08b1ade96816a167a82a`
- V11.1 amendment: `18da58a52e37453311d1f4bfd12510ee7010e92639a888cb931a1854db04541f`
- V11.1 selection manifest: `78d3e660ccbfba777296a433283f1c0045cbec280181211266280b686a5771f6`
- V11.1 run manifest: `82833c93c49cf7ddc8f4245444552fc043b2b462b047a92797b9a18b62da4f75`
- Formal records: `21cc81283bbb688236534295de93c8438d4cdb5c2915ad0d8e857de8d4d8220f`
- Formal summary: `847023eb31c188730c78fb7fcf9f264f20386006bc4f606bea4417d606e11e87`
- Formal report: `a4500c3a7c8ac0f2c6e8f812dd480daf989d521ea66d7c309b0cff6907613222`
