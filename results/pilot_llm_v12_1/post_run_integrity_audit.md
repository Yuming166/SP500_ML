# Pilot-LLM V12.1 post-run integrity and validity audit

Date: 2026-09-02 (Asia/Shanghai)

## Protocol lineage

V12 froze all 358 eligible BoolQ validation provenance roots not used by V11.1. Its auxiliary rewrite gate produced 1,073/1,074 usable substitutes and therefore aborted before any validation-agent output, as required by the frozen rule.

V12.1 was preregistered before validation-agent output. It inherited the exact selection, 1,073 usable substitutes, model, prompts, agents, conditions, score, endpoint, bootstrap, and four-worker assignment. It made exactly one bounded second repair call for the sole frozen unusable evidence ID; the repair was valid without truncation. All 1,074 substitutes then passed the preformal gate.

## Parallel execution integrity

- Execution used four endpoint-concurrent client workers. Physical GPU allocation behind the endpoint was not observable and is not claimed.
- A non-BoolQ preformal probe measured 2.26x wall-clock speedup for four concurrent versus four serial short requests.
- Formal execution ran from approximately 00:13:30 to 00:30:00, about 16.5 minutes, for an aggregate average near 7.2 logical calls per second.
- Shard record counts were 1,800, 1,800, 1,780, and 1,780.
- The merged file contains exactly 7,160 records and 7,160 unique `(cqid, agent_index, condition)` keys.
- Every condition has 1,790 records; every agent has 1,432 records; every question has 20 records.
- All 7,160 records carry `pilot-llm-v12.1-2026-09-02` and `success=true`.
- First-pass validity was 1.0. No failed record, replacement question, duplicate tuple, second runner, or post-result retry was used.

## Frozen confirmatory result

The single primary endpoint was `AUROC(R_PI, consensus_wrong | original_agreement >= 0.8)`, with the unchanged pre-outcome score

`R_PI = 0.1 * D_inert + 0.3 * flip_inertia + 0.6 * frac_shared`.

- High-consensus subset: N=300 of 358 questions.
- Consensus-wrong outcomes: 66; correct outcomes: 234. Both count gates pass.
- AUROC: **0.7051605801605801**.
- Frozen 1,000-question-bootstrap 95% CI: **[0.6200274348422496, 0.7813093289689035]**.
- The lower endpoint is strictly above 0.5.
- Frozen verdict: **PASS**.

This is confirmatory support for the narrowly defined claim that the frozen paired-intervention risk score ranks high-consensus BoolQ errors above correct high consensus for this model, prompt, evidence, and endpoint regime.

## Independent recomputation and leakage checks

- A separate pairwise, tie-aware AUROC implementation reproduced the point estimate exactly.
- A separate implementation of the frozen bootstrap with seed 20260921 reproduced both interval endpoints exactly.
- All 358 questions yielded complete risk rows.
- Recomputed `R_PI` had maximum absolute formula error 0.0.
- Poisoning `any_wrong`, `correct`, `gold_binary`, `harmful_fc`, and `label` fields did not change any risk score.
- `shared_weighted` is absent from the formal summary and risk contract.

## Preregistered secondary results

- `D_inert` high-consensus AUROC: 0.7905, 95% CI [0.7235, 0.8502].
- `flip_inertia` high-consensus AUROC: 0.8067, 95% CI [0.7443, 0.8627].
- `frac_shared` high-consensus AUROC: 0.4978, 95% CI [0.4194, 0.5706].
- `R_PI` for harmful false consensus over all 358 questions: AUROC 0.7118, 95% CI [0.6309, 0.7956].
- Frozen 80%-coverage router: error fell from 0.2200 to 0.1333, an absolute reduction of 0.0867 with bootstrap interval [0.0458, 0.0975].
- Intervention flip rates were 0.4698 for removal, 0.3274 for reversal, and 0.3006 for substitution.
- Secondary pooled V11.1 plus V12 high-consensus analysis: N=480, wrong=96, AUROC 0.6689, 95% CI [0.6022, 0.7382].

## Important heterogeneity boundary

The preregistered descriptive BoolQ-label subgroups point in opposite directions:

- `yes`: N=210, wrong=55, AUROC 0.8337, 95% CI [0.7466, 0.9131].
- `no`: N=90, wrong=11, AUROC 0.2135, 95% CI [0.1125, 0.3387].

This does not retroactively change the frozen aggregate PASS, but it rules out a label-invariant or universally reliable interpretation. The aggregate result is strongly heterogeneous and is driven by the `yes` regime; the mechanism behind the reversed `no` ranking requires a new, independently registered study. The subgroup finding must be reported with the primary result rather than hidden.

No endpoint, sample, threshold, feature weight, pass rule, or record was changed after formal outcomes were available.

## Artifact hashes

- V12 preregistration: `45f317f72b47b91b42fd35196a0a50ee6b914a54e1a16e07f1414f0d7cbbabc5`
- V12.1 amendment: `bd17be49d8aed79084b771a416ffcf9fb3a853bb67e3ae66f38c8a6f39caf01b`
- V12 preformal abort: `5a5dc877f3a7d4a6b1b6a5e1938d4d0aeb08620645872191962d11e87142809e`
- V12.1 selection manifest: `ac1f15283ad719aaa6da969aabce34f59e1da6872a01890912c23fc1519d674b`
- V12.1 substitute manifest: `2572aaf384aa08c00ae0f697200ebbfa03d825c0746369eb0965ace944e33750`
- V12.1 run manifest: `dea715447a2f7e3f32db22fc8a2bc04fa17920611ff90293ccd06bd72717ef68`
- Formal records: `1c9a1e4d01b198a5e4e6f7527500c44b9058a772a93afb9844c3ef3b22d7e5a8`
- Formal summary: `6a2b90969cac1fc1a8af69bd448a198c627b3fb4ef213ae435bf16e838f4b9e8`
- Formal report: `394149d651204faa328b8dbee0d40b104f2515b71df40e9d88d930ad4ed58c09`
- Preregistered secondary analysis: `62a288bb792d29761c8ebd01b2f1b4eb9832da117a1b7f9d7a64dea2f10ce283`
