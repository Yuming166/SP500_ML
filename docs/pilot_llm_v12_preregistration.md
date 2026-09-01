# Pilot-LLM V12 preregistration

Protocol version: `pilot-llm-v12-2026-09-01`

Frozen before any V12 auxiliary rewrite or validation-agent output. V11.1 outcomes were already known when this prospective replication was designed; they may motivate sample size, but they may not alter V12's score, endpoint, threshold, or decision rule.

## 1. Question set

- Dataset: official BoolQ validation parquet, SHA-256 `52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8`.
- Apply the unchanged V11 eligibility gate to the raw dataset.
- The eligible universe contains 558 provenance roots: 346 `yes` and 212 `no`.
- Exclude exactly the 200 roots frozen in V11.1 selection manifest SHA-256 `78d3e660ccbfba777296a433283f1c0045cbec280181211266280b686a5771f6`.
- Use every remaining eligible root: N=358, comprising 246 `yes` and 112 `no`. No balancing, rewrite-success selection, model-output selection, or outcome filtering is allowed.
- The V12 and V11.1 provenance-root overlap must be zero.

## 2. Frozen model experiment

- Model, endpoint, five agent personas, evidence partition, prompt contract, and four conditions (`original`, `remove`, `reverse`, `substitute`) are unchanged from V11.1.
- Formal logical calls: `358 * 5 * 4 = 7,160`.
- Each question remains a paired block of 20 calls.

## 3. Frozen auxiliary rewrite rule

Each of the 1,074 evidence sentences receives one initial opposite-answer rewrite call with seed 20260922. The exact V10.4 single-line parser and length window are retained.

If and only if that initial candidate is unusable, it receives exactly one repair call with seed 20260923. A short repair receives repeated copies of the fixed neutral suffix `in the described local situation.` until the first point inside the original length window. Overlong repairs are never truncated. If any evidence remains unusable, V12 aborts before validation-agent calls; no evidence item or question may be dropped or replaced.

## 4. Four-worker execution contract

- Four client workers target the single frozen OpenAI-compatible endpoint.
- Physical GPU allocation behind that endpoint is not observable from the client and will not be claimed without server-side evidence.
- Questions are assigned as complete 20-call blocks. Within each BoolQ label, questions sorted by `cqid` are distributed round-robin over shards 0 through 3.
- Each shard has its own content-addressed cache, partial record file, progress file, and final shard record file.
- Resume is allowed only from those frozen caches and partial records. Duplicate logical tuples are forbidden.
- Results are merged only after every shard completes. The merged file must contain exactly 7,160 unique `(cqid, agent_index, condition)` tuples, 1,790 records per condition, 1,432 per agent, and 20 per question.
- No metric or outcome aggregation is permitted during execution. Parallel completion order has no statistical role; merged records use canonical key order.

A pre-registration concurrency probe used eight synthetic, non-BoolQ prompts: four serial and four concurrent. All returned HTTP 200 with valid schemas; four-way concurrent wall time was 2.26 times faster. This probe did not touch any V12 question or outcome.

## 5. Primary endpoint and decision rule

The only primary endpoint remains

`AUROC(R_PI, consensus_wrong | original_agreement >= 0.8)`

where the pre-outcome score is unchanged:

`R_PI = 0.1 * D_inert + 0.3 * flip_inertia + 0.6 * frac_shared`.

The score may read only these three inputs. It may not read `any_wrong`, `correct`, `gold_binary`, `harmful_fc`, `label`, or any equivalent outcome field.

Use a 1,000-replicate question bootstrap with seed 20260921. V12 passes only if:

1. at least 80 questions meet original agreement >= 0.8;
2. that subset contains at least 10 wrong and 10 correct consensus outcomes; and
3. the lower endpoint of the bootstrap 95% AUROC interval is strictly greater than 0.5.

No sample-size extension, early stopping for success, weight tuning, threshold tuning, endpoint promotion, or post-result exclusion is allowed.

## 6. Secondary analyses

- Individual `D_inert`, `flip_inertia`, and `frac_shared` AUROCs on the high-consensus subset.
- `R_PI` AUROC for harmful false consensus over all V12 questions.
- Frozen rank router at 80% coverage.
- Descriptive `yes`/`no` subgroup results, explicitly non-confirmatory.
- A V11.1 plus V12 cumulative or meta-analytic estimate, explicitly secondary and incapable of rescuing a failed V12 primary endpoint.

## 7. Preformal gates

- Exact frozen selection and zero V11.1 root overlap.
- All 1,074 substitute sentences usable under the frozen rule.
- Offline risk-contract and shard-balance audit passes.
- A four-question smoke run exercises all four workers and produces exactly 80 unique, successful logical records.
- Full test suite and `git diff --check` pass.

Any gate failure is recorded and the formal run does not start.
