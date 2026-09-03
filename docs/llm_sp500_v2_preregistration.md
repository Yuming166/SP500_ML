# LLM-S&P500 V2 preregistration — prompt-hardening re-run (paired with V1)

- **Protocol version**: `llm-sp500-v2-2026-09-03`
- **Status**: frozen before the V2 formal run (smoke accepted at 100%).
- **Inherits**: `docs/llm_sp500_v1_preregistration.md` (all sections) and
  the frozen V1 manifest (`results/llm_sp500_v1/manifest.json`, sha256
  `6fd3c3edbfa8253246252f658fbc220367d4f596568096bd53ac885c788d477f`).
- **Prompt module**: `src/sp500_forecastability/llm_sp500_v2_prompts.py`
  (frozen at commit-time of this document).
- **Driver**: `scripts/run_llm_sp500_v2.sh`.

---

## 1. Motivation

V1 completed its formal run (2,500 calls) with a first-pass valid-response
yield of **68.2% (1,705/2,500)**. Failure-mode analysis of the 795 failed
calls (registered as the basis for this prereg) showed:

| Failure shape | n | % of failures | Roles |
| --- | ---: | ---: | --- |
| Empty `claims` array on abstention | 585 | 73.6% | consistency 458, skeptic 115, literal 12 |
| evidence_id not in packet catalog | 195 | 24.5% | cf 104, literal 54, min 19, skeptic 18 |
| JSON truncation / malformed | 15 | 1.9% | cf 6, others |

The empty-claims failure is a **prompt/contract conflict**, not a model
capability failure: the V1 `consistency` and `skeptic` system prompts
instruct the model to abstain (`action='cash'`, low confidence) when
evidence disagrees, and the model emits `"claims": []` on abstention —
but `parse_agent_decision` requires ≥ 1 claim with nonempty
`evidence_ids`. V1 therefore **fail-closed precisely when its agents
behaved as designed** (abstained under conflict), biasing every router's
effective coverage.

V2 fixes exactly these two failure modes at the prompt layer. **No other
protocol element changes.**

## 2. Frozen changes (exhaustive)

### F1 (abstention-must-cite). Every role's system prompt is appended
with a mandatory rule: an abstain/`cash` decision still carries ≥ 1 claim
citing the conflicting or distrusted evidence with `stance='attacks'`;
empty `claims` arrays are invalid. The `skeptic` and `consistency` prompts
additionally name the evidence to cite (the contradicting roots). The
`parse_agent_decision` contract, the D9_v1 per-claim filter, and all
parsers are **unchanged** — abstention becomes representable without any
contract relaxation.

### F2 (verbatim-evidence-first). The user template presents the packet's
evidence ids as a numbered `EVIDENCE CATALOG` and requires character-for-
character copies; the retry reminder names the two dominant failure modes
explicitly. The `min` role prompt (updated after a 4/5 smoke revealed it
citing `min_base_rate` as an evidence id) requires the FIRST catalog id,
verbatim, and forbids citing `min_base_rate` itself.

Nothing else is modified: roles, horizon (5 trading days), temperature 0.0,
max_tokens 800, timeout, packet builder, manifest, split, seeds, retries
(1), workers (8), routers, AMIR fit, AURC CI, report structure.

## 3. Inherited frozen design (unchanged from V1)

| Element | Value |
| --- | --- |
| Manifest | V1's: salt `llm-sp500-v1-2026-09-03`, 500 dates (350 train / 150 test), sha `6fd3c3ed…` |
| Window | 2021-05-10 → 2026-04-27 (1,247 eligible rows; D6_v1) |
| Packet | 29 features / 7 roots, as-of `available_at == publication_time == decision_time`, z-scores over 252-day lookback |
| Roles | literal, skeptic, consistency, cf, min (V5 §7 inheritance) |
| Decoding | temperature 0.0, max_tokens 800 (D8_v1), 1 retry with schema reminder |
| Concurrency | 8 workers (D11_v1) |
| Routers | majority, mean_confidence_long, v5_provenance_baseline, single_min_agent, amir_router_v5 (D12_v1 LogisticRegression on 10 role features) |
| Primary endpoint | **H1_v2 ≡ H1_v1**: AURC(amir_router_v5) − AURC(majority) < 0, PASS iff 95% paired moving-block bootstrap CI upper bound < 0 (block length 21) |

## 4. Backend (D1_v2)

The backend is injected via CLI (`--endpoint`, `--model`) so the frozen
code runs unchanged against:

- **default**: local vLLM `http://localhost:31519/v1/chat/completions`,
  model `Hy-MT2-7B` — the same backend as V1's formal run, making V2 a
  **single-variable (prompt-only) paired re-run**;
- **relay**: `https://openapi.center/v1/chat/completions`, model
  `gpt-5.4-mini` (V1's original gpt-4o target is not offered by the
  relay; the GPT-5.x family is). Selected with the driver's `--relay`.

Primary analysis is the default-backend run. A relay run, if executed,
is reported as a separate section (backend-generalization check), never
substituted into the primary table.

At planning time (2026-09-03) the relay's inference upstream returned
502 `upstream_error` for every model; the relay run executes only when
the relay is restored. The key is supplied via `OPENAI_API_KEY` and is
never logged or written to disk (D4_v1 carries over; **the operator must
rotate the relay key** because it was exposed in the planning transcript).

## 5. Hypotheses

- **H1_v2** (primary, frozen = H1_v1): AURC(amir) − AURC(majority) < 0
  with 95% CI upper bound < 0 on the 150-date test window.
- **H2_v2** (registered secondary, one-sided McNemar on paired records):
  V2 first-pass valid-response yield > V1 first-pass valid-response yield
  on the same 2,500 calls. Success criterion: p < 0.01. This tests the
  prompt fix itself, not market predictability.
- **H3_v2** (exploratory, no criterion): per-router metrics under V2
  prompts vs. V1 report values.

## 6. Failure-mode budget (pre-registered expectation)

Given the V1 failure decomposition, V2 projects first-pass yield ≥ 85%
(removing ~585 empty-claims failures and ≥ half of the 195
evidence-id failures; JSON truncation unaffected). A V2 yield below 75%
falsifies the mechanism claim and triggers investigation before any
interpretation.

## 7. Deviations registered

| # | Deviation | Rationale |
| --- | --- | --- |
| D1_v2 | Backend (endpoint + model) becomes a CLI parameter rather than a frozen constant; default remains V1's Hy-MT2-7B local vLLM | enables the relay run without code edits after freeze |
| D2_v2 | `min` role prompt hardened to cite the first catalog evidence id (changed after the first smoke showed `min_base_rate` cited as evidence) | same failure class F2; fixed pre-freeze, documented here |
| D3_v2 | V1-vs-V2 per-role yield comparison added to the report as an informational table (H2_v2 is computed offline, not in `report.md`) | paired-design bookkeeping |
| D4_v2 | V1's `--relay` models are the GPT-5.x family, not gpt-4o | the relay does not offer gpt-4o |

All V1 deviations (D1_v1 … D12_v1) carry over unchanged.

## 8. Interpretation boundary

V2 tests whether prompt-level hardening recovers agent yield and, with it,
whether the AMIR-style router's AURC advantage over the majority baseline
becomes resolvable on the same 150 test dates. A PASS under H1_v2 closes
the V5 signpost under the improved protocol; a FAIL replicates V1's
negative result under higher yield, which is itself informative (the V1
CI width was not a yield artifact). Neither outcome establishes S&P 500
predictability, investment performance, or cross-model generalization.
