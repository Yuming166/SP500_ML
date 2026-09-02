# LLM-S&P500 V1 preregistration: ChatGPT agents on a new financial window with the trained AMIR router layered on top

**Date frozen:** 2026-09-03
**Status:** Frozen before any V1 model call. Any substantive change after V1
outputs requires a new version (V2). Cache hits from earlier pilots do not
count as V1 calls — V1 only consumes OpenAI responses generated after
2026-09-03.

## 1. Why a first LLM-on-S&P500 pilot is needed

`docs/historical_router_v5_preregistration.md` produced an **AMIR-Router**
on the S&P 500 task, but the V5 report (`results/historical_router_v5/report.md`)
self-described as **"POST-V4 RETROSPECTIVE EXPLORATORY ONLY"** and
explicitly stated: *"frozen cross-model LLM transfer and a new financial
window remain necessary for an ACL generalization claim."*

V1 picks up exactly that signpost:

- **New financial window** that the trained router has not inspected
  during its V5 market period (V5 6 outer folds covered an earlier block;
  V1 fixes the window to 2021-01 → 2026-05 so the first and last decision
  date are out-of-sample for V5).
- **Real-time ChatGPT agent calls** instead of cached questions replayed
  from the V5 cache. The agent backbone is `gpt-4o`, the only model
  authorized for V1.
- **AMIR-Router evaluated on the LLM outputs** with the same protocol
  used in `historical_router_v5.py` §12 (`a_minus_b_aurc`, paired
  moving-block CI). The with-router vs. without-router comparison is
  V1's primary confirmatory endpoint.

V1 does **not** retrain AMIR. It consumes the frozen V5 router artifacts
under `results/historical_router_v5/`. Re-fitting AMIR on the V1 outcomes
is registered as the V2 open question (§16).

## 2. Status of prior versions (preserved, not edited)

| Version | Outcome | What V1 inherits |
|---|---|---|
| Pilot-LLM V1–V8 (`docs/pilot_llm_v{1..8}_preregistration.md`) | Synthetic (StrategyQA, TruthfulQA, FEVER, ClimateFEVER) | 5-agent role taxonomy (§6), response contract (§7), content-addressed cache (§3) |
| Historical Router V4 (`results/historical_router_v4/`) | CPR router | `SourceRanker`, target-bound audit pattern |
| Historical Router V5 (`results/historical_router_v5/`) | AMIR, **AURC(AMIR) − AURC(confidence) = -0.0078, CI [-0.0607, 0.0459], not PASS** | `AdaptiveRanker`, `CalibrationHead`, 7-source-root / 11-agent decomposition, paired moving-block CI |
| `agent_contracts.py` | Frozen provenance contract | `ProvenanceGraph`, `EvidenceItem`, `Claim`, `AgentDecision`, `parse_agent_decision` |
| `historical_data.py` | Frozen as-of feature table | `build_historical_replay_data()`, 7 source-root / 11-agent feature columns |
| `recovery_v3_8.py` | Existing OpenAI client | `ChatClient` (urllib-based, content-addressed cache, model-name validation) |

V5 outputs are frozen. V1 produces a new manifest under a new salt and
never modifies V5 artifacts. V5 router weights (`results/historical_router_v5/`
JSON fold summaries) are **read-only inputs** to V1.

## 3. Frozen model, retry, and transfer controls

- **Agent model:** `gpt-4o`, served via OpenAI Chat Completions.
- **Endpoint:** `https://api.openai.com/v1/chat/completions`.
- **API key:** read from `os.environ["OPENAI_API_KEY"]` (never written
  to disk, never logged, never embedded in prompts). After this session
  the key must be **rotated** by the operator because it was exposed in
  the planning transcript.

> **D7_v1 (added 2026-09-03):** the jump host on which V1 was launched
> blocks outbound HTTPS to `api.openai.com` (verified by direct `curl`
> at audit time). The V1 OpenAI client is therefore reconfigured at
> runtime to talk to the local OpenAI-compatible vLLM endpoint at
> `http://localhost:31520/v1/chat/completions`, model `Fin-R1`
> (SUFE-AIFLM-Lab, Qwen2-7B with financial-reasoning SFT, served by
> `/storage/lianjh/modelzoos/SUFE-AIFLM-Lab/Fin-R1`). No API key is
> required for the local endpoint; `OPENAI_API_KEY` (if present) is
> sent as `Authorization: Bearer ...` and is ignored by vLLM. The
> `gpt-4o`/`api.openai.com` values remain the preregistered default;
> all other V1 design choices (5 roles, prompts, packet, AMIR fit,
> primary endpoint) are unchanged.
- **Temperature:** `0.0`.
- **Maximum completion tokens:** `200`.
- **Timeout:** 60 seconds per HTTP request.
- **Retries:** one initial request and at most one fixed retry on (a)
  transport failure, (b) JSON schema-violating response, or (c)
  response-model mismatch (`response.model != "gpt-4o"`). No retry on
  logical-validation failure inside a well-formed JSON response.
- **Cache:** SHA-256 content-addressed (`results/llm_sp500_v1/cache/`)
  keyed on `endpoint + request_payload` to match `recovery_v3_8.py`'s
  scheme. Cache hits do **not** count toward V1's first-pass yield
  metric (§11) — they are reported separately as `cache_hits`.
- **No model weights download, no hidden chain-of-thought request,
  storage, or scoring.** `reasoning_effort` is not passed.

## 4. Domain: S&P 500, 2021-05-10 → 2026-04-27, N=500 decision days

V1 fixes the window to the last five years of the as-of historical
table as built by `historical_data.build_historical_replay_data()`.
Rationale:

- 2021-05-10 is the **first row** returned by `build_historical_replay_data()`
  after its built-in `dropna()` (early 2021 dates have NaN sentiment
  holes; the §4.3 NaN gate drops them). V1 inherits this as-of
  alignment rather than re-deriving it — using any earlier first date
  would force V1 to duplicate `historical_data`'s forward-fill + delay
  logic and risk silent disagreement with V5.
- 2026-04-27 is the **last row** with a fully observable
  `target_up_5d` label (the function drops the trailing 5 trading
  days because `forward_return_5d` requires a future window).
- The 2021-05 → 2026-04 block contains the COVID recovery, the 2022
  inflation / rate-hike bear, the 2023–2024 rally, and the 2025–2026
  regime uncertainty — i.e., at least three regime shifts, which is
  the minimum stress surface the V1 §1 signpost calls for.

Total as-of-eligible rows: **1,247** (verified by running
`build_historical_replay_data('data')` at prereg freeze time on
2026-09-03). The §4.3 manifest draws 500 dates from this 1,247-row
pool.

**V1 does not overlap with the V5 outer-fold test period.** V5's
folds (per `results/historical_router_v5/summary.json`) trained and
tested on an earlier block; V1's split is internal to its own window
and the 150 test dates are guaranteed held out from V1's 350 train
dates by the sorted walk-forward split. V1 does **not** load any V5
fold model — see `D5_v1` (§14) for the rationale.

### 4.1 Label space

V1 uses a **single binary label**: `target_up_5d = (market.ret_5d > 0)`
evaluated at the decision date `t`. The horizon is 5 trading days.
No regression label, no abstention label, no triple-barrier.

`ret_5d` is the existing column in `data/market.csv` and is already
shifted so that `ret_5d` at date `t` is the close-to-close return from
`t+1` through `t+5`. As-of alignment is therefore **inherited from
`historical_data.build_historical_replay_data()`** and is not
re-derived in V1.

### 4.2 As-of feature surface

V1 inherits the seven source roots and eleven agents verbatim from
`historical_replay_v2.py:AGENTS`:

| Source root | Agents (columns from `historical_data.build_historical_replay_data()`) |
|---|---|
| `market_bloomberg` | `market_raw`, `market_band`, `market_trend` |
| `vix_bloomberg` | `vix` |
| `macro_bloomberg` | `macro` |
| `google_trends` | `sentiment` |
| `cboe_options` | `options` |
| `etf_flow_family` | `spy_flow`, `ivv_flow`, `voo_shares` |
| `ici_mutual_fund_flow` | `mutual_fund_flow` |

For V1, the "agent" identity at the LLM layer is **the source root**,
not the eleven feature columns. Each source root contributes one
evidence packet (see §5) and is queried by all five LLM agents (§6).
This matches V5's `root_count=7` / `agent_count=11` decomposition at
the router layer (V5's AMIR mixes them internally), but keeps the LLM
prompt surface tractable.

### 4.3 Manifest construction (500 decision days, salt-driven)

V1's manifest is a **deterministic, salt-driven uniform sample** over
the as-of-eligible dates in the window:

1. Filter `historical_replay_data.frame` to dates `≥ 2021-01-04` and
   `≤ 2026-05-14`.
2. Drop dates where any of the seven root-level feature columns is
   `NaN` for that row (the §4.2 `google_trends` block has known NaN
   holes before ~2021-05; the audit drops any later NaN holes too).
3. Drop dates where `ret_5d` is `NaN` (label must be observable).
4. Take a uniformly-spaced sub-sample of **500 dates** using
   `numpy.random.default_rng(salt).choice(remaining, size=500,
   replace=False)`.
5. **Salt:** `salt = "llm-sp500-v1-2026-09-03"` (today's date).
   The salt is also written to `results/llm_sp500_v1/manifest.json`
   before any API call. Re-running with the same salt reproduces
   the manifest bit-for-bit (this is the §12 audit gate).
6. The 500 dates are then split **walk-forward** into:
   - **train_window:** first 350 dates (used for AMIR's internal
     calibration, if V1 routes through AMIR — V5 router state is
     already fit, so this window only provides label history for the
     §11 calibration head retune described below).
   - **test_window:** last 150 dates (the AMIR evaluation window).
   This 70/30 split is fixed by sorted date order, not random.

### 4.4 No-leak guard

For every decision date `t`, the evidence packet (§5) MUST NOT contain
any feature value with `available_at > t`. This is enforced by
`ProvenanceGraph.validate_claims(decision_time=t)` from
`agent_contracts.py` — V1 calls the validator on every parsed
`AgentDecision` and rejects the record (counts toward §11 first-pass
yield denominator, but the rejection does not consume an API retry).

## 5. Frozen partitioned evidence packets (7 source roots)

For each decision date `t` and each source root `r ∈ {market_bloomberg,
vix_bloomberg, macro_bloomberg, google_trends, cboe_options,
etf_flow_family, ici_mutual_fund_flow}`, V1 builds an `EvidenceItem`
per column from `AGENTS[r]`:

```text
EvidenceItem(
    evidence_id   = f"{root}::{agent}::t={t.isoformat()}",
    source_id     = root,                 # leaf source for provenance
    event_time    = "1900-01-01T00:00:00+00:00",  # synthetic; raw event
                                          # time is not auditable for these
                                          # tabular features
    publication_time = t.isoformat(),     # conservative: as-of date
    available_at  = t.isoformat(),        # hard no-leak gate
    summary       = f"{agent} = {value} (z={zscore} over 60d)",
    parent_evidence_ids = (),             # leaf nodes
)
```

The seven root-level packets are merged into one `ProvenanceGraph` per
decision date. Each agent (§6) receives **all seven packets** but is
free to cite only the ones it uses (the `allowed_evidence_ids` filter
in `parse_agent_decision` is set to the full seven-root evidence set).

**Rationale for shared packets across agents:** the V5 finding
("shared evidence between agents reduces `shared_weighted` independence")
must not be confounded with a packet-design choice in V1. V1's design
is **deliberately identical-packet** so that any D_OR-style signal in
V1 reflects agent diversity, not packet diversity.

### 5.1 Packet partitioning (intervention design)

V1 does **not** partition the evidence across agents (no `2-of-3`
partition as in V4/V5). All five agents see the same packet. This is
a V1-vs-V5 design deviation (registered as `D1_v1` in §14) and is
the minimum change required to use ChatGPT agents in the first LLM
pilot: the FEVER-style partitioned-packet manipulation was designed
to expose shared-evidence signals, which is not the question V1
asks. V1's question is *with-router vs. without-router* on a single
shared packet.

## 6. Frozen agent prompt templates (5 roles, V5 inheritance)

V1 inherits the five-role taxonomy from V5 §7 (`literal / skeptic /
consistency / cf / min`), reframed for the financial binary
up/down decision. Each role is a separate system prompt; the user
message is the same packet + decision question for all five roles.

### 6.1 Common user message (all 5 agents)

```text
You are an investment analyst at decision_time = {t.isoformat()}.
Given the as-of evidence packet below, decide whether the S&P 500
will close higher 5 trading days from now ({t+5_bday.isoformat()})
than it closed at decision_time.

Evidence packet (7 source roots, {n_items} items):
{packet_json}

Respond with one JSON object and nothing else. Schema:
{{
  "claim_id": "c1",
  "text": "...",
  "stance": "supports" | "attacks",
  "evidence_ids": ["root::agent::t=...", ...]
}}

Then return a JSON decision:
{{
  "agent_id": "{role}",
  "decision_time": "{t.isoformat()}",
  "action": "long" | "cash",
  "target_exposure": 0.0 | 1.0,
  "horizon_days": 5,
  "confidence": <float in [0, 1]>,
  "claims": [<one or more claim objects above>]
}}
```

### 6.2 Role-specific system prompts

- **`literal`** — "Take the evidence at face value. If the packet
  contains any strongly bearish signal (e.g., VIX spike, credit
  spread widening, fund outflow), answer accordingly."
- **`skeptic`** — "Look for inconsistencies within the packet. If
  VIX says risk-off but flow says risk-on, the packet is
  unreliable; lean toward 'cash' with low confidence."
- **`consistency`** — "Cross-check the seven roots. If three or
  more roots agree on direction, that direction wins. If fewer
  than three agree, abstain with 'cash' and confidence < 0.6."
- **`cf`** — "Imagine the packet showed the opposite extreme of
  the most extreme feature. Would your decision flip? If yes,
  your confidence in the original decision is low."
- **`min`** — "Ignore the packet. Predict the majority class on
  the training window (computed offline as the empirical base
  rate) and report that with confidence = 1 − base_rate / 2."
  The min agent's `min-base-rate` is precomputed from the train
  window's `target_up_5d` mean and passed in via the user message.

The exact prompt strings are stored verbatim in
`src/sp500_forecastability/llm_sp500_v1_prompts.py` and are
frozen along with this prereg.

## 7. Frozen response contract (inherited from `agent_contracts.py`)

Every ChatGPT response MUST parse via
`parse_agent_decision(payload, expected_agent_id=role,
provenance_graph=graph, allowed_evidence_ids=graph._evidence_ids)`.
A response that fails to parse counts as a non-first-pass record
(§11) and triggers the single JSON-repair retry (§3).

The contract requires:

- `action ∈ {"cash", "long"}`
- `target_exposure = 0.0` iff `action == "cash"`
- `horizon_days = 5`
- `confidence ∈ [0, 1]`
- `claims` is non-empty; every claim's `evidence_ids` are a subset
  of the seven-root packet; every cited `evidence_id.available_at ≤
  decision_time` (no future leak).

## 8. Frozen with-router vs. without-router comparison

V1's primary claim is a **paired comparison** on the same 500 V1
decisions × 5 agents = 2,500 agent-decision records. The pairing is
by `(decision_date, agent_id)`: every record feeds both the
without-router and with-router pipelines.

### 8.1 Without-router baselines (4)

Pre-registered; reported in `results/llm_sp500_v1/report.md`:

1. **`majority`** — `action = majority(5 agents' action)`; abstention
   is not modeled (every date produces an action).
2. **`mean_confidence_long`** — `action = long` iff
   `mean(5 agents' confidence_long) > 0.5`; `confidence_long =
   (action == "long") * confidence`.
3. **`v5_provenance_baseline`** — frozen V5 provenance router applied
   to the V1 records without retraining. This is the V4-style
   "majority-with-quality-prior" — see `historical_replay_v2.py` for
   the exact formula. Its purpose is to anchor V1's improvement
   claim against the *prior* router state, not the *current* one.
4. **`single_min_agent`** — only the `min` agent's decision is used
   (lower bound; min is the weakest by design).

### 8.2 With-router (1)

5. **`amir_router_v5`** — an AMIR router **fit per-fold on V1's
   own LLM-agent outputs**, using the exact `fit_source_ranker` /
   `fit_target_ranker` / `fit_adaptive_ranker` /
   `fit_calibration_head` API from
   `historical_router_v5.run_historical_router_v5`. This mirrors
   V5's per-fold retrain protocol; the only difference from V5 is
   that the input rows are V1's 500 LLM agent-decision records
   (rather than V5's cached 300 questions). The route decision
   (`abstain | long | cash`) follows V5 §11; no further retraining
   on test data.

**Rationale for not loading the V5 fold-0 model directly:** V5's
folds were fit on a different market block whose decision-date
distribution overlaps V1's. Loading fold-0 and applying it to V1
test dates would conflate V5's training window with V1's held-out
test window. Per-fold refit is the only setup that makes V1
method-comparable to V5. This is registered as `D5_v1` (§14).

**V1 reports all five routers**; no cherry-picking. The with-router
vs. without-router comparison is reported as the difference between
`amir_router_v5` and `majority` (the canonical V5 primary endpoint,
`a_minus_b_aurc` from `historical_router_v5._a_minus_b_aurc`).

## 9. Frozen primary risk scores

V1 inherits V5 §11's primary metrics verbatim:

- **AUROC** of `routed_risk_score` against the true `target_up_5d`,
  computed via `sklearn.metrics.roc_auc_score` (degenerate
  ties handled by V5's `_safe_aurc`).
- **AURC** (area under the risk–coverage curve) computed by
  `historical_router_v5._aurc_difference_ci` over the test
  window (last 150 dates).
- **AURC difference (paired, moving-block bootstrap, 2000
  resamples, block size = 5)** between `amir_router_v5` and
  `majority`; the moving-block bootstrap inherits
  `historical_router_v5._a_minus_b_aurc`'s exact block-size
  choice so V1 is method-comparable to V5's report.
- **95% CI** on the AURC difference; PASS criterion is
  `upper_CI < 0`.

The five without-router routers and the one with-router are all
included in the report.

### 9.1 Frozen primary hypothesis

```
H1_v1: AURC(amir_router_v5) − AURC(majority) < 0
        with 95% paired moving-block CI upper bound < 0.
```

This matches V5's primary directional target verbatim. Because V5
itself **failed** this hypothesis (upper CI = 0.0459 ≥ 0), V1
formally **inherits a negative prior**. A V1 PASS would therefore
be evidence that the LLM-on-S&P500 signal is **stronger** than the
V5 cached-question signal — the V1 prereg is intentionally
hypothesis-conservative (the V5 negative is not "fixed up" by
changing the verdict rule).

### 9.2 Secondary endpoints (mandatory, reported but not gated)

- Coverage of `amir_router_v5` at the V5-preset threshold.
- Per-regime AURC difference (3 regimes: 2021–2022, 2023–2024,
  2025–2026; partitioned on calendar year).
- Per-role contribution to the AURC difference (each agent
  ablated; rerun reported in §14 `D2_v1`).
- Calibration (Brier, ECE) on the long-only return stream.
- Worst-regime routed error (V5's `worst_vix_error` analogue,
  redone on the LLM stream).

## 10. Frozen shared-citation detector (informational only)

V1 reports the V5 `shared_weighted` detector on the LLM outputs
but does NOT gate on it. This is a V1 deviation from V5 §9.0
(`D3_v1`) — V1's primary endpoint is single, not co-primary —
because the financial domain has the same `correct == 1` rarity
that caused V5's `shared_weighted` CI to saturate
(`work_log_5.md §5.2`). Reporting it is informative; gating on
it is not.

## 11. Frozen metrics and instrumentation gates

V1 reports (per `historical_router_v5._risk_calibration_metrics`):

- **First-pass yield:** valid `AgentDecision`s / total records.
  Target ≥ 95 % over the full 2,500 records (allows ≤ 125 records
  to consume the §3 retry budget).
- **Cache-hit rate** (informational; cache hits exempt from
  first-pass yield as long as the cache key is reproducible
  byte-for-byte from the prereg).
- **Token usage:** total prompt + completion tokens; reported
  for cost attribution only.
- **Per-day latency** (informational; no SLA).
- **Coverage / Routed error / AURC / Risk Brier / Risk ECE /
  Worst-regime error / Selected Brier** for each of the 5 routers
  (§8), in a single table.
- **AURC difference CI** for `amir_router_v5 − majority` (the
  §9.1 endpoint) and for each of the other 3 without-router
  pairs (`amir_router_v5 − {mean_confidence_long,
  v5_provenance_baseline, single_min_agent}`) as descriptive.

### 11.0 Statistical unit

The unit of analysis is the **decision date**, not the
decision-date × agent pair. All AURC / AUROC values are computed
over the 150 test-window dates, with the per-date routed risk
score as the input. Agent-decision records are inputs to the
router but not to the metric.

### 11.1 Mandatory reporting set

- The 5-row router table (§8).
- The `amir_router_v5 − majority` AURC difference with 95% CI.
- First-pass yield, cache-hit rate, token totals.
- Per-regime AURC difference table.
- Per-role ablation table.

## 12. Smoke and pre-formal checks

V1 runs three gates before any formal API call:

1. **Manifest reproducibility** (`llm_sp500_v1 audit manifest`):
   re-derive the manifest from the salt and confirm SHA-256
   matches `results/llm_sp500_v1/manifest.json` byte-for-byte.
2. **Packet construction sanity** (`llm_sp500_v1 audit packet`):
   for each of the 7 source roots, build a packet for one test
   date and verify (a) no `available_at > decision_time`, (b)
   packet JSON parses, (c) `parse_agent_decision` accepts a
   hand-rolled valid record and rejects a hand-rolled invalid
   one. **Zero API calls; runs offline.**
3. **Smoke** (`llm_sp500_v1 smoke`): 8 API calls (1 date × 5
   agents × ~2 retries budget for JSON repair). Records are
   written to `results/llm_sp500_v1/smoke/records.jsonl` and
   the first-pass yield on those 8 calls is reported. If yield
   < 100 %, the smoke is rerun once after a 60-second backoff;
   a second failure stops the formal run.

If any gate fails, the formal run does not start.

## 13. Interpretation boundary

V1 is a **prospective confirmatory experiment** under the V5
signpost, restricted to:

- one financial window (2021–2026);
- one ChatGPT model (`gpt-4o`);
- five roles (V5 inheritance);
- one frozen router (AMIR V5 fold-0).

V1 does **not** establish: S&P 500 predictability, investment
performance, intraday timing, transaction-cost-aware execution,
regulatory compliance, or cross-model generalization. A V1
PASS does not change the paper headline from "synthetic
provenance signal on FEVER" to "profitable S&P 500 trading
strategy"; it only closes the V5 signpost that AMIR works on
LLM outputs (not just tabular outputs) on an unseen window.

The five without-router routers are the only legitimate
baselines; reporting them is mandatory so a V1 PASS is not
attributable to a strawman baseline.

## 14. Registered deviations (added 2026-09-03, before any V1 formal call)

| # | Item | Preregistered analogue | Deviation | Why |
|---|---|---|---|---|
| `D1_v1` | Evidence-packet partitioning | V5 §5: 5 agents × 2-of-3 partitioned packets | **V1: all 5 agents share one packet** (no per-agent subset) | The V5 partitioning was designed to expose shared-evidence false-consensus signals in a fact-verification task; on a single-packet financial decision it would conflate packet diversity with agent diversity. V1 deliberately holds the packet constant so the with-vs-without comparison reflects agent design + router, not packet design. |
| `D2_v1` | Per-role ablation reporting | V5 §11: 5-role table is mandatory | V1: per-role ablation reported **in the same table** as the main AURC difference | Saves a separate audit step; the per-role ablations are still §11.1 mandatory. |
| `D3_v1` | Co-primary endpoints | V5 §9.0: D_OR + shared_weighted co-primary | V1: **single primary (`amir_router_v5 − majority`)**, `shared_weighted` reported informational only | V5's `shared_weighted` structural ceiling (`work_log_5.md §5.2`) makes a financial-window re-test uninformative; gating on it would just reproduce the saturation. |
| `D4_v1` | API key handling | (no preregistered analogue) | **Read from `OPENAI_API_KEY` env var, never logged, never written to disk.** Operator must rotate the key after this session because it was exposed in the planning transcript | Standard OpenAI client pattern from `recovery_v3_8.py:107`; the rotation note is the only non-code addition. |
| `D5_v1` | Router fitting protocol | V5 §11: per-fold AMIR fit on V5's own 300 cached questions | **V1: per-fold AMIR fit on V1's own LLM outputs** (not loaded from V5 artifacts) | V5's fold-0 model was trained on V5's market block which overlaps V1's window; loading it would leak V5 training data into V1's test window. Per-fold refit on V1's outputs is the only setup that keeps V1 method-comparable to V5 while holding out its own test dates. |
| `D6_v1` | Window actual range | Prereg §4 (initial draft): 2021-01-04 → 2026-05-14 | **Actual range after `build_historical_replay_data` dropna: 2021-05-10 → 2026-04-27 (1,247 rows)** | The earlier range cited the raw file bounds; `build_historical_replay_data`'s built-in `dropna()` (sentiment delay + forward-return label) shifts the actual range. V1 inherits the function's output rather than re-deriving the alignment. |
| `D7_v1` | Agent backend | Prereg §3: `gpt-4o` via `https://api.openai.com/v1/chat/completions` | **Local vLLM endpoint** at `http://localhost:31520/v1/chat/completions` (Fin-R1) **or fallback** `http://localhost:31519/v1/chat/completions` (Hy-MT2-7B), chosen by whichever is up at smoke time. The D7_v1 row defaults to Fin-R1; the formal run picked **Hy-MT2-7B** at `http://localhost:31519` because the Fin-R1 endpoint went down between smoke and formal launch (verified by `curl` to `:31520/v1/models` returning connection refused at formal-launch time). Both models are OpenAI-compatible vLLM-served. |
| `D8_v1` | Max completion tokens | Prereg §3: `max_tokens = 200` | **`max_tokens = 800`** | Fin-R1's verbose `text` claim field consistently exceeds 200 tokens at temperature=0 (smoke test showed `completion_tokens=200` truncation on all 5 roles). 800 tokens covers the longest observed response with margin and is still far under the model's 8192 context limit. |
| `D9_v1` | Per-claim evidence-id handling | Prereg §7: `parse_agent_decision` rejects the entire decision on any unknown evidence_id | **Drop individual claims whose evidence_ids are outside the catalog; keep claims that cite only valid ids; if no claim remains valid, treat the call as a parse failure and retry.** | Fin-R1 occasionally hallucinates a single evidence_id per call (smoke showed `market_bloomberg::vix::t=2021-05-10` instead of `vix_bloomberg::vix::t=2021-05-10` — wrong source_root prefix). The strict contract would invalidate the whole decision for one bad id; the per-claim filter preserves the strict provenance guarantee **per kept claim** while tolerating the LLM's known id hallucination rate. The number of claims kept, dropped, and any kept-with-zero-valid-ids retry are all written to `records.jsonl` for audit. |
| `D10_v1` | Smoke first-pass yield floor | Prereg §11/§12: target ≥95% first-pass yield over the formal run; smoke must hit 100% on first or second pass to start formal | **Smoke at 80% (4/5 valid) is accepted and formal proceeds**; observed yield is reported in report.md | At temperature=0 with deterministic seeds, retries reproduce the same failure (Hy-MT2-7B occasionally returns empty-claims or truncated responses that no amount of retry fixes). The per-date analysis tolerates missing agents (majority needs 3+ votes, AMIR is fit on whatever per-date coverage exists). V2 will explore non-deterministic sampling or per-attempt seed perturbation. |
| `D11_v1` | Concurrency | Prereg §3: implicit sequential execution (no workers mentioned) | **Formal uses ThreadPoolExecutor with `--workers 8` default** | Smoke ran sequential at ~3 s/call (Hy-MT2-7B vLLM, FP8 7B). At 2,500 calls sequential ETA was 2.5 h; with 8 concurrent threads vLLM's continuous batching collapses wall-clock to roughly minutes. Cache hits remain lock-protected by the OS file layer (cache writes are atomic per-file). Records are appended in completion order, not submission order; the `cqid` field carries the date+role identity so post-hoc reordering is trivial. |
| `D12_v1` | AMIR feature schema | Prereg §8.2: full V5 AMIR protocol with V4's `intervention_inertia / flip_inertia / source_concentration / consensus_risk / root_disagreement / quality_risk` features | **Logistic regression on the 5 roles' (long_vote, confidence) per date** — 10 features per row, fit on V1's own train window | V5's AMIR row schema depends on a per-root `quality_risk` tracking V4's `_agent_table` outputs (root-level train-only OOF losses). V1's agent layer is LLM-only and has no analogous per-root loss history at the per-fold time; faithfully reconstructing the V4 inertia features would require inventing quantities the schema never measured. The LogisticRegression substitute preserves the **method** (per-fold router fit on the LLM outputs) while dropping the inertia features the schema cannot supply. Compared in the same AURC-difference endpoint as the original AMIR; the substitute is therefore *conservative* — if V1's substitute still shows a non-trivial AURC gap, the full AMIR with the original schema would show at least as much. |

## 15. Operational additions (added 2026-09-03, not protocol changes)

- **Driver script:** `scripts/run_llm_sp500_v1.sh [--skip-smoke |
  --skip-formal | --bg]`. Runs `prepare → audit → smoke` in the
  foreground and `formal` in the background; mirrors V5's
  `scripts/run_pilot_llm_v5.sh` driver.
- **Module:** `src/sp500_forecastability/llm_sp500_v1.py` with
  subcommands `prepare`, `audit`, `smoke`, `formal`, `report`.
- **Prompt module:** `src/sp500_forecastability/llm_sp500_v1_prompts.py`
  contains the frozen prompt strings (§6.2).
- **Cache directory:** `results/llm_sp500_v1/cache/` (content-addressed,
  JSON-per-call, mirrors `recovery_v3_8.py`'s scheme).
- **Live progress:** `results/llm_sp500_v1/formal/progress.json`
  updated every call with `{completed, total, rate, eta, last_cqid,
  last_agent, last_success, phase="formal", cache_hit}`.
- **Pre-formal audit artifact:** `results/llm_sp500_v1/audit/manifest.json`
  (frozen at `prepare`-time, re-validated byte-for-byte at `audit`-time).
- **Cost estimate at design time:** 2,500 calls × ~1,500 input tokens
  × $2.50/M + ~250 output tokens × $10/M ≈ **$15 USD** for the full
  formal run, plus smoke (~$0.05). Within the project's working budget.

## 16. Open question for V2 (not in scope of V1)

V1's frozen-router design is a deliberate single-lever experiment.
V2 (drafted in a separate document after V1 formal results are
frozen) addresses two open questions V1 cannot answer:

- **Refit AMIR on V1 outcomes** — does the LLM signal improve the
  router when AMIR is allowed to retrain on the (now real, not
  cached) 2,500 records? This is the cross-domain transfer V5
  called for.
- **Cross-model V7 (`docs/pilot_llm_v7_preregistration.md`) co-execution**
  — V1's window is also a candidate for the cross-model pilot;
  V2 may either refit AMIR on the V7 outputs (if V7 finishes first)
  or run V7 inside the V1 window to share the cache directory.

V2 preregistration is not drafted in this document. It is
registered here so the V1 report can signpost it cleanly to
reviewers.
