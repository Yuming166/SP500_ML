# Work Log 3 — Pilot-LLM V4 pre-formal hardening (2026-08-31)

## 1. Diagnosis of the V4 pilot code vs the frozen preregistration

Five gaps between `docs/pilot_llm_v4_preregistration.md` and
`src/sp500_forecastability/pilot_llm_v4.py` were identified by direct
diff-reading of the two files:

1. **Permutation robustness (§9.4) was not implemented.** The previous
   `_permutation_aurocs` shuffled labels within rows rather than permuting
   the 5-agent ↔ 2-of-3-subset assignment, so the reported median / p05 /
   p95 was an upper bound on label noise, not on partition sensitivity.
2. **No bootstrap CIs (§11.1).** The preregistered 95 % question-cluster
   bootstrap CIs were not attached to AUROC / AUPRC / Risk@80 in the
   summary. Reports shipped point estimates only.
3. **No Platt LOO calibration (§9.3 S3).** S3 demands "Brier/ECE after
   Platt scaling fit on leave-one-question-out". The previous
   `_safe_brier` / `_safe_ece` used raw D_OR, which is structurally
   miscalibrated for low-prevalence outcomes.
4. **Silent substitute manifest failure.** `build_substitute_manifest`
   left `substitute_qid=""` for items without a same-cluster
   opposite-label candidate within ±50 %, and
   `build_composite_questions` then quietly dropped them from the
   eligible pool. Audit found 5/200 items affected (src_len 7 vs
   candidates 11–30, ratio 1.57–4.29). Silent drop would have hidden
   5 items from the manifest.
5. **Composite tie audit missing.** `validate_manifest` accepted a
   composite with a 1-1-1 label tie, which would only be discovered when
   `summarize_records` raised mid-run. Fail-fast was possible but not
   realised.

## 2. Implemented changes (single commit, before any V4 formal call)

- **Substitute manifest (§6, D1):** kept the ±50 % window as the primary
  match; added a **nearest-length fallback** when no in-window candidate
  exists in the same cluster. Records `in_length_window: bool` and three
  deviation lines in the manifest. All 200 items now have a substitute
  (audit confirms 100 % yield).
- **Substitute manifest fail-fast (D4):** if a source item has zero
  opposite-label same-cluster candidates at all, raise at `prepare`
  time with the offending qid, cluster, and candidate count.
- **Composite tie audit (D5):** `validate_manifest` now rejects any
  composite whose three TruthfulQA labels are not 2-1 or 1-2 before any
  call.
- **Permutation robustness (§9.4, D3):** replaced the broken
  `_permutation_aurocs` with **`_loao_aurocs`** — leave-one-agent-out
  AUROC across 5 variants (one per held-out agent index). Median and
  [p05, p95] reported alongside the deterministic AUROC with a
  registered deviation note explaining that the preregistered subset
  permutation requires re-calling per (cqid, agent, subset) tuple.
- **Bootstrap CIs (§11.1):** added `auroc_ci`, `auprc_ci`, and
  `risk_at_80_ci` to every metric entry. CIs use the registered seed
  `20260901` and 1,000 question-level replicates (not call-level;
  call-level would inflate n by 20×).
- **Platt LOO calibration (§9.3 S3):** added `_platt_loo_brier_ece`
  using `sklearn.linear_model.LogisticRegression(C=1.0, max_iter=200)`
  fit on n−1 questions per fold. Reports `brier_platt`, `ece_platt`,
  and `brier_raw` for comparison. Degenerate folds (single-class
  training) fall back to the training prior and are flagged.
- **Per-agent signal retention:** `_per_question_risks` now stores
  `_agent_inert` and `_agent_conf_stable` lists so LOAO and any future
  per-agent decomposition can be recomputed without re-calling.
- **`audit` subcommand:** runs all pre-formal gates and prints
  `{n_items, n_substitute_hits, substitute_yield, n_composites,
  balance, passes_yield_threshold}`.
- **`all` subcommand:** `prepare → audit → smoke → formal` in one
  process. Resumable by default. `--yes` removes the single prompt
  before formal. `--skip-smoke` reuses the latest smoke records;
  `--skip-formal` stops after smoke.
- **Resume from `records.partial.jsonl`:** `execute_run` now loads any
  partial records at start, builds a `(cqid, agent_index, condition)`
  done-set, and skips them. `--no-resume` on `smoke` / `run` opts out.
- **`progress.json`:** written every call with
  `{completed, total, elapsed_seconds, rate_per_second, eta_seconds,
  last_cqid, last_agent, last_condition, last_success,
  updated_at}`.
- **Bash driver:** `scripts/run_pilot_llm_v4.sh` runs prepare + audit
  + smoke in the foreground and the formal run in the background via
  `nohup`. `scripts/wait_pilot_llm_v4.sh` blocks on
  `run.pid` and prints the final `D_OR__harmful_fc` summary when done.

## 3. Verification

- `pytest -q tests/test_pilot_llm_v4.py` → 15/15 pass.
- `pytest -q` → 65/65 pass (no other tests broken).
- `python -m sp500_forecastability.pilot_llm_v4 audit` →
  `{n_items: 200, n_substitute_hits: 200, substitute_yield: 1.0,
  n_composites: 50, balance: {False: 25, True: 25},
  passes_yield_threshold: True}`.
- `python -m sp500_forecastability.pilot_llm_v4 prepare` →
  manifest.json now 128 KB (was 115 KB), 200 substitute entries
  (was 150); composite balance unchanged.

## 4. What this does NOT change

- The three preregistered hypotheses (§9.2 D_OR AUROC > 0.5; §9.3 S1/S2/S3;
  §10 shared_citation_signal AUROC) are unchanged. The implementation now
  reports them honestly with CIs and LOO calibration.
- V3 outputs are not edited or re-scored.
- No new LLM calls were made for this hardening pass.

## 5. Next action

`bash scripts/run_pilot_llm_v4.sh --yes` to run the audit + smoke + formal
pipeline end-to-end. Formal run takes ~5–10 min wall-clock (≈ 2.2 MB
transfer, 1,000 calls). Resume is automatic on retry; progress can be
polled with `cat results/pilot_llm_v4/formal/progress.json` or
`bash scripts/wait_pilot_llm_v4.sh`.
