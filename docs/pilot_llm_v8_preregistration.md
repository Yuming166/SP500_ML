# Pilot-LLM V8 preregistration: FEVER N=100 with V5 salt + Ling-3.0-tiny-int4 (cross-model)

**Date frozen:** 2026-09-01
**Status:** Frozen before any V8 model call. Any substantive change after V8
outputs requires a new version (V9).

## 1. Why an eighth pilot is needed

V7 (FEVER, N=100, V5 salt, Qwen3.5-4B) confirms V5's `shared_weighted`
signal holds at N = 100 with the same selection rule:
`shared_weighted` AUROC = 0.816 [0.567, 1.000] (CI lo 0.567 > 0.5). V5 ⊂
V7 by construction at the question level (V7 uses V5's salt).

V7's `D_OR` (0.621) is selection-fixed stable but selection-sensitive
across selections (V5 → V6 with fresh salt: 0.656 → 0.388, drift
0.268). V8's central question:

> **Does the `shared_weighted` signal transfer across model families?**

V7's strongest finding is that `shared_weighted` is the
selection-robust signal. But V7 only tests it on **one model**
(`Qwen3.5-4B`). The strongest threat to the paper's headline is
"this only works on Qwen3.5-4B". V8 addresses this by running the
**same V7 protocol** on **Ling-3.0-tiny-int4** (inclusionAI /
antgroup), a fundamentally different architecture:

| | V7 (`Qwen3.5-4B`) | V8 (`Ling-3.0-tiny-int4`) |
|---|---|---|
| Family | Alibaba Qwen (Alibaba DAMO) | inclusionAI / antgroup |
| Architecture | Standard transformer decoder | **Hybrid KDA-MLA + MoE** (3 KDA + 1 MLA per 4-layer block, 128 experts) |
| Total params | 4 B | 7.9 B (1.3 B activated per token) |
| Precision | bfloat16 | int4 quantized |
| Native runtime | vLLM (stock) | **SGLang** (Ling README §install) |
| Chat template | vLLM default | Ling chat_template.jinja (`<role>SYSTEM</role>`-style) |
| Thinking mode | disabled by default | **enabled by default** — must set `enable_thinking=False` |

Cross-model signal transfer is the **single most important
external-validity claim** of the paper. V8 directly tests it.

## 2. Why Ling specifically (vs other cross-model options)

Other cross-model candidates were considered and rejected:

| Candidate | Reject reason |
|---|---|
| `Fin-R1` (SUFE-AIFLM-Lab, 15 GB, Qwen2-based) | Same architectural family (Qwen2 lineage); cross-model claim is weak |
| `Qwen3-8B` (zhangsg's modelzoo) | Same Qwen3 family; same architectural lineage |
| `ChatGLM3-6b` (zhaoyw, 12 GB) | Released 2023, two-year-old model; lianjh noted it would not be a fair comparison to Qwen3.5-4B (2025) |

Ling-3.0-tiny-int4 is the **best available cross-model target**:
different research lab, different architecture (hybrid linear-attention
MoE vs standard decoder), different precision regime (int4 vs bf16),
different native runtime (SGLang vs vLLM).

## 3. Status of prior pilots (preserved, not edited)

| Version | Outcome | Stop reason |
|---|---|---|
| Pilot-LLM V1 | Stopped at smoke | Generated StrategyQA claim can flip polarity while label is attached to original question |
| Pilot-LLM V2 | Stopped at smoke | Agent-level abstention erased false-consensus observations |
| Pilot-LLM V3 | Formal run completed (750/750) | Protocol-limited negative result (AUROC = 0.4125); not retried |
| Pilot-LLM V4 | Formal run completed (1,000/1,000) | §9.2 + 2/3 secondaries passed; `D_OR` 0.676; `shared_weighted` 0.785 (discovered post-hoc) |
| Pilot-LLM V5 | Formal run completed (1,000/1,000) | PARTIAL_PASS: `D_OR` cleared, `shared_weighted` CI lo = 0.359 < 0.5; `brier_platt` 0.040, `ece_platt` 0.000 |
| Pilot-LLM V6 | Formal run completed (2,000/2,000) | Original §9.2 FAIL (D_OR = 0.388); amended §9.2 PASS_SINGLE on `shared_weighted` (0.820 [0.571, 0.995]) per `D6_v6` |
| Pilot-LLM V7 | Formal run completed (2,000/2,000) | PASS_SINGLE_SHARED_WEIGHTED on Qwen3.5-4B: `shared_weighted` = 0.816 [0.567, 1.000]; V5 ⊂ V7 by construction |
| **Pilot-LLM V8** | **This document** | — |

## 4. Frozen model, retry, and transfer controls

| Item | V7 value | V8 value |
|---|---|---|
| Endpoint | `http://10.63.0.88:31519/v1/chat/completions` | `http://localhost:31520/v1/chat/completions` |
| Model | `Qwen3.5-4B` | `Ling-3.0-tiny-int4` |
| Temperature | 0.0 | 0.0 |
| Maximum completion tokens | 160 | 160 |
| Timeout | 60 s | 60 s |
| Retries | 1 initial + 1 JSON-repair | 1 initial + 1 JSON-repair |
| Cache | content-addressed, sha256(endpoint + payload) | content-addressed, sha256(endpoint + payload), separate cache_dir per model |
| Thinking mode | n/a | **`chat_template_kwargs={"enable_thinking": False}`** mandatory (Ling's chat_template enables thinking by default — agents would emit reasoning tokens before answering, breaking the JSON contract) |

V8's serve stack is **SGLang** (not vLLM). SGLang is launched via:

```bash
CUDA_VISIBLE_DEVICES=3 python -m sglang.launch_server \
    --model-path /storage/lianjh/modelzoos/inclusionAI/Ling-3.0-tiny-int4 \
    --port 31520 \
    --host 0.0.0.0 \
    --served-model-name Ling-3.0-tiny \
    --max-model-len 8192 \
    --trust-remote-code
```

(V8 must run on a 4090 GPU; Ling's hybrid architecture is incompatible
with V100 CC 7.0 when SGLang forces torch ≥ 2.13.)

## 5. Domain: FEVER (same as V5 / V6 / V7)

Inherited verbatim from V5 §4. FEVER `valid.jsonl` is already on the
jump host:

```text
/storage/gaoym/sp500-forecastability-lab/data/fever/fever-validation.jsonl
SHA-256: 5da0ccc0ccf77f974611de13f8aac6f78c6bba6293912835099eb6029baa85d9
```

### 5.1 Manifest construction (V5 salt, N = 100)

V8 inherits V5's salt (`pilot-llm-v5-2026-08-31`) and the same N = 100
construction rule as V7. **V5 ⊂ V7 ⊂ V8** by construction at the
question level. V8 = V7's manifest run on a different model; V5 ⊂ V8
is also true (transitive).

V8 uses the same `V7 §4.3` selection rule with V5's salt, so V8's
selection is **identical** to V7's selection. V8 therefore tests
**model-agnosticism on the exact same question set as V7**, isolating
model from selection.

## 6. Frozen paired evidence conditions (4, same as V5 / V6 / V7)

Inherited verbatim from V5 §6 / V6 §6 / V7 §6: `original`, `remove`,
`reverse`, `substitute`. The substitute condition uses LLM-rewritten
negative paraphrase (V5 §6.3).

V8 does **not** regenerate substitute manifests. V8 reuses V5's
substitute manifest via the content-addressed cache (`D1_v7`, V7 §14).
The LLM-rewritten substitutes are deterministic functions of
(prompt, seed) and are model-independent; the same substitute
sentence will be used for both Qwen3.5-4B (V5/V6/V7) and
Ling-3.0-tiny-int4 (V8) evaluations.

## 7. Frozen agents and response contract

Identical to V5 §7. Five agents: `literal_evidence`,
`skeptical_auditor`, `consistency_checker`,
`counterfactual_reasoner`, `minimal_judge`. Per-claim `yes`/`no`
answer, confidence in `[0, 1]`, citation-packet validation.

**V8-specific response parsing**: Ling's chat_template emits reasoning
blocks (`<think>...</think>`) when thinking mode is on. V8's parse
must extract only the JSON object AFTER the closing `</think>`. If
thinking mode is left on, the first 200-500 generated tokens are
reasoning, leaving the actual JSON in the tail — the existing
JSON-only parser would reject them. The mandatory
`enable_thinking=False` at request time prevents this.

## 8. Frozen intervention and consensus definitions

Inherited verbatim from V5 §8.

## 9. **Frozen primary risk scores** (D_OR + shared_weighted, co-primary, any-passes)

Same as V7 §9 (any-passes verdict logic). Both D_OR and
`shared_weighted` are co-primary; §9.2 passes if **either** clears
its 95% CI lo > 0.5 bar. The cross-model claim is centered on
`shared_weighted`'s robustness:

- **V5 (Qwen3.5-4B, FEVER, N=50, V5 salt)**: `shared_weighted` = 0.698 [0.359, 1.000]
- **V7 (Qwen3.5-4B, FEVER, N=100, V5 salt)**: `shared_weighted` = 0.816 [0.567, 1.000]
- **V8 (Ling-3.0-tiny-int4, FEVER, N=100, V5 salt)**: `shared_weighted` = ? [?, ?]

If V8's `shared_weighted` AUROC has CI lo > 0.5, the cross-model
signal transfer claim is confirmed: `shared_weighted` generalizes
across architectures (Qwen dense transformer → Ling hybrid KDA-MLA
MoE). If V8's `shared_weighted` AUROC has CI lo ≤ 0.5, the
cross-model claim is refuted; V8 reports the failure and discusses
whether the signal is model-architecture-specific.

### 9.1 Frozen co-primary endpoints

Same as V7 §9.1.

### 9.2 Frozen co-primary hypothesis

Same as V7 §9.2 (any-passes).

### 9.3 Pre-registered secondary hypotheses

| # | Secondary hypothesis | Why it is required |
|---|---|---|
| S1 | **AUPRC(`D_OR`) > AUPRC(`D_majority`)** | Same as V7 |
| S2 | **`Risk@80%Coverage`(`D_OR`) does not exceed prevalence baseline by more than 0.05** | Same as V7 |
| S3 | **Calibration**: Brier(`D_OR`) and ECE(`D_OR`) both < 0.30 after Platt scaling fit on leave-one-question-out | Same as V7 |
| S4 | **Cross-model transfer**: `shared_weighted` AUROC on Ling (V8) > 0.5 with CI lo > 0.5, AND the V8 point estimate is within ±0.2 of the V7 point estimate (0.816 ± 0.2 → [0.616, 1.016]) | Central V8 question: is the selection-robust `shared_weighted` signal **model-agnostic**? |
| S5 | **V5 + V7 + V8 joint**: `shared_weighted` AUROC > 0.5 with CI lo > 0.5 on the joint N = 100 set (V5 ⊂ V7 ⊂ V8; all three runs share the same 50 V5 questions plus V7/V8's 50 fresh questions) | Tests whether the cross-model signal is consistent across both Qwen3.5-4B and Ling-3.0-tiny-int4 on a *common* question set |
| S6 | **Per-condition flip rate parity**: V8's `substitute_flip` rate within ±0.15 of V5's 0.336 | Sanity check: Ling's substitute should produce similar per-agent fragility as Qwen3.5-4B if the methodology is model-agnostic at the agent level |

Pass criteria: §9.2 passes (any-passes) **and** at least two of {S1, S2, S3} pass. **S4, S5, S6 are reported but not gating.** If §9.2 fails, V8 reports the methodology as failing to transfer across model families.

### 9.4 Partition robustness

LOAO median + [p05, p95] AUROC across 5 variants, reported for
`D_OR` and `shared_weighted`. Same as V7 §9.4.

## 10. Frozen shared-citation detectors

Same as V7 §10. `shared_weighted` is the central cross-model probe.

## 11. Frozen metrics and instrumentation gates

### 11.0 Statistical unit

Same as V7 §11.0 (n = 100, 2000 calls, question-level bootstrap, seed
`20260902`, 1,000 replicates).

### 11.1 Mandatory reporting set

For `D_OR` and `shared_weighted`:
- AUROC, AUPRC, Risk@80%Coverage, Brier+ECE (Platt LOO), each with
  95% CI.

For `D_inert`, `D_conf`: AUROC + CI.

For each of the four shared-citation detectors: AUROC + CI.

Plus: `D_majority` baseline, per-condition flip rates, harmful_fc
prevalence, LOAO, substitute-generation pass stats.

**Cross-model comparison table (central V8 deliverable)**:

| Endpoint | V5 (Qwen, N=50) | V7 (Qwen, N=100) | V8 (Ling, N=100) |
|---|---|---|---|
| `D_OR` | 0.656 [0.508, 0.787] | 0.621 [0.441, 0.793] | (V8) |
| `shared_weighted` | 0.698 [0.359, 1.000] | 0.816 [0.567, 1.000] | (V8) |
| `brier_platt` | 0.040 | 0.071 | (V8) |
| `ece_platt` | 0.000 | 0.0001 | (V8) |

The formal run passes only if:
- the FEVER dataset digest matches the frozen SHA-256 (V5 §4.2);
- the V8 manifest of 100 composites is reproducible from V5's salt
  (V5 ⊂ V7 ⊂ V8 by construction);
- the substitute manifest is reproducible from V5's salt (V8 reuses
  V5's substitute manifest via content-addressed cache);
- the substitute-generation pass left < 10% items unusable;
- at least 98% of 2,000 calls have an HTTP response or valid cache;
- at least 95% produce a strict yes/no decision after the fixed retry
  (with `enable_thinking=False` enforced at request time);
- at least 95% of (100 × 5 = 500) question-agent quadruplets are
  complete;
- every accepted citation is packet-validated.

These are operational gates, not superiority criteria.

## 12. Smoke and pre-formal checks

The only permitted pre-formal V8 smoke is **two composite questions,
one agent (`literal_evidence`), the four conditions**: 8 calls.
Smoke outputs remain separate from formal results.

Pre-formal audit must confirm:
- SGLang endpoint reachable at `http://localhost:31520/v1/models` and
  `Ling-3.0-tiny` is listed;
- the V8 manifest of 100 composites is reproducible from V5's salt
  (= V7's manifest by construction, V5 ⊂ V7 ⊂ V8);
- the substitute manifest is reproducible from V5's salt;
- **the smoke run produces clean JSON without `<think>...</think>`
  blocks** (validates that `enable_thinking=False` is wired
  through the client correctly);
- all 8 smoke calls pass instrumentation gates.

## 13. Interpretation boundary

V8 reports cross-model signal transfer. If V8 passes §9.2 on
`shared_weighted` and the V8 point estimate is within ±0.2 of V7's
0.816, the methodology generalizes across model architectures. If V8
fails §9.2, the cross-model claim is refuted; V8 reports the failure
mode (Ling's hybrid KDA-MLA-MoE may be too aggressive / not aggressive
enough on the substitute intervention to surface shared citations).

V8 does not establish S&P 500 predictability, investment performance,
or LLM faithfulness in general.

## 14. Registered deviations

The following deviations are registered before any V8 formal call.

| # | Item | Preregistered analogue | Deviation | Why |
|---|---|---|---|---|
| `D1_v8` | Model | V7: `Qwen3.5-4B` on `http://10.63.0.88:31519` | **`Ling-3.0-tiny-int4` on `http://localhost:31520`** (SGLang server, GPU 3 RTX 4090) | V8's central question is cross-model signal transfer. Ling is inclusionAI's hybrid KDA-MLA MoE int4 model, the strongest available cross-model target. SGLang is Ling's official runtime per the model README. |
| `D2_v8` | `enable_thinking` | (n/a in V7; Qwen3.5-4B has no thinking mode) | **Mandatory `chat_template_kwargs={"enable_thinking": False}` in every request** | Ling's chat_template emits `<think>...</think>` blocks by default. With thinking on, the first 200-500 generated tokens are reasoning; the JSON contract fails. Disabling thinking is required for protocol comparability. |
| `D3_v8` | Cache | V7: shared cache across V5/V6/V7 | **Separate cache_dir for V8** (`results/pilot_llm_v8/cache/`) keyed on V8's endpoint URL | Different endpoint URL → different cache_key → old cache won't collide. |
| `D4_v8` | Substitute manifest | V7: shared substitute manifest from V5 | **V8 inherits V5's substitute manifest** (same content_addressed cache scheme, content-derived) | The substitute prompts and seed (20_260_903) are model-independent; the same substitute sentence is used for both Qwen3.5-4B and Ling-3.0-tiny-int4 evaluations. Re-running substitute generation against Ling would be wasteful and produce identical output. |
| `D5_v8` | Endpoint validation in `CachedChatClient` | V1: `endpoint != DEFAULT_ENDPOINT or model != DEFAULT_MODEL` raises | **V8 uses a custom `LingChatClient` that bypasses the V1 endpoint/model validation** | The V1 validation is a hard guard for the original V1 reproducibility. V8 is a new preregistered experiment with its own endpoint and model, so a separate client is needed. |
| `D6_v8` | GPU | V7: V100 (lianjh's vllm on V100 CC 7.0) | **RTX 4090 (CC 8.9, GPU 3) via SGLang** | Ling's hybrid KDA-MLA MoE kernel requires torch ≥ 2.13 (SGLang requirement), which drops V100 CC 7.0 support. RTX 4090 CC 8.9 supports torch 2.13+ natively. |

## 15. Operational additions

- `scripts/run_pilot_llm_v8.sh`: prepares manifest, runs
  substitute-gen (idempotent), pre-formal audit, smoke (8 calls),
  formal (2,000 calls) — same structure as V7 driver.
- `scripts/wait_pilot_llm_v8.sh`: blocks on background formal and prints
  the §9.2 verdict + cross-model comparison table when done.
- SGLang is launched **separately** before the V8 driver is invoked
  (different process tree from the V8 pilot).

## 16. Open question for V9

V8 does **not** address selection-robustness across selections on
Ling (V6-style question set). V9 would be Ling on a fresh selection
(N = 100, fresh salt, FEVER). V9 preregistration is **not** drafted
in this document.

V9's central question is: is Ling's selection-sensitivity (analogous
to Qwen3.5-4B's V5 → V6 regression) also small, or does it amplify?
If small, the selection-variance is a model-property; if amplified,
it is an architecture-by-selection interaction.