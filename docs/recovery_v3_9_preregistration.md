# Recovery V3.9 preregistration: zero-shot Qwen-to-Fin-R1 ELAR

Protocol version: `recovery-v3.9-qwen-to-finr1-elar-2026-09-03`

Status: **frozen by `recovery_v3_9 prepare` before any task-bearing Fin-R1
formal call**. The only permitted task calls before formal execution are the
same two fixed development schema-smoke examples used by V3.8.

## 1. Registered continuation and non-retroactivity

V3.8 Section 7 registered Fin-R1 as the next cross-model replication before
the Ling result was known. V3.9 tests whether the frozen Qwen3.5-4B V3.7.1
evidence-ledger action router (ELAR) transfers without target fitting to the
SUFE-AIFLM-Lab Fin-R1 checkpoint.

The Ling V3.8.3 result is preserved as a coverage-failure negative result. Its
outcomes do not fit, calibrate, select, or gate V3.9. V3.9 does not modify any
earlier code, manifest, response, route, or reported result.

## 2. Frozen source router and target

- Source: `Qwen3.5-4B` and the frozen V3.7.1 ELAR manifest.
- Policy thresholds: ledger confidence at least `0.8`, lexical coverage at
  least `0.0`, and at most one unsupported term.
- Target: `Fin-R1`, the local SUFE-AIFLM-Lab 7B BF16 Qwen2 checkpoint.
- Runtime: vLLM 0.28.0, Torch 2.13.0, Transformers 5.16.1, GPU 4, 8,192-token
  context, no checkpoint quantization, and `--generation-config vllm`.
- Endpoint: `http://127.0.0.1:31520/v1/chat/completions`.
- Decoding: temperature 0, fixed inherited seeds, thinking disabled where the
  chat template accepts the flag, and one inherited schema-repair attempt.

The manifest hashes all small model artifacts, records every weight shard's
size, hashes the server and runner scripts, and checks that `/v1/models`
contains exactly `Fin-R1`.

V3.9 inherits the final V3.8.3 closed action-transport conformance layer:
yes/no case folding, unambiguous evidence-key aliasing, empty/singleton
citation-list normalization, and finite numeric percentage confidence
normalization. No additional parser rule may be added after V3.9 freezes.

## 3. Evaluation universe and execution

V3.9 reuses the exact ordered V3.7.1 400-example formal selection: 200
Supported and 200 Refuted claims, 1,200 globally unique page roots, and zero
claim/root overlap with source-router development folds. All target-dependent
observables are regenerated into a fresh Fin-R1 cache:

1. five anchor-only persona decisions;
2. all three recovery actions (`candidate_0`, `candidate_1`, `both`);
3. two single-root atomic certificates;
4. ledgers for every proof-eligible action; and
5. route selection written before outcome construction.

Any terminal action failure aborts before evaluation. Certificate and ledger
failures are retained and fail closed to KEEP. The already registered
atomic-proof-only diagnostic treats a missing fail-closed ledger as an empty
diagnostic ranking object; this cannot affect primary ELAR, which requires a
valid ledger.

## 4. Primary estimand and fixed gates

The primary estimand is native-label macro accuracy gain over Fin-R1 KEEP.
The question bootstrap uses seed `20261102`, 10,000 replicates, and a 95%
percentile interval. V3.9 passes only if all inherited gates pass:

1. macro-gain CI lower bound is above zero;
2. damage among high-consensus-correct cases is at most 5%;
3. both native-label groups have nonnegative gain;
4. at least ten repairs select the held-out annotated root; and
5. net fixes exceed KEEP and every root-budget-matched baseline.

No subgroup or unlimited-routing diagnostic can replace the primary result.

## 5. Claim boundary

A pass supports zero-shot transfer across two separately trained checkpoints
from different organizations, but both belong to the broad Qwen lineage. It
is weaker evidence of architectural invariance than a successful Ling result.
It does not establish universal transfer, publisher independence, live
retrieval robustness, or performance on translation-specialized Hy-MT2.
