# Recovery V3.10 preregistration: schema-constrained cross-model ELAR

Protocol version: `recovery-v3.10-qwen-to-finr1-guided-elar-2026-09-03`

Status: **frozen before any V3.10 task-bearing Fin-R1 call**.

## 1. Outcome-blind transport diagnosis

V3.9.2 stopped after 22 complete formal action bundles and before any outcome
evaluation. Eighteen of 176 action rows remained invalid because Fin-R1 wrote
bare evidence identifiers in JSON-shaped arrays. No formal certificate,
ledger, route, correctness, gain, or subgroup statistic was computed.

The manifest content-addresses the V3.9.2 manifest, abort record, and 176-row
partial action artifact. V3.10 uses a fresh cache and reruns all 400 examples;
no prior target response is reused at inference.

## 2. Frozen schema-constrained interface

Every target request includes vLLM's OpenAI-compatible
`response_format=json_schema`, selected only by artifact kind:

- action: exactly `answer`, `confidence`, and `cited_evidence_ids`;
- certificate: exactly `atomic_checks`, `coverage_complete`, and `confidence`,
  with fixed check fields and status enum; and
- ledger: exactly `entries` and `challenge`, with fixed nested fields and
  verdict/challenge enums.

The schemas constrain JSON field names and primitive/container types. They do
not establish whether a claim span is genuine, an evidence ID is packet-local,
a quote is exact, a relation is entailed, or a certificate is complete. Every
output therefore still passes the frozen V3.8.3/V3.9.2 semantic parser and
fail-closed checks. Temperature, prompts, seeds, token limits, and the single
repair attempt are unchanged.

The exact schemas, dispatcher, request-body construction, runtime version, and
server script are hashed in the manifest. No schema or parser extension is
permitted after V3.10 freezes. Any terminal formal action failure aborts.

## 3. Frozen scientific protocol

V3.10 changes no scientific choice from V3.9:

- Qwen3.5-4B V3.7.1 ELAR router and thresholds `0.8`, `0.0`, and `1`;
- local SUFE Fin-R1 BF16 Qwen2 target on vLLM 0.28.0;
- exact ordered 400-example balanced formal set and 1,200 unique roots;
- five baseline decisions, three actions, two certificates, and eligible
  ledgers regenerated for every example;
- pre-outcome route construction and no target fitting/calibration;
- question bootstrap seed `20261102` with 10,000 replicates; and
- all five primary gates and registered matched baselines.

Ling outcomes and V3.9-series formal outcomes are not used for fitting,
thresholding, schema selection, or route selection.

## 4. Claim and originality boundary

Structured decoding is an interface reliability control, not the claimed
novel contribution. The contribution under test remains ELAR's pre-outcome,
provenance-root-aware, proof-carrying action choice. A pass supports transfer
to a separately trained SUFE checkpoint within the broad Qwen lineage; it does
not establish cross-architecture or universal transfer.
