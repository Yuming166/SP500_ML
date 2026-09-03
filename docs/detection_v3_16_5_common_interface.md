# Detection V3.16.5 common cross-model interface

Date: 2026-09-03 (Asia/Shanghai)

Status: **frozen after three Ling structured-decoding transport diagnostics and
before any V3.16.5 call**.

The earlier Ling smokes showed that vLLM JSON-schema constrained decoding can
enter a whitespace loop after emitting an answer and evidence ID. Raising the
completion budget from 128 to 256 and 512 did not isolate the effect because
the transport protocol version also changed the seed. These runs are retained
as instrumentation evidence and no Ling outcome metric was computed.

V3.16.5 establishes one clean model-independent interface:

- Qwen3.5-4B and Ling-3.0-tiny both receive the same prompt-only JSON contract;
- neither request uses server-side `response_format`;
- both use the same strict local parser and 256-token limit;
- the seed is derived from the immutable string
  `detection-v3.16-common-seed-2026-09-03`, model ID, item ID, agent, and
  condition, and will not change under later transport version labels;
- substitute may decline to cite unrelated evidence; original/reverse require
  the exact visible ID and remove requires no citation; and
- a completely fresh `calls_4` cache is used for both models.

Because the interface changed for both models, Qwen development and weight
selection must be rerun before Ling transfer. The sample partitions, natural
contrastive pairs, risk-coordinate definitions, weight-selection algorithm,
pilot gates, and 500 formal candidate items remain unchanged. Formal calls are
not authorized.
