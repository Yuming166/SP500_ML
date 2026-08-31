from sp500_forecastability.pilot_llm_v1 import (
    StrategyQAExample,
    evidence_view,
)
from sp500_forecastability.pilot_llm_v1 import (
    select_frozen_examples as select_v1,
)
from sp500_forecastability.pilot_llm_v2 import build_messages, select_frozen_examples


def _example(index: int, label: bool) -> StrategyQAExample:
    return StrategyQAExample(
        qid=f"qid-{label}-{index}",
        question=f"Is item {index} valid?",
        claim=f"GENERATED CLAIM {index}",
        label=label,
        facts=(f"Fact {index}.",),
    )


def test_v2_prompt_exposes_original_question_but_never_generated_claim() -> None:
    example = _example(1, True)
    messages = build_messages(
        example,
        evidence_view(example, "original"),
        agent_id="literal_evidence",
        persona="Use evidence literally.",
    )
    prompt = messages[1]["content"]

    assert example.question in prompt
    assert example.claim not in prompt
    assert '"claim"' not in prompt


def test_v2_uses_an_independent_selection_salt() -> None:
    examples = [_example(index, False) for index in range(100)] + [
        _example(index, True) for index in range(100)
    ]

    v1_qids = {example.qid for example in select_v1(examples)}
    v2 = select_frozen_examples(examples)

    assert len(v2) == 50
    assert sum(example.label for example in v2) == 25
    assert {example.qid for example in v2} != v1_qids
