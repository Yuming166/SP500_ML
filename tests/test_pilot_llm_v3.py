import json

import pytest

from sp500_forecastability.pilot_llm_v1 import StrategyQAExample, evidence_view
from sp500_forecastability.pilot_llm_v2 import select_frozen_examples as select_v2
from sp500_forecastability.pilot_llm_v3 import (
    build_messages,
    parse_forced_qa_decision,
    select_frozen_examples,
)


def _example(index: int = 1, label: bool = True) -> StrategyQAExample:
    return StrategyQAExample(
        qid=f"qid-{label}-{index}",
        question=f"Is item {index} valid?",
        claim=f"GENERATED CLAIM {index}",
        label=label,
        facts=(f"Fact {index}.",),
    )


def _response(**updates: object) -> str:
    payload: dict[str, object] = {
        "agent_id": "literal_evidence",
        "answer": "yes",
        "confidence": 0.8,
        "cited_evidence_ids": ["E01"],
    }
    payload.update(updates)
    return json.dumps(payload)


def test_v3_prompt_excludes_generated_claim_and_requires_answer() -> None:
    example = _example()
    messages = build_messages(
        example,
        evidence_view(example, "original"),
        agent_id="literal_evidence",
        persona="Choose yes or no.",
    )
    prompt = "\n".join(message["content"] for message in messages)

    assert example.question in prompt
    assert example.claim not in prompt
    assert "you cannot abstain" in prompt


def test_v3_parser_accepts_only_forced_yes_no_schema() -> None:
    decision = parse_forced_qa_decision(
        _response(),
        expected_agent_id="literal_evidence",
        allowed_evidence_ids=("E01",),
    )
    assert decision.answer == "yes"
    assert decision.to_internal_payload()["decision"] == "answer"

    with pytest.raises(ValueError, match="answer must be yes or no"):
        parse_forced_qa_decision(
            _response(answer=None),
            expected_agent_id="literal_evidence",
            allowed_evidence_ids=("E01",),
        )

    with pytest.raises(ValueError, match="unexpected decision fields"):
        parse_forced_qa_decision(
            _response(decision="abstain"),
            expected_agent_id="literal_evidence",
            allowed_evidence_ids=("E01",),
        )

    with pytest.raises(ValueError, match="outside the packet"):
        parse_forced_qa_decision(
            _response(cited_evidence_ids=["invented"]),
            expected_agent_id="literal_evidence",
            allowed_evidence_ids=("E01",),
        )


def test_v3_uses_a_new_selection_salt() -> None:
    examples = [_example(index, False) for index in range(100)] + [
        _example(index, True) for index in range(100)
    ]

    v2_qids = {example.qid for example in select_v2(examples)}
    v3 = select_frozen_examples(examples)

    assert len(v3) == 50
    assert sum(example.label for example in v3) == 25
    assert {example.qid for example in v3} != v2_qids
