import json

import pytest

from sp500_forecastability.pilot_llm_v1 import (
    PROTOCOL_VERSION,
    StrategyQAExample,
    build_messages,
    evidence_view,
    parse_qa_decision,
    parse_serialized_string_array,
    select_frozen_examples,
    summarize_records,
)


def _example(*, qid: str = "qid-1", label: bool = True) -> StrategyQAExample:
    return StrategyQAExample(
        qid=qid,
        question="Can the claim be true?",
        claim="The claim is true.",
        label=label,
        facts=("The first fact is true.", "The second fact is true."),
    )


def _decision(
    *,
    answer: str | None = "yes",
    decision: str = "answer",
    confidence: float = 0.8,
    citations: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "agent_id": "literal_evidence",
            "decision": decision,
            "answer": answer,
            "confidence": confidence,
            "cited_evidence_ids": ["E01"] if citations is None else citations,
        }
    )


def test_strategyqa_serialized_facts_remain_separate() -> None:
    value = "['A fact with an apostrophe: it\\'s fine.'\n 'A second fact.']"

    assert parse_serialized_string_array(value) == (
        "A fact with an apostrophe: it's fine.",
        "A second fact.",
    )


def test_frozen_selection_is_balanced_and_deterministic() -> None:
    examples = [
        _example(qid=f"false-{index}", label=False) for index in range(30)
    ] + [_example(qid=f"true-{index}", label=True) for index in range(30)]

    first = select_frozen_examples(examples)
    second = select_frozen_examples(list(reversed(examples)))

    assert [example.qid for example in first] == [example.qid for example in second]
    assert sum(example.label for example in first) == 25
    assert len(first) == 50


def test_interventions_keep_environment_owned_ids_and_hide_condition_name() -> None:
    example = _example()
    original = evidence_view(example, "original")
    removed = evidence_view(example, "remove")
    reversed_view = evidence_view(example, "reverse")

    assert original.allowed_evidence_ids == ("E01", "E02")
    assert removed.allowed_evidence_ids == ()
    assert reversed_view.allowed_evidence_ids == original.allowed_evidence_ids
    assert all("counterfactual" in text for _, text in reversed_view.items)

    messages = build_messages(
        example,
        reversed_view,
        agent_id="literal_evidence",
        persona="Use the packet literally.",
    )
    assert '"label"' not in messages[1]["content"]
    assert '"condition"' not in messages[1]["content"]


def test_qa_parser_rejects_schema_drift_and_packet_external_citations() -> None:
    valid = parse_qa_decision(
        _decision(),
        expected_agent_id="literal_evidence",
        allowed_evidence_ids=("E01",),
    )
    assert valid.answer == "yes"

    payload = json.loads(_decision())
    payload["reasoning"] = "hidden text is outside the contract"
    with pytest.raises(ValueError, match="unexpected decision fields"):
        parse_qa_decision(
            json.dumps(payload),
            expected_agent_id="literal_evidence",
            allowed_evidence_ids=("E01",),
        )

    with pytest.raises(ValueError, match="outside the packet"):
        parse_qa_decision(
            _decision(citations=["invented"]),
            expected_agent_id="literal_evidence",
            allowed_evidence_ids=("E01",),
        )


def test_qa_parser_enforces_abstention_contract() -> None:
    parsed = parse_qa_decision(
        _decision(answer=None, decision="abstain", confidence=0.0, citations=[]),
        expected_agent_id="literal_evidence",
        allowed_evidence_ids=(),
    )
    assert parsed.decision == "abstain"

    with pytest.raises(ValueError, match="abstain requires"):
        parse_qa_decision(
            _decision(answer=None, decision="abstain", confidence=0.2, citations=[]),
            expected_agent_id="literal_evidence",
            allowed_evidence_ids=(),
        )


def _record(condition: str, decision: dict[str, object]) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "qid": "qid-1",
        "label": True,
        "agent_id": "literal_evidence",
        "condition": condition,
        "success": True,
        "first_pass_valid": True,
        "attempts": [
            {
                "cache_hit": False,
                "cache_key": "key",
                "http_status": 200,
                "request_bytes": 10,
                "response_bytes": 10,
                "latency_seconds": 0.1,
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
                "parse_error": None,
                "transport_error": None,
            }
        ],
        "decision": decision,
        "final_error": None,
    }


def test_summary_uses_observable_decision_change_for_responsiveness() -> None:
    answer_yes = {
        "agent_id": "literal_evidence",
        "decision": "answer",
        "answer": "yes",
        "confidence": 0.8,
        "cited_evidence_ids": ["E01"],
    }
    abstain = {
        "agent_id": "literal_evidence",
        "decision": "abstain",
        "answer": None,
        "confidence": 0.0,
        "cited_evidence_ids": [],
    }
    answer_no = {
        "agent_id": "literal_evidence",
        "decision": "answer",
        "answer": "no",
        "confidence": 0.7,
        "cited_evidence_ids": ["E01"],
    }
    records = [
        _record("original", answer_yes),
        _record("remove", abstain),
        _record("reverse", answer_no),
    ]

    summary = summarize_records(records, mode="smoke", expected_examples=1, agent_count=1)

    assert summary["interventions"]["complete_triplets"] == 1
    assert summary["interventions"]["paired_responsiveness"]["estimate"] == 1.0
    assert summary["instrumentation"]["transferred_total_bytes"] == 60
