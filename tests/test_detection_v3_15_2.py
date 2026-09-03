from __future__ import annotations

import json

import pytest

from sp500_forecastability import detection_v3_15_2 as subject


def test_parent_formal_abort_is_outcome_blind_empty_packet_only() -> None:
    audit = subject._formal_abort_audit()
    assert audit["rows"] == 7160
    assert audit["successful"] == 7140
    assert audit["first_pass_valid"] == 7133
    assert audit["all_terminal_failures_are_empty_packet_remove"] is True
    assert audit["record_fields_contain_no_outcomes"] is True


def test_empty_packet_placeholder_citations_are_cleared() -> None:
    payload = {
        "answer": "no",
        "confidence": 0.9,
        "cited_evidence_ids": ["evidence_packet"],
    }
    parsed = subject.parse_ling_decision(
        json.dumps(payload),
        expected_agent_id="minimal_judge",
        allowed_evidence_ids=[],
    )
    assert parsed["answer"] == "no"
    assert parsed["confidence"] == 0.9
    assert parsed["cited_evidence_ids"] == []
    assert "empty_packet_citations_cleared" in parsed["parse_mode"]


def test_nonempty_packet_placeholder_is_still_rejected() -> None:
    payload = {
        "answer": "no",
        "confidence": 0.9,
        "cited_evidence_ids": ["evidence_packet"],
    }
    with pytest.raises(ValueError, match="outside packet"):
        subject.parse_ling_decision(
            json.dumps(payload),
            expected_agent_id="minimal_judge",
            allowed_evidence_ids=["E01"],
        )
