from __future__ import annotations

import json

import pytest

from sp500_forecastability import detection_v3_15_1 as subject


def test_parent_smoke_abort_has_only_missing_agent_id() -> None:
    audit = subject._smoke_abort_audit()
    assert audit["rows"] == 80
    assert audit["successful"] == 29
    assert audit["terminal_error_counts"] == {
        "ValueError: missing decision fields: ['agent_id']": 51
    }
    assert audit["record_fields_contain_no_outcomes"] is True


def test_missing_agent_id_is_inserted_from_environment() -> None:
    payload = {
        "answer": "yes",
        "confidence": 0.9,
        "cited_evidence_ids": ["E01"],
    }
    parsed = subject.parse_ling_decision(
        json.dumps(payload),
        expected_agent_id="literal_evidence",
        allowed_evidence_ids=["E01"],
    )
    assert parsed["agent_id"] == "literal_evidence"
    assert "insert_expected_agent_id" in parsed["parse_mode"]


def test_present_wrong_agent_id_is_still_rejected() -> None:
    payload = {
        "agent_id": "wrong",
        "answer": "yes",
        "confidence": 0.9,
        "cited_evidence_ids": ["E01"],
    }
    with pytest.raises(ValueError, match="agent_id"):
        subject.parse_ling_decision(
            json.dumps(payload),
            expected_agent_id="literal_evidence",
            allowed_evidence_ids=["E01"],
        )
