from __future__ import annotations

import json

import pytest

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_8 as v38
from sp500_forecastability import recovery_v3_8_3 as v383
from sp500_forecastability import recovery_v3_9 as v39
from sp500_forecastability import recovery_v3_9_1 as amended


def test_leading_json_with_trailing_prose_is_accepted() -> None:
    result = amended.parse_action_decision(
        '{"answer":"yes","confidence":0.9,"cited_evidence_ids":["A00"]}\n'
        "Explanation: A00 supports the answer.",
        ["A00"],
    )
    assert result["answer"] == "yes"
    assert result["cited_evidence_ids"] == ["A00"]
    assert result["parse_mode"] == "v3_9_1_leading_json_with_trailing_text"


def test_prior_conformance_is_applied_to_leading_json() -> None:
    result = amended.parse_action_decision(
        '{"answer":"No","confidence":95,"evidence_ids":"A00"}\nReason.',
        ["A00"],
    )
    assert result["answer"] == "no"
    assert result["confidence"] == pytest.approx(0.95)
    assert result["cited_evidence_ids"] == ["A00"]
    assert "v3_8_3_answer_casefold" in result["parse_mode"]


@pytest.mark.parametrize(
    "content",
    [
        'Before {"answer":"yes","confidence":1,"cited_evidence_ids":[]}',
        '{"answer":"yes","confidence":1,"cited_evidence_ids":[]} {"x":1}',
        '{"answer":"yes","confidence":1,"cited_evidence_ids":[]',
    ],
)
def test_unsafe_salvage_cases_remain_rejected(content: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        amended.parse_action_decision(content, [])


def test_smoke_abort_audit_is_outcome_blind_and_all_replayable() -> None:
    audit = amended._audit_v39_smoke_abort()
    assert audit["responses"] == 16
    assert audit["strict_json"] == 4
    assert audit["leading_json_with_trailing_text"] == 12
    assert audit["formal_target_calls"] == 0
    serialized = json.dumps(audit)
    assert "accuracy" not in serialized
    assert "gold" not in serialized


def test_manifest_preserves_finr1_and_binds_prior_abort() -> None:
    manifest = amended.build_protocol_manifest()
    assert manifest["protocol_version"] == amended.PROTOCOL_VERSION
    assert manifest["target"]["model"] == "Fin-R1"
    assert manifest["source"]["policy"] == {
        "confidence_threshold": 0.8,
        "lexical_threshold": 0.0,
        "unsupported_term_cap": 1,
    }
    transport = manifest["transport_amendment"]
    assert transport["no_further_parser_extension_after_freeze"] is True
    assert transport["uses_target_correctness_or_action_outcomes"] is False
    assert transport["prior_protocol_manifest_sha256"] == base._sha256_path(
        amended.V39_MANIFEST
    )
    assert manifest["implementation_path"].endswith("recovery_v3_9_1.py")


def test_configuration_is_scoped_and_restored() -> None:
    old_version = v39.PROTOCOL_VERSION
    old_parser = v383.parse_action_decision
    with amended._configured_base():
        assert v38.PROTOCOL_VERSION == amended.PROTOCOL_VERSION
        assert v38.TARGET_MODEL == "Fin-R1"
        assert v383.parse_action_decision is amended.parse_action_decision
    assert v39.PROTOCOL_VERSION == old_version
    assert v383.parse_action_decision is old_parser
