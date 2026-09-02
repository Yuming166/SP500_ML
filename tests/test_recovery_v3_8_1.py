from __future__ import annotations

import json

import pytest

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_8 as v38
from sp500_forecastability import recovery_v3_8_1 as amended


def test_answer_casefold_is_semantic_preserving() -> None:
    result = amended.parse_action_decision(
        '{"answer":"No","confidence":0.7,"cited_evidence_ids":["A00"]}',
        ["A00"],
    )
    assert result["answer"] == "no"
    assert result["parse_mode"] == "v3_8_1_answer_casefold"


def test_unambiguous_evidence_alias_is_accepted() -> None:
    result = amended.parse_action_decision(
        '{"answer":"yes","confidence":0.7,"evidence_ids":["A00"]}',
        ["A00"],
    )
    assert result["cited_evidence_ids"] == ["A00"]
    assert result["parse_mode"] == "v3_8_1_evidence_ids_alias"


def test_ambiguous_alias_and_unknown_values_remain_rejected() -> None:
    with pytest.raises(ValueError, match="unknown=.*evidence_ids"):
        amended.parse_action_decision(
            '{"answer":"yes","confidence":0.7,"evidence_ids":[],"cited_evidence_ids":[]}',
            [],
        )
    with pytest.raises(ValueError, match="answer must be yes or no"):
        amended.parse_action_decision(
            '{"answer":"maybe","confidence":0.7,"cited_evidence_ids":[]}',
            [],
        )


def test_manifest_preserves_v38_and_freezes_amendment() -> None:
    manifest = amended.build_protocol_manifest()
    assert manifest["protocol_version"] == amended.PROTOCOL_VERSION
    assert manifest["source"]["policy"] == {
        "confidence_threshold": 0.8,
        "lexical_threshold": 0.0,
        "unsupported_term_cap": 1,
    }
    assert manifest["amendment"]["fresh_target_cache"] is True
    assert manifest["amendment"]["uses_target_correctness_or_action_outcomes"] is False
    assert manifest["amendment"]["prior_protocol_manifest_sha256"] == base._sha256_path(
        amended.V38_MANIFEST
    )
    assert manifest["amendment"]["runner_script_sha256"] == base._sha256_path(
        amended.RUN_SCRIPT
    )
    assert manifest["amendment"]["prior_abort_audit"]["rows"] == 640
    assert manifest["amendment"]["prior_abort_audit"]["terminal_failures"] == 3
    assert manifest["implementation_path"].endswith("recovery_v3_8_1.py")


def test_configured_base_restores_imported_modules() -> None:
    old_version = v38.PROTOCOL_VERSION
    with amended._configured_base():
        assert v38.PROTOCOL_VERSION == amended.PROTOCOL_VERSION
    assert v38.PROTOCOL_VERSION == old_version


def test_attempt_stats_reports_normalization_without_outcomes() -> None:
    stats = amended._attempt_stats(
        [
            {
                "success": True,
                "first_pass_valid": True,
                "attempts": [],
                "decision": {"parse_mode": "v3_8_1_answer_casefold"},
                "gold_binary": 1,
            }
        ]
    )
    assert stats["decision_parse_modes"] == {"v3_8_1_answer_casefold": 1}
    assert "gold_binary" not in json.dumps(stats)


def test_v38_abort_audit_is_outcome_blind_and_content_addressed() -> None:
    audit = amended._audit_v38_abort()
    assert audit["rows"] == 640
    assert audit["complete_example_bundles"] == 80
    assert audit["terminal_failures"] == 3
    assert audit["partial_action_records_sha256"] == base._sha256_path(
        amended.V38_PARTIAL
    )
    serialized = json.dumps(audit)
    assert "gold_binary" not in serialized
    assert "accuracy" not in serialized
