from __future__ import annotations

import json

import pytest

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_8 as v38
from sp500_forecastability import recovery_v3_8_2 as amended


def test_empty_citation_string_is_empty_list() -> None:
    result = amended.parse_action_decision(
        '{"answer":"No","confidence":0.0,"cited_evidence_ids":""}',
        ["A00"],
    )
    assert result["answer"] == "no"
    assert result["cited_evidence_ids"] == []
    assert result["parse_mode"] == (
        "v3_8_2_answer_casefold_and_empty_citation_string"
    )


def test_exact_allowed_scalar_citation_is_singleton_list() -> None:
    result = amended.parse_action_decision(
        '{"answer":"yes","confidence":0.5,"evidence_ids":"A00"}',
        ["A00"],
    )
    assert result["cited_evidence_ids"] == ["A00"]
    assert result["parse_mode"] == (
        "v3_8_2_evidence_ids_alias_and_singleton_citation_string"
    )


def test_unknown_scalar_citation_and_ambiguous_alias_remain_rejected() -> None:
    with pytest.raises(TypeError, match="must be a list of strings"):
        amended.parse_action_decision(
            '{"answer":"yes","confidence":0.5,"cited_evidence_ids":"A99"}',
            ["A00"],
        )
    with pytest.raises(ValueError, match="unknown=.*evidence_ids"):
        amended.parse_action_decision(
            '{"answer":"yes","confidence":0.5,"evidence_ids":[],"cited_evidence_ids":[]}',
            ["A00"],
        )


def test_manifest_preserves_router_and_binds_v381_abort() -> None:
    manifest = amended.build_protocol_manifest()
    assert manifest["protocol_version"] == amended.PROTOCOL_VERSION
    assert manifest["source"]["policy"] == {
        "confidence_threshold": 0.8,
        "lexical_threshold": 0.0,
        "unsupported_term_cap": 1,
    }
    amendment = manifest["amendment"]
    assert amendment["fresh_target_cache"] is True
    assert amendment["uses_target_correctness_or_action_outcomes"] is False
    assert amendment["prior_protocol_manifest_sha256"] == base._sha256_path(
        amended.V381_MANIFEST
    )
    assert amendment["runner_script_sha256"] == base._sha256_path(
        amended.RUN_SCRIPT
    )
    assert amendment["prior_abort_audit"]["rows"] == 1040
    assert amendment["prior_abort_audit"]["terminal_failures"] == 1
    assert len(
        amendment["prior_abort_audit"]["failed_response_cache_artifacts"]
    ) == 2
    assert manifest["implementation_path"].endswith("recovery_v3_8_2.py")


def test_configured_base_restores_imported_modules() -> None:
    old_version = v38.PROTOCOL_VERSION
    with amended._configured_base():
        assert v38.PROTOCOL_VERSION == amended.PROTOCOL_VERSION
    assert v38.PROTOCOL_VERSION == old_version


def test_v381_abort_audit_is_outcome_blind() -> None:
    audit = amended._audit_v381_abort()
    assert audit["rows"] == 1040
    assert audit["complete_example_bundles"] == 130
    assert audit["terminal_failures"] == 1
    serialized = json.dumps(audit)
    assert "gold_binary" not in serialized
    assert "accuracy" not in serialized
