from __future__ import annotations

import json
import math

import pytest

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_8 as v38
from sp500_forecastability import recovery_v3_8_3 as amended


def test_percentage_confidence_is_fraction() -> None:
    result = amended.parse_action_decision(
        '{"answer":"yes","confidence":95,"cited_evidence_ids":["A00"]}',
        ["A00"],
    )
    assert result["confidence"] == pytest.approx(0.95)
    assert result["parse_mode"] == "v3_8_3_confidence_percent"


@pytest.mark.parametrize("confidence", [True, "95", -1, 101, math.inf])
def test_unsafe_confidence_values_remain_rejected(confidence: object) -> None:
    content = json.dumps(
        {"answer": "yes", "confidence": confidence, "cited_evidence_ids": []}
    )
    with pytest.raises((TypeError, ValueError)):
        amended.parse_action_decision(content, [])


def test_prior_canonicalizations_are_preserved() -> None:
    result = amended.parse_action_decision(
        '{"answer":"No","confidence":100,"evidence_ids":"A00"}',
        ["A00"],
    )
    assert result["answer"] == "no"
    assert result["confidence"] == 1.0
    assert result["cited_evidence_ids"] == ["A00"]
    assert result["parse_mode"] == (
        "v3_8_3_answer_casefold_and_evidence_ids_alias_and_"
        "singleton_citation_string_and_confidence_percent"
    )


def test_manifest_binds_dev_smoke_and_finality() -> None:
    manifest = amended.build_protocol_manifest()
    assert manifest["protocol_version"] == amended.PROTOCOL_VERSION
    amendment = manifest["amendment"]
    assert amendment["no_further_parser_extension_after_freeze"] is True
    assert amendment["uses_target_correctness_or_action_outcomes"] is False
    assert amendment["prior_protocol_manifest_sha256"] == base._sha256_path(
        amended.V382_MANIFEST
    )
    assert amendment["prior_abort_audit"]["rows"] == 16
    assert amendment["prior_abort_audit"]["development_examples"] == 2
    assert amendment["prior_abort_audit"]["terminal_failures"] == 1
    assert len(
        amendment["prior_abort_audit"]["failed_response_cache_artifacts"]
    ) == 2
    assert manifest["implementation_path"].endswith("recovery_v3_8_3.py")


def test_configured_base_restores_imported_modules() -> None:
    old_version = v38.PROTOCOL_VERSION
    with amended._configured_base():
        assert v38.PROTOCOL_VERSION == amended.PROTOCOL_VERSION
    assert v38.PROTOCOL_VERSION == old_version


def test_v382_abort_audit_is_outcome_blind() -> None:
    audit = amended._audit_v382_abort()
    serialized = json.dumps(audit)
    assert "gold_binary" not in serialized
    assert "accuracy" not in serialized
