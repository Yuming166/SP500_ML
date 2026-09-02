from __future__ import annotations

import json

import pytest

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_8 as v38
from sp500_forecastability import recovery_v3_9 as v39
from sp500_forecastability import recovery_v3_10 as guided
from sp500_forecastability.pilot_llm_v1 import MAX_COMPLETION_TOKENS


def test_artifact_kind_is_bound_to_existing_token_contract() -> None:
    assert guided._artifact_kind(MAX_COMPLETION_TOKENS) == "action"
    assert guided._artifact_kind(512) == "certificate"
    assert guided._artifact_kind(768) == "ledger"
    with pytest.raises(ValueError, match="does not identify"):
        guided._artifact_kind(999)


@pytest.mark.parametrize("kind", ["action", "certificate", "ledger"])
def test_response_format_is_strict_json_schema(kind: str) -> None:
    response_format = guided._response_format(kind)
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


def test_action_schema_prevents_bare_or_unknown_fields() -> None:
    assert guided.ACTION_SCHEMA["properties"]["cited_evidence_ids"] == {
        "type": "array",
        "items": {"type": "string"},
    }
    assert guided.ACTION_SCHEMA["properties"]["answer"]["enum"] == ["yes", "no"]


def test_abort_audit_is_outcome_blind_and_binds_all_failed_attempts() -> None:
    audit = guided._audit_v392_abort()
    assert audit["rows"] == 176
    assert audit["complete_example_bundles"] == 22
    assert audit["terminal_failures"] == 18
    assert audit["failed_attempts"] == audit["invalid_json_attempts"] == 36
    serialized = json.dumps(audit)
    assert "accuracy" not in serialized
    assert "gold" not in serialized


def test_manifest_preserves_elar_and_freezes_guided_interface() -> None:
    manifest = guided.build_protocol_manifest()
    assert manifest["protocol_version"] == guided.PROTOCOL_VERSION
    assert manifest["target"]["model"] == "Fin-R1"
    assert manifest["source"]["policy"] == {
        "confidence_threshold": 0.8,
        "lexical_threshold": 0.0,
        "unsupported_term_cap": 1,
    }
    interface = manifest["schema_constrained_interface"]
    assert interface["semantic_validators_changed"] is False
    assert interface["prompts_or_decoding_parameters_changed"] is False
    assert interface["no_further_schema_or_parser_extension_after_freeze"] is True
    assert interface["prior_protocol_manifest_sha256"] == base._sha256_path(
        guided.V392_MANIFEST
    )
    assert manifest["implementation_path"].endswith("recovery_v3_10.py")


def test_configuration_scopes_guided_client_and_restores() -> None:
    old = v39.FinR1ChatClient
    with guided._configured_base():
        assert v38.PROTOCOL_VERSION == guided.PROTOCOL_VERSION
        assert v38.CrossModelChatClient is guided.GuidedFinR1ChatClient
    assert v39.FinR1ChatClient is old
