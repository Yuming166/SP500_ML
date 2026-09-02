from __future__ import annotations

import json

import pytest

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_6_2 as v362
from sp500_forecastability import recovery_v3_8 as v38
from sp500_forecastability import recovery_v3_8_3 as v383
from sp500_forecastability import recovery_v3_9 as finr1


def test_manifest_freezes_finr1_and_original_elar_policy() -> None:
    manifest = finr1.build_protocol_manifest()
    assert manifest["protocol_version"] == finr1.PROTOCOL_VERSION
    assert manifest["target"]["model"] == "Fin-R1"
    assert manifest["target"]["architecture"] == "Qwen2ForCausalLM"
    assert manifest["target"]["runtime"]["quantization"] == "none"
    assert manifest["source"]["policy"] == {
        "confidence_threshold": 0.8,
        "lexical_threshold": 0.0,
        "unsupported_term_cap": 1,
    }
    assert manifest["registered_continuation"][
        "ling_outcomes_used_for_finr1_fit_or_selection"
    ] is False
    assert manifest["evaluation_diagnostic_fix"]["primary_elar_changed"] is False
    assert manifest["runner_script_sha256"] == base._sha256_path(finr1.RUN_SCRIPT)


def test_configuration_scopes_target_client_and_parser() -> None:
    old_model = v38.TARGET_MODEL
    old_endpoint = v38.TARGET_ENDPOINT
    old_client = v38.CrossModelChatClient
    old_parser = v362.parse_action_decision
    with finr1._configured_base():
        assert v38.TARGET_MODEL == "Fin-R1"
        assert v38.TARGET_ENDPOINT == finr1.TARGET_ENDPOINT
        assert v38.CrossModelChatClient is finr1.FinR1ChatClient
        assert v362.parse_action_decision is v383.parse_action_decision
    assert v38.TARGET_MODEL == old_model
    assert v38.TARGET_ENDPOINT == old_endpoint
    assert v38.CrossModelChatClient is old_client
    assert v362.parse_action_decision is old_parser


def test_finr1_client_rejects_target_drift(tmp_path) -> None:
    with finr1._configured_base():
        client = finr1.FinR1ChatClient(cache_dir=tmp_path)
        assert client.model == "Fin-R1"
        with pytest.raises(ValueError, match="endpoint and model are frozen"):
            finr1.FinR1ChatClient(cache_dir=tmp_path, model="other")


def test_endpoint_inventory_uses_finr1_scope(monkeypatch) -> None:
    observed = {}

    def fake_inventory() -> set[str]:
        observed["model"] = v38.TARGET_MODEL
        observed["endpoint"] = v38.TARGET_ENDPOINT
        return {v38.TARGET_MODEL}

    monkeypatch.setattr(v38, "endpoint_model_ids", fake_inventory)
    with finr1._configured_base():
        assert v38.endpoint_model_ids() == {"Fin-R1"}
    assert observed == {
        "model": "Fin-R1",
        "endpoint": finr1.TARGET_ENDPOINT,
    }


def test_transport_parser_is_identical_to_final_closed_conformance() -> None:
    content = '{"answer":"No","confidence":95,"evidence_ids":"A00"}'
    with finr1._configured_base():
        actual = v362.parse_action_decision(content, ["A00"])
    expected = v383.parse_action_decision(content, ["A00"])
    assert actual == expected


def test_manifest_contains_no_ling_metric_values() -> None:
    manifest = finr1.build_protocol_manifest()
    continuation = json.dumps(manifest["registered_continuation"])
    assert "accuracy" not in continuation
    assert "gain" not in continuation
    assert "verdict" not in continuation
