from __future__ import annotations

import json

import pytest

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_6_2 as v362
from sp500_forecastability import recovery_v3_7 as frozen
from sp500_forecastability import recovery_v3_8 as v38
from sp500_forecastability import recovery_v3_8_3 as v383
from sp500_forecastability import recovery_v3_9_2 as amended


def test_certificate_envelope_keeps_semantic_validation() -> None:
    content = json.dumps(
        {
            "atomic_checks": [
                {
                    "claim_span": "Alpha is blue",
                    "status": "supported",
                    "evidence_ids": ["C000"],
                }
            ],
            "coverage_complete": True,
            "confidence": 0.9,
        }
    ) + "\nExplanation: the JSON above is authoritative."
    result = amended.parse_certificate(
        content,
        "Alpha is blue.",
        ["A00", "C000"],
        ["C000"],
    )
    assert result["confidence"] == 0.9
    assert result["transport_parse_mode"] == "leading_json_with_trailing_text"


def test_ledger_envelope_calls_unchanged_parser_on_leading_json(monkeypatch) -> None:
    observed = {}

    def fake_parser(content, example, action, certificate, consensus):
        observed["payload"] = json.loads(content)
        observed["arguments"] = (example, action, certificate, consensus)
        return {"entries": [], "challenge": {}}

    monkeypatch.setattr(amended, "_ORIGINAL_LEDGER_PARSER", fake_parser)
    result = amended.parse_ledger(
        '{"entries":[],"challenge":{}}\nExplanation only.',
        {"example_id": "x"},
        "candidate_0",
        {"atomic_checks": []},
        "no",
    )
    assert observed["payload"] == {"entries": [], "challenge": {}}
    assert result["transport_parse_mode"] == "leading_json_with_trailing_text"


@pytest.mark.parametrize(
    "content",
    [
        'Before {"x":1}',
        '{"x":1} {"y":2}',
        '{"x":1',
        '{"x":1}',
    ],
)
def test_envelope_rejects_unsafe_or_unneeded_fallback(content: str) -> None:
    original = ValueError("strict failure")
    with pytest.raises(ValueError, match="strict failure"):
        amended._leading_mapping_json(content, original)


def test_attempt_stats_records_artifact_envelope_modes() -> None:
    stats = amended._attempt_stats(
        [
            {
                "success": True,
                "first_pass_valid": True,
                "attempts": [],
                "certificate": {
                    "transport_parse_mode": "leading_json_with_trailing_text"
                },
            },
            {
                "success": True,
                "first_pass_valid": True,
                "attempts": [],
                "certificate": {"atomic_checks": []},
            },
        ]
    )
    assert stats["artifact_parse_modes"] == {
        "leading_json_with_trailing_text": 1,
        "strict": 1,
    }


def test_smoke_abort_audit_replays_all_envelope_failures() -> None:
    audit = amended._audit_v391_smoke_abort()
    assert audit["actions"] == audit["successful_actions"] == 16
    assert audit["certificates"] == 4
    assert audit["successful_certificates"] == 2
    assert audit["failed_attempts_replayed_with_envelope_only"] == 4
    serialized = json.dumps(audit)
    assert "accuracy" not in serialized
    assert "gold" not in serialized


def test_manifest_preserves_policy_and_binds_uniform_amendment() -> None:
    manifest = amended.build_protocol_manifest()
    assert manifest["protocol_version"] == amended.PROTOCOL_VERSION
    assert manifest["target"]["model"] == "Fin-R1"
    assert manifest["source"]["policy"] == {
        "confidence_threshold": 0.8,
        "lexical_threshold": 0.0,
        "unsupported_term_cap": 1,
    }
    envelope = manifest["uniform_envelope_amendment"]
    assert envelope["certificate_or_ledger_semantic_validator_changed"] is False
    assert envelope["no_further_transport_extension_after_freeze"] is True
    assert envelope["prior_protocol_manifest_sha256"] == base._sha256_path(
        amended.V391_MANIFEST
    )
    assert manifest["implementation_path"].endswith("recovery_v3_9_2.py")


def test_configuration_scopes_and_restores_all_parsers() -> None:
    old_certificate = v362.parse_certificate
    old_ledger = frozen.parse_ledger
    old_stats = v383._attempt_stats
    with amended._configured_base():
        assert v38.PROTOCOL_VERSION == amended.PROTOCOL_VERSION
        assert v362.parse_certificate is amended.parse_certificate
        assert frozen.parse_ledger is amended.parse_ledger
        assert v383._attempt_stats is amended._attempt_stats
    assert v362.parse_certificate is old_certificate
    assert frozen.parse_ledger is old_ledger
    assert v383._attempt_stats is old_stats
