import json
from pathlib import Path

import pytest

from sp500_forecastability import recovery_v1 as recovery


def _example() -> recovery.RecoveryExample:
    return recovery.RecoveryExample(
        cqid="q1",
        source_root="root-a",
        question="Is the statement supported?",
        evidence=(("E01", "The statement is supported."), ("E02", "More context."), ("E03", "Background.")),
    )


def _context() -> dict[str, object]:
    return {
        "original_consensus": "no",
        "original_agreement": 1.0,
        "D_inert": 1.0,
        "flip_inertia": 1.0,
        "frac_shared": 0.8,
        "R_PI": 0.88,
        "ledger": [],
    }


def test_recovery_prompt_never_accepts_outcome_fields() -> None:
    context = _context()
    context["gold_binary"] = 1
    with pytest.raises(ValueError, match="outcome fields"):
        recovery.build_messages(_example(), context, "full_evidence")


def test_full_evidence_prompt_is_blind_to_old_consensus() -> None:
    messages = recovery.build_messages(_example(), _context(), "full_evidence")
    content = "\n".join(message["content"] for message in messages)
    assert "previous team consensus was" not in content
    assert "The statement is supported." in content


def test_parse_recovery_decision_enforces_action_and_citations() -> None:
    valid = json.dumps({
        "action_id": "full_evidence",
        "answer": "yes",
        "confidence": 0.8,
        "cited_evidence_ids": ["E01"],
    })
    parsed = recovery.parse_recovery_decision(
        valid, expected_action="full_evidence", allowed_evidence_ids=["E01", "E02"],
    )
    assert parsed["answer"] == "yes"
    invalid = valid.replace('"E01"', '"OUTSIDE"')
    with pytest.raises(ValueError, match="outside"):
        recovery.parse_recovery_decision(
            invalid, expected_action="full_evidence", allowed_evidence_ids=["E01", "E02"],
        )


def test_root_fold_is_deterministic_and_bounded() -> None:
    assert recovery.root_fold("root-a") == recovery.root_fold("root-a")
    assert 0 <= recovery.root_fold("root-a") < recovery.N_FOLDS


def test_manifest_preserves_parent_hashes() -> None:
    manifest = recovery.build_manifest()
    assert manifest["parent"]["records_sha256"] == recovery._sha256_path(recovery.PARENT_RECORDS)
    assert manifest["expected_questions"] == 300
    assert manifest["expected_recovery_calls"] == 900
    assert manifest["claim_boundary"]["confirmatory"] is False
    assert manifest["claim_boundary"]["provenance_disjoint_recovery"] is False


def test_protocol_document_exists() -> None:
    path = Path("docs/recovery_v1_protocol.md")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "does not alter V12.1" in text
    assert "does not test provenance-disjoint" in text
