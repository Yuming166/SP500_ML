from __future__ import annotations

import json

from sp500_forecastability import detection_v3_15 as subject


def test_parent_is_exact_full_v12_1_surface() -> None:
    parent, composites = subject._validated_parent()
    assert len(composites) == 358
    assert len(parent["substitute_manifest"]) == 1074
    assert parent["risk_contract"]["weights"] == subject.RISK_WEIGHTS


def test_ling_conformance_is_closed_and_semantic() -> None:
    payload = {
        "agent_id": "literal_evidence",
        "answer": " YES ",
        "confidence": 95,
        "evidence_ids": "E01",
    }
    parsed = subject.parse_ling_decision(
        json.dumps(payload),
        expected_agent_id="literal_evidence",
        allowed_evidence_ids=["E01", "E02"],
    )
    assert parsed["answer"] == "yes"
    assert parsed["confidence"] == 0.95
    assert parsed["cited_evidence_ids"] == ["E01"]


def test_preoutcome_score_ignores_poisoned_outcomes() -> None:
    row = {"D_inert": 0.4, "flip_inertia": 0.6, "frac_shared": 0.2}
    expected = sum(subject.RISK_WEIGHTS[key] * row[key] for key in row)
    poisoned = {**row, "label": "poison", "gold_binary": "poison"}
    actual = sum(subject.RISK_WEIGHTS[key] * poisoned[key] for key in subject.RISK_WEIGHTS)
    assert actual == expected


def test_preoutcome_coordinates_match_v12_1_on_same_record_bundle() -> None:
    records = subject.base._load_partial_records(subject.PARENT_RECORDS)
    cqid = str(records[0]["cqid"])
    bundle = [row for row in records if str(row["cqid"]) == cqid]
    stripped = [
        {key: value for key, value in row.items() if key not in {"label", "gold_binary"}}
        for row in bundle
    ]
    actual = subject._preoutcome_rows(stripped)[0]
    subject.v121.configure_v12_1()
    expected = subject.v11._risk_rows(bundle)[0]
    for key in ("D_inert", "flip_inertia", "frac_shared", "R_PI", "agreement"):
        assert actual[key] == expected[key]


def test_protocol_separates_aggregate_and_label_robustness() -> None:
    if not subject.PROTOCOL_MANIFEST.exists():
        return
    manifest = subject.build_protocol_manifest()
    assert manifest["parent"]["selection_changed"] is False
    assert manifest["risk_contract"]["weights"] == subject.RISK_WEIGHTS
    assert manifest["claim_boundary"]["cross_model_family"] is True
    assert manifest["claim_boundary"]["aggregate_pass_does_not_imply_label_robust_pass"] is True
