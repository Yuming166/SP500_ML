from __future__ import annotations

import json

import numpy as np

from sp500_forecastability import recovery_v3_11 as subject


def _action_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for agent_index in range(5):
        rows.append(
            {
                "example_id": "example-1",
                "phase": "baseline",
                "action": "KEEP",
                "agent_index": agent_index,
                "success": True,
                "decision": {
                    "answer": "no",
                    "confidence": 0.9,
                    "cited_evidence_ids": ["A00"],
                },
                "gold_binary": "poison",
                "annotated_evidence_ids": ["poison"],
            }
        )
    for action, answer in (
        ("candidate_0", "yes"),
        ("candidate_1", "no"),
        ("both", "yes"),
    ):
        rows.append(
            {
                "example_id": "example-1",
                "phase": "recovery",
                "action": action,
                "agent_index": None,
                "success": True,
                "decision": {
                    "answer": answer,
                    "confidence": 0.9,
                    "cited_evidence_ids": ["C000"],
                },
                "gold_binary": "poison",
                "packet_contains_annotated_root": "poison",
                "annotated_evidence_ids": ["poison"],
            }
        )
    return rows


def test_frozen_selection_passes_structural_audit() -> None:
    selection = json.loads(subject.SELECTION_PATH.read_text(encoding="utf-8"))
    audit = subject.audit_selection(selection)
    assert audit["passed"] is True
    assert audit["counts"] == {"formal": 188}
    assert audit["score_direction_counts"] == {
        "Refuted:distractor_at_least_annotated": 47,
        "Refuted:distractor_below_annotated": 47,
        "Supported:distractor_at_least_annotated": 47,
        "Supported:distractor_below_annotated": 47,
    }


def test_formal_embedding_artifact_contains_only_preoutcome_fields() -> None:
    subject._validate_router_inputs()
    arrays = np.load(subject.ROUTER_INPUTS)
    assert set(arrays.files) == {
        "example_ids",
        "splits",
        "scores",
        "relation_vectors",
    }


def test_route_selection_ignores_poisoned_outcome_and_annotation_fields() -> None:
    selected, diagnostics = subject._select_routes(
        ["example-1"],
        np.asarray([[0.8, 0.2]]),
        np.zeros((1, 2, 4)),
        _action_rows(),
        np.asarray([[0.9, 0.1]]),
    )
    assert selected == {"example-1": "candidate_0"}
    assert diagnostics["example-1"]["gate_components"] == {
        "high_consensus": True,
        "provenance_margin": True,
        "relation_margin": True,
        "target_action_confidence": True,
        "changes_consensus": True,
        "target_relation_agreement": True,
    }


def test_serialized_relation_head_has_expected_dimension_and_boundary() -> None:
    subject.validate_router_manifest()
    head = json.loads(subject.ROUTER_HEAD.read_text(encoding="utf-8"))
    assert len(head["coefficient"]) == 1024
    assert head["negative_class"] == "Refuted"
    assert head["positive_class"] == "Supported"
    assert subject.PROVENANCE_SCORE_MARGIN == 0.30
    assert subject.RELATION_CONFIDENCE_MARGIN == 0.10
    assert subject.TARGET_ACTION_CONFIDENCE == 0.80
