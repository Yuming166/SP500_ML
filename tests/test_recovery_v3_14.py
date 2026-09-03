from __future__ import annotations

import json

import numpy as np

from sp500_forecastability import recovery_v3_14 as subject


def test_frozen_selection_passes_all_structural_gates() -> None:
    selection = json.loads(subject.SELECTION_PATH.read_text(encoding="utf-8"))
    audit = subject.audit_selection(selection)
    assert audit["passed"] is True
    assert audit["counts"] == {"formal": 300}
    assert audit["labels"] == {"Supported": 150, "NotSupported": 150}
    assert audit["direction_counts"] == {
        "NotSupported:distractor_at_least_annotated": 75,
        "NotSupported:distractor_below_annotated": 75,
        "Supported:distractor_at_least_annotated": 75,
        "Supported:distractor_below_annotated": 75,
    }
    assert audit["retrieval_role_auc"] <= subject.MAX_ORIENTED_ROLE_AUC


def test_router_is_exact_target_free_v311_transfer() -> None:
    manifest = subject.build_router_manifest()
    source = json.loads(subject.FROZEN_ROUTER_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["relation_head_sha256"] == source["relation_head_sha256"]
    assert manifest["thresholds"] == source["thresholds"]
    assert manifest["target_model_absent_from_training_models"] is True
    assert manifest["no_refit"] is True
    assert manifest["no_target_calibration"] is True


def test_formal_embedding_artifact_is_inference_only() -> None:
    if not subject.ROUTER_INPUTS.exists():
        return
    subject._validate_router_inputs()
    arrays = np.load(subject.ROUTER_INPUTS)
    assert set(arrays.files) == {
        "example_ids",
        "splits",
        "scores",
        "relation_vectors",
    }


def test_protocol_claim_boundary_is_zero_shot_but_not_cross_family() -> None:
    if not subject.PROTOCOL_MANIFEST.exists():
        return
    manifest = subject.build_protocol_manifest()
    boundary = manifest["claim_boundary"]
    assert boundary["target_fit_or_calibration"] is False
    assert boundary["target_model_used_for_method_selection"] is False
    assert boundary["cross_dataset_transfer"] is True
    assert boundary["same_qwen_model_family"] is True
    assert boundary["cross_family_transfer"] is False
