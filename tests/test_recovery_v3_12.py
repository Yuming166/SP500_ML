from __future__ import annotations

import json

from sp500_forecastability import recovery_v3_11 as v311
from sp500_forecastability import recovery_v3_12 as subject


def test_validation_selection_is_fresh_balanced_and_shortcut_safe() -> None:
    selection = json.loads(subject.SELECTION_PATH.read_text(encoding="utf-8"))
    audit = subject.audit_selection(selection)
    assert audit["passed"] is True
    assert audit["counts"] == {"formal": 112}
    assert audit["labels"] == {"Supported": 56, "Refuted": 56}
    assert audit["candidate_0_annotated_fraction"] == 0.5
    assert max(audit["retrieval_role_auc"], 1.0 - audit["retrieval_role_auc"]) <= 0.65


def test_cosign_requires_answer_confidence_and_selected_root_citation() -> None:
    provisional = {"accepted": "candidate_0", "rejected": "candidate_1"}
    diagnostics = {
        "accepted": {"target_action_answer": "yes"},
        "rejected": {"target_action_answer": "yes"},
    }
    teacher_rows = [
        {
            "example_id": "accepted",
            "decision": {
                "answer": "yes",
                "confidence": 0.9,
                "cited_evidence_ids": ["C000"],
            },
            "allowed_selected_packet_evidence_ids": ["C000"],
        },
        {
            "example_id": "rejected",
            "decision": {
                "answer": "yes",
                "confidence": 0.9,
                "cited_evidence_ids": ["A00"],
            },
            "allowed_selected_packet_evidence_ids": ["C100"],
        },
    ]
    selected, detail = subject._cosign_routes(provisional, diagnostics, teacher_rows)
    assert selected == {"accepted": "candidate_0", "rejected": "KEEP"}
    assert detail["accepted"]["gate_components"] == {
        "teacher_same_answer": True,
        "teacher_confidence": True,
        "teacher_cites_selected_root": True,
    }
    assert detail["rejected"]["gate_components"]["teacher_cites_selected_root"] is False


def test_target_configuration_restores_v311_globals() -> None:
    old = (v311.PROTOCOL_VERSION, v311.TARGET_MODEL, v311.TARGET_ENDPOINT)
    with subject._configured_target():
        assert v311.PROTOCOL_VERSION == subject.PROTOCOL_VERSION
        assert v311.TARGET_MODEL == subject.TARGET_MODEL
        assert v311.TARGET_ENDPOINT == subject.TARGET_ENDPOINT
    assert (v311.PROTOCOL_VERSION, v311.TARGET_MODEL, v311.TARGET_ENDPOINT) == old


def test_development_models_have_nontrivial_zero_harm_cosigned_routes() -> None:
    subject.validate_router_manifest()
    diagnostics = json.loads(subject.ROUTER_MANIFEST.read_text(encoding="utf-8"))[
        "development_diagnostics"
    ]
    for metrics in diagnostics["model_metrics"].values():
        assert metrics["fixes"] >= 5
        assert metrics["harms"] == 0
        assert min(metrics["by_native_label_net_gain"].values()) >= 0
