import json

from sp500_forecastability import recovery_v3_5_1 as pace


def _inference_example() -> dict:
    return {
        "example_id": "pace-x",
        "claim": "Alpha implies beta.",
        "anchor": {
            "retrieval_score": 0.4,
            "evidence": [{"evidence_id": "A00", "text": "Alpha is present."}],
        },
        "candidates": [
            {
                "retrieval_score": 0.8,
                "title_overlap": 0.5,
                "evidence": [{"evidence_id": "C00", "text": "Beta is present."}],
            },
            {
                "retrieval_score": 0.2,
                "title_overlap": 0.0,
                "evidence": [{"evidence_id": "C10", "text": "Beta is absent."}],
            },
        ],
    }


def _baseline_records() -> list[dict]:
    return [
        {
            "example_id": "pace-x",
            "phase": "baseline",
            "action": "KEEP",
            "agent_index": index,
            "decision": {
                "answer": "no",
                "confidence": 0.7 + index * 0.01,
                "cited_evidence_ids": ["A00"],
            },
        }
        for index in range(5)
    ]


def _certificates() -> list[dict]:
    return [
        {
            "example_id": "pace-x",
            "action": action,
            "certificate": {
                "relation": relation,
                "support_strength": support,
                "refute_strength": 1.0 - support,
                "confidence": 0.9,
                "new_evidence_ids": [evidence_id],
                "missing_bridge": False,
            },
        }
        for action, relation, support, evidence_id in (
            ("candidate_0", "supports", 0.9, "C00"),
            ("candidate_1", "refutes", 0.1, "C10"),
        )
    ]


def test_protocol_has_fixed_balanced_development_and_test_sizes() -> None:
    assert pace.EXPECTED_COUNTS == {"development": 600, "test": 500}
    assert pace.EXPECTED_LABELS["test"] == {"Supported": 250, "Refuted": 250}
    assert pace.CERTIFICATE_ACTIONS == ("candidate_0", "candidate_1")


def test_certificate_parser_projects_to_new_root_grounding() -> None:
    content = json.dumps(
        {
            "relation": "support",
            "support_strength": 0.9,
            "refute_strength": 0.1,
            "confidence": 0.8,
            "new_evidence_ids": ["C00"],
            "missing_bridge": False,
        }
    )
    parsed = pace.parse_certificate(content, ["C00"])
    assert parsed["relation"] == "supports"
    projected = pace.parse_certificate(content.replace("C00", "A00"), ["C00"])
    assert projected["new_evidence_ids"] == []
    assert projected["dropped_evidence_ids"] == ["A00"]


def test_certificate_gate_requires_counter_consensus_grounding() -> None:
    rows = pace._certificate_groups(_certificates())
    assert pace._certificate_gate(rows[("pace-x", "candidate_0")], "no")
    assert not pace._certificate_gate(rows[("pace-x", "candidate_1")], "no")


def test_inference_features_do_not_require_gold_or_recovery_outcomes() -> None:
    names, matrix, keys = pace._feature_matrix(
        [_inference_example()], _baseline_records(), _certificates()
    )
    assert matrix.shape == (2, len(names))
    assert keys == [("pace-x", "candidate_0"), ("pace-x", "candidate_1")]
    assert not any(
        fragment in name
        for name in names
        for fragment in ("gold", "label", "outcome", "annotation")
    )


def test_pace_selects_only_a_certified_counter_consensus_action() -> None:
    example = _inference_example()
    grouped = {"pace-x": _baseline_records()}
    certificates = pace._certificate_groups(_certificates())
    predictions = {
        ("pace-x", "candidate_0"): (0.8, 0.1),
        ("pace-x", "candidate_1"): (0.9, 0.0),
    }
    selected = pace._select_policy(
        [example],
        grouped,
        certificates,
        predictions,
        fix_threshold=0.5,
        harm_cap=0.2,
        utility_threshold=0.2,
    )
    assert selected == {"pace-x": "candidate_0"}


def test_ex_fever_selection_passes_all_structural_gates() -> None:
    selection = pace.build_selection()
    report = pace.audit_selection(selection)
    assert report["passed"] is True
    assert report["counts"] == pace.EXPECTED_COUNTS
    assert report["page_root_overlap_between_splits"] == 0
    assert report["test_distinct_page_roots"] == 1_000
