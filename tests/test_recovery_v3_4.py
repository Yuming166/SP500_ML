import numpy as np

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_4 as jury


def _example() -> dict:
    return {
        "example_id": "x",
        "claim": "A claim",
        "anchor": {"evidence": [{"evidence_id": "A00", "text": "anchor"}]},
        "candidates": [
            {"evidence": [{"evidence_id": "C00", "text": "first"}]},
            {"evidence": [{"evidence_id": "C10", "text": "second"}]},
        ],
    }


def _baseline(answer: str) -> list[dict]:
    return [
        {
            "phase": "baseline",
            "agent_index": index,
            "decision": {"answer": answer, "confidence": 0.9, "cited_evidence_ids": ["A00"]},
        }
        for index in range(5)
    ]


def _stance(candidate_0: list[float], candidate_1: list[float]) -> dict:
    def rows(opposition: list[float]) -> np.ndarray:
        return np.asarray([[0.05, value, value] for value in opposition])

    return {
        ("x", "anchor"): rows([0.1] * 5),
        ("x", "candidate_0"): rows(candidate_0),
        ("x", "candidate_1"): rows(candidate_1),
    }


def test_frozen_jury_policy_constants() -> None:
    assert jury.ENSEMBLE_SIZE == 5
    assert jury.OPPOSITION_THRESHOLD == 0.4
    assert jury.DELTA_THRESHOLD == -0.2
    assert jury.DISPERSION_MULTIPLIER == 1.0
    assert jury.QUORUM == {"no": 0.8, "yes": 1.0}


def test_external_eligibility_is_fixed() -> None:
    examples = jury.load_eligible()
    assert len(examples) == 483
    assert sum(row["label"] == "Supported" for row in examples) == 361
    assert sum(row["label"] == "Refuted" for row in examples) == 122
    assert all(len(row["annotated_evidence"]) >= 2 for row in examples)


def test_inference_packet_builder_does_not_require_a_label() -> None:
    packets = jury._inference_packets([_example()])
    assert [row[1] for row in packets] == ["anchor", "candidate_0", "candidate_1"]


def test_jury_routes_with_four_of_five_votes_for_no_consensus() -> None:
    example = _example()
    stance = _stance([0.8, 0.8, 0.8, 0.8, 0.3], [0.2, 0.2, 0.2, 0.2, 0.4])
    selected, diagnostics = jury._jury_policy(
        [example], {"x": _baseline("no")}, stance
    )
    assert selected["x"] == "candidate_0"
    assert diagnostics["x"]["jury_agreement"] == 0.8


def test_yes_consensus_requires_unanimous_candidate_vote() -> None:
    example = _example()
    stance = _stance([0.8, 0.8, 0.8, 0.8, 0.3], [0.2, 0.2, 0.2, 0.2, 0.4])
    selected, diagnostics = jury._jury_policy(
        [example], {"x": _baseline("yes")}, stance
    )
    assert selected["x"] == "KEEP"
    assert "jury_disagreement" in diagnostics["x"]["veto_reasons"]


def test_dispersion_can_veto_an_unanimous_candidate() -> None:
    example = _example()
    stance = _stance([1.0, 1.0, 1.0, 1.0, 0.01], [0.0] * 5)
    selected, diagnostics = jury._jury_policy(
        [example], {"x": _baseline("no")}, stance
    )
    assert selected["x"] == "KEEP"
    assert "weak_counter_consensus_witness" in diagnostics["x"]["veto_reasons"]


def test_publisher_groups_are_not_split_and_folds_are_balanced() -> None:
    examples = [
        {"example_id": f"{publisher}-{index}", "fact_check_root": publisher}
        for publisher, size in (("a", 4), ("b", 3), ("c", 2), ("d", 2), ("e", 1), ("f", 1))
        for index in range(size)
    ]
    assignments = jury._publisher_folds(examples)
    for publisher in {row["fact_check_root"] for row in examples}:
        assert len({assignments[row["example_id"]] for row in examples if row["fact_check_root"] == publisher}) == 1
    sizes = [sum(fold == index for fold in assignments.values()) for index in range(5)]
    assert max(sizes) - min(sizes) <= 2


def test_stance_class_order_is_unchanged() -> None:
    assert base.STANCE_CLASSES == ("irrelevant", "refutes", "supports")
