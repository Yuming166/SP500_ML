import json
from pathlib import Path

import numpy as np

from sp500_forecastability import recovery_v3 as recovery


def _frozen_example() -> dict[str, object]:
    selection = json.loads(
        Path("results/recovery_v3_2/selection_manifest.json").read_text(encoding="utf-8")
    )
    return selection["examples"][0]


def test_wayback_source_root_is_unwrapped() -> None:
    url = "https://web.archive.org/web/20200101000000/https://www.bbc.co.uk/news/example"
    assert recovery.canonical_source_root(url) == "bbc.co.uk"


def test_frozen_selection_passes_all_precall_gates() -> None:
    selection = json.loads(
        Path("results/recovery_v3_2/selection_manifest.json").read_text(encoding="utf-8")
    )
    report = recovery.audit_selection(selection)
    assert report["passed"] is True
    assert report["counts"] == recovery.EXPECTED_COUNTS
    assert not any(report["claim_overlap"].values())


def test_prompt_hides_outcome_and_provenance_role_fields() -> None:
    example = _frozen_example()
    messages = [
        *recovery.build_baseline_messages(example, 0),
        *recovery.build_recovery_messages(example, "candidate_0", "no"),
    ]
    prompt = "\n".join(message["content"] for message in messages)
    assert str(example["label"]) not in prompt
    assert str(example["fact_check_root"]) not in prompt
    assert "gold_binary" not in prompt
    assert "annotation_role" not in prompt
    assert "held_out_annotated_root" not in prompt


def test_candidate_order_is_balanced_in_every_partition() -> None:
    selection = json.loads(
        Path("results/recovery_v3_2/selection_manifest.json").read_text(encoding="utf-8")
    )
    report = recovery.audit_selection(selection)
    assert all(
        0.45 <= fraction <= 0.55 for fraction in report["candidate_0_annotated_fraction"].values()
    )


def test_budget_matching_respects_root_cost() -> None:
    examples = [
        {"example_id": "one", "candidates": [{"retrieval_score": 0.1}]},
        {"example_id": "two", "candidates": [{"retrieval_score": 0.2}]},
    ]
    selected = recovery._truncate_to_budget(
        examples,
        {"one": "both", "two": "candidate_0"},
        budget=1,
        name="fixed_both",
    )
    spent = sum(
        0 if action == "KEEP" else 2 if action == "both" else 1 for action in selected.values()
    )
    assert spent <= 1


def test_retrieval_baseline_spends_budget_on_highest_score_first() -> None:
    examples = [
        {"example_id": "low", "candidates": [{"retrieval_score": 0.1}]},
        {"example_id": "high", "candidates": [{"retrieval_score": 0.9}]},
    ]
    selected = recovery._truncate_to_budget(
        examples,
        {"low": "candidate_0", "high": "candidate_0"},
        budget=1,
        name="retrieval_score",
    )
    assert selected == {"low": "KEEP", "high": "candidate_0"}


def test_cape_cost_curve_uses_predicted_utility_without_outcomes() -> None:
    examples = [{"example_id": "low"}, {"example_id": "high"}]
    selected = {"low": "candidate_0", "high": "candidate_0"}
    predictions = {
        ("low", "candidate_0"): (0.1, 0.0),
        ("high", "candidate_0"): (0.9, 0.0),
    }
    budgeted = recovery._truncate_cape_to_budget(examples, selected, predictions, budget=1)
    assert budgeted == {"low": "KEEP", "high": "candidate_0"}


def test_smoke_records_form_complete_action_matrix() -> None:
    selection = json.loads(
        Path("results/recovery_v3_2/selection_manifest.json").read_text(encoding="utf-8")
    )
    examples = [row for row in selection["examples"] if row["split"] == "train"][:2]
    records = recovery._load_jsonl(Path("results/recovery_v3_2/smoke/records.jsonl"))
    recovery._validate_action_matrix(examples, records, split="train")


def test_route_can_be_selected_before_recovery_outcomes_are_revealed() -> None:
    example = _frozen_example()
    all_records = recovery._load_jsonl(Path("results/recovery_v3_2/smoke/records.jsonl"))
    baseline = [
        row
        for row in all_records
        if row["example_id"] == example["example_id"] and row["phase"] == "baseline"
    ]
    stance = {
        (str(example["example_id"]), role): np.asarray([0.2, 0.4, 0.4])
        for role in ("anchor", "candidate_0", "candidate_1")
    }
    names, matrix, keys = recovery._feature_matrix([example], baseline, stance)
    predictions = {key: (0.2, 0.0) for key in keys}
    selected = recovery._select_policy(
        [example],
        {str(example["example_id"]): baseline},
        predictions,
        threshold_yes=0.0,
        threshold_no=0.0,
        harm_cap=0.05,
    )
    assert len(names) == matrix.shape[1]
    assert selected[str(example["example_id"])] in recovery.RECOVERY_ACTIONS


def test_stance_encoder_supports_three_class_probabilities() -> None:
    texts = [
        "claim alpha evidence irrelevant",
        "claim alpha evidence irrelevant",
        "claim beta evidence refutes",
        "claim beta evidence refutes",
        "claim gamma evidence supports",
        "claim gamma evidence supports",
    ]
    labels = ["irrelevant", "irrelevant", "refutes", "refutes", "supports", "supports"]
    model = recovery._new_stance_model()
    model.fit(texts, labels)
    probabilities = recovery._standard_stance_probabilities(model, texts)
    assert probabilities.shape == (6, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
