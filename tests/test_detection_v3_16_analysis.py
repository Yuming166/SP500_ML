from __future__ import annotations

import json
from pathlib import Path

import pytest

from sp500_forecastability import detection_v3_16_analysis as analysis

RECORDS = Path("results/detection_v3_16_development/calls_1/qwen/development/records.jsonl")


def test_preoutcome_rows_hide_labels_and_label_bearing_ids() -> None:
    records = [json.loads(line) for line in RECORDS.read_text().splitlines()]
    rows = analysis.build_preoutcome_rows(records)
    assert len(rows) == 60
    forbidden = {"gold_label", "label", "correct", "error", "consensus_wrong", "item_id"}
    assert all(not (forbidden & set(row)) for row in rows)
    assert all(
        "support" not in row["opaque_id"] and "refute" not in row["opaque_id"] for row in rows
    )


def test_risk_coordinates_ignore_item_id_semantics() -> None:
    records = [json.loads(line) for line in RECORDS.read_text().splitlines()]
    original = analysis.build_preoutcome_rows(records)
    replacements = {
        item_id: f"opaque-input-{index}"
        for index, item_id in enumerate(sorted({row["item_id"] for row in records}))
    }
    poisoned = [{**row, "item_id": replacements[row["item_id"]]} for row in records]
    changed = analysis.build_preoutcome_rows(poisoned)
    keep = {
        "consensus",
        "agreement",
        "mean_consensus_confidence",
        *analysis.COORDINATES,
    }
    original_features = sorted(
        json.dumps({key: row[key] for key in keep}, sort_keys=True) for row in original
    )
    changed_features = sorted(
        json.dumps({key: row[key] for key in keep}, sort_keys=True) for row in changed
    )
    assert original_features == changed_features


def test_outcome_join_is_exactly_balanced() -> None:
    records = [json.loads(line) for line in RECORDS.read_text().splitlines()]
    payload = {"split": "development", "rows": analysis.build_preoutcome_rows(records)}
    rows = analysis.join_outcomes(payload)
    assert sum(row["gold_label"] == "SUPPORTS" for row in rows) == 30
    assert sum(row["gold_label"] == "REFUTES" for row in rows) == 30
    assert sum(row["error"] for row in rows if row["gold_label"] == "SUPPORTS") == 4
    assert sum(row["error"] for row in rows if row["gold_label"] == "REFUTES") == 6


def test_simplex_is_nonnegative_and_sums_to_one() -> None:
    weights = analysis._simplex_weights()
    assert len(weights) == 1001
    assert all(all(value >= 0 for value in row) for row in weights)
    assert all(sum(row) == pytest.approx(1.0) for row in weights)
