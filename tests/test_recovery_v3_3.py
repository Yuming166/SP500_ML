import json
from pathlib import Path

import numpy as np

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_3 as witness


def test_witness_grid_is_frozen_and_small() -> None:
    assert witness.WITNESS_PROBABILITY_GRID == (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    assert witness.WITNESS_DELTA_GRID == (-0.2, 0.0, 0.1, 0.2, 0.3, 0.4)


def test_witness_selects_counter_consensus_root_without_outcomes() -> None:
    selection = json.loads(
        Path("results/recovery_v3_2/selection_manifest.json").read_text(encoding="utf-8")
    )
    example = selection["examples"][0]
    records = base._load_jsonl(Path("results/recovery_v3_2/smoke/records.jsonl"))
    baseline = [
        row
        for row in records
        if row["example_id"] == example["example_id"] and row["phase"] == "baseline"
    ]
    stance = {
        (example["example_id"], "anchor"): np.asarray([0.1, 0.8, 0.1]),
        (example["example_id"], "candidate_0"): np.asarray([0.1, 0.2, 0.7]),
        (example["example_id"], "candidate_1"): np.asarray([0.7, 0.2, 0.1]),
    }
    selected = witness._witness_policy(
        [example],
        {example["example_id"]: baseline},
        stance,
        probability_threshold=0.4,
        delta_threshold=-0.2,
    )
    assert selected[example["example_id"]] == "candidate_0"
