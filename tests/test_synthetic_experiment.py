from pathlib import Path

import numpy as np

from sp500_forecastability.synthetic_experiment import (
    CORRUPTION_MECHANISMS,
    METHODS,
    V3_CORRUPTION_MECHANISMS,
    V3_METHODS,
    generate_protocol_rows,
    mechanism_heldout_partitions,
    run_synthetic_v1,
    run_synthetic_v2,
    run_synthetic_v3,
)
from sp500_forecastability.synthetic_v4_experiment import (
    V4_FEATURES,
    V4_METHODS,
    V4_TEST_SEEDS,
    _fit_monotonic_logistic,
    run_synthetic_v4,
    run_v3_matched_coverage_posthoc,
)


def test_protocol_generates_independent_rows_for_every_mechanism() -> None:
    rows = generate_protocol_rows((11, 22, 33))

    assert {row["scenario"] for row in rows} == {
        "independent_clean",
        "shared_clean",
        "shared_corruption",
        "stale_evidence",
        "partial_corruption",
    }
    assert len(rows) == 672
    assert all(set(METHODS).issubset(row) for row in rows)
    partitions = mechanism_heldout_partitions(rows)
    assert set(partitions) == {scenario.value for scenario in CORRUPTION_MECHANISMS}
    for held_out, (train, test) in partitions.items():
        assert all(row["scenario"] != held_out for row in train)
        assert any(row["scenario"] == held_out for row in test)


def test_formal_runner_writes_report_json_and_all_figures(tmp_path: Path) -> None:
    payload = run_synthetic_v1(tmp_path, base_seeds=(11, 22, 33), bootstrap_repeats=8)

    assert payload["row_count"] == 864
    assert set(payload["pooled_results"]) == set(METHODS)
    assert (tmp_path / "synthetic_v1.md").is_file()
    assert (tmp_path / "synthetic_v1" / "results.json").is_file()
    for name in (
        "risk_coverage.png",
        "reliability_diagram.png",
        "mechanism_heatmap.png",
        "provenance_noise_curve.png",
        "agent_count_curve.png",
    ):
        assert (tmp_path / "synthetic_v1" / name).is_file()


def test_v2_keeps_behavior_baselines_decoupled_and_writes_separate_artifacts(tmp_path: Path) -> None:
    rows = generate_protocol_rows((11, 22, 33), profile="v2")

    assert any(row["confidence"] != row["recent_performance"] for row in rows)
    payload = run_synthetic_v2(tmp_path, base_seeds=(11, 22, 33), bootstrap_repeats=8)

    assert payload["row_count"] == 864
    assert (tmp_path / "synthetic_v2.md").is_file()
    assert (tmp_path / "synthetic_v2" / "results.json").is_file()


def test_v3_adds_fair_provenance_ablations_and_writes_separate_artifacts(tmp_path: Path) -> None:
    rows = generate_protocol_rows((11, 22, 33), profile="v3")

    assert {row["scenario"] for row in rows} == {
        "independent_clean",
        "shared_clean",
        "shared_corruption",
        "stale_evidence",
        "partial_corruption",
        "evidence_inertia",
    }
    assert all(set(V3_METHODS).issubset(row) for row in rows)
    partitions = mechanism_heldout_partitions(
        rows, corruption_mechanisms=V3_CORRUPTION_MECHANISMS
    )
    assert set(partitions) == {scenario.value for scenario in V3_CORRUPTION_MECHANISMS}

    payload = run_synthetic_v3(tmp_path, base_seeds=(11, 22, 33), bootstrap_repeats=8)

    assert payload["row_count"] == 1152
    assert set(payload["pooled_results"]) == set(V3_METHODS)
    assert (tmp_path / "synthetic_v3.md").is_file()
    assert (tmp_path / "synthetic_v3" / "results.json").is_file()


def test_v4_rows_add_noisy_pre_outcome_features() -> None:
    rows = generate_protocol_rows((11, 22, 33), profile="v4")

    assert all(set(V4_FEATURES).issubset(row) for row in rows)
    inertia = [row for row in rows if row["scenario"] == "evidence_inertia"]
    assert {row["causal_effect_risk"] for row in inertia} != {1.0}
    assert any(row["consensus_error"] == 0 for row in inertia)


def test_monotonic_logistic_coefficients_are_non_negative() -> None:
    matrix = np.asarray([[0.0, 1.0], [0.2, 0.8], [0.8, 0.2], [1.0, 0.0]])
    labels = np.asarray([0.0, 0.0, 1.0, 1.0])

    fitted = _fit_monotonic_logistic(matrix, labels, l2=0.01)

    assert all(value >= 0.0 for value in fitted.coefficients)


def test_v4_runner_and_v3_posthoc_write_separate_artifacts(tmp_path: Path) -> None:
    test_seeds = {
        scenario: (seeds[0],) for scenario, seeds in V4_TEST_SEEDS.items()
    }
    payload = run_synthetic_v4(
        tmp_path,
        bootstrap_repeats=4,
        train_seeds=(11, 22, 33, 44, 55),
        test_seeds_by_mechanism=test_seeds,
    )
    posthoc = run_v3_matched_coverage_posthoc(tmp_path, bootstrap_repeats=4)

    assert payload["row_count"] == 384
    assert set(payload["pooled_results"]) == set(V4_METHODS)
    assert set(payload["macro_aurc"]) == set(V4_METHODS)
    assert posthoc["status"] == "post_hoc_does_not_replace_v3_primary_result"
    assert (tmp_path / "synthetic_v4.md").is_file()
    assert (tmp_path / "synthetic_v4" / "results.json").is_file()
    assert (tmp_path / "synthetic_v3_posthoc_matched_coverage.md").is_file()
