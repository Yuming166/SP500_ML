from pathlib import Path

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
