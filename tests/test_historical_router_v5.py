from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from sp500_forecastability import historical_router_v5 as v5


def _target_rows(n_timestamps: int = 60) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    methods = ("majority", "recent_performance", "provenance")
    for timestamp in range(n_timestamps):
        for method_index, method in enumerate(methods):
            risk = ((timestamp * 3 + method_index) % 13) / 13
            rows.append(
                {
                    "timestamp": timestamp,
                    "method": method,
                    "intervention_inertia": risk,
                    "flip_inertia": ((timestamp + method_index) % 11) / 11,
                    "source_concentration": ((timestamp * 2 + method_index) % 9) / 9,
                    "consensus_risk": ((timestamp + 2 * method_index) % 8) / 8,
                    "action_confidence_risk": ((timestamp + method_index) % 7) / 7,
                    "root_disagreement": ((timestamp * 2) % 10) / 10,
                    "quality_risk": ((timestamp + 3) % 6) / 6,
                    "error": int((timestamp + method_index) % 4 == 0),
                    "group": f"g{timestamp % 3}",
                }
            )
    return pd.DataFrame(rows)


def _source_rows() -> pd.DataFrame:
    values = np.linspace(0.0, 1.0, 30)
    return pd.DataFrame(
        {
            "intervention_inertia": values,
            "flip_inertia": np.roll(values, 3),
            "source_concentration": np.roll(values, 7),
            "error": (values > 0.55).astype(int),
        }
    )


def test_actual_contract_remains_eleven_agents_and_seven_roots() -> None:
    assert len(v5.AGENTS) == 11
    assert len({root for root, _columns in v5.AGENTS.values()}) == 7


def test_pair_sampling_is_deterministic_and_capped_per_group() -> None:
    labels = np.tile([0, 1], 200)
    groups = np.asarray([f"g{(index // 2) % 2}" for index in range(len(labels))])
    first = v5._pair_indices(labels, groups, cap_per_group=23)
    second = v5._pair_indices(labels, groups, cap_per_group=23)

    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    assert len(first[0]) <= 46


def test_hard_target_constraints_guarantee_intervention_logit_ordering() -> None:
    rows = _target_rows()
    ranker = v5.fit_target_ranker(rows)
    assert min(ranker.coefficients[: len(v5.COMMON_FEATURES)]) >= v5.HARD_COMMON_MINIMUM

    clean = rows.iloc[:12].copy()
    for feature in v5.COMMON_FEATURES:
        stressed = clean.copy()
        stressed[feature] = stressed[feature] + v5.INTERVENTION_INCREMENT
        rise = ranker.predict_logit(stressed) - ranker.predict_logit(clean)
        assert np.all(rise + 1e-12 >= v5.HARD_COMMON_MINIMUM * v5.INTERVENTION_INCREMENT)


def test_adaptive_gate_cannot_increase_with_source_shift() -> None:
    source = v5.SourceRanker((0.2, 0.3, 0.4), 0.0, True, 10)
    target = v5.TargetRanker((0.25,) * 7, (0.0, 0.0), 0.0, True, 10, 0.25)
    ranker = v5.AdaptiveRanker(
        source=source,
        target=target,
        feature_mean=(0.0, 0.0, 0.0),
        feature_sd=(1.0, 1.0, 1.0),
        source_score_mean=0.0,
        source_score_sd=1.0,
        target_score_mean=0.0,
        target_score_sd=1.0,
        gate_intercept=1.0,
        gate_shift_slope=3.0,
        gate_objective=0.0,
        gate_converged=True,
        gate_pair_count=10,
    )
    rows = _target_rows(3).iloc[[0, 3, 6]].copy()
    rows.loc[:, list(v5.COMMON_FEATURES)] = np.asarray(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [2.0, 2.0, 2.0]]
    )
    order = np.argsort(ranker.shift(rows))
    assert np.all(np.diff(ranker.gate(rows)[order]) <= 1e-12)


def test_full_ranker_and_monotone_calibration_fit_on_synthetic_rows() -> None:
    source_rows = _source_rows()
    target_rows = _target_rows()
    source = v5.fit_source_ranker(source_rows)
    target = v5.fit_target_ranker(target_rows)
    adaptive = v5.fit_adaptive_ranker(source_rows, target_rows, source, target)
    assert adaptive.gate_shift_slope >= 0.0

    chosen = v5._choose_by_rank(target_rows, adaptive)
    calibration = v5.fit_calibration_head(chosen)
    assert calibration.slope >= 0.0
    assert np.all(np.diff(calibration.predict([-1.0, 0.0, 1.0])) >= 0.0)


def test_calibration_head_and_threshold_slices_are_temporally_disjoint() -> None:
    rows = _target_rows(20)
    head, threshold = v5._split_calibration_rows(rows)
    assert head["timestamp"].max() < threshold["timestamp"].min()
    assert set(head["timestamp"]).isdisjoint(set(threshold["timestamp"]))


def test_identity_calibration_is_strict_json_serializable() -> None:
    payload = asdict(v5._identity_calibration(12))
    assert json.loads(json.dumps(payload, allow_nan=False))["objective"] is None
