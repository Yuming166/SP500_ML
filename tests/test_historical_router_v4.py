from __future__ import annotations

import numpy as np
import pandas as pd

from sp500_forecastability import historical_router_v4 as v4


def _synthetic_market_fit_rows(n_timestamps: int = 40) -> pd.DataFrame:
    rows = []
    methods = ("majority", "recent_performance", "provenance")
    for timestamp in range(n_timestamps):
        for method_index, method in enumerate(methods):
            base = (timestamp % 10) / 10
            rows.append(
                {
                    "timestamp": timestamp,
                    "method": method,
                    "intervention_inertia": base,
                    "flip_inertia": (timestamp % 7) / 7,
                    "source_concentration": (timestamp % 5) / 5,
                    "consensus_risk": (timestamp % 6) / 6,
                    "action_confidence_risk": (timestamp % 8) / 8,
                    "root_disagreement": (timestamp % 9) / 9,
                    "quality_risk": (timestamp % 4) / 4,
                    "error": int((timestamp + method_index) % 4 == 0),
                    "group": f"g{timestamp % 3}",
                }
            )
    return pd.DataFrame(rows)


def test_actual_market_contract_has_eleven_agents_and_seven_roots() -> None:
    assert len(v4.AGENTS) == 11
    assert len({root for root, _columns in v4.AGENTS.values()}) == 7


def test_source_prior_and_market_model_are_monotone_in_declared_risks() -> None:
    source = pd.DataFrame(
        {
            "intervention_inertia": [0.0, 0.1, 0.2, 0.8, 0.9, 1.0],
            "flip_inertia": [0.1, 0.0, 0.2, 0.7, 1.0, 0.9],
            "source_concentration": [0.0, 0.2, 0.1, 0.8, 0.9, 1.0],
            "error": [0, 0, 0, 1, 1, 1],
        }
    )
    _intercept, anchor, _objective = v4.fit_source_prior(source)
    assert (anchor >= 0).all()

    fit_rows = _synthetic_market_fit_rows()
    model = v4.fit_market_router(fit_rows, anchor)
    assert min(model.coefficients) >= 0.0

    clean = fit_rows.iloc[:20].copy()
    stressed = clean.copy()
    for feature in v4.MARKET_FEATURES:
        stressed[feature] = np.clip(stressed[feature] + 0.1, 0.0, 1.0)
    assert np.all(model.predict(stressed) + 1e-12 >= model.predict(clean))


def test_market_candidate_signature_is_bounded_and_has_three_actions() -> None:
    probabilities = pd.Series(
        np.linspace(0.1, 0.9, len(v4.AGENTS)),
        index=list(v4.AGENTS),
        name=pd.Timestamp("2026-01-02"),
    )
    roots = sorted({root for root, _columns in v4.AGENTS.values()})
    candidates = v4.market_candidate_rows(
        probabilities,
        agent_weights=pd.Series(1 / len(v4.AGENTS), index=list(v4.AGENTS)),
        root_weights=pd.Series(1 / len(roots), index=roots),
        root_losses=pd.Series(0.25, index=roots),
    )

    assert {row["method"] for row in candidates} == {
        "majority",
        "recent_performance",
        "provenance",
    }
    for row in candidates:
        for feature in v4.MARKET_FEATURES:
            assert 0.0 <= float(row[feature]) <= 1.0


def test_future_unmatured_labels_cannot_change_current_signature() -> None:
    dates = pd.date_range("2026-01-01", periods=12, freq="B")
    probability_values = np.linspace(0.2, 0.8, len(dates))
    probabilities = pd.DataFrame(
        {
            agent: np.clip(probability_values + 0.01 * index, 0.01, 0.99)
            for index, agent in enumerate(v4.AGENTS)
        },
        index=dates,
    )
    market_a = pd.DataFrame(
        {
            "target_up_5d": np.zeros(len(dates), dtype=int),
            "forward_return_5d": np.zeros(len(dates)),
            "vix": np.full(len(dates), 20.0),
        },
        index=dates,
    )
    market_b = market_a.copy()
    market_b.loc[dates[7]:, "target_up_5d"] = 1

    rows_a = v4.online_market_rows(probabilities, market_a)
    rows_b = v4.online_market_rows(probabilities, market_b)
    signature = ["timestamp", "method", *v4.MARKET_FEATURES]
    pd.testing.assert_frame_equal(
        rows_a.loc[:, signature].reset_index(drop=True),
        rows_b.loc[:, signature].reset_index(drop=True),
    )


def test_conformal_quantile_uses_frozen_finite_sample_index() -> None:
    scores = np.arange(10, dtype=float) / 10
    assert v4._conformal_quantile(scores, 0.8) == 0.8
