import pandas as pd

from sp500_forecastability.features import (
    cross_market_disagreement,
    direction_target,
    forward_return,
)
from sp500_forecastability.signal_processing import causal_filter_bank


def test_forward_target_does_not_fill_unavailable_future_labels() -> None:
    close = pd.Series([100.0, 101.0, 103.0, 102.0], name="close")

    target = forward_return(close, horizon=2)
    direction = direction_target(close, horizon=2)

    assert abs(target.iloc[0] - 0.03) < 1e-12
    assert target.iloc[-1] != target.iloc[-1]
    assert direction.iloc[0] == 1.0
    assert direction.iloc[-1] != direction.iloc[-1]


def test_disagreement_uses_trailing_values_only() -> None:
    frame = pd.DataFrame(
        {
            "return": [1.0, 1.0, 1.0, 2.0, 2.0],
            "vix": [1.0, 1.0, 1.0, 1.0, 5.0],
        }
    )

    disagreement = cross_market_disagreement(
        frame, ["return", "vix"], window=3, min_periods=3
    )

    assert disagreement.iloc[0] != disagreement.iloc[0]
    assert disagreement.iloc[2] == 0.0
    assert disagreement.iloc[4] > 0.0


def test_causal_filter_bank_reconstructs_the_input() -> None:
    signal = pd.Series([1.0, 2.0, 1.5, 3.0, 2.5, 4.0], name="signal")

    bands = causal_filter_bank(signal, spans=(2, 3))

    reconstructed = bands.sum(axis=1)
    assert all(abs(left - right) < 1e-12 for left, right in zip(reconstructed, signal))


def test_causal_filter_bank_does_not_use_future_values() -> None:
    original = pd.Series([1.0, 2.0, 1.5, 3.0, 2.5, 4.0])
    changed_future = original.copy()
    changed_future.iloc[-1] = 1000.0

    first = causal_filter_bank(original, spans=(2, 3))
    second = causal_filter_bank(changed_future, spans=(2, 3))

    assert first.iloc[:-1].equals(second.iloc[:-1])
