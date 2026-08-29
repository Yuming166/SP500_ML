"""Leakage-aware target and cross-market feature helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


def forward_return(close: pd.Series, horizon: int = 5) -> pd.Series:
    """Return the close-to-close forward return for a trading-day horizon."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    values = pd.Series(close, copy=True, dtype="float64")
    result = values.shift(-horizon).div(values).sub(1.0)
    return result.rename("forward_return")


def direction_target(close: pd.Series, horizon: int = 5) -> pd.Series:
    """Return a nullable 0/1 target for the sign of the forward return."""

    target = forward_return(close, horizon=horizon)
    return target.gt(0.0).astype("float64").where(target.notna()).rename("target_up")


def rolling_zscore(
    series: pd.Series, window: int = 252, min_periods: int | None = None
) -> pd.Series:
    """Standardize a series using only its trailing observations."""

    if window < 2:
        raise ValueError("window must be at least 2")
    min_periods = window if min_periods is None else min_periods
    if not 1 <= min_periods <= window:
        raise ValueError("min_periods must be between 1 and window")
    values = pd.Series(series, copy=True, dtype="float64")
    rolling = values.rolling(window=window, min_periods=min_periods)
    mean = rolling.mean()
    # A constant trailing window has no directional information. Represent it
    # as a zero z-score instead of propagating an avoidable division-by-zero NaN.
    standard_deviation = rolling.std(ddof=0).mask(lambda value: value == 0.0, 1.0)
    return ((values - mean) / standard_deviation).rename(series.name)


def cross_market_disagreement(
    frame: pd.DataFrame,
    columns: Sequence[str],
    signs: Mapping[str, int] | None = None,
    window: int = 252,
    min_periods: int | None = None,
) -> pd.Series:
    """Measure dispersion among trailing standardized market signals.

    ``signs`` can orient variables into the same economic direction before the
    cross-sectional standard deviation. For example, VIX and credit spreads
    can use ``-1`` while returns and fund flows use ``+1``. The feature is a
    diagnostic of disagreement, not a claim that every input has a stable
    causal interpretation.
    """

    selected = list(columns)
    if not selected:
        raise ValueError("columns must not be empty")
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise KeyError(f"columns not found: {missing}")
    signs = {} if signs is None else dict(signs)

    standardized = {}
    for column in selected:
        direction = signs.get(column, 1)
        if direction not in (-1, 1):
            raise ValueError("signs must contain only -1 or 1")
        standardized[column] = rolling_zscore(
            frame[column], window=window, min_periods=min_periods
        ) * direction
    return pd.DataFrame(standardized, index=frame.index).std(axis=1, ddof=0).rename(
        "cross_market_disagreement"
    )
