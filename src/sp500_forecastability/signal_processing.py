"""Causal, leakage-aware signal-processing baselines.

These filters are deliberately simple. They provide an interpretable baseline
before adding fixed wavelets or a learnable filter bank.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def causal_ema(series: pd.Series, span: int) -> pd.Series:
    """Return an exponentially weighted moving average using past values only."""

    if span < 1:
        raise ValueError("span must be positive")
    values = pd.Series(series, copy=True, dtype="float64")
    return values.ewm(span=span, adjust=False, min_periods=1).mean().rename(series.name)


def causal_filter_bank(series: pd.Series, spans: Sequence[int] = (5, 20, 60)) -> pd.DataFrame:
    """Decompose a signal into high, mid, and low causal bands.

    The columns telescope back to the input signal:

    ``detail + bands + trend == series``

    Every component at time ``t`` depends only on observations at or before
    ``t``. This is a filter-bank baseline, not a claim that it is optimal.
    """

    ordered_spans = [int(span) for span in spans]
    if not ordered_spans or any(span < 1 for span in ordered_spans):
        raise ValueError("spans must contain positive integers")
    if ordered_spans != sorted(set(ordered_spans)):
        raise ValueError("spans must be strictly increasing")

    values = pd.Series(series, copy=True, dtype="float64")
    smooth = [causal_ema(values, span) for span in ordered_spans]
    components: dict[str, pd.Series] = {
        f"detail_0_{ordered_spans[0]}": values - smooth[0]
    }
    for left_span, right_span, left, right in zip(
        ordered_spans, ordered_spans[1:], smooth, smooth[1:]
    ):
        components[f"band_{left_span}_{right_span}"] = left - right
    components[f"trend_{ordered_spans[-1]}"] = smooth[-1]
    return pd.DataFrame(components, index=values.index)


def apply_causal_filter_bank(
    frame: pd.DataFrame, columns: Sequence[str], spans: Sequence[int] = (5, 20, 60)
) -> pd.DataFrame:
    """Apply the same causal filter bank to several aligned signal columns."""

    selected = list(columns)
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise KeyError(f"columns not found: {missing}")
    if not selected:
        raise ValueError("columns must not be empty")

    transformed = []
    for column in selected:
        result = causal_filter_bank(frame[column], spans=spans)
        result.columns = [f"{column}__{name}" for name in result.columns]
        transformed.append(result)
    return pd.concat(transformed, axis=1)

