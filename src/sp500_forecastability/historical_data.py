"""As-of construction for the first real-data historical replay.

This module builds evidence features, not a trading claim.  It makes every
availability assumption explicit and keeps the root source of causal transforms
so that a router cannot count derived price signals as independent evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from pathlib import Path

import pandas as pd

from sp500_forecastability.features import direction_target, forward_return
from sp500_forecastability.signal_processing import causal_filter_bank


@dataclass(frozen=True)
class HistoricalReplayData:
    """Leakage-aware features, labels, and root-source lineage."""

    frame: pd.DataFrame
    feature_roots: dict[str, str]
    availability_assumptions: dict[str, str]


def _read_daily(path: Path, date_column: str | int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    column = frame.columns[date_column] if isinstance(date_column, int) else date_column
    if column not in frame:
        raise KeyError(f"missing date column {column!r} in {path.name}")
    dates = pd.to_datetime(frame[column], errors="raise")
    result = frame.drop(columns=column).copy()
    result.index = pd.DatetimeIndex(dates, name="date")
    if result.index.has_duplicates:
        raise ValueError(f"duplicate dates in {path.name}")
    return result.sort_index()


def _read_bloomberg_export(path: Path, value_name: str) -> pd.DataFrame:
    """Read the local Bloomberg export format whose table header starts at row 7."""

    frame = pd.read_csv(path, skiprows=6)
    result = pd.DataFrame(
        {value_name: pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy()},
        index=pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0], errors="raise"), name="date"),
    )
    return result.sort_index()


def _read_bloomberg_excel(path: Path, value_name: str) -> pd.DataFrame:
    """Read the matching Bloomberg Excel export whose table header is row 7."""

    frame = pd.read_excel(path, skiprows=6)
    return pd.DataFrame(
        {value_name: pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy()},
        index=pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0], errors="raise"), name="date"),
    ).sort_index()


def _read_wind_excel(path: Path, value_names: tuple[str, ...]) -> pd.DataFrame:
    """Read a local Wind workbook whose data table begins on Excel row 32.

    The surrounding title, range, and source rows are intentionally discarded.
    Only rows with a parseable date enter the replay; this prevents the range
    metadata from being mistaken for a late observation.
    """

    frame = pd.read_excel(path, skiprows=31)
    if frame.shape[1] != len(value_names) + 1:
        raise ValueError(f"unexpected column count in {path.name}")
    dates = pd.to_datetime(frame.iloc[:, 0], errors="coerce", format="mixed")
    observed = frame.loc[dates.notna()].copy()
    index = pd.DatetimeIndex(dates.loc[dates.notna()].to_numpy(), name="date")
    if index.has_duplicates:
        raise ValueError(f"duplicate dates in {path.name}")
    values = observed.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    values.columns = value_names
    values.index = index
    return values.sort_index()


def build_historical_replay_data(
    data_directory: Path | str,
    *,
    horizon_days: int = 5,
    sentiment_delay_days: int = 5,
    mutual_fund_flow_delay_days: int = 5,
) -> HistoricalReplayData:
    """Build a conservative close-to-next-session, causal replay table.

    Daily sources are treated as visible only after their date's close. Weekly
    sentiment is shifted by five trading rows after its last known observation.
    Forward-fill carries a previously available value forward; no back-fill is
    performed. The final horizon rows with unavailable labels are removed.
    """

    if min(horizon_days, sentiment_delay_days, mutual_fund_flow_delay_days) < 1:
        raise ValueError("horizon_days and feature-release delays must be positive")
    root = Path(data_directory)
    market = _read_daily(root / "market.csv", "Date")[["sp500"]]
    vix = _read_daily(root / "vix.csv", "Date")[["vix"]]
    macro = _read_daily(root / "macro.csv", 0)[["10Y", "credit"]]
    sentiment = _read_daily(root / "sentiment.csv", "date")[
        ["recession", "inflation", "unemployment"]
    ]
    cboe = _read_wind_excel(
        root / "CBOE_full.xlsx",
        ("cboe_total_pcr", "cboe_index_pcr", "put_call_ratio"),
    )
    mutual_fund_flow = _read_wind_excel(
        root / "ICI_mutual_fund_flow.xlsx",
        (
            "mutual_fund_total_flow",
            "mutual_fund_domestic_flow",
            "mutual_fund_foreign_flow",
        ),
    )
    spy_flow = _read_bloomberg_export(root / "SPY_FUND_FLOW.csv", "spy_fund_flow")
    ivv_flow = _read_bloomberg_export(root / "IVV_FUND_FLOW.csv", "ivv_fund_flow")
    voo_shares = _read_bloomberg_excel(root / "VOO_US_EQUITY_DAILY.xlsx", "voo_shares")
    index = market.index
    daily = market.join(vix.reindex(index), how="left").join(macro.reindex(index), how="left")
    # These values have already been published on earlier dates. Forward-fill
    # only propagates the last as-of observation and never imports a future one.
    daily[["vix", "10Y", "credit"]] = daily[["vix", "10Y", "credit"]].ffill()
    delayed_sentiment = sentiment.reindex(index, method="ffill").shift(sentiment_delay_days)
    daily = daily.join(delayed_sentiment)
    delayed_mutual_fund_flow = mutual_fund_flow.reindex(index, method="ffill").shift(
        mutual_fund_flow_delay_days
    )
    daily = (
        daily.join(cboe.reindex(index))
        .join(spy_flow.reindex(index))
        .join(ivv_flow.reindex(index))
        .join(voo_shares.reindex(index))
        .join(delayed_mutual_fund_flow)
    )
    daily[
        [
            "cboe_total_pcr",
            "cboe_index_pcr",
            "put_call_ratio",
            "spy_fund_flow",
            "ivv_fund_flow",
            "voo_shares",
        ]
    ] = daily[
        [
            "cboe_total_pcr",
            "cboe_index_pcr",
            "put_call_ratio",
            "spy_fund_flow",
            "ivv_fund_flow",
            "voo_shares",
        ]
    ].ffill()

    log_price = daily["sp500"].astype(float).apply(log).rename("log_price")
    price_bands = causal_filter_bank(log_price, spans=(5, 20, 60)).add_prefix("market_")
    features = pd.concat(
        (
            daily,
            price_bands,
            daily["sp500"].pct_change(5).rename("market_return_5d"),
            daily["vix"].pct_change(5).rename("vix_change_5d"),
            daily["credit"].diff(5).rename("credit_change_5d"),
            daily["put_call_ratio"].pct_change(5).rename("put_call_change_5d"),
            daily["cboe_total_pcr"].pct_change(5).rename("cboe_total_change_5d"),
            daily["cboe_index_pcr"].pct_change(5).rename("cboe_index_change_5d"),
            (daily["cboe_index_pcr"] - daily["put_call_ratio"]).rename(
                "cboe_index_stock_spread"
            ),
            daily[["spy_fund_flow", "ivv_fund_flow"]].rolling(5).sum().rename(
                columns={"spy_fund_flow": "spy_flow_5d", "ivv_fund_flow": "ivv_flow_5d"}
            ),
            daily["voo_shares"].pct_change(5).rename("voo_shares_change_5d"),
            daily["mutual_fund_total_flow"].diff(5).rename("mutual_fund_total_change_5d"),
            daily["mutual_fund_domestic_flow"].div(daily["mutual_fund_total_flow"]).rename(
                "mutual_fund_domestic_share"
            ),
        ),
        axis=1,
    )
    features["forward_return_5d"] = forward_return(features["sp500"], horizon=horizon_days)
    features["target_up_5d"] = direction_target(features["sp500"], horizon=horizon_days)
    features = features.dropna().copy()
    roots = {
        "sp500": "market_bloomberg",
        "vix": "vix_bloomberg",
        "10Y": "macro_bloomberg",
        "credit": "macro_bloomberg",
        "recession": "google_trends",
        "inflation": "google_trends",
        "unemployment": "google_trends",
        "market_return_5d": "market_bloomberg",
        "vix_change_5d": "vix_bloomberg",
        "credit_change_5d": "macro_bloomberg",
        "cboe_total_pcr": "cboe_options",
        "cboe_index_pcr": "cboe_options",
        "put_call_ratio": "cboe_options",
        "put_call_change_5d": "cboe_options",
        "cboe_total_change_5d": "cboe_options",
        "cboe_index_change_5d": "cboe_options",
        "cboe_index_stock_spread": "cboe_options",
        "spy_fund_flow": "spy_etf_flow",
        "ivv_fund_flow": "ivv_etf_flow",
        "spy_flow_5d": "spy_etf_flow",
        "ivv_flow_5d": "ivv_etf_flow",
        "voo_shares": "voo_etf_flow",
        "voo_shares_change_5d": "voo_etf_flow",
        "mutual_fund_total_flow": "ici_mutual_fund_flow",
        "mutual_fund_domestic_flow": "ici_mutual_fund_flow",
        "mutual_fund_foreign_flow": "ici_mutual_fund_flow",
        "mutual_fund_total_change_5d": "ici_mutual_fund_flow",
        "mutual_fund_domestic_share": "ici_mutual_fund_flow",
        **{column: "market_bloomberg" for column in price_bands.columns},
    }
    return HistoricalReplayData(
        frame=features,
        feature_roots=roots,
        availability_assumptions={
            "market_vix_macro": "visible after same-day close; usable next session",
            "sentiment": f"last known weekly value delayed by {sentiment_delay_days} trading days",
            "cboe_options": "visible after the recorded day's close; usable next session",
            "mutual_fund_flow": (
                "weekly timestamp has no release-time metadata in this checkout; last known value "
                f"is delayed by {mutual_fund_flow_delay_days} trading days"
            ),
            "missing_values": "forward-fill only after a source becomes available; never back-fill",
            "label": f"forward {horizon_days}-trading-day return, withheld at decision time",
        },
    )
