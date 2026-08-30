"""Rule-agent expanding walk-forward replay on the as-of historical table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sp500_forecastability.historical_data import build_historical_replay_data
from sp500_forecastability.splits import walk_forward_splits

METHODS = ("majority", "confidence", "recent_performance", "provenance")
ROOTS = ("market_bloomberg", "vix_bloomberg", "macro_bloomberg", "google_trends")


@dataclass(frozen=True)
class ReplaySummary:
    method: str
    coverage: float
    routed_error: float
    false_rejection: float
    mean_trade_return: float
    max_drawdown: float
    turnover: float


def _zscore(values: pd.Series, window: int = 60) -> pd.Series:
    mean = values.rolling(window, min_periods=window).mean()
    std = values.rolling(window, min_periods=window).std(ddof=0).replace(0.0, 1.0)
    return (values - mean) / std


def _agent_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Create five deterministic, as-of rule agents and their source roots."""

    signals = pd.DataFrame(index=frame.index)
    signals["market_short"] = _zscore(frame["market_return_5d"])
    signals["market_trend"] = _zscore(frame["market_trend_60"].diff(5))
    signals["vix"] = -_zscore(frame["vix_change_5d"])
    signals["macro"] = -_zscore(frame["credit_change_5d"])
    signals["sentiment"] = -_zscore(frame["recession"].diff(5))
    actions = signals.gt(0.0).astype(int).add_prefix("action_")
    confidence = signals.abs().clip(upper=2.0).div(4.0).add(0.5).add_prefix("confidence_")
    result = pd.concat((actions, confidence), axis=1).dropna()
    roots = {
        "market_short": "market_bloomberg",
        "market_trend": "market_bloomberg",
        "vix": "vix_bloomberg",
        "macro": "macro_bloomberg",
        "sentiment": "google_trends",
    }
    return result, roots


def _online_quality(
    actions: pd.DataFrame, labels: pd.Series, horizon: int = 5, window: int = 126
) -> pd.DataFrame:
    """Estimate agent correctness only from labels matured before each decision."""

    matured = actions.eq(labels, axis=0).shift(horizon)
    return matured.rolling(window, min_periods=window).mean()


def _risk_rows(frame: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    agent_data, roots = _agent_table(frame)
    common = frame.join(agent_data, how="inner")
    action_columns = [column for column in common if column.startswith("action_")]
    confidence_columns = [column for column in common if column.startswith("confidence_")]
    actions = common[action_columns].rename(columns=lambda value: value.removeprefix("action_"))
    confidences = common[confidence_columns].rename(columns=lambda value: value.removeprefix("confidence_"))
    quality = _online_quality(actions, common["target_up_5d"].astype(int), horizon=horizon)
    rows = []
    for timestamp in common.index:
        if quality.loc[timestamp].isna().any():
            continue
        votes = actions.loc[timestamp]
        decision = int(votes.mean() >= 0.5)
        agreeing = votes.index[votes == decision]
        agreement = len(agreeing) / len(votes)
        root_counts = pd.Series([roots[agent] for agent in agreeing]).value_counts()
        concentration = root_counts.max() / len(agreeing)
        root_quality = []
        for root in ROOTS:
            members = [agent for agent, agent_root in roots.items() if agent_root == root]
            root_quality.append(float(quality.loc[timestamp, members].mean()))
        quality_by_root = dict(zip(ROOTS, root_quality, strict=True))
        consensus_quality = float(quality.loc[timestamp, list(agreeing)].mean())
        source_quality = float(
            np.mean([quality_by_root[roots[agent]] for agent in agreeing])
        )
        row = {
            "date": timestamp,
            "target": int(common.loc[timestamp, "target_up_5d"]),
            "forward_return": float(common.loc[timestamp, "forward_return_5d"]),
            "majority_action": decision,
            "correct": int(decision == int(common.loc[timestamp, "target_up_5d"])),
            "risk_majority": 1.0 - agreement,
            "risk_confidence": 1.0 - float(confidences.loc[timestamp, list(agreeing)].mean()),
            "risk_recent_performance": 1.0 - consensus_quality,
            "risk_provenance": min(1.0, 0.6 * concentration + 0.4 * (1.0 - source_quality)),
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def _drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def run_historical_replay(
    data_directory: Path | str = "data", output_root: Path | str = "results"
) -> dict[str, ReplaySummary]:
    """Run fixed expanding walk-forward routing without tuning on test outcomes."""

    replay = build_historical_replay_data(data_directory)
    rows = _risk_rows(replay.frame)
    all_results: dict[str, list[pd.DataFrame]] = {method: [] for method in METHODS}
    for train_indices, test_indices in walk_forward_splits(
        len(rows), train_size=504, test_size=126, step=126, gap=5
    ):
        train, test = rows.iloc[list(train_indices)], rows.iloc[list(test_indices)].copy()
        for method in METHODS:
            risk = f"risk_{method}"
            threshold = float(train[risk].quantile(0.75))
            test[method] = test[risk] <= threshold
            all_results[method].append(
                test[["correct", "forward_return", "majority_action", method]].copy()
            )
    summaries: dict[str, ReplaySummary] = {}
    for method, pieces in all_results.items():
        result = pd.concat(pieces)
        routed = result[result[method]]
        trades = routed.iloc[::5]
        position = trades["majority_action"].astype(float)
        strategy_return = trades["forward_return"] * position
        summaries[method] = ReplaySummary(
            method=method,
            coverage=float(result[method].mean()),
            routed_error=float(1.0 - routed["correct"].mean()),
            false_rejection=float((~result.loc[result["correct"] == 1, method]).mean()),
            mean_trade_return=float(strategy_return.mean()),
            max_drawdown=_drawdown(strategy_return),
            turnover=float(position.diff().abs().fillna(position.iloc[0]).mean()),
        )
    _write_report(Path(output_root) / "historical_replay_v0.md", replay, rows, summaries)
    return summaries


def _write_report(path: Path, replay, rows: pd.DataFrame, summaries: dict[str, ReplaySummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Historical replay V0",
        "",
        (
            "Rule-agent, fixed-price expanding walk-forward replay under the assumptions in "
            "[`docs/historical_replay_protocol.md`](../docs/historical_replay_protocol.md)."
        ),
        "",
        f"- As-of feature rows: {len(replay.frame)} ({replay.frame.index.min().date()} to {replay.frame.index.max().date()}).",
        f"- Mature online-quality decisions: {len(rows)} ({rows.index.min().date()} to {rows.index.max().date()}).",
        "- Splits: expanding 504-day train, 5-day gap, 126-day test; each threshold is the train-only risk 75th percentile.",
        "- Portfolio figures use every fifth routed decision to avoid overlapping five-day label returns; no transaction costs yet.",
        "",
        "| Router | Coverage | Routed error | False rejection | Mean 5d trade return | Max drawdown | Turnover |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries.values():
        lines.append(
            f"| {summary.method} | {summary.coverage:.3f} | {summary.routed_error:.3f} | "
            f"{summary.false_rejection:.3f} | {summary.mean_trade_return:.4f} | "
            f"{summary.max_drawdown:.3f} | {summary.turnover:.3f} |"
        )
    lines.extend(("", "## Boundary", "", "This is a rule-agent replay, not an LLM evaluation or a claim that the historical data supplies audited Bloomberg publication timestamps."))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_historical_replay()
