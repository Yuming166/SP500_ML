"""V1 replay: train-only probability agents with root-source aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sp500_forecastability.historical_data import build_historical_replay_data
from sp500_forecastability.splits import walk_forward_splits

AGENTS = {
    "market_raw": ("market_bloomberg", ["market_return_5d", "market_detail_0_5"]),
    "market_band": ("market_bloomberg", ["market_band_5_20", "market_band_20_60"]),
    "market_trend": ("market_bloomberg", ["market_trend_60", "market_return_5d"]),
    "vix": ("vix_bloomberg", ["vix", "vix_change_5d"]),
    "macro": ("macro_bloomberg", ["10Y", "credit", "credit_change_5d"]),
    "sentiment": ("google_trends", ["recession", "inflation", "unemployment"]),
    "put_call": ("cboe_stock_pcr", ["put_call_ratio", "put_call_change_5d"]),
    "spy_flow": ("etf_flow_family", ["spy_fund_flow", "spy_flow_5d"]),
    "ivv_flow": ("etf_flow_family", ["ivv_fund_flow", "ivv_flow_5d"]),
    "voo_shares": ("etf_flow_family", ["voo_shares", "voo_shares_change_5d"]),
}


@dataclass(frozen=True)
class V1Summary:
    router: str
    coverage: float
    error: float
    mean_trade_return: float
    max_drawdown: float


def _drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=test.index)
    for agent, (_, columns) in AGENTS.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.2, max_iter=1000))
        model.fit(train[columns], train["target_up_5d"].astype(int))
        output[agent] = model.predict_proba(test[columns])[:, 1]
    return output


def run_historical_replay_v1(
    data_directory: Path | str = "data", output_root: Path | str = "results"
) -> dict[str, V1Summary]:
    """Compare duplicated-agent voting against root-source probability grouping."""

    frame = build_historical_replay_data(data_directory).frame
    rows: dict[str, list[pd.DataFrame]] = {"majority": [], "confidence": [], "provenance": []}
    for train_ix, test_ix in walk_forward_splits(len(frame), 504, 126, step=126, gap=5):
        train, test = frame.iloc[list(train_ix)], frame.iloc[list(test_ix)].copy()
        probabilities = _fit_predict(train, test)
        actions = probabilities.ge(0.5).astype(int)
        test["majority_action"] = (actions.mean(axis=1) >= 0.5).astype(int)
        test["majority_confidence"] = probabilities.sub(0.5).abs().mul(2).mean(axis=1)
        root_probabilities = pd.DataFrame(
            {
                root: probabilities[[name for name, (agent_root, _) in AGENTS.items() if agent_root == root]].mean(axis=1)
                for root in sorted({root for root, _ in AGENTS.values()})
            }
        )
        test["provenance_action"] = (root_probabilities.mean(axis=1) >= 0.5).astype(int)
        test["provenance_confidence"] = root_probabilities.sub(0.5).abs().mul(2).mean(axis=1)
        market_probability = root_probabilities["market_bloomberg"]
        other_probability = root_probabilities.drop(columns="market_bloomberg").mean(axis=1)
        test["soft_cap_action"] = (0.45 * market_probability + 0.55 * other_probability >= 0.5).astype(int)
        for router, action, confidence in (
            ("majority", "majority_action", "majority_confidence"),
            ("confidence", "majority_action", "majority_confidence"),
            ("provenance", "provenance_action", "provenance_confidence"),
            ("soft_root_cap", "soft_cap_action", "provenance_confidence"),
        ):
            selected = pd.Series(True, index=test.index)
            out = test[["forward_return_5d", "target_up_5d", action]].copy()
            out["selected"] = selected
            out["action"] = out[action]
            rows.setdefault(router, []).append(out)
    summaries = {}
    for router, pieces in rows.items():
        result = pd.concat(pieces)
        routed = result[result["selected"]]
        trades = routed.iloc[::5]
        returns = trades["forward_return_5d"] * trades["action"]
        summaries[router] = V1Summary(
            router, float(result["selected"].mean()),
            float((routed["action"] != routed["target_up_5d"]).mean()),
            float(returns.mean()), _drawdown(returns),
        )
    _report(Path(output_root) / "historical_replay_v1.md", summaries)
    return summaries


def _report(path: Path, summaries: dict[str, V1Summary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Historical replay V1: source grouping", "",
        (
            "Each agent is a logistic model fit only on its expanding training window. "
            "Three market agents share one root source and SPY/IVV/VOO flow agents share an ETF-flow "
            "family; Provenance averages within each source group before combining groups."
        ), "",
        "| Router | Coverage | Error | Mean non-overlapping 5d return | Max drawdown |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries.values():
        lines.append(f"| {item.router} | {item.coverage:.3f} | {item.error:.3f} | {item.mean_trade_return:.4f} | {item.max_drawdown:.3f} |")
    lines.extend(("", "No transaction costs or LLM calls are included. This is a source-duplication ablation, not a final trading result."))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_historical_replay_v1()
