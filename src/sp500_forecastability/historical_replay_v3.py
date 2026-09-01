"""V3 historical replay: cross-domain reliability routing (V10 transfer).

Adds 3 pre-registered routers to the V2 ablation:
  R_v10:  weighted vote with V10 per-agent AUROC (frozen, pre-registered)
  R_brier: weighted vote with 1/(OOF Brier + 0.01)
  R_equal: equal-weight (control)
V0/V1/V2's 4 routers (majority, confidence, recent_performance, provenance)
are kept unchanged. V3 reports all 7 routers in one table.

Pre-registered in `docs/historical_replay_v3_protocol.md` (frozen
2026-09-01). No parameter tuning on V3 outcomes. All 7 routers reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sp500_forecastability.historical_data import build_historical_replay_data
from sp500_forecastability.splits import walk_forward_splits

HORIZON_DAYS = 5
QUALITY_WINDOW = 126

# V3 D2_v3: V10 per-agent AUROC_fragility, frozen from
# analysis/individual_agent_reliability.json (V10, N=100, V5 salt).
# These weights are LLM-derived, NOT retrained on market data.
V10_PER_AGENT_AUROC = {
    "market_raw":            0.423,  # literal_evidence
    "market_band":           0.493,  # skeptical_auditor
    "market_trend":          0.439,  # consistency_checker
    "vix":                   0.427,  # counterfactual_reasoner
    "macro":                 0.468,  # minimal_judge
    "sentiment":             0.423,  # cycle to literal_evidence
    "options":               0.493,  # cycle to skeptical_auditor
    "spy_flow":              0.439,  # cycle to consistency_checker
    "ivv_flow":              0.427,  # cycle to counterfactual_reasoner
    "voo_shares":            0.468,  # cycle to minimal_judge
    "mutual_fund_flow":      0.423,  # cycle to literal_evidence
}
V10_WEIGHT_FLOOR = 0.5  # D2_v3: clip to [0.5, 1.0] to avoid zeroing any agent

AGENTS = {
    "market_raw": ("market_bloomberg", ["market_return_5d", "market_detail_0_5"]),
    "market_band": ("market_bloomberg", ["market_band_5_20", "market_band_20_60"]),
    "market_trend": ("market_bloomberg", ["market_trend_60", "market_return_5d"]),
    "vix": ("vix_bloomberg", ["vix", "vix_change_5d"]),
    "macro": ("macro_bloomberg", ["10Y", "credit", "credit_change_5d"]),
    "sentiment": ("google_trends", ["recession", "inflation", "unemployment"]),
    "options": (
        "cboe_options",
        ["cboe_total_pcr", "cboe_index_pcr", "cboe_index_stock_spread", "cboe_index_change_5d"],
    ),
    "spy_flow": ("etf_flow_family", ["spy_fund_flow", "spy_flow_5d"]),
    "ivv_flow": ("etf_flow_family", ["ivv_fund_flow", "ivv_flow_5d"]),
    "voo_shares": ("etf_flow_family", ["voo_shares", "voo_shares_change_5d"]),
    "mutual_fund_flow": (
        "ici_mutual_fund_flow",
        [
            "mutual_fund_total_flow",
            "mutual_fund_domestic_flow",
            "mutual_fund_foreign_flow",
            "mutual_fund_total_change_5d",
            "mutual_fund_domestic_share",
        ],
    ),
}


@dataclass(frozen=True)
class V2Summary:
    router: str
    coverage: float
    routed_error: float
    false_rejection: float
    mean_trade_return: float
    max_drawdown: float


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=test.index)
    for agent, (_, columns) in AGENTS.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.2, max_iter=1000))
        model.fit(train[columns], train["target_up_5d"].astype(int))
        output[agent] = model.predict_proba(test[columns])[:, 1]
    return output


def _nested_oof_probabilities(train: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for fit_ix, test_ix in walk_forward_splits(
        len(train), train_size=252, test_size=63, step=63, gap=HORIZON_DAYS
    ):
        pieces.append(_fit_predict(train.iloc[list(fit_ix)], train.iloc[list(test_ix)]))
    return pd.concat(pieces)


def _inverse_loss_weights(losses: pd.Series) -> pd.Series:
    bounded = losses.clip(lower=0.01, upper=1.0)
    weights = bounded.pow(-1.0)
    return weights / weights.sum()


def _root_probabilities(probabilities: pd.DataFrame) -> pd.DataFrame:
    roots = sorted({root for root, _ in AGENTS.values()})
    return pd.DataFrame(
        {
            root: probabilities[
                [agent for agent, (agent_root, _) in AGENTS.items() if agent_root == root]
            ].mean(axis=1)
            for root in roots
        },
        index=probabilities.index,
    )


def _losses(probabilities: pd.DataFrame, labels: pd.Series) -> tuple[pd.Series, pd.Series]:
    agent_losses = probabilities.sub(labels, axis=0).pow(2).tail(QUALITY_WINDOW).mean()
    root_losses = _root_probabilities(probabilities).sub(labels, axis=0).pow(2).tail(QUALITY_WINDOW).mean()
    return agent_losses, root_losses


def _router_rows(
    probabilities: pd.DataFrame, agent_weights: pd.Series, root_weights: pd.Series
) -> pd.DataFrame:
    actions = probabilities.ge(0.5).astype(int)
    majority_action = (actions.mean(axis=1) >= 0.5).astype(int)
    majority_risk = 1.0 - actions.eq(majority_action, axis=0).mean(axis=1)
    agreeing = probabilities.where(actions.eq(majority_action, axis=0))
    confidence_risk = 1.0 - agreeing.sub(0.5).abs().mul(2.0).mean(axis=1)
    recent_probability = probabilities.mul(agent_weights, axis=1).sum(axis=1)
    roots = _root_probabilities(probabilities)
    provenance_probability = roots.mul(root_weights, axis=1).sum(axis=1)
    provenance_risk = (
        1.0 - provenance_probability.sub(0.5).abs().mul(2.0) + roots.std(axis=1, ddof=0)
    )

    # V3 D2_v3: R_v10 weighted vote with frozen V10 per-agent AUROC
    v10_weights = probabilities.columns.map(
        lambda a: max(V10_WEIGHT_FLOOR, V10_PER_AGENT_AUROC.get(a, 0.5))
    )
    v10_weights = pd.Series(v10_weights.values, index=probabilities.columns)
    v10_probability = probabilities.mul(v10_weights, axis=1).div(
        v10_weights.sum()
    ).sum(axis=1)
    v10_risk = 1.0 - v10_probability.sub(0.5).abs().mul(2.0)

    # V3 D3_v3: R_brier weighted vote with 1 / (Brier + 0.01)
    brier_weights = agent_weights.copy()
    brier_probability = probabilities.mul(brier_weights, axis=1).div(
        brier_weights.sum()
    ).sum(axis=1)
    brier_risk = 1.0 - brier_probability.sub(0.5).abs().mul(2.0)

    # V3 D4_v3: R_equal weighted vote (all weights = 1.0)
    equal_weights = pd.Series(1.0, index=probabilities.columns)
    equal_probability = probabilities.mean(axis=1)
    equal_risk = 1.0 - equal_probability.sub(0.5).abs().mul(2.0)

    return pd.DataFrame(
        {
            "majority_action": majority_action,
            "majority_risk": majority_risk,
            "confidence_action": majority_action,
            "confidence_risk": confidence_risk,
            "recent_performance_action": (recent_probability >= 0.5).astype(int),
            "recent_performance_risk": 1.0 - recent_probability.sub(0.5).abs().mul(2.0),
            "provenance_action": (provenance_probability >= 0.5).astype(int),
            "provenance_risk": provenance_risk,
            "v10_action": (v10_probability >= 0.5).astype(int),
            "v10_risk": v10_risk,
            "brier_action": (brier_probability >= 0.5).astype(int),
            "brier_risk": brier_risk,
            "equal_action": (equal_probability >= 0.5).astype(int),
            "equal_risk": equal_risk,
        },
        index=probabilities.index,
    )


def _drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    wealth = (1.0 + returns).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def _test_rows(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    oof = _nested_oof_probabilities(train)
    oof_labels = train.loc[oof.index, "target_up_5d"].astype(float)
    agent_loss, root_loss = _losses(oof, oof_labels)
    calibration = _router_rows(oof, _inverse_loss_weights(agent_loss), _inverse_loss_weights(root_loss))
    thresholds = {
        method: float(calibration[f"{method}_risk"].quantile(0.75))
        for method in (
            "majority", "confidence", "recent_performance", "provenance",
            "v10", "brier", "equal",
        )
    }
    predicted = _fit_predict(train, test)
    history_probabilities, history_labels = oof.copy(), oof_labels.copy()
    rows = []
    for position, timestamp in enumerate(test.index):
        if position >= HORIZON_DAYS:
            matured = test.index[position - HORIZON_DAYS]
            history_probabilities.loc[matured] = predicted.loc[matured]
            history_labels.loc[matured] = float(test.loc[matured, "target_up_5d"])
        agent_loss, root_loss = _losses(history_probabilities, history_labels)
        row = _router_rows(
            predicted.loc[[timestamp]], _inverse_loss_weights(agent_loss), _inverse_loss_weights(root_loss)
        )
        row["target"] = int(test.loc[timestamp, "target_up_5d"])
        row["forward_return"] = float(test.loc[timestamp, "forward_return_5d"])
        for method, threshold in thresholds.items():
            row[f"{method}_selected"] = row[f"{method}_risk"] <= threshold
        rows.append(row)
    return pd.concat(rows), thresholds


def run_historical_replay_v3(
    data_directory: Path | str = "data", output_root: Path | str = "results"
) -> dict[str, V2Summary]:
    """Run the pre-specified V3 cross-domain reliability ablation."""

    frame = build_historical_replay_data(data_directory).frame
    methods = (
        "majority", "confidence", "recent_performance", "provenance",
        "v10", "brier", "equal",
    )
    pieces = []
    threshold_rows = []
    for train_ix, test_ix in walk_forward_splits(
        len(frame), train_size=504, test_size=126, step=126, gap=HORIZON_DAYS
    ):
        rows, thresholds = _test_rows(frame.iloc[list(train_ix)], frame.iloc[list(test_ix)].copy())
        pieces.append(rows)
        threshold_rows.append(thresholds)
    result = pd.concat(pieces)
    summaries = {}
    for method in methods:
        selected = result[f"{method}_selected"]
        routed = result.loc[selected]
        calendar_trades = result.iloc[::HORIZON_DAYS]
        trades = calendar_trades.loc[calendar_trades[f"{method}_selected"]]
        returns = trades["forward_return"] * trades[f"{method}_action"]
        summaries[method] = V2Summary(
            method,
            float(selected.mean()),
            float((routed[f"{method}_action"] != routed["target"]).mean()),
            float((~selected.loc[result["target"] == result[f"{method}_action"]]).mean()),
            float(returns.mean()),
            _drawdown(returns),
        )
    _report(Path(output_root) / "historical_replay_v3.md", frame, summaries, threshold_rows)
    return summaries


def _report(
    path: Path,
    frame: pd.DataFrame,
    summaries: dict[str, V2Summary],
    threshold_rows: list[dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    threshold_table = pd.DataFrame(threshold_rows)
    lines = [
        "# Historical replay V3: cross-domain reliability routing",
        "",
        (
            "V3 is specified in "
            "[`docs/historical_replay_v3_protocol.md`](../docs/historical_replay_v3_protocol.md). "
            "It keeps the V0/V1/V2 reports frozen and adds 3 pre-registered "
            "routers (R_v10, R_brier, R_equal) to the V2 ablation. "
            "All 7 routers reported; no cherry-picking."
        ),
        "",
        f"- As-of feature rows: {len(frame)} ({frame.index.min().date()} to {frame.index.max().date()}).",
        "- Splits: expanding 504-day train, five-day gap, 126-day test.",
        "- Every abstention threshold is the train-only nested-OOF 75th-percentile risk.",
        "- Portfolio figures use every fifth calendar decision, so five-day returns do not overlap.",
        "",
        "**7 routers compared (pre-registered; all reported):**",
        "- `majority` / `confidence` / `recent_performance` / `provenance`: inherited from V2",
        "- `v10`: V10 per-agent AUROC_fragility (literal=0.423, skept=0.493, consist=0.439, cf=0.427, min=0.468), clipped to [0.5, 1.0]",
        "- `brier`: market-OOF per-agent Brier, weight = 1 / (Brier + 0.01)",
        "- `equal`: uniform weights (control = simple mean)",
        "",
        "| Router | Coverage | Routed error | False rejection | Mean 5d trade return | Max drawdown |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries.values():
        lines.append(
            f"| {summary.router} | {summary.coverage:.3f} | {summary.routed_error:.3f} | "
            f"{summary.false_rejection:.3f} | {summary.mean_trade_return:.4f} | "
            f"{summary.max_drawdown:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Mean train-only risk thresholds",
            "",
            "| Majority | Confidence | Recent performance | Provenance | V10 | Brier | Equal |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            "| " + " | ".join(f"{threshold_table[column].mean():.3f}"
                              for column in threshold_table) + " |",
            "",
            "## Boundary",
            "",
            (
                "No transaction costs or intraday release-time audit are included. "
                "This replay is evidence about selective routing under stated as-of "
                "assumptions, not investment advice or proof of causal market impact. "
                "R_v10 weights are inherited frozen from V10's per-agent AUROC; "
                "R_brier weights are computed from train-only OOF predictions; "
                "R_equal is a uniform-weight control. None of the three new routers "
                "is tuned on V3 outcomes."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_historical_replay_v3()