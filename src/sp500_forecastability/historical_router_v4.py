"""V4 cross-domain causal-provenance selective router.

The router transfers a compact intervention signature from the frozen real-LLM
V12.1 records and adapts it inside each market outer-training window.  Market
test outcomes never enter fitting, calibration, quality weights, or routing.

The complete frozen contract lives in
``docs/historical_router_v4_preregistration.md``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from math import ceil, log
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from sp500_forecastability import pilot_llm_v12_1
from sp500_forecastability.historical_data import build_historical_replay_data
from sp500_forecastability.historical_replay_v3 import (
    AGENTS,
    HORIZON_DAYS,
    _fit_predict,
    _inverse_loss_weights,
    _losses,
    _nested_oof_probabilities,
    _root_probabilities,
)
from sp500_forecastability.splits import walk_forward_splits

PROTOCOL_VERSION = "historical-router-v4-2026-09-02"
DEFAULT_SOURCE_RECORDS = Path("results/pilot_llm_v12_1/formal/records.jsonl")
DEFAULT_OUTPUT_ROOT = Path("results/historical_router_v4")

TARGET_COVERAGE = 0.80
FIT_FRACTION = 0.70
GROUP_TEMPERATURE = 0.10
L2_PENALTY = 0.01
ANCHOR_PENALTY = 0.10
STRESS_PENALTY = 0.10
STRESS_INCREMENT = 0.20
STRESS_MARGIN = 0.02
MIN_QUALITY_HISTORY = 10
QUALITY_WINDOW = 126
BLOCK_LENGTH = 21
BOOTSTRAP_REPLICATES = 1_000
BOOTSTRAP_SEED = 20_260_902

COMMON_FEATURES = (
    "intervention_inertia",
    "flip_inertia",
    "source_concentration",
)
MARKET_FEATURES = COMMON_FEATURES + (
    "consensus_risk",
    "action_confidence_risk",
    "root_disagreement",
    "quality_risk",
)
METHOD_ORDER = ("provenance", "recent_performance", "majority")
METHOD_OFFSET_NAMES = ("method_recent_performance", "method_majority")
BASELINE_METHODS = ("majority", "confidence", "recent_performance", "provenance")
ABLATION_METHODS = (
    "cpr_no_anchor",
    "cpr_no_group_dro",
    "cpr_no_stress",
    "fixed_structural",
)


@dataclass(frozen=True)
class RiskModel:
    """Small interpretable risk model with non-negative structural coefficients."""

    intercept: float
    coefficients: tuple[float, ...]
    method_offsets: tuple[float, float]
    objective: float
    converged: bool

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        matrix = rows.loc[:, MARKET_FEATURES].to_numpy(dtype=float)
        methods = _method_matrix(rows["method"])
        logits = (
            self.intercept
            + matrix @ np.asarray(self.coefficients, dtype=float)
            + methods @ np.asarray(self.method_offsets, dtype=float)
        )
        return expit(logits)

    def as_dict(self) -> dict[str, object]:
        return {
            "intercept": self.intercept,
            "coefficients": dict(zip(MARKET_FEATURES, self.coefficients, strict=True)),
            "method_offsets": dict(zip(METHOD_OFFSET_NAMES, self.method_offsets, strict=True)),
            "objective": self.objective,
            "converged": self.converged,
        }


@dataclass(frozen=True)
class RouterSummary:
    router: str
    coverage: float
    routed_error: float
    false_rejection: float
    selected_brier: float
    aurc: float
    worst_vix_regime_error: float
    mean_5d_return: float
    max_drawdown: float
    turnover: float
    mean_5d_return_net_5bps: float
    mean_5d_return_net_10bps: float


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_source_signature(path: Path | str = DEFAULT_SOURCE_RECORDS) -> pd.DataFrame:
    """Load V12.1's outcome-labelled, high-consensus pre-outcome signatures."""

    pilot_llm_v12_1.configure_v12_1()
    records = _read_jsonl(Path(path))
    rows = pilot_llm_v12_1.v12.v11._risk_rows(records)
    high = [row for row in rows if float(row["agreement"]) >= 0.8]
    frame = pd.DataFrame(
        {
            "intervention_inertia": [float(row["D_inert"]) for row in high],
            "flip_inertia": [float(row["flip_inertia"]) for row in high],
            "source_concentration": [float(row["frac_shared"]) for row in high],
            "error": [int(row["consensus_wrong"]) for row in high],
            "cqid": [str(row["cqid"]) for row in high],
        }
    )
    if len(frame) < 2 or frame["error"].nunique() != 2:
        raise ValueError("source signature needs both error classes")
    return frame


def _binary_log_loss(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, logits) - labels * logits


def fit_source_prior(rows: pd.DataFrame) -> tuple[float, np.ndarray, float]:
    """Fit the frozen non-negative three-coordinate real-LLM source prior."""

    matrix = rows.loc[:, COMMON_FEATURES].to_numpy(dtype=float)
    labels = rows["error"].to_numpy(dtype=float)
    prevalence = float(np.clip(labels.mean(), 1e-6, 1 - 1e-6))
    initial = np.asarray([log(prevalence / (1 - prevalence)), 0.1, 0.1, 0.1])

    def objective(parameters: np.ndarray) -> float:
        logits = parameters[0] + matrix @ parameters[1:]
        return float(
            _binary_log_loss(logits, labels).mean() + L2_PENALTY * np.sum(parameters[1:] ** 2)
        )

    fitted = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=((None, None), (0.0, None), (0.0, None), (0.0, None)),
        options={"maxiter": 1_000, "ftol": 1e-12},
    )
    if not fitted.success:
        raise RuntimeError(f"source-prior fit failed: {fitted.message}")
    return float(fitted.x[0]), np.asarray(fitted.x[1:], dtype=float), float(fitted.fun)


def _method_matrix(methods: Iterable[object]) -> np.ndarray:
    values = [str(value) for value in methods]
    return np.asarray(
        [[float(method == "recent_performance"), float(method == "majority")] for method in values],
        dtype=float,
    )


def _design_matrix(rows: pd.DataFrame) -> np.ndarray:
    features = rows.loc[:, MARKET_FEATURES].to_numpy(dtype=float)
    return np.column_stack((np.ones(len(rows)), features, _method_matrix(rows["method"])))


def _soft_group_loss(losses: np.ndarray, groups: Sequence[object]) -> float:
    group_array = np.asarray([str(group) for group in groups])
    group_losses = np.asarray(
        [losses[group_array == group].mean() for group in sorted(set(group_array))],
        dtype=float,
    )
    return float(
        GROUP_TEMPERATURE * (logsumexp(group_losses / GROUP_TEMPERATURE) - log(len(group_losses)))
    )


def fit_market_router(
    rows: pd.DataFrame,
    source_coefficients: Sequence[float],
    *,
    use_anchor: bool = True,
    use_group_dro: bool = True,
    use_stress: bool = True,
) -> RiskModel:
    """Fit one outer-fold CPR model under the frozen constrained objective."""

    if rows.empty or rows["error"].nunique() != 2:
        raise ValueError("market router fit requires both error classes")
    design = _design_matrix(rows)
    labels = rows["error"].to_numpy(dtype=float)
    anchor = np.asarray(source_coefficients, dtype=float)
    prevalence = float(np.clip(labels.mean(), 1e-6, 1 - 1e-6))
    initial = np.zeros(design.shape[1], dtype=float)
    initial[0] = log(prevalence / (1 - prevalence))
    initial[1 : 1 + len(COMMON_FEATURES)] = anchor
    initial[1 + len(COMMON_FEATURES) : 1 + len(MARKET_FEATURES)] = 0.1

    stressed = design.copy()
    common_slice = slice(1, 1 + len(COMMON_FEATURES))
    stressed[:, common_slice] = np.clip(stressed[:, common_slice] + STRESS_INCREMENT, 0.0, 1.0)

    def objective(parameters: np.ndarray) -> float:
        logits = design @ parameters
        losses = _binary_log_loss(logits, labels)
        value = (
            _soft_group_loss(losses, rows["group"].tolist())
            if use_group_dro
            else float(losses.mean())
        )
        value += L2_PENALTY * float(np.sum(parameters[1:] ** 2))
        if use_anchor:
            value += ANCHOR_PENALTY * float(np.sum((parameters[common_slice] - anchor) ** 2))
        if use_stress:
            clean_risk = expit(logits)
            stressed_risk = expit(stressed @ parameters)
            shortfall = np.maximum(0.0, STRESS_MARGIN - (stressed_risk - clean_risk))
            value += STRESS_PENALTY * float(np.mean(shortfall**2))
        return float(value)

    free_offsets = len(METHOD_OFFSET_NAMES)
    bounds = (
        ((None, None),)
        + tuple((0.0, None) for _ in MARKET_FEATURES)
        + tuple((None, None) for _ in range(free_offsets))
    )
    fitted = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1_000, "ftol": 1e-12},
    )
    if not fitted.success:
        raise RuntimeError(f"CPR fit failed: {fitted.message}")
    feature_end = 1 + len(MARKET_FEATURES)
    return RiskModel(
        intercept=float(fitted.x[0]),
        coefficients=tuple(float(value) for value in fitted.x[1:feature_end]),
        method_offsets=tuple(float(value) for value in fitted.x[feature_end:]),
        objective=float(fitted.fun),
        converged=bool(fitted.success),
    )


def _equal_weights(names: Sequence[str]) -> pd.Series:
    return pd.Series(1.0 / len(names), index=list(names), dtype=float)


def _quality_state(
    probabilities: pd.DataFrame, labels: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if len(probabilities) < MIN_QUALITY_HISTORY:
        agent_names = list(AGENTS)
        root_names = sorted({root for root, _columns in AGENTS.values()})
        agent_losses = pd.Series(0.25, index=agent_names, dtype=float)
        root_losses = pd.Series(0.25, index=root_names, dtype=float)
        return (
            _equal_weights(agent_names),
            _equal_weights(root_names),
            agent_losses,
            root_losses,
        )
    bounded_probabilities = probabilities.tail(QUALITY_WINDOW)
    bounded_labels = labels.loc[bounded_probabilities.index]
    agent_losses, root_losses = _losses(bounded_probabilities, bounded_labels)
    return (
        _inverse_loss_weights(agent_losses),
        _inverse_loss_weights(root_losses),
        agent_losses,
        root_losses,
    )


def _root_intervention_signature(root_values: np.ndarray) -> tuple[float, float]:
    original_probability = float(root_values.mean())
    original_action = int(original_probability >= 0.5)
    variants: list[float] = []
    for index, probability in enumerate(root_values):
        others = np.delete(root_values, index)
        variants.append(float(others.mean()))
        variants.append(
            float((root_values.sum() - probability + (1.0 - probability)) / len(root_values))
        )
        variants.append(float((root_values.sum() - probability + 0.5) / len(root_values)))
    variant_array = np.asarray(variants, dtype=float)
    flips = (variant_array >= 0.5).astype(int) != original_action
    stable = (~flips) & (np.abs(variant_array - original_probability) < 0.05)
    return float(stable.mean()), float(1.0 - flips.mean())


def market_candidate_rows(
    probability_row: pd.Series,
    *,
    agent_weights: pd.Series,
    root_weights: pd.Series,
    root_losses: pd.Series,
) -> list[dict[str, object]]:
    """Create three candidate actions and the shared outcome-free signature."""

    probabilities = probability_row.reindex(list(AGENTS)).astype(float)
    actions = probabilities.ge(0.5).astype(int)
    majority_action = int(actions.mean() >= 0.5)
    majority_probability = float(actions.mean())
    agreement = float((actions == majority_action).mean())
    agreeing_confidence = probabilities.loc[actions == majority_action].sub(0.5).abs().mul(2.0)
    confidence_risk = float(1.0 - agreeing_confidence.mean())

    recent_probability = float(probabilities.mul(agent_weights.reindex(probabilities.index)).sum())
    root_frame = _root_probabilities(pd.DataFrame([probabilities], index=[probability_row.name]))
    roots = root_frame.iloc[0].astype(float)
    provenance_probability = float(roots.mul(root_weights.reindex(roots.index)).sum())
    intervention_inertia, flip_inertia = _root_intervention_signature(roots.to_numpy())

    agreeing_roots = [
        AGENTS[agent][0] for agent in probabilities.index if int(actions[agent]) == majority_action
    ]
    root_counts = pd.Series(agreeing_roots, dtype=str).value_counts()
    source_concentration = float(root_counts.max() / max(1, len(agreeing_roots)))
    root_disagreement = float(np.clip(2.0 * roots.std(ddof=0), 0.0, 1.0))
    quality_risk = float(np.clip(root_losses.reindex(roots.index).mean(), 0.0, 1.0))
    consensus_risk = 1.0 - agreement

    candidates = {
        "majority": (majority_probability, majority_action),
        "recent_performance": (recent_probability, int(recent_probability >= 0.5)),
        "provenance": (provenance_probability, int(provenance_probability >= 0.5)),
    }
    shared = {
        "intervention_inertia": intervention_inertia,
        "flip_inertia": flip_inertia,
        "source_concentration": source_concentration,
        "consensus_risk": consensus_risk,
        "root_disagreement": root_disagreement,
        "quality_risk": quality_risk,
        "majority_risk": consensus_risk,
        "confidence_risk": float(np.clip(confidence_risk, 0.0, 1.0)),
        "recent_performance_risk": float(
            np.clip(1.0 - 2.0 * abs(recent_probability - 0.5), 0.0, 1.0)
        ),
        "provenance_risk": float(
            np.clip(
                1.0 - 2.0 * abs(provenance_probability - 0.5) + root_disagreement,
                0.0,
                1.0,
            )
        ),
    }
    return [
        {
            **shared,
            "method": method,
            "action_probability": probability,
            "action": action,
            "action_confidence_risk": float(np.clip(1.0 - 2.0 * abs(probability - 0.5), 0.0, 1.0)),
        }
        for method, (probability, action) in candidates.items()
    ]


def online_market_rows(
    probabilities: pd.DataFrame,
    market: pd.DataFrame,
    *,
    initial_probabilities: pd.DataFrame | None = None,
    initial_labels: pd.Series | None = None,
) -> pd.DataFrame:
    """Build timestamped signatures with quality updated only after label maturity."""

    history_probabilities = (
        initial_probabilities.copy()
        if initial_probabilities is not None
        else pd.DataFrame(columns=probabilities.columns, dtype=float)
    )
    history_labels = (
        initial_labels.astype(float).copy()
        if initial_labels is not None
        else pd.Series(dtype=float)
    )
    output: list[dict[str, object]] = []
    timestamps = list(probabilities.index)
    for position, timestamp in enumerate(timestamps):
        if position >= HORIZON_DAYS:
            matured = timestamps[position - HORIZON_DAYS]
            history_probabilities.loc[matured] = probabilities.loc[matured]
            history_labels.loc[matured] = float(market.loc[matured, "target_up_5d"])
        agent_weights, root_weights, _agent_losses, root_losses = _quality_state(
            history_probabilities, history_labels
        )
        candidates = market_candidate_rows(
            probabilities.loc[timestamp],
            agent_weights=agent_weights,
            root_weights=root_weights,
            root_losses=root_losses,
        )
        target = int(market.loc[timestamp, "target_up_5d"])
        for candidate in candidates:
            output.append(
                {
                    **candidate,
                    "timestamp": timestamp,
                    "target": target,
                    "error": int(int(candidate["action"]) != target),
                    "forward_return": float(market.loc[timestamp, "forward_return_5d"]),
                    "vix": float(market.loc[timestamp, "vix"]),
                }
            )
    return pd.DataFrame(output)


def _split_router_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = sorted(pd.unique(rows["timestamp"]))
    if len(timestamps) < 10:
        raise ValueError("not enough OOF timestamps for router fit/calibration")
    cut = min(len(timestamps) - 1, max(1, int(len(timestamps) * FIT_FRACTION)))
    fit_timestamps = set(timestamps[:cut])
    return rows.loc[rows["timestamp"].isin(fit_timestamps)].copy(), rows.loc[
        ~rows["timestamp"].isin(fit_timestamps)
    ].copy()


def assign_groups(rows: pd.DataFrame, outer_train: pd.DataFrame) -> pd.DataFrame:
    """Assign train-only VIX-tertile x chronological-third robust groups."""

    output = rows.copy()
    low, high = outer_train["vix"].quantile([1 / 3, 2 / 3]).tolist()
    unique_timestamps = sorted(pd.unique(output["timestamp"]))
    position = {timestamp: index for index, timestamp in enumerate(unique_timestamps)}
    denominator = max(1, len(unique_timestamps))

    def group(row: pd.Series) -> str:
        regime = "low" if row["vix"] <= low else ("mid" if row["vix"] <= high else "high")
        segment = min(2, int(3 * position[row["timestamp"]] / denominator))
        return f"{regime}_t{segment}"

    output["group"] = output.apply(group, axis=1)
    return output


def _conformal_quantile(scores: Sequence[float], coverage: float = TARGET_COVERAGE) -> float:
    values = np.sort(np.asarray(scores, dtype=float))
    if not len(values):
        raise ValueError("cannot calibrate an empty score sequence")
    index = min(len(values) - 1, max(0, ceil((len(values) + 1) * coverage) - 1))
    return float(values[index])


def _select_cpr(rows: pd.DataFrame, model: RiskModel) -> pd.DataFrame:
    scored = rows.copy()
    scored["risk"] = model.predict(scored)
    order = {method: index for index, method in enumerate(METHOD_ORDER)}
    scored["tie_order"] = scored["method"].map(order)
    chosen = (
        scored.sort_values(["timestamp", "risk", "tie_order"])
        .groupby("timestamp", sort=True, as_index=False)
        .first()
    )
    return chosen.drop(columns="tie_order")


def _select_fixed_structural(rows: pd.DataFrame) -> pd.DataFrame:
    selected = rows.loc[rows["method"] == "provenance"].copy()
    selected["risk"] = (
        0.1 * selected["intervention_inertia"]
        + 0.3 * selected["flip_inertia"]
        + 0.6 * selected["source_concentration"]
    )
    return selected


def _select_baseline(rows: pd.DataFrame, method: str) -> pd.DataFrame:
    if method in {"majority", "confidence"}:
        selected = rows.loc[rows["method"] == "majority"].copy()
    else:
        selected = rows.loc[rows["method"] == method].copy()
    selected["risk"] = selected[f"{method}_risk"]
    return selected


def _apply_threshold(rows: pd.DataFrame, threshold: float, router: str, fold: int) -> pd.DataFrame:
    output = rows.copy()
    output["selected"] = output["risk"] <= threshold
    output["router"] = router
    output["outer_fold"] = fold
    output["threshold"] = threshold
    return output


def _aurc(rows: pd.DataFrame) -> float:
    ordered = rows.sort_values(["risk", "timestamp"])
    cumulative = ordered["error"].to_numpy(dtype=float).cumsum()
    return float(np.mean(cumulative / np.arange(1, len(ordered) + 1)))


def _drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    wealth = (1.0 + returns).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def summarize_policy(rows: pd.DataFrame, router: str) -> RouterSummary:
    selected = rows["selected"].astype(bool)
    routed = rows.loc[selected]
    correct = rows["error"] == 0
    false_rejection = float((~selected & correct).sum() / max(1, int(correct.sum())))
    regime_errors = [
        float(group.loc[group["selected"], "error"].mean())
        for _regime, group in rows.groupby("vix_regime")
        if bool(group["selected"].any())
    ]

    calendar = rows.sort_values("timestamp").iloc[::HORIZON_DAYS].copy()
    calendar["position"] = calendar["action"].where(calendar["selected"], 0).astype(float)
    changes = calendar["position"].diff().abs().fillna(calendar["position"].abs())
    gross = calendar["forward_return"] * calendar["position"]
    net_5 = gross - 0.0005 * changes
    net_10 = gross - 0.0010 * changes
    return RouterSummary(
        router=router,
        coverage=float(selected.mean()),
        routed_error=float(routed["error"].mean()) if len(routed) else float("nan"),
        false_rejection=false_rejection,
        selected_brier=(
            float(((routed["action_probability"] - routed["target"]) ** 2).mean())
            if len(routed)
            else float("nan")
        ),
        aurc=_aurc(rows),
        worst_vix_regime_error=max(regime_errors) if regime_errors else float("nan"),
        mean_5d_return=float(gross.mean()),
        max_drawdown=_drawdown(gross),
        turnover=float(changes.mean()),
        mean_5d_return_net_5bps=float(net_5.mean()),
        mean_5d_return_net_10bps=float(net_10.mean()),
    )


def _moving_block_indices(
    n_rows: int, *, rng: np.random.Generator, block_length: int = BLOCK_LENGTH
) -> np.ndarray:
    indices: list[int] = []
    while len(indices) < n_rows:
        start = int(rng.integers(0, n_rows))
        indices.extend((start + offset) % n_rows for offset in range(block_length))
    return np.asarray(indices[:n_rows], dtype=int)


def primary_error_difference_ci(
    cpr_rows: pd.DataFrame,
    confidence_rows: pd.DataFrame,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, tuple[float, float]]:
    """Moving-block CI for CPR minus confidence routed error."""

    left = cpr_rows.sort_values("timestamp").reset_index(drop=True)
    right = confidence_rows.sort_values("timestamp").reset_index(drop=True)
    if left["timestamp"].tolist() != right["timestamp"].tolist():
        raise ValueError("primary policies must share identical timestamps")

    def difference(a: pd.DataFrame, b: pd.DataFrame) -> float:
        a_selected = a.loc[a["selected"]]
        b_selected = b.loc[b["selected"]]
        if a_selected.empty or b_selected.empty:
            return float("nan")
        return float(a_selected["error"].mean() - b_selected["error"].mean())

    observed = difference(left, right)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        index = _moving_block_indices(len(left), rng=rng)
        value = difference(left.iloc[index], right.iloc[index])
        if np.isfinite(value):
            samples.append(value)
    if not samples:
        return observed, (float("nan"), float("nan"))
    low, high = np.quantile(np.asarray(samples), [0.025, 0.975])
    return observed, (float(low), float(high))


def _vix_regime(values: pd.Series, train: pd.DataFrame) -> pd.Series:
    low, high = train["vix"].quantile([1 / 3, 2 / 3]).tolist()
    return values.map(lambda value: "low" if value <= low else ("mid" if value <= high else "high"))


def _stress_audit(model: RiskModel, rows: pd.DataFrame) -> dict[str, object]:
    clean = model.predict(rows)
    stressed = rows.copy()
    for feature in COMMON_FEATURES:
        stressed[feature] = np.clip(stressed[feature] + STRESS_INCREMENT, 0.0, 1.0)
    stressed_risk = model.predict(stressed)
    deltas = stressed_risk - clean
    return {
        "monotonicity_violations": int((deltas < -1e-12).sum()),
        "paired_margin_satisfaction": float((deltas + 1e-12 >= STRESS_MARGIN).mean()),
        "mean_stress_risk_increase": float(deltas.mean()),
    }


def _write_outputs(
    output_root: Path,
    *,
    source: dict[str, object],
    summaries: Mapping[str, RouterSummary],
    primary: dict[str, object],
    fold_details: list[dict[str, object]],
    audits: dict[str, object],
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "RETROSPECTIVE_DEVELOPMENT_ONLY",
        "source_prior": source,
        "primary": primary,
        "summaries": {name: asdict(summary) for name, summary in summaries.items()},
        "folds": fold_details,
        "audits": audits,
        "claim_boundary": (
            "This is a retrospective cross-domain routing development experiment. "
            "It is not evidence of investment alpha, causal market impact, or "
            "prospective cross-model generalization."
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Historical Router V4: CPR-Router",
        "",
        f"- Protocol: `{PROTOCOL_VERSION}`",
        "- Status: **RETROSPECTIVE DEVELOPMENT ONLY**",
        f"- Source LLM questions: {source['n_questions']}",
        f"- Market outer folds: {len(fold_details)}",
        f"- Actual market agents / roots: {audits['agent_count']} / {audits['root_count']}",
        "",
        "## Primary developmental endpoint",
        "",
        (
            f"`routed_error(CPR) - routed_error(confidence)` = "
            f"{primary['difference']:.4f} "
            f"(95% moving-block CI [{primary['ci'][0]:.4f}, {primary['ci'][1]:.4f}])."
        ),
        f"Coverage gate [0.70, 0.90]: **{primary['coverage_gate']}**.",
        "This endpoint is diagnostic, not eligible for a confirmatory PASS claim.",
        "",
        "## All routers and frozen ablations",
        "",
        (
            "| Router | Coverage | Routed error | Brier | AURC | Worst VIX error | "
            "Mean 5d return | Max drawdown | Turnover | Net 5bps | Net 10bps |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {summary.coverage:.3f} | {summary.routed_error:.3f} | "
            f"{summary.selected_brier:.3f} | {summary.aurc:.3f} | "
            f"{summary.worst_vix_regime_error:.3f} | {summary.mean_5d_return:.4f} | "
            f"{summary.max_drawdown:.3f} | {summary.turnover:.3f} | "
            f"{summary.mean_5d_return_net_5bps:.4f} | "
            f"{summary.mean_5d_return_net_10bps:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Structural audit",
            "",
            f"- Monotonicity violations: {audits['monotonicity_violations']}",
            (
                "- Mean paired-stress margin satisfaction: "
                f"{audits['paired_margin_satisfaction']:.3f}"
            ),
            (
                "- Mean risk increase under frozen paired stress: "
                f"{audits['mean_stress_risk_increase']:.4f}"
            ),
            "",
            "## Interpretation boundary",
            "",
            (
                "CPR-Router transfers an abstract intervention signature from real LLM-agent "
                "records and adapts it using only matured market-training labels. The current "
                "date range was previously inspected by V0-V3, so this run is evidence about "
                "method feasibility and failure modes, not a prospective financial result."
            ),
        ]
    )
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def run_historical_router_v4(
    data_directory: Path | str = "data",
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    source_records: Path | str = DEFAULT_SOURCE_RECORDS,
) -> dict[str, object]:
    """Run the frozen retrospective V4 cross-domain router experiment."""

    source_rows = load_source_signature(source_records)
    source_intercept, source_coefficients, source_objective = fit_source_prior(source_rows)
    market = build_historical_replay_data(data_directory).frame
    policy_pieces: dict[str, list[pd.DataFrame]] = {
        name: [] for name in (*BASELINE_METHODS, "cpr_router", *ABLATION_METHODS)
    }
    fold_details: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    outer_splits = list(
        walk_forward_splits(len(market), train_size=504, test_size=126, step=126, gap=HORIZON_DAYS)
    )
    for fold, (train_indices, test_indices) in enumerate(outer_splits):
        outer_train = market.iloc[list(train_indices)]
        outer_test = market.iloc[list(test_indices)]
        oof_probabilities = _nested_oof_probabilities(outer_train)
        oof_rows = online_market_rows(oof_probabilities, outer_train)
        fit_rows, calibration_rows = _split_router_rows(oof_rows)
        fit_rows = assign_groups(fit_rows, outer_train)

        models = {
            "cpr_router": fit_market_router(fit_rows, source_coefficients),
            "cpr_no_anchor": fit_market_router(fit_rows, source_coefficients, use_anchor=False),
            "cpr_no_group_dro": fit_market_router(
                fit_rows, source_coefficients, use_group_dro=False
            ),
            "cpr_no_stress": fit_market_router(fit_rows, source_coefficients, use_stress=False),
        }
        predicted = _fit_predict(outer_train, outer_test)
        oof_labels = outer_train.loc[oof_probabilities.index, "target_up_5d"].astype(float)
        test_rows = online_market_rows(
            predicted,
            outer_test,
            initial_probabilities=oof_probabilities,
            initial_labels=oof_labels,
        )
        test_rows["vix_regime"] = _vix_regime(test_rows["vix"], outer_train)
        calibration_rows["vix_regime"] = _vix_regime(calibration_rows["vix"], outer_train)

        thresholds: dict[str, float] = {}
        for method in BASELINE_METHODS:
            calibration_policy = _select_baseline(calibration_rows, method)
            thresholds[method] = _conformal_quantile(calibration_policy["risk"])
            test_policy = _select_baseline(test_rows, method)
            policy_pieces[method].append(
                _apply_threshold(test_policy, thresholds[method], method, fold)
            )
        for name, model in models.items():
            calibration_policy = _select_cpr(calibration_rows, model)
            thresholds[name] = _conformal_quantile(calibration_policy["risk"])
            test_policy = _select_cpr(test_rows, model)
            policy_pieces[name].append(_apply_threshold(test_policy, thresholds[name], name, fold))
        calibration_fixed = _select_fixed_structural(calibration_rows)
        thresholds["fixed_structural"] = _conformal_quantile(calibration_fixed["risk"])
        test_fixed = _select_fixed_structural(test_rows)
        policy_pieces["fixed_structural"].append(
            _apply_threshold(test_fixed, thresholds["fixed_structural"], "fixed_structural", fold)
        )
        fold_audit = _stress_audit(models["cpr_router"], fit_rows)
        audit_rows.append(fold_audit)
        fold_details.append(
            {
                "fold": fold,
                "train_start": str(outer_train.index.min().date()),
                "train_end": str(outer_train.index.max().date()),
                "test_start": str(outer_test.index.min().date()),
                "test_end": str(outer_test.index.max().date()),
                "fit_timestamps": int(fit_rows["timestamp"].nunique()),
                "calibration_timestamps": int(calibration_rows["timestamp"].nunique()),
                "thresholds": thresholds,
                "models": {name: model.as_dict() for name, model in models.items()},
                "audit": fold_audit,
            }
        )

    policy_rows = {
        name: pd.concat(pieces).sort_values("timestamp").reset_index(drop=True)
        for name, pieces in policy_pieces.items()
    }
    summaries = {name: summarize_policy(rows, name) for name, rows in policy_rows.items()}
    difference, interval = primary_error_difference_ci(
        policy_rows["cpr_router"], policy_rows["confidence"]
    )
    primary = {
        "name": "routed_error_cpr_minus_confidence",
        "difference": difference,
        "ci": list(interval),
        "coverage_gate": bool(0.70 <= summaries["cpr_router"].coverage <= 0.90),
        "directional_target_met": bool(
            np.isfinite(interval[1])
            and interval[1] < 0.0
            and 0.70 <= summaries["cpr_router"].coverage <= 0.90
        ),
        "confirmatory_eligible": False,
    }
    audits = {
        "agent_count": len(AGENTS),
        "root_count": len({root for root, _columns in AGENTS.values()}),
        "outer_folds": len(outer_splits),
        "outer_test_rows": len(policy_rows["cpr_router"]),
        "monotonicity_violations": int(
            sum(int(row["monotonicity_violations"]) for row in audit_rows)
        ),
        "paired_margin_satisfaction": float(
            np.mean([float(row["paired_margin_satisfaction"]) for row in audit_rows])
        ),
        "mean_stress_risk_increase": float(
            np.mean([float(row["mean_stress_risk_increase"]) for row in audit_rows])
        ),
        "legacy_protocol_documented_agents": 6,
        "legacy_protocol_documented_roots": 5,
        "legacy_contract_mismatch_preserved_and_corrected_in_v4": True,
    }
    source = {
        "records": str(source_records),
        "n_questions": len(source_rows),
        "error_prevalence": float(source_rows["error"].mean()),
        "intercept": source_intercept,
        "coefficients": dict(zip(COMMON_FEATURES, source_coefficients, strict=True)),
        "objective": source_objective,
    }
    return _write_outputs(
        Path(output_root),
        source=source,
        summaries=summaries,
        primary=primary,
        fold_details=fold_details,
        audits=audits,
    )


if __name__ == "__main__":
    run_historical_router_v4()
