"""V5 adaptive mechanism-invariant selective router.

V5 is a post-V4 retrospective development experiment.  It replaces the fixed
cross-domain coefficient anchor with a shift-monotone gate, fits risk ordering
with pairwise losses, imposes exact target-mechanism bounds, and calibrates the
chosen candidate in a later temporal slice.  The frozen contract is documented
in ``docs/historical_router_v5_preregistration.md``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import log
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from sp500_forecastability import historical_router_v4 as v4
from sp500_forecastability.historical_data import build_historical_replay_data
from sp500_forecastability.historical_replay_v3 import (
    AGENTS,
    HORIZON_DAYS,
    _fit_predict,
    _nested_oof_probabilities,
)
from sp500_forecastability.splits import walk_forward_splits

PROTOCOL_VERSION = "historical-router-v5-2026-09-02"
DEFAULT_SOURCE_RECORDS = Path("results/pilot_llm_v12_1/formal/records.jsonl")
DEFAULT_V4_SUMMARY = Path("results/historical_router_v4/summary.json")
DEFAULT_OUTPUT_ROOT = Path("results/historical_router_v5")

TARGET_COVERAGE = 0.80
CALIBRATION_HEAD_FRACTION = 0.50
GROUP_TEMPERATURE = 0.10
RANK_L2_PENALTY = 0.01
GATE_L2_PENALTY = 0.001
PAIR_CAP_PER_GROUP = 4_096
PAIR_SEED = 20_260_903
HARD_COMMON_MINIMUM = 0.25
INTERVENTION_INCREMENT = 0.20
FEATURE_SD_FLOOR = 0.05
SCORE_SD_FLOOR = 0.05
BLOCK_LENGTH = 21
BOOTSTRAP_REPLICATES = 1_000
BOOTSTRAP_SEED = 20_260_903

COMMON_FEATURES = v4.COMMON_FEATURES
MARKET_FEATURES = v4.MARKET_FEATURES
METHOD_ORDER = v4.METHOD_ORDER
METHOD_OFFSET_NAMES = v4.METHOD_OFFSET_NAMES
BASELINE_METHODS = v4.BASELINE_METHODS
AMIR_METHODS = (
    "amir_router",
    "amir_target_only",
    "amir_fixed_gate",
    "amir_no_hard_constraint",
    "amir_no_calibration",
)


@dataclass(frozen=True)
class SourceRanker:
    coefficients: tuple[float, float, float]
    objective: float
    converged: bool
    pair_count: int

    def predict_logit(self, rows: pd.DataFrame) -> np.ndarray:
        matrix = rows.loc[:, COMMON_FEATURES].to_numpy(dtype=float)
        return matrix @ np.asarray(self.coefficients, dtype=float)

    def as_dict(self) -> dict[str, object]:
        return {
            "coefficients": dict(zip(COMMON_FEATURES, self.coefficients, strict=True)),
            "objective": self.objective,
            "converged": self.converged,
            "pair_count": self.pair_count,
        }


@dataclass(frozen=True)
class TargetRanker:
    coefficients: tuple[float, ...]
    method_offsets: tuple[float, float]
    objective: float
    converged: bool
    pair_count: int
    hard_common_minimum: float

    def predict_logit(self, rows: pd.DataFrame) -> np.ndarray:
        features = rows.loc[:, MARKET_FEATURES].to_numpy(dtype=float)
        methods = v4._method_matrix(rows["method"])
        return (
            features @ np.asarray(self.coefficients, dtype=float)
            + methods @ np.asarray(self.method_offsets, dtype=float)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "coefficients": dict(zip(MARKET_FEATURES, self.coefficients, strict=True)),
            "method_offsets": dict(
                zip(METHOD_OFFSET_NAMES, self.method_offsets, strict=True)
            ),
            "objective": self.objective,
            "converged": self.converged,
            "pair_count": self.pair_count,
            "hard_common_minimum": self.hard_common_minimum,
        }


@dataclass(frozen=True)
class AdaptiveRanker:
    source: SourceRanker
    target: TargetRanker
    feature_mean: tuple[float, float, float]
    feature_sd: tuple[float, float, float]
    source_score_mean: float
    source_score_sd: float
    target_score_mean: float
    target_score_sd: float
    gate_intercept: float
    gate_shift_slope: float
    gate_objective: float
    gate_converged: bool
    gate_pair_count: int
    forced_gate: float | None = None

    def shift(self, rows: pd.DataFrame) -> np.ndarray:
        matrix = rows.loc[:, COMMON_FEATURES].to_numpy(dtype=float)
        mean = np.asarray(self.feature_mean, dtype=float)
        sd = np.asarray(self.feature_sd, dtype=float)
        distance = np.mean(((matrix - mean) / sd) ** 2, axis=1)
        return distance / (1.0 + distance)

    def gate(self, rows: pd.DataFrame) -> np.ndarray:
        if self.forced_gate is not None:
            return np.full(len(rows), self.forced_gate, dtype=float)
        return expit(self.gate_intercept - self.gate_shift_slope * self.shift(rows))

    def component_scores(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        source = (
            self.source.predict_logit(rows) - self.source_score_mean
        ) / self.source_score_sd
        target = (
            self.target.predict_logit(rows) - self.target_score_mean
        ) / self.target_score_sd
        return source, target

    def predict_logit(self, rows: pd.DataFrame) -> np.ndarray:
        source, target = self.component_scores(rows)
        gate = self.gate(rows)
        return gate * source + (1.0 - gate) * target

    def with_forced_gate(self, value: float | None) -> AdaptiveRanker:
        return AdaptiveRanker(**{**asdict(self), "source": self.source, "target": self.target,
                                 "forced_gate": value})

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source.as_dict(),
            "target": self.target.as_dict(),
            "feature_mean": dict(zip(COMMON_FEATURES, self.feature_mean, strict=True)),
            "feature_sd": dict(zip(COMMON_FEATURES, self.feature_sd, strict=True)),
            "source_score_mean": self.source_score_mean,
            "source_score_sd": self.source_score_sd,
            "target_score_mean": self.target_score_mean,
            "target_score_sd": self.target_score_sd,
            "gate_intercept": self.gate_intercept,
            "gate_shift_slope": self.gate_shift_slope,
            "gate_objective": self.gate_objective,
            "gate_converged": self.gate_converged,
            "gate_pair_count": self.gate_pair_count,
            "forced_gate": self.forced_gate,
        }


@dataclass(frozen=True)
class CalibrationHead:
    intercept: float
    slope: float
    objective: float | None
    converged: bool
    n_rows: int

    def predict(self, scores: Sequence[float]) -> np.ndarray:
        values = np.asarray(scores, dtype=float)
        return expit(self.intercept + self.slope * values)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _pair_indices(
    labels: Sequence[int | float],
    groups: Sequence[object],
    *,
    seed: int = PAIR_SEED,
    cap_per_group: int = PAIR_CAP_PER_GROUP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic error/correct pair indices and their group labels."""

    label_array = np.asarray(labels, dtype=int)
    group_array = np.asarray([str(group) for group in groups])
    rng = np.random.default_rng(seed)
    positive_parts: list[np.ndarray] = []
    negative_parts: list[np.ndarray] = []
    pair_groups: list[np.ndarray] = []
    for group in sorted(set(group_array)):
        members = np.flatnonzero(group_array == group)
        positive = members[label_array[members] == 1]
        negative = members[label_array[members] == 0]
        if not len(positive) or not len(negative):
            continue
        total = len(positive) * len(negative)
        if total <= cap_per_group:
            flat = np.arange(total, dtype=int)
        else:
            flat = np.sort(rng.choice(total, size=cap_per_group, replace=False))
        positive_parts.append(positive[flat // len(negative)])
        negative_parts.append(negative[flat % len(negative)])
        pair_groups.append(np.full(len(flat), group, dtype=object))
    if not positive_parts:
        raise ValueError("pairwise ranking requires an error/correct pair in at least one group")
    return (
        np.concatenate(positive_parts),
        np.concatenate(negative_parts),
        np.concatenate(pair_groups),
    )


def _smooth_group_loss(losses: np.ndarray, groups: Sequence[object]) -> float:
    return v4._soft_group_loss(losses, groups)


def fit_source_ranker(rows: pd.DataFrame) -> SourceRanker:
    matrix = rows.loc[:, COMMON_FEATURES].to_numpy(dtype=float)
    positive, negative, groups = _pair_indices(
        rows["error"].to_numpy(), np.full(len(rows), "source", dtype=object)
    )
    differences = matrix[positive] - matrix[negative]

    def objective(parameters: np.ndarray) -> float:
        losses = np.logaddexp(0.0, -(differences @ parameters))
        return _smooth_group_loss(losses, groups) + RANK_L2_PENALTY * float(
            np.sum(parameters**2)
        )

    fitted = minimize(
        objective,
        np.full(len(COMMON_FEATURES), 0.1, dtype=float),
        method="L-BFGS-B",
        bounds=tuple((0.0, None) for _ in COMMON_FEATURES),
        options={"maxiter": 1_000, "ftol": 1e-12},
    )
    if not fitted.success:
        raise RuntimeError(f"source ranking fit failed: {fitted.message}")
    return SourceRanker(
        coefficients=tuple(float(value) for value in fitted.x),
        objective=float(fitted.fun),
        converged=bool(fitted.success),
        pair_count=len(positive),
    )


def _target_design(rows: pd.DataFrame) -> np.ndarray:
    features = rows.loc[:, MARKET_FEATURES].to_numpy(dtype=float)
    return np.column_stack((features, v4._method_matrix(rows["method"])))


def fit_target_ranker(
    rows: pd.DataFrame, *, hard_common_minimum: float = HARD_COMMON_MINIMUM
) -> TargetRanker:
    if rows.empty or rows["error"].nunique() != 2:
        raise ValueError("target ranker fit requires both error classes")
    design = _target_design(rows)
    positive, negative, groups = _pair_indices(rows["error"], rows["group"])
    differences = design[positive] - design[negative]

    def objective(parameters: np.ndarray) -> float:
        losses = np.logaddexp(0.0, -(differences @ parameters))
        return _smooth_group_loss(losses, groups) + RANK_L2_PENALTY * float(
            np.sum(parameters**2)
        )

    initial = np.zeros(design.shape[1], dtype=float)
    initial[: len(COMMON_FEATURES)] = max(0.1, hard_common_minimum)
    initial[len(COMMON_FEATURES) : len(MARKET_FEATURES)] = 0.1
    bounds = (
        tuple((hard_common_minimum, None) for _ in COMMON_FEATURES)
        + tuple((0.0, None) for _ in MARKET_FEATURES[len(COMMON_FEATURES) :])
        + tuple((None, None) for _ in METHOD_OFFSET_NAMES)
    )
    fitted = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1_000, "ftol": 1e-12},
    )
    if not fitted.success:
        raise RuntimeError(f"target ranking fit failed: {fitted.message}")
    feature_end = len(MARKET_FEATURES)
    return TargetRanker(
        coefficients=tuple(float(value) for value in fitted.x[:feature_end]),
        method_offsets=tuple(float(value) for value in fitted.x[feature_end:]),
        objective=float(fitted.fun),
        converged=bool(fitted.success),
        pair_count=len(positive),
        hard_common_minimum=hard_common_minimum,
    )


def _safe_moments(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(max(array.std(ddof=0), SCORE_SD_FLOOR))


def fit_adaptive_ranker(
    source_rows: pd.DataFrame,
    target_rows: pd.DataFrame,
    source: SourceRanker,
    target: TargetRanker,
) -> AdaptiveRanker:
    feature_mean = source_rows.loc[:, COMMON_FEATURES].mean().to_numpy(dtype=float)
    feature_sd = np.maximum(
        source_rows.loc[:, COMMON_FEATURES].std(ddof=0).to_numpy(dtype=float),
        FEATURE_SD_FLOOR,
    )
    source_mean, source_sd = _safe_moments(source.predict_logit(source_rows))
    target_mean, target_sd = _safe_moments(target.predict_logit(target_rows))
    base = AdaptiveRanker(
        source=source,
        target=target,
        feature_mean=tuple(float(value) for value in feature_mean),
        feature_sd=tuple(float(value) for value in feature_sd),
        source_score_mean=source_mean,
        source_score_sd=source_sd,
        target_score_mean=target_mean,
        target_score_sd=target_sd,
        gate_intercept=0.0,
        gate_shift_slope=1.0,
        gate_objective=float("nan"),
        gate_converged=False,
        gate_pair_count=0,
    )
    positive, negative, pair_groups = _pair_indices(
        target_rows["error"], target_rows["group"]
    )
    source_scores, target_scores = base.component_scores(target_rows)
    shifts = base.shift(target_rows)

    def objective(parameters: np.ndarray) -> float:
        gates = expit(parameters[0] - parameters[1] * shifts)
        scores = gates * source_scores + (1.0 - gates) * target_scores
        differences = scores[positive] - scores[negative]
        losses = np.logaddexp(0.0, -differences)
        return _smooth_group_loss(losses, pair_groups) + GATE_L2_PENALTY * float(
            np.sum(parameters**2)
        )

    fitted = minimize(
        objective,
        np.asarray([0.0, 1.0]),
        method="L-BFGS-B",
        bounds=((-6.0, 6.0), (0.0, 10.0)),
        options={"maxiter": 1_000, "ftol": 1e-12},
    )
    if not fitted.success:
        raise RuntimeError(f"adaptive gate fit failed: {fitted.message}")
    return AdaptiveRanker(
        **{
            **asdict(base),
            "source": source,
            "target": target,
            "gate_intercept": float(fitted.x[0]),
            "gate_shift_slope": float(fitted.x[1]),
            "gate_objective": float(fitted.fun),
            "gate_converged": bool(fitted.success),
            "gate_pair_count": len(positive),
        }
    )


def _split_calibration_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = sorted(pd.unique(rows["timestamp"]))
    if len(timestamps) < 4:
        raise ValueError("not enough timestamps to separate calibration head and threshold")
    cut = min(
        len(timestamps) - 1,
        max(1, int(len(timestamps) * CALIBRATION_HEAD_FRACTION)),
    )
    head_timestamps = set(timestamps[:cut])
    return (
        rows.loc[rows["timestamp"].isin(head_timestamps)].copy(),
        rows.loc[~rows["timestamp"].isin(head_timestamps)].copy(),
    )


def _choose_by_rank(rows: pd.DataFrame, ranker: AdaptiveRanker) -> pd.DataFrame:
    scored = rows.copy()
    scored["rank_score"] = ranker.predict_logit(scored)
    scored["source_gate"] = ranker.gate(scored)
    scored["source_shift"] = ranker.shift(scored)
    order = {method: index for index, method in enumerate(METHOD_ORDER)}
    scored["tie_order"] = scored["method"].map(order)
    chosen = (
        scored.sort_values(["timestamp", "rank_score", "tie_order"])
        .groupby("timestamp", sort=True, as_index=False)
        .first()
    )
    return chosen.drop(columns="tie_order")


def fit_calibration_head(chosen_rows: pd.DataFrame) -> CalibrationHead:
    scores = chosen_rows["rank_score"].to_numpy(dtype=float)
    labels = chosen_rows["error"].to_numpy(dtype=float)
    prevalence = float(np.clip(labels.mean(), 1e-6, 1.0 - 1e-6))
    initial = np.asarray([log(prevalence / (1.0 - prevalence)), 1.0])

    def objective(parameters: np.ndarray) -> float:
        logits = parameters[0] + parameters[1] * scores
        return float(v4._binary_log_loss(logits, labels).mean())

    fitted = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=((None, None), (0.0, None)),
        options={"maxiter": 1_000, "ftol": 1e-12},
    )
    if not fitted.success:
        raise RuntimeError(f"calibration-head fit failed: {fitted.message}")
    return CalibrationHead(
        intercept=float(fitted.x[0]),
        slope=float(fitted.x[1]),
        objective=float(fitted.fun),
        converged=bool(fitted.success),
        n_rows=len(chosen_rows),
    )


def _score_adaptive(
    rows: pd.DataFrame,
    ranker: AdaptiveRanker,
    calibration: CalibrationHead,
) -> pd.DataFrame:
    chosen = _choose_by_rank(rows, ranker)
    chosen["risk"] = calibration.predict(chosen["rank_score"])
    return chosen


def _identity_calibration(n_rows: int) -> CalibrationHead:
    return CalibrationHead(
        intercept=0.0,
        slope=1.0,
        objective=None,
        converged=True,
        n_rows=n_rows,
    )


def _risk_calibration_metrics(rows: pd.DataFrame, bins: int = 10) -> dict[str, float]:
    risk = rows["risk"].to_numpy(dtype=float)
    labels = rows["error"].to_numpy(dtype=float)
    brier = float(np.mean((risk - labels) ** 2))
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(risk, edges[1:-1], right=True), bins - 1)
    ece = 0.0
    for index in range(bins):
        mask = assignments == index
        if mask.any():
            ece += float(mask.mean()) * abs(float(risk[mask].mean() - labels[mask].mean()))
    return {"risk_brier": brier, "risk_ece_10": float(ece)}


def _aurc_difference_ci(
    left_rows: pd.DataFrame,
    right_rows: pd.DataFrame,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, tuple[float, float]]:
    left = left_rows.sort_values("timestamp").reset_index(drop=True)
    right = right_rows.sort_values("timestamp").reset_index(drop=True)
    if left["timestamp"].tolist() != right["timestamp"].tolist():
        raise ValueError("AURC policies must share identical timestamps")

    def difference(a: pd.DataFrame, b: pd.DataFrame) -> float:
        return float(v4._aurc(a) - v4._aurc(b))

    observed = difference(left, right)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        indices = v4._moving_block_indices(
            len(left), rng=rng, block_length=BLOCK_LENGTH
        )
        samples[replicate] = difference(left.iloc[indices], right.iloc[indices])
    low, high = np.quantile(samples, [0.025, 0.975])
    return observed, (float(low), float(high))


def _correlation_or_none(left: Sequence[float], right: Sequence[float]) -> float | None:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if len(a) < 2 or a.std(ddof=0) < 1e-12 or b.std(ddof=0) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _write_outputs(
    output_root: Path,
    *,
    source: dict[str, object],
    summaries: Mapping[str, dict[str, object]],
    primary: dict[str, object],
    fold_details: list[dict[str, object]],
    audits: dict[str, object],
    v4_comparison: dict[str, object],
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "POST_V4_RETROSPECTIVE_EXPLORATORY_ONLY",
        "source_ranker": source,
        "primary": primary,
        "summaries": summaries,
        "v4_descriptive_comparison": v4_comparison,
        "folds": fold_details,
        "audits": audits,
        "claim_boundary": (
            "V5 was designed after V4 on a reused market period. It is not a "
            "confirmatory, prospective, investment-alpha, or cross-model claim."
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Historical Router V5: AMIR-Router",
        "",
        f"- Protocol: `{PROTOCOL_VERSION}`",
        "- Status: **POST-V4 RETROSPECTIVE EXPLORATORY ONLY**",
        f"- Source LLM questions: {source['n_questions']}",
        f"- Market outer folds: {len(fold_details)}",
        "",
        "## Primary exploratory endpoint",
        "",
        (
            "`AURC(AMIR) - AURC(confidence)` = "
            f"{primary['difference']:.4f} (95% paired moving-block CI "
            f"[{primary['ci'][0]:.4f}, {primary['ci'][1]:.4f}])."
        ),
        f"Directional target (upper CI < 0): **{primary['directional_target_met']}**.",
        "This endpoint is not eligible for a confirmatory PASS claim.",
        "",
        "## Routers and frozen ablations",
        "",
        (
            "| Router | Coverage | Routed error | AURC | Risk Brier | Risk ECE | "
            "Worst VIX error | Selected Brier |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {summary['coverage']:.3f} | {summary['routed_error']:.3f} | "
            f"{summary['aurc']:.3f} | {summary['risk_brier']:.3f} | "
            f"{summary['risk_ece_10']:.3f} | "
            f"{summary['worst_vix_regime_error']:.3f} | "
            f"{summary['selected_brier']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## V4-linked descriptive comparison",
            "",
            (
                f"Frozen V4 CPR AURC = {v4_comparison['v4_cpr_aurc']:.4f}; "
                f"V5 AMIR AURC difference = {v4_comparison['v5_minus_v4']:.4f}."
            ),
            "This comparison is descriptive because V4 directly motivated V5.",
            "",
            "## Mechanism and transfer audits",
            "",
            f"- Hard target-bound violations: {audits['hard_bound_violations']}",
            (
                "- Minimum guaranteed target-logit rise for an isolated +0.20 "
                f"common-feature intervention: {audits['minimum_hard_logit_rise']:.4f}"
            ),
            f"- Mean source gate on outer tests: {audits['mean_test_source_gate']:.4f}",
            (
                "- Mean shift/gate correlation: "
                f"{audits['mean_shift_gate_correlation']}"
            ),
            "",
            "## Interpretation boundary",
            "",
            (
                "AMIR tests whether cross-domain mechanism scores can be used conditionally "
                "rather than imposed as a fixed prior. All market results reuse a period "
                "already inspected by earlier versions; frozen cross-model LLM transfer and "
                "a new financial window remain necessary for an ACL generalization claim."
            ),
        ]
    )
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def run_historical_router_v5(
    data_directory: Path | str = "data",
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    source_records: Path | str = DEFAULT_SOURCE_RECORDS,
    v4_summary_path: Path | str = DEFAULT_V4_SUMMARY,
) -> dict[str, object]:
    """Run the frozen post-V4 V5 retrospective experiment."""

    source_rows = v4.load_source_signature(source_records)
    source_ranker = fit_source_ranker(source_rows)
    market = build_historical_replay_data(data_directory).frame
    policy_pieces: dict[str, list[pd.DataFrame]] = {
        name: [] for name in (*BASELINE_METHODS, *AMIR_METHODS)
    }
    fold_details: list[dict[str, object]] = []
    hard_coefficients: list[float] = []
    test_gates: list[float] = []
    shift_gate_correlations: list[float] = []

    outer_splits = list(
        walk_forward_splits(
            len(market), train_size=504, test_size=126, step=126, gap=HORIZON_DAYS
        )
    )
    for fold, (train_indices, test_indices) in enumerate(outer_splits):
        outer_train = market.iloc[list(train_indices)]
        outer_test = market.iloc[list(test_indices)]
        oof_probabilities = _nested_oof_probabilities(outer_train)
        oof_rows = v4.online_market_rows(oof_probabilities, outer_train)
        fit_rows, calibration_rows = v4._split_router_rows(oof_rows)
        fit_rows = v4.assign_groups(fit_rows, outer_train)
        calibration_head_rows, threshold_rows = _split_calibration_rows(calibration_rows)

        target_full = fit_target_ranker(fit_rows)
        ranker_full = fit_adaptive_ranker(
            source_rows, fit_rows, source_ranker, target_full
        )
        target_no_hard = fit_target_ranker(fit_rows, hard_common_minimum=0.0)
        ranker_no_hard = fit_adaptive_ranker(
            source_rows, fit_rows, source_ranker, target_no_hard
        )
        rankers = {
            "amir_router": ranker_full,
            "amir_target_only": ranker_full.with_forced_gate(0.0),
            "amir_fixed_gate": ranker_full.with_forced_gate(0.5),
            "amir_no_hard_constraint": ranker_no_hard,
            "amir_no_calibration": ranker_full,
        }

        predicted = _fit_predict(outer_train, outer_test)
        oof_labels = outer_train.loc[oof_probabilities.index, "target_up_5d"].astype(float)
        test_rows = v4.online_market_rows(
            predicted,
            outer_test,
            initial_probabilities=oof_probabilities,
            initial_labels=oof_labels,
        )
        calibration_head_rows["vix_regime"] = v4._vix_regime(
            calibration_head_rows["vix"], outer_train
        )
        threshold_rows["vix_regime"] = v4._vix_regime(
            threshold_rows["vix"], outer_train
        )
        test_rows["vix_regime"] = v4._vix_regime(test_rows["vix"], outer_train)

        thresholds: dict[str, float] = {}
        calibrations: dict[str, CalibrationHead] = {}
        for method in BASELINE_METHODS:
            threshold_policy = v4._select_baseline(threshold_rows, method)
            thresholds[method] = v4._conformal_quantile(
                threshold_policy["risk"], TARGET_COVERAGE
            )
            test_policy = v4._select_baseline(test_rows, method)
            policy_pieces[method].append(
                v4._apply_threshold(test_policy, thresholds[method], method, fold)
            )

        for name, ranker in rankers.items():
            head_chosen = _choose_by_rank(calibration_head_rows, ranker)
            calibration = (
                _identity_calibration(len(head_chosen))
                if name == "amir_no_calibration"
                else fit_calibration_head(head_chosen)
            )
            calibrations[name] = calibration
            threshold_policy = _score_adaptive(threshold_rows, ranker, calibration)
            thresholds[name] = v4._conformal_quantile(
                threshold_policy["risk"], TARGET_COVERAGE
            )
            test_policy = _score_adaptive(test_rows, ranker, calibration)
            policy_pieces[name].append(
                v4._apply_threshold(test_policy, thresholds[name], name, fold)
            )

        common_coefficients = list(target_full.coefficients[: len(COMMON_FEATURES)])
        hard_coefficients.extend(common_coefficients)
        full_test_chosen = policy_pieces["amir_router"][-1]
        test_gates.extend(full_test_chosen["source_gate"].astype(float).tolist())
        correlation = _correlation_or_none(
            full_test_chosen["source_shift"], full_test_chosen["source_gate"]
        )
        if correlation is not None:
            shift_gate_correlations.append(correlation)

        fold_details.append(
            {
                "fold": fold,
                "train_start": str(outer_train.index.min().date()),
                "train_end": str(outer_train.index.max().date()),
                "test_start": str(outer_test.index.min().date()),
                "test_end": str(outer_test.index.max().date()),
                "fit_timestamps": int(fit_rows["timestamp"].nunique()),
                "calibration_head_timestamps": int(
                    calibration_head_rows["timestamp"].nunique()
                ),
                "threshold_timestamps": int(threshold_rows["timestamp"].nunique()),
                "thresholds": thresholds,
                "rankers": {name: ranker.as_dict() for name, ranker in rankers.items()},
                "calibrations": {
                    name: calibration.as_dict()
                    for name, calibration in calibrations.items()
                },
                "mean_test_source_gate": float(full_test_chosen["source_gate"].mean()),
                "shift_gate_correlation": correlation,
                "hard_common_coefficients": dict(
                    zip(COMMON_FEATURES, common_coefficients, strict=True)
                ),
            }
        )

    policy_rows = {
        name: pd.concat(pieces).sort_values("timestamp").reset_index(drop=True)
        for name, pieces in policy_pieces.items()
    }
    summaries: dict[str, dict[str, object]] = {}
    for name, rows in policy_rows.items():
        inherited = asdict(v4.summarize_policy(rows, name))
        inherited.update(_risk_calibration_metrics(rows))
        summaries[name] = inherited

    difference, interval = _aurc_difference_ci(
        policy_rows["amir_router"], policy_rows["confidence"]
    )
    primary = {
        "name": "aurc_amir_minus_confidence",
        "difference": difference,
        "ci": list(interval),
        "directional_target_met": bool(np.isfinite(interval[1]) and interval[1] < 0.0),
        "confirmatory_eligible": False,
    }

    v4_payload = json.loads(Path(v4_summary_path).read_text(encoding="utf-8"))
    v4_cpr_aurc = float(v4_payload["summaries"]["cpr_router"]["aurc"])
    v4_comparison = {
        "v4_summary": str(v4_summary_path),
        "v4_cpr_aurc": v4_cpr_aurc,
        "v5_amir_aurc": float(summaries["amir_router"]["aurc"]),
        "v5_minus_v4": float(summaries["amir_router"]["aurc"]) - v4_cpr_aurc,
        "inferential": False,
    }

    hard_array = np.asarray(hard_coefficients, dtype=float)
    audits = {
        "agent_count": len(AGENTS),
        "root_count": len({root for root, _columns in AGENTS.values()}),
        "outer_folds": len(outer_splits),
        "outer_test_rows": len(policy_rows["amir_router"]),
        "hard_common_minimum": HARD_COMMON_MINIMUM,
        "hard_bound_violations": int((hard_array < HARD_COMMON_MINIMUM - 1e-10).sum()),
        "minimum_fitted_common_coefficient": float(hard_array.min()),
        "minimum_hard_logit_rise": float(hard_array.min() * INTERVENTION_INCREMENT),
        "mean_test_source_gate": float(np.mean(test_gates)),
        "mean_shift_gate_correlation": (
            float(np.mean(shift_gate_correlations)) if shift_gate_correlations else None
        ),
        "all_gate_shift_slopes_nonnegative": bool(
            all(
                float(fold["rankers"]["amir_router"]["gate_shift_slope"]) >= 0.0
                for fold in fold_details
            )
        ),
    }
    source = {
        "records": str(source_records),
        "n_questions": len(source_rows),
        "error_prevalence": float(source_rows["error"].mean()),
        **source_ranker.as_dict(),
    }
    return _write_outputs(
        Path(output_root),
        source=source,
        summaries=summaries,
        primary=primary,
        fold_details=fold_details,
        audits=audits,
        v4_comparison=v4_comparison,
    )


if __name__ == "__main__":
    run_historical_router_v5()
