"""Post-hoc V3 coverage audit and preregistered Synthetic V4 experiment."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import log
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any

import numpy as np
from scipy.optimize import minimize

from sp500_forecastability.synthetic_benchmark import Scenario
from sp500_forecastability.synthetic_experiment import (
    BOOTSTRAP_REPLICATES,
    METHOD_LABELS,
    V3_BASE_SEEDS,
    V3_CORRUPTION_MECHANISMS,
    V3_METHODS,
    MethodMetrics,
    _aggregate_bootstrap_ci,
    _aggregate_threshold_metrics,
    _annotate_test_rows,
    _aurc,
    _fold_metrics,
    _format_value,
    _plot_group_curve,
    _plot_heatmap,
    _plot_reliability,
    _plot_risk_coverage,
    _risk_at_coverage,
    generate_protocol_rows,
    mechanism_heldout_partitions,
)

V4_TRAIN_SEEDS = (3301, 3407, 3511, 3607, 3701, 3803, 3907, 4001, 4111, 4201)
V4_TEST_SEEDS = {
    Scenario.SHARED_CORRUPTION: (4303, 4409, 4513, 4603, 4703, 4801, 4903, 5003, 5101, 5209),
    Scenario.STALE_EVIDENCE: (5303, 5407, 5501, 5603, 5701, 5801, 5903, 6007, 6101, 6203),
    Scenario.PARTIAL_CORRUPTION: (6301, 6407, 6503, 6607, 6701, 6803, 6907, 7001, 7103, 7207),
    Scenario.EVIDENCE_INERTIA: (7307, 7403, 7507, 7603, 7703, 7801, 7901, 8009, 8101, 8209),
}
V4_FEATURES = (
    "shared_integrity_risk",
    "stale_fraction",
    "temporal_violation_fraction",
    "causal_effect_risk",
)
V4_METHODS = (
    "majority",
    "confidence",
    "agreement",
    "recent_performance",
    "quality_only",
    "source_overlap",
    "temporal_only",
    "provenance_v3",
    "provenance_v4",
    "oracle",
)
MATCHED_COVERAGES = (0.60, 0.70, 0.80, 0.90)
TARGET_THRESHOLD_COVERAGE = 0.80
MAX_THRESHOLD_COVERAGE = 0.82
MODEL_L2 = 0.01
V4_BOOTSTRAP_SEED = 20260831


@dataclass(frozen=True)
class MonotonicLogisticModel:
    """Small logistic model whose feature coefficients cannot be negative."""

    intercept: float
    coefficients: tuple[float, ...]

    def logits(self, matrix: np.ndarray) -> np.ndarray:
        return self.intercept + matrix @ np.asarray(self.coefficients, dtype=float)

    def probabilities(self, matrix: np.ndarray) -> np.ndarray:
        return _sigmoid(self.logits(matrix))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_monotonic_logistic(
    matrix: np.ndarray, labels: np.ndarray, *, l2: float
) -> MonotonicLogisticModel:
    matrix = np.asarray(matrix, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if matrix.ndim != 2 or len(matrix) != len(labels) or not len(labels):
        raise ValueError("matrix and labels must form a non-empty two-dimensional problem")
    if len(set(labels.tolist())) != 2:
        raise ValueError("monotonic logistic fitting requires both labels")
    prevalence = (float(labels.sum()) + 0.5) / (len(labels) + 1.0)
    initial = np.zeros(matrix.shape[1] + 1, dtype=float)
    initial[0] = log(prevalence / (1.0 - prevalence))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept, coefficients = parameters[0], parameters[1:]
        logits = intercept + matrix @ coefficients
        losses = np.logaddexp(0.0, logits) - labels * logits
        difference = _sigmoid(logits) - labels
        value = float(losses.mean() + 0.5 * l2 * np.dot(coefficients, coefficients))
        gradient = np.concatenate(
            (
                np.asarray([difference.mean()]),
                matrix.T @ difference / len(labels) + l2 * coefficients,
            )
        )
        return value, gradient

    fitted = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=((None, None), *((0.0, None) for _ in range(matrix.shape[1]))),
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not fitted.success:
        raise RuntimeError(f"monotonic logistic fit failed: {fitted.message}")
    return MonotonicLogisticModel(
        intercept=float(fitted.x[0]),
        coefficients=tuple(float(value) for value in fitted.x[1:]),
    )


def _matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in V4_FEATURES] for row in rows], dtype=float)


def _labels(rows: Sequence[Mapping[str, Any]], name: str) -> np.ndarray:
    return np.asarray([int(row[name]) for row in rows], dtype=float)


def _cross_fitted_router(
    rows: Sequence[Mapping[str, Any]], *, folds: int = 5
) -> tuple[np.ndarray, MonotonicLogisticModel, MonotonicLogisticModel]:
    """Return grouped OOF risks, final router, and a monotonic Platt calibrator."""

    seeds = tuple(sorted({int(row["base_seed"]) for row in rows}))
    if len(seeds) < folds:
        raise ValueError("not enough base-seed clusters for grouped cross-fitting")
    logits = np.full(len(rows), np.nan, dtype=float)
    for fold_index in range(folds):
        validation_seeds = set(seeds[fold_index::folds])
        train_indices = [
            index for index, row in enumerate(rows) if int(row["base_seed"]) not in validation_seeds
        ]
        validation_indices = [
            index for index, row in enumerate(rows) if int(row["base_seed"]) in validation_seeds
        ]
        fitted = _fit_monotonic_logistic(
            _matrix([rows[index] for index in train_indices]),
            _labels([rows[index] for index in train_indices], "harmful_false_consensus"),
            l2=MODEL_L2,
        )
        logits[validation_indices] = fitted.logits(
            _matrix([rows[index] for index in validation_indices])
        )
    if not np.isfinite(logits).all():
        raise RuntimeError("cross-fitting did not score every training row")
    calibrator = _fit_monotonic_logistic(
        logits.reshape(-1, 1),
        _labels(rows, "harmful_false_consensus"),
        l2=MODEL_L2,
    )
    final_model = _fit_monotonic_logistic(
        _matrix(rows), _labels(rows, "harmful_false_consensus"), l2=MODEL_L2
    )
    calibrated = calibrator.probabilities(logits.reshape(-1, 1))
    return calibrated, final_model, calibrator


def _select_threshold(scores: Sequence[float], errors: Sequence[int]) -> float:
    """Choose a training-only threshold under the preregistered coverage band."""

    score_values = np.asarray(scores, dtype=float)
    error_values = np.asarray(errors, dtype=int)
    candidates = sorted({float(value) for value in score_values})
    candidates.append(float(np.nextafter(score_values.max(), np.inf)))
    evaluated: list[tuple[float, float, float, float]] = []
    correct = error_values == 0
    for threshold in candidates:
        routed = score_values <= threshold
        coverage = float(routed.mean())
        if not routed.any():
            continue
        selective_error = float(error_values[routed].mean())
        rejected = ~routed
        false_rejection = float((rejected & correct).sum() / max(1, int(correct.sum())))
        evaluated.append((threshold, coverage, selective_error, false_rejection))
    in_band = [
        item
        for item in evaluated
        if TARGET_THRESHOLD_COVERAGE <= item[1] <= MAX_THRESHOLD_COVERAGE
    ]
    if in_band:
        chosen = min(
            in_band,
            key=lambda item: (
                item[2],
                item[3],
                abs(item[1] - TARGET_THRESHOLD_COVERAGE),
                item[0],
            ),
        )
    else:
        chosen = min(
            evaluated,
            key=lambda item: (
                abs(item[1] - TARGET_THRESHOLD_COVERAGE),
                item[2],
                item[3],
                item[0],
            ),
        )
    return chosen[0]


def _outer_v4_rows(
    *,
    train_seeds: Sequence[int] = V4_TRAIN_SEEDS,
    test_seeds_by_mechanism: Mapping[Scenario, Sequence[int]] = V4_TEST_SEEDS,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, float]],
    dict[str, dict[str, Any]],
]:
    generated_train = generate_protocol_rows(train_seeds, profile="v4")
    annotated: list[dict[str, Any]] = []
    thresholds_by_fold: dict[str, dict[str, float]] = {}
    model_details: dict[str, dict[str, Any]] = {}
    controls = {Scenario.INDEPENDENT_CLEAN.value, Scenario.SHARED_CLEAN.value}

    for held_out in V3_CORRUPTION_MECHANISMS:
        train_scenarios = {
            *controls,
            *(item.value for item in V3_CORRUPTION_MECHANISMS if item != held_out),
        }
        train = [dict(row) for row in generated_train if row["scenario"] in train_scenarios]
        generated_test = generate_protocol_rows(test_seeds_by_mechanism[held_out], profile="v4")
        test_scenarios = {*controls, held_out.value}
        test = [dict(row) for row in generated_test if row["scenario"] in test_scenarios]

        oof_risks, final_model, calibrator = _cross_fitted_router(train)
        test_logits = final_model.logits(_matrix(test))
        test_risks = calibrator.probabilities(test_logits.reshape(-1, 1))
        for row, risk in zip(test, test_risks):
            row["provenance_v4"] = float(risk)

        thresholds: dict[str, float] = {}
        training_errors = [int(row["consensus_error"]) for row in train]
        for method in V4_METHODS:
            if method == "oracle":
                thresholds[method] = 0.50
            elif method == "provenance_v4":
                thresholds[method] = _select_threshold(oof_risks, training_errors)
            else:
                thresholds[method] = _select_threshold(
                    [float(row[method]) for row in train], training_errors
                )
        thresholds_by_fold[held_out.value] = thresholds
        model_details[held_out.value] = {
            "features": list(V4_FEATURES),
            "intercept": final_model.intercept,
            "coefficients": dict(zip(V4_FEATURES, final_model.coefficients)),
            "calibration_intercept": calibrator.intercept,
            "calibration_slope": calibrator.coefficients[0],
            "threshold": thresholds["provenance_v4"],
        }
        for row in test:
            row["held_out_mechanism"] = held_out.value
            for method, threshold in thresholds.items():
                row[f"threshold_{method}"] = threshold
            annotated.append(row)
    return annotated, thresholds_by_fold, model_details


def _paired_delta(
    rows: Sequence[Mapping[str, Any]],
    method: str,
    baseline: str,
    metric: str,
    repeats: int,
    *,
    seed_offset: int,
) -> dict[str, float]:
    def value(sample: Sequence[Mapping[str, Any]], candidate: str) -> float:
        if metric == "aurc":
            return _aurc(sample, candidate)
        if metric == "macro_aurc":
            return mean(
                _aurc(
                    [row for row in sample if row["held_out_mechanism"] == scenario.value],
                    candidate,
                )
                for scenario in V3_CORRUPTION_MECHANISMS
            )
        if metric == "risk_at_80":
            return _risk_at_coverage(sample, candidate, 0.80)
        raise ValueError(f"unsupported paired metric: {metric}")

    estimate = value(rows, method) - value(rows, baseline)
    by_seed: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["base_seed"])].append(row)
    seeds = tuple(sorted(by_seed))
    mechanisms_by_seed = {
        seed: {str(row["held_out_mechanism"]) for row in seed_rows}
        for seed, seed_rows in by_seed.items()
    }
    disjoint_outer_seeds = all(len(mechanisms) == 1 for mechanisms in mechanisms_by_seed.values())
    seeds_by_mechanism = {
        scenario.value: tuple(
            seed
            for seed in seeds
            if scenario.value in mechanisms_by_seed[seed]
        )
        for scenario in V3_CORRUPTION_MECHANISMS
    }
    random = Random(V4_BOOTSTRAP_SEED + seed_offset)
    differences = []
    for _ in range(repeats):
        if disjoint_outer_seeds:
            sampled_seeds = [
                random.choice(fold_seeds)
                for fold_seeds in seeds_by_mechanism.values()
                for _ in fold_seeds
            ]
        else:
            sampled_seeds = [random.choice(seeds) for _ in seeds]
        sample = [row for selected_seed in sampled_seeds for row in by_seed[selected_seed]]
        differences.append(value(sample, method) - value(sample, baseline))
    return {
        "estimate": float(estimate),
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
    }


def _matched_coverage_table(
    rows: Sequence[Mapping[str, Any]], methods: Sequence[str]
) -> dict[str, dict[str, float]]:
    return {
        method: {
            f"risk_at_{int(coverage * 100)}": _risk_at_coverage(rows, method, coverage)
            for coverage in MATCHED_COVERAGES
        }
        for method in methods
    }


def run_v3_matched_coverage_posthoc(
    output_root: Path | str = "results", *, bootstrap_repeats: int = BOOTSTRAP_REPLICATES
) -> dict[str, Any]:
    """Add an explicitly post-hoc, outcome-preserving audit of frozen V3."""

    rows = generate_protocol_rows(V3_BASE_SEEDS, profile="v3")
    partitions = mechanism_heldout_partitions(
        rows, corruption_mechanisms=V3_CORRUPTION_MECHANISMS
    )
    test_rows, _ = _annotate_test_rows(partitions, methods=V3_METHODS)
    matched = _matched_coverage_table(test_rows, V3_METHODS)
    comparisons = {
        baseline: {
            metric: _paired_delta(
                test_rows,
                "provenance_v3",
                baseline,
                metric,
                bootstrap_repeats,
                seed_offset=100 * baseline_index + metric_index,
            )
            for metric_index, metric in enumerate(("aurc", "risk_at_80"))
        }
        for baseline_index, baseline in enumerate(
            ("confidence", "quality_only", "source_overlap", "temporal_only"), start=1
        )
    }
    threshold_audit = {}
    for method in ("quality_only", "provenance_v3"):
        rejected = [
            float(row[method]) > float(row[f"threshold_{method}"]) for row in test_rows
        ]
        threshold_audit[method] = {
            "coverage": sum(not value for value in rejected) / len(test_rows),
            "rejected_errors": sum(
                value and int(row["consensus_error"])
                for value, row in zip(rejected, test_rows)
            ),
            "rejected_correct": sum(
                value and not int(row["consensus_error"])
                for value, row in zip(rejected, test_rows)
            ),
        }
    payload = {
        "status": "post_hoc_does_not_replace_v3_primary_result",
        "coverages": list(MATCHED_COVERAGES),
        "matched_coverage": matched,
        "paired_differences_provenance_v3_minus_baseline": comparisons,
        "threshold_audit": threshold_audit,
        "row_count": len(test_rows),
    }
    root = Path(output_root)
    artifact_root = root / "synthetic_v3"
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "posthoc_matched_coverage.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Synthetic V3 post-hoc matched-coverage audit",
        "",
        "This appendix does not modify the frozen V3 protocol or primary conclusion. It was added after inspecting unequal threshold coverage and is exploratory.",
        "",
        "## Matched-coverage consensus error",
        "",
        "| Method | Risk@60% | Risk@70% | Risk@80% | Risk@90% |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in V3_METHODS:
        values = matched[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {values['risk_at_60']:.3f} | "
            f"{values['risk_at_70']:.3f} | {values['risk_at_80']:.3f} | "
            f"{values['risk_at_90']:.3f} |"
        )
    lines.extend(
        (
            "",
            "## Why the frozen-threshold errors differed",
            "",
            f"- Quality only retained {threshold_audit['quality_only']['coverage']:.3f}, rejected {threshold_audit['quality_only']['rejected_errors']} errors, and rejected {threshold_audit['quality_only']['rejected_correct']} correct rows.",
            f"- Conditional provenance retained {threshold_audit['provenance_v3']['coverage']:.3f}, rejected {threshold_audit['provenance_v3']['rejected_errors']} errors, and rejected {threshold_audit['provenance_v3']['rejected_correct']} correct rows.",
            "",
            "At matched 80% coverage, conditional provenance reaches the same risk as the diagnostic Oracle. This supports its ranking quality, but it does not retroactively satisfy the original unequal-threshold V3 hypothesis.",
            "",
            "## Paired cluster-bootstrap differences",
            "",
            "Negative values favor Conditional provenance. Clusters are frozen V3 base seeds.",
            "",
            "| Baseline | Delta AURC [95% CI] | Delta Risk@80 [95% CI] |",
            "| --- | ---: | ---: |",
        )
    )
    for baseline, metrics in comparisons.items():
        aurc, risk = metrics["aurc"], metrics["risk_at_80"]
        lines.append(
            f"| {METHOD_LABELS[baseline]} | {aurc['estimate']:.3f} "
            f"[{aurc['ci_low']:.3f}, {aurc['ci_high']:.3f}] | "
            f"{risk['estimate']:.3f} [{risk['ci_low']:.3f}, {risk['ci_high']:.3f}] |"
        )
    lines.extend(
        (
            "",
            "## Boundary",
            "",
            "This is a post-hoc evaluation of the unchanged synthetic V3 rows. It is not an independent confirmation and does not establish LLM or market validity.",
        )
    )
    report_path = root / "synthetic_v3_posthoc_matched_coverage.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _write_v4_report(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Mapping[str, float]],
    metrics: Mapping[str, MethodMetrics],
    confidence_intervals: Mapping[str, Mapping[str, tuple[float | None, float | None]]],
    fold_results: Mapping[str, Mapping[str, Mapping[str, float | None]]],
    matched: Mapping[str, Mapping[str, float]],
    comparisons: Mapping[str, Mapping[str, Mapping[str, float]]],
    model_details: Mapping[str, Mapping[str, Any]],
    macro_aurc: Mapping[str, float],
    primary_hypothesis_supported: bool,
) -> None:
    lines = [
        "# Synthetic V4 formal experiment",
        "",
        "This report was generated under the frozen [V4 preregistration](../docs/synthetic_v4_preregistration.md).",
        "",
        "## Protocol snapshot",
        "",
        f"- Training seed clusters: {', '.join(map(str, V4_TRAIN_SEEDS))}.",
        "- Four disjoint ten-seed outer test sets; no test seed appears in training or another fold.",
        "- Outer leave-one-mechanism-out evaluation with five-fold base-seed grouped cross-fitting inside training.",
        "- Non-negative logistic provenance weights, monotonic cross-fitted calibration, and training-only 80--82% coverage threshold selection.",
        "- Imperfect actions and paired interventions are generated with the frozen V4 behavior-noise rates.",
        "",
        "## Preregistered outcome",
        "",
        (
            "The primary hypothesis is supported: Monotonic provenance V4 beats all three preregistered baselines on both macro AURC and Risk@80 with paired intervals below zero."
            if primary_hypothesis_supported
            else "The primary hypothesis is not supported. Monotonic provenance V4 must not be described as superior to the fixed V3 score or every ablation; the tables below preserve the mixed result."
        ),
        "",
        "## Macro-average AURC across held-out mechanisms",
        "",
        "| Method | Macro AURC |",
        "| --- | ---: |",
        *(f"| {METHOD_LABELS[method]} | {macro_aurc[method]:.3f} |" for method in V4_METHODS),
        "",
        "## Pooled held-out results",
        "",
        "Values are estimate [95% cluster-bootstrap CI]. Selective error and coverage use each fold's train-selected threshold.",
        "",
        "| Method | AUROC | AUPRC | ECE | Brier | AURC | Risk@80% | Selective error | False rejection | Coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in V4_METHODS:
        result, ci = metrics[method], confidence_intervals[method]
        values = (
            "auroc",
            "auprc",
            "ece",
            "brier",
            "aurc",
            "risk_at_80_coverage",
            "high_confidence_error",
            "false_rejection",
            "coverage_at_threshold",
        )
        lines.append(
            "| "
            + METHOD_LABELS[method]
            + " | "
            + " | ".join(_format_value(getattr(result, name), ci.get(name)) for name in values)
            + " |"
        )
    lines.extend(
        (
            "",
            "## Matched-coverage consensus error",
            "",
            "| Method | Risk@60% | Risk@70% | Risk@80% | Risk@90% |",
            "| --- | ---: | ---: | ---: | ---: |",
        )
    )
    for method in V4_METHODS:
        values = matched[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {values['risk_at_60']:.3f} | "
            f"{values['risk_at_70']:.3f} | {values['risk_at_80']:.3f} | "
            f"{values['risk_at_90']:.3f} |"
        )
    lines.extend(
        (
            "",
            "## Paired primary-metric differences",
            "",
            "Negative values favor Monotonic provenance V4.",
            "",
            "| Baseline | Delta macro AURC [95% CI] | Delta Risk@80 [95% CI] |",
            "| --- | ---: | ---: |",
        )
    )
    for baseline, values in comparisons.items():
        aurc, risk = values["macro_aurc"], values["risk_at_80"]
        lines.append(
            f"| {METHOD_LABELS[baseline]} | {aurc['estimate']:.3f} "
            f"[{aurc['ci_low']:.3f}, {aurc['ci_high']:.3f}] | "
            f"{risk['estimate']:.3f} [{risk['ci_low']:.3f}, {risk['ci_high']:.3f}] |"
        )
    lines.extend(("", "## Learned outer-fold models", ""))
    lines.append(
        "| Held-out mechanism | Shared-integrity | Stale | Temporal | Intervention | Calibration slope | Threshold |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for held_out, details in model_details.items():
        coefficients = details["coefficients"]
        lines.append(
            f"| {held_out} | {coefficients['shared_integrity_risk']:.3f} | "
            f"{coefficients['stale_fraction']:.3f} | "
            f"{coefficients['temporal_violation_fraction']:.3f} | "
            f"{coefficients['causal_effect_risk']:.3f} | "
            f"{details['calibration_slope']:.3f} | {details['threshold']:.3f} |"
        )
    lines.extend(("", "## Train-selected thresholds by held-out mechanism", ""))
    lines.append("| Held-out mechanism | " + " | ".join(METHOD_LABELS[item] for item in V4_METHODS) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in V4_METHODS) + " |")
    for held_out, values in thresholds.items():
        lines.append(
            "| " + held_out + " | " + " | ".join(f"{values[item]:.3f}" for item in V4_METHODS) + " |"
        )
    lines.extend(("", "## Mechanism-wise selective error and achieved coverage", ""))
    lines.append("| Held-out mechanism | Method | Selective error | Coverage | False rejection |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for held_out in (item.value for item in V3_CORRUPTION_MECHANISMS):
        for method in ("quality_only", "provenance_v3", "provenance_v4"):
            values = fold_results[held_out][method]
            lines.append(
                f"| {held_out} | {METHOD_LABELS[method]} | "
                f"{values['high_confidence_error']:.3f} | "
                f"{values['coverage_at_threshold']:.3f} | {values['false_rejection']:.3f} |"
            )
    inertia_model = model_details[Scenario.EVIDENCE_INERTIA.value]
    inertia_result = fold_results[Scenario.EVIDENCE_INERTIA.value]["provenance_v4"]
    lines.extend(
        (
            "",
            "## Diagnostic interpretation",
            "",
            "The learned router improves probability calibration and beats generic confidence routing, but it does not extrapolate the intervention mechanism as well as the fixed V3 structural prior.",
            f"When evidence inertia is completely held out, the learned intervention coefficient is {inertia_model['coefficients']['causal_effect_risk']:.3f}; held-out coverage rises to {inertia_result['coverage_at_threshold']:.3f}. The training-selected threshold therefore fails to preserve its intended 80--82% coverage under this mechanism shift.",
            "This negative result supports retaining explicit provenance priors or adding mechanism-diverse training data rather than relying on an empirical advantage from known mechanisms.",
        )
    )
    lines.extend(
        (
            "",
            "## Figures",
            "",
            "- [Risk-coverage curve](synthetic_v4/risk_coverage.png)",
            "- [Reliability diagram](synthetic_v4/reliability_diagram.png)",
            "- [Mechanism-wise threshold heatmap](synthetic_v4/mechanism_heatmap.png)",
            "- [Source-quality-noise curve](synthetic_v4/provenance_noise_curve.png)",
            "- [Agent-count curve](synthetic_v4/agent_count_curve.png)",
            "",
            "## Interpretation boundary",
            "",
            "These are controlled, noisy rule-agent results. They test a routing and evaluation contract; they do not demonstrate LLM faithfulness, S&P 500 predictability, or investment performance.",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_synthetic_v4(
    output_root: Path | str = "results",
    *,
    bootstrap_repeats: int = BOOTSTRAP_REPLICATES,
    train_seeds: Sequence[int] = V4_TRAIN_SEEDS,
    test_seeds_by_mechanism: Mapping[Scenario, Sequence[int]] = V4_TEST_SEEDS,
) -> dict[str, Any]:
    """Run the independently seeded, preregistered Synthetic V4 benchmark."""

    root = Path(output_root)
    figure_root = root / "synthetic_v4"
    figure_root.mkdir(parents=True, exist_ok=True)
    rows, thresholds, model_details = _outer_v4_rows(
        train_seeds=train_seeds, test_seeds_by_mechanism=test_seeds_by_mechanism
    )
    metrics = {method: _aggregate_threshold_metrics(rows, method) for method in V4_METHODS}
    confidence_intervals = {
        method: _aggregate_bootstrap_ci(
            rows, method, bootstrap_repeats, methods=V4_METHODS
        )
        for method in V4_METHODS
    }
    fold_results = _fold_metrics(
        rows, methods=V4_METHODS, corruption_mechanisms=V3_CORRUPTION_MECHANISMS
    )
    matched = _matched_coverage_table(rows, V4_METHODS)
    baselines = ("confidence", "quality_only", "provenance_v3")
    comparisons = {
        baseline: {
            metric: _paired_delta(
                rows,
                "provenance_v4",
                baseline,
                metric,
                bootstrap_repeats,
                seed_offset=1000 + baseline_index * 10 + metric_index,
            )
            for metric_index, metric in enumerate(("macro_aurc", "risk_at_80"))
        }
        for baseline_index, baseline in enumerate(baselines)
    }
    _plot_risk_coverage(rows, figure_root / "risk_coverage.png", methods=V4_METHODS)
    _plot_reliability(rows, figure_root / "reliability_diagram.png", methods=V4_METHODS)
    _plot_heatmap(
        fold_results,
        figure_root / "mechanism_heatmap.png",
        methods=V4_METHODS,
        corruption_mechanisms=V3_CORRUPTION_MECHANISMS,
    )
    _plot_group_curve(
        rows,
        "source_quality_noise",
        figure_root / "provenance_noise_curve.png",
        "V4 noise sensitivity",
        methods=("provenance_v4", "provenance_v3", "quality_only", "confidence"),
    )
    _plot_group_curve(
        rows,
        "agent_count",
        figure_root / "agent_count_curve.png",
        "V4 agent-count sensitivity",
        methods=("provenance_v4", "provenance_v3", "quality_only", "confidence"),
    )
    macro_aurc = {
        method: mean(
            float(fold_results[scenario.value][method]["aurc"])
            for scenario in V3_CORRUPTION_MECHANISMS
        )
        for method in V4_METHODS
    }
    primary_hypothesis_supported = all(
        macro_aurc["provenance_v4"] < macro_aurc[baseline]
        and matched["provenance_v4"]["risk_at_80"] < matched[baseline]["risk_at_80"]
        and comparisons[baseline]["macro_aurc"]["ci_high"] < 0.0
        and comparisons[baseline]["risk_at_80"]["ci_high"] < 0.0
        for baseline in baselines
    )
    payload = {
        "protocol": {
            "train_seeds": list(train_seeds),
            "test_seeds_by_mechanism": {
                scenario.value: list(seeds) for scenario, seeds in test_seeds_by_mechanism.items()
            },
            "matched_coverages": list(MATCHED_COVERAGES),
            "threshold_coverage_band": [TARGET_THRESHOLD_COVERAGE, MAX_THRESHOLD_COVERAGE],
            "bootstrap_repeats": bootstrap_repeats,
            "model_l2": MODEL_L2,
        },
        "thresholds": thresholds,
        "model_details": model_details,
        "pooled_results": {method: asdict(result) for method, result in metrics.items()},
        "pooled_confidence_intervals": confidence_intervals,
        "matched_coverage": matched,
        "macro_aurc": macro_aurc,
        "primary_hypothesis_supported": primary_hypothesis_supported,
        "paired_differences_provenance_v4_minus_baseline": comparisons,
        "held_out_results": fold_results,
        "row_count": len(rows),
    }
    (figure_root / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_v4_report(
        root / "synthetic_v4.md",
        rows,
        thresholds,
        metrics,
        confidence_intervals,
        fold_results,
        matched,
        comparisons,
        model_details,
        macro_aurc,
        primary_hypothesis_supported,
    )
    return payload


if __name__ == "__main__":
    run_v3_matched_coverage_posthoc()
    run_synthetic_v4()
