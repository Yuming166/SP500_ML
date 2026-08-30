"""Reproducible formal evaluation for the synthetic false-consensus benchmark.

The protocol is frozen in ``docs/synthetic_v1_preregistration.md``.  This
module deliberately keeps risk scoring, threshold selection, post-outcome
metrics, bootstrap uncertainty, and plotting separate so that no outcome can
enter a deployable score or its threshold.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import blake2b
from math import ceil
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any

import matplotlib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from sp500_forecastability.metrics import brier_score, expected_calibration_error
from sp500_forecastability.synthetic_benchmark import (
    BenchmarkConfig,
    ParameterizedEpisode,
    ProvenanceVisibility,
    Scenario,
    conditional_provenance_risk,
    generate_parameterized_episode,
    provenance_risk,
)

BASE_SEEDS = (101, 211, 307, 401, 509, 601, 709, 809, 907, 1009)
V2_BASE_SEEDS = (1103, 1201, 1301, 1409, 1511, 1601, 1709, 1801, 1901, 2003)
V3_BASE_SEEDS = (2203, 2309, 2411, 2503, 2609, 2707, 2801, 2903, 3001, 3109)
AGENT_COUNTS = (3, 5, 7, 9)
SOURCE_QUALITY_NOISE = (0.00, 0.10, 0.20, 0.35)
CORRUPTION_STRENGTHS = (0.40, 0.60, 0.80, 1.00)
CONTROLS = (Scenario.INDEPENDENT_CLEAN, Scenario.SHARED_CLEAN)
CORRUPTION_MECHANISMS = (
    Scenario.SHARED_CORRUPTION,
    Scenario.STALE_EVIDENCE,
    Scenario.PARTIAL_CORRUPTION,
)
V3_CORRUPTION_MECHANISMS = (
    *CORRUPTION_MECHANISMS,
    Scenario.EVIDENCE_INERTIA,
)
METHODS = (
    "majority",
    "confidence",
    "agreement",
    "recent_performance",
    "provenance",
    "oracle",
)
V3_METHODS = (
    "majority",
    "confidence",
    "agreement",
    "recent_performance",
    "quality_only",
    "source_overlap",
    "temporal_only",
    "provenance_v3",
    "oracle",
)
METHOD_LABELS = {
    "majority": "Majority",
    "confidence": "Confidence",
    "agreement": "Agreement",
    "recent_performance": "Recent performance",
    "provenance": "Provenance",
    "oracle": "Oracle (diagnostic)",
}
METHOD_LABELS.update(
    {
        "quality_only": "Quality only",
        "source_overlap": "Source overlap only",
        "temporal_only": "Temporal only",
        "provenance_v3": "Conditional provenance",
    }
)
TARGET_TRAIN_RISK_QUANTILE = 0.75
RISK_AT_COVERAGE = 0.80
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20260830


@dataclass(frozen=True)
class MethodMetrics:
    """Post-outcome metrics for one method on one evaluation partition."""

    auroc: float
    auprc: float
    ece: float
    brier: float
    aurc: float
    risk_at_80_coverage: float
    high_confidence_error: float | None
    false_rejection: float
    independent_correct_rejection: float
    coverage_at_threshold: float


def _stable_seed(base_seed: int, *parts: object) -> int:
    """Derive independent deterministic streams without Python hash randomization."""

    digest = blake2b(
        ":".join((str(base_seed), *(str(part) for part in parts))).encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def _profile_methods(profile: str) -> tuple[str, ...]:
    return V3_METHODS if profile == "v3" else METHODS


def _profile_corruption_mechanisms(profile: str) -> tuple[Scenario, ...]:
    return V3_CORRUPTION_MECHANISMS if profile == "v3" else CORRUPTION_MECHANISMS


def _is_corruption(scenario: Scenario, *, profile: str) -> bool:
    return scenario in _profile_corruption_mechanisms(profile)


def _episode_risk_scores(
    episode: ParameterizedEpisode, *, profile: str = "v1"
) -> dict[str, float]:
    """Produce only pre-outcome risk estimates; the oracle is diagnostics-only."""

    decisions = episode.observation.decisions
    vote_counts = {
        action: sum(decision.action == action for decision in decisions)
        for action in ("cash", "long")
    }
    total = len(decisions)
    majority_fraction = max(vote_counts.values()) / total
    pair_total = total * (total - 1) / 2
    agreeing_pairs = sum(count * (count - 1) / 2 for count in vote_counts.values())
    mean_confidence = mean(decision.confidence for decision in decisions)
    agent_quality = []
    for decision in decisions:
        evidence_id = decision.claims[0].evidence_ids[0]
        root_source = next(
            iter(episode.observation.provenance_graph.root_source_ids(evidence_id))
        )
        agent_quality.append(episode.observation.source_quality[root_source])
    recent_performance_risk = 1.0 - mean(agent_quality)
    if profile in {"v2", "v3"}:
        recent_performance_risk = 1.0 - mean(episode.agent_historical_performance.values())
    scores = {
        "majority": 1.0 - majority_fraction,
        "confidence": 1.0 - mean_confidence,
        "agreement": 1.0 - agreeing_pairs / pair_total,
        "recent_performance": recent_performance_risk,
        "oracle": float(episode.harmful_false_consensus),
    }
    if profile == "v3":
        base = provenance_risk(episode.observation)
        conditional = conditional_provenance_risk(episode)
        scores.update(
            {
                "quality_only": base.source_quality_risk,
                "source_overlap": base.source_concentration,
                "temporal_only": min(1.0, base.stale_fraction + base.temporal_violation_fraction),
                "provenance_v3": conditional.score,
            }
        )
    else:
        scores["provenance"] = provenance_risk(episode.observation).score
    return scores


def generate_protocol_rows(
    base_seeds: Sequence[int] = BASE_SEEDS, *, profile: str = "v1"
) -> list[dict[str, Any]]:
    """Generate every protocol episode independently by mechanism and config."""

    if not base_seeds:
        raise ValueError("base_seeds must not be empty")
    if profile not in {"v1", "v2", "v3"}:
        raise ValueError("profile must be v1, v2, or v3")
    rows: list[dict[str, Any]] = []
    for scenario in (*CONTROLS, *_profile_corruption_mechanisms(profile)):
        strengths = CORRUPTION_STRENGTHS if _is_corruption(scenario, profile=profile) else (0.60,)
        for agent_count in AGENT_COUNTS:
            for source_noise in SOURCE_QUALITY_NOISE:
                for strength in strengths:
                    config_kwargs: dict[str, Any] = {
                        "agent_count": agent_count,
                        "corruption_strength": strength,
                        "source_quality_noise": source_noise,
                        "provenance_visibility": ProvenanceVisibility.ALIASED,
                        "renamed_transformations": True,
                    }
                    if profile in {"v2", "v3"}:
                        config_kwargs.update(
                            confidence_quality_coupling=0.15,
                            confidence_noise=0.08,
                        )
                    config = BenchmarkConfig(
                        **config_kwargs
                    )
                    for base_seed in base_seeds:
                        seed = _stable_seed(
                            base_seed, profile, scenario.value, agent_count, source_noise, strength
                        )
                        episode = generate_parameterized_episode(scenario, seed=seed, config=config)
                        rows.append(_row_from_episode(episode, base_seed, source_noise, profile))
    return rows


def _row_from_episode(
    episode: ParameterizedEpisode, base_seed: int, source_noise: float, profile: str
) -> dict[str, Any]:
    row = {
        "episode_key": (
            f"{episode.scenario.value}:{base_seed}:{len(episode.observation.decisions)}:"
            f"{source_noise:.2f}:{episode.corruption_strength:.2f}"
        ),
        "base_seed": base_seed,
        "scenario": episode.scenario.value,
        "agent_count": len(episode.observation.decisions),
        "source_quality_noise": source_noise,
        "corruption_strength": episode.corruption_strength,
        "harmful_false_consensus": int(episode.harmful_false_consensus),
        "consensus_error": int(episode.consensus_action != episode.outcome_action),
        "independent_clean": int(episode.scenario is Scenario.INDEPENDENT_CLEAN),
    }
    row.update(_episode_risk_scores(episode, profile=profile))
    return row


def _quantile(values: Sequence[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), quantile, method="linear"))


def mechanism_heldout_partitions(
    rows: Sequence[Mapping[str, Any]],
    *,
    corruption_mechanisms: Sequence[Scenario] = CORRUPTION_MECHANISMS,
) -> dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Create one leave-one-corruption-mechanism-out partition per mechanism."""

    partitions: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for held_out in corruption_mechanisms:
        test_names = {held_out.value, *(scenario.value for scenario in CONTROLS)}
        train_names = {
            *(scenario.value for scenario in CONTROLS),
            *(scenario.value for scenario in corruption_mechanisms if scenario != held_out),
        }
        train = [dict(row) for row in rows if row["scenario"] in train_names]
        test = [dict(row) for row in rows if row["scenario"] in test_names]
        if not train or not test:
            raise RuntimeError("mechanism-held-out partition is unexpectedly empty")
        partitions[held_out.value] = (train, test)
    return partitions


def frozen_thresholds(
    train_rows: Sequence[Mapping[str, Any]], *, methods: Sequence[str] = METHODS
) -> dict[str, float]:
    """Select outcome-free thresholds with the preregistered training quantile."""

    thresholds = {
        method: _quantile([float(row[method]) for row in train_rows], TARGET_TRAIN_RISK_QUANTILE)
        for method in methods
        if method != "oracle"
    }
    thresholds["oracle"] = 0.50
    return thresholds


def _aurc(rows: Sequence[Mapping[str, Any]], method: str) -> float:
    ordered = sorted(rows, key=lambda row: (float(row[method]), str(row["episode_key"])))
    errors = np.asarray([int(row["consensus_error"]) for row in ordered], dtype=float)
    return float(np.mean(np.cumsum(errors) / np.arange(1, len(errors) + 1)))


def _risk_at_coverage(rows: Sequence[Mapping[str, Any]], method: str, coverage: float) -> float:
    selected = sorted(rows, key=lambda row: (float(row[method]), str(row["episode_key"])))[: max(1, ceil(len(rows) * coverage))]
    return float(mean(int(row["consensus_error"]) for row in selected))


def evaluate_method(
    rows: Sequence[Mapping[str, Any]], method: str, threshold: float
) -> MethodMetrics:
    """Evaluate a frozen pre-outcome risk score after outcome revelation."""

    labels = [int(row["harmful_false_consensus"]) for row in rows]
    scores = [float(row[method]) for row in rows]
    if len(set(labels)) != 2:
        raise ValueError("formal evaluation requires positive and negative labels")
    rejected = [score > threshold for score in scores]
    routed = [not value for value in rejected]
    routed_errors = [
        int(row["consensus_error"])
        for row, is_routed in zip(rows, routed)
        if is_routed
    ]
    correct = [int(row["consensus_error"]) == 0 for row in rows]
    independent_correct = [
        int(row["independent_clean"]) == 1 and int(row["consensus_error"]) == 0 for row in rows
    ]
    false_rejection_denominator = sum(correct)
    independent_denominator = sum(independent_correct)
    return MethodMetrics(
        auroc=float(roc_auc_score(labels, scores)),
        auprc=float(average_precision_score(labels, scores)),
        ece=expected_calibration_error(labels, scores),
        brier=brier_score(labels, scores),
        aurc=_aurc(rows, method),
        risk_at_80_coverage=_risk_at_coverage(rows, method, RISK_AT_COVERAGE),
        high_confidence_error=float(mean(routed_errors)) if routed_errors else None,
        false_rejection=(
            sum(value and is_rejected for value, is_rejected in zip(correct, rejected))
            / false_rejection_denominator
        ),
        independent_correct_rejection=(
            sum(value and is_rejected for value, is_rejected in zip(independent_correct, rejected))
            / independent_denominator
        ),
        coverage_at_threshold=sum(routed) / len(rows),
    )


def _bootstrap_ci(
    rows: Sequence[Mapping[str, Any]],
    method: str,
    threshold: float,
    repeats: int,
    *,
    methods: Sequence[str] = METHODS,
) -> dict[str, tuple[float | None, float | None]]:
    """Cluster bootstrap on the preregistered base-seed unit."""

    by_seed: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["base_seed"])].append(row)
    seeds = tuple(sorted(by_seed))
    random = Random(BOOTSTRAP_SEED + tuple(methods).index(method))
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(repeats):
        sampled_rows = [
            row for seed in (random.choice(seeds) for _ in seeds) for row in by_seed[seed]
        ]
        metrics = evaluate_method(sampled_rows, method, threshold)
        for name, value in asdict(metrics).items():
            if value is not None:
                samples[name].append(float(value))
    return {
        name: (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
        for name, values in samples.items()
    }


def _annotate_test_rows(
    partitions: Mapping[str, tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]],
    *,
    methods: Sequence[str] = METHODS,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    annotated: list[dict[str, Any]] = []
    thresholds_by_fold: dict[str, dict[str, float]] = {}
    for held_out, (train, test) in partitions.items():
        thresholds = frozen_thresholds(train, methods=methods)
        thresholds_by_fold[held_out] = thresholds
        for row in test:
            expanded = dict(row)
            expanded["held_out_mechanism"] = held_out
            for method, threshold in thresholds.items():
                expanded[f"threshold_{method}"] = threshold
            annotated.append(expanded)
    return annotated, thresholds_by_fold


def _fold_metrics(
    annotated_rows: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str] = METHODS,
    corruption_mechanisms: Sequence[Scenario] = CORRUPTION_MECHANISMS,
) -> dict[str, dict[str, dict[str, float | None]]]:
    output: dict[str, dict[str, dict[str, float | None]]] = {}
    for held_out in (scenario.value for scenario in corruption_mechanisms):
        rows = [row for row in annotated_rows if row["held_out_mechanism"] == held_out]
        output[held_out] = {}
        for method in methods:
            threshold = float(rows[0][f"threshold_{method}"])
            output[held_out][method] = asdict(evaluate_method(rows, method, threshold))
    return output


def _aggregate_threshold_metrics(
    rows: Sequence[Mapping[str, Any]], method: str
) -> MethodMetrics:
    """Evaluate rows whose threshold was already selected in their own train fold."""

    labels = [int(row["harmful_false_consensus"]) for row in rows]
    scores = [float(row[method]) for row in rows]
    rejected = [score > float(row[f"threshold_{method}"]) for row, score in zip(rows, scores)]
    routed = [not value for value in rejected]
    routed_errors = [
        int(row["consensus_error"])
        for row, is_routed in zip(rows, routed)
        if is_routed
    ]
    correct = [int(row["consensus_error"]) == 0 for row in rows]
    independent_correct = [
        int(row["independent_clean"]) == 1 and int(row["consensus_error"]) == 0 for row in rows
    ]
    return MethodMetrics(
        auroc=float(roc_auc_score(labels, scores)),
        auprc=float(average_precision_score(labels, scores)),
        ece=expected_calibration_error(labels, scores),
        brier=brier_score(labels, scores),
        aurc=_aurc(rows, method),
        risk_at_80_coverage=_risk_at_coverage(rows, method, RISK_AT_COVERAGE),
        high_confidence_error=float(mean(routed_errors)) if routed_errors else None,
        false_rejection=sum(value and reject for value, reject in zip(correct, rejected)) / sum(correct),
        independent_correct_rejection=(
            sum(value and reject for value, reject in zip(independent_correct, rejected))
            / sum(independent_correct)
        ),
        coverage_at_threshold=sum(routed) / len(rows),
    )


def _aggregate_bootstrap_ci(
    rows: Sequence[Mapping[str, Any]],
    method: str,
    repeats: int,
    *,
    methods: Sequence[str] = METHODS,
) -> dict[str, tuple[float | None, float | None]]:
    by_seed: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["base_seed"])].append(row)
    seeds = tuple(sorted(by_seed))
    random = Random(BOOTSTRAP_SEED + 100 + tuple(methods).index(method))
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(repeats):
        sample = [row for seed in (random.choice(seeds) for _ in seeds) for row in by_seed[seed]]
        for name, value in asdict(_aggregate_threshold_metrics(sample, method)).items():
            if value is not None:
                samples[name].append(float(value))
    return {
        name: (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
        for name, values in samples.items()
    }


def _risk_coverage_points(rows: Sequence[Mapping[str, Any]], method: str) -> tuple[list[float], list[float]]:
    coverages = np.linspace(0.05, 1.0, 20)
    return list(coverages), [_risk_at_coverage(rows, method, float(value)) for value in coverages]


def _plot_risk_coverage(
    rows: Sequence[Mapping[str, Any]], output: Path, *, methods: Sequence[str] = METHODS
) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for method in methods:
        coverage, risk = _risk_coverage_points(rows, method)
        axis.plot(coverage, risk, label=METHOD_LABELS[method], linewidth=2)
    axis.set(xlabel="Coverage (least-risky consensus retained)", ylabel="Consensus error")
    axis.set_title("Risk-coverage curve: pooled mechanism-held-out test rows")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_reliability(
    rows: Sequence[Mapping[str, Any]], output: Path, *, methods: Sequence[str] = METHODS
) -> None:
    columns = 3
    row_count = ceil(len(methods) / columns)
    figure, axes = plt.subplots(row_count, columns, figsize=(10, 3 * row_count), sharex=True, sharey=True)
    flattened_axes = np.asarray(axes).reshape(-1)
    labels = np.asarray([int(row["harmful_false_consensus"]) for row in rows])
    for axis, method in zip(flattened_axes, methods):
        scores = np.asarray([float(row[method]) for row in rows])
        centers, observed = [], []
        for index in range(10):
            low, high = index / 10, (index + 1) / 10
            mask = (scores >= low) & ((scores < high) | ((index == 9) & (scores == high)))
            if mask.any():
                centers.append(float(scores[mask].mean()))
                observed.append(float(labels[mask].mean()))
        axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        axis.plot(centers, observed, marker="o", linewidth=1.8)
        axis.set_title(METHOD_LABELS[method], fontsize=10)
        axis.grid(alpha=0.2)
    for axis in flattened_axes[len(methods) :]:
        axis.set_visible(False)
    figure.supxlabel("Predicted harmful-consensus risk")
    figure.supylabel("Observed frequency")
    figure.suptitle("Reliability diagrams (pooled held-out test rows)", y=1.01)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_heatmap(
    fold_results: Mapping[str, Mapping[str, Mapping[str, float | None]]],
    output: Path,
    *,
    methods: Sequence[str] = METHODS,
    corruption_mechanisms: Sequence[Scenario] = CORRUPTION_MECHANISMS,
) -> None:
    values = np.asarray(
        [
            [fold_results[scenario.value][method]["high_confidence_error"] for scenario in corruption_mechanisms]
            for method in methods
        ],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(7, 4.5))
    image = axis.imshow(values, cmap="magma_r", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(range(len(corruption_mechanisms)), [item.value for item in corruption_mechanisms], rotation=20)
    axis.set_yticks(range(len(methods)), [METHOD_LABELS[item] for item in methods])
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            axis.text(column_index, row_index, f"{values[row_index, column_index]:.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="High-confidence error")
    axis.set_title("Mechanism-wise held-out error at frozen threshold")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_group_curve(
    rows: Sequence[Mapping[str, Any]],
    group: str,
    output: Path,
    title: str,
    *,
    methods: Sequence[str] = ("provenance", "recent_performance", "confidence"),
) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    for method in methods:
        x_values = sorted({float(row[group]) for row in rows})
        y_values = []
        for value in x_values:
            group_rows = [row for row in rows if float(row[group]) == value]
            y_values.append(_aurc(group_rows, method))
        axis.plot(x_values, y_values, marker="o", label=METHOD_LABELS[method])
    axis.set(xlabel=group.replace("_", " "), ylabel="AURC", title=title)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _format_value(value: float | None, ci: tuple[float | None, float | None] | None) -> str:
    if value is None:
        return "NA"
    if ci is None or ci[0] is None or ci[1] is None:
        return f"{value:.3f}"
    return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def _write_report(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Mapping[str, float]],
    metrics: Mapping[str, MethodMetrics],
    confidence_intervals: Mapping[str, Mapping[str, tuple[float | None, float | None]]],
    fold_results: Mapping[str, Mapping[str, Mapping[str, float | None]]],
    *,
    experiment_name: str,
    preregistration_path: str,
    base_seeds: Sequence[int],
    profile_note: str,
    methods: Sequence[str] = METHODS,
    corruption_mechanisms: Sequence[Scenario] = CORRUPTION_MECHANISMS,
) -> None:
    lines = [
        f"# {experiment_name.replace('_', ' ').title()} formal experiment",
        "",
        (
            "This report is generated by `python -m sp500_forecastability.synthetic_experiment`. "
            f"The protocol was frozen in [the preregistration](../docs/{preregistration_path})."
        ),
        "",
        "## Protocol snapshot",
        "",
        f"- Base-seed clusters: {', '.join(str(seed) for seed in base_seeds)}.",
        f"- Generated independent episodes: {len(rows) // len(corruption_mechanisms)} before held-out test expansion; {len(rows)} pooled held-out test rows.",
        "- Evaluation: leave-one-corruption-mechanism-out; both clean controls appear in each test fold.",
        "- Frozen threshold: deployable methods use the 75th percentile of training-only pre-outcome risks; abstain iff risk is strictly greater.",
        f"- Uncertainty: {BOOTSTRAP_REPLICATES} base-seed-cluster bootstrap resamples; percentile 95% CI.",
        "- Oracle is diagnostic only and is excluded from deployable-method interpretation.",
        f"- Profile: {profile_note}",
        "",
        "## Pooled held-out results",
        "",
        (
            "Values are estimate [95% CI]. Risk@80% is consensus-action error after retaining the "
            "least-risky 80%. High-confidence error and false rejection use the frozen fold-specific "
            "threshold."
        ),
        "",
        "| Method | AUROC | AUPRC | ECE | Brier | AURC | Risk@80% | High-confidence error | False rejection | Coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        result = metrics[method]
        ci = confidence_intervals[method]
        lines.append(
            "| "
            + METHOD_LABELS[method]
            + " | "
            + " | ".join(
                _format_value(getattr(result, name), ci.get(name))
                for name in (
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
            )
            + " |"
        )
    lines.extend(("", "## Frozen thresholds by held-out mechanism", ""))
    lines.extend(("| Held-out mechanism | " + " | ".join(METHOD_LABELS[item] for item in methods) + " |", "| --- | " + " | ".join("---:" for _ in methods) + " |"))
    for held_out, values in thresholds.items():
        lines.append(
            "| " + held_out + " | " + " | ".join(f"{values[method]:.3f}" for method in methods) + " |"
        )
    lines.extend(("", "## Held-out mechanism high-confidence error", ""))
    lines.extend(("| Method | " + " | ".join(item.value for item in corruption_mechanisms) + " |", "| --- | " + " | ".join("---:" for _ in corruption_mechanisms) + " |"))
    for method in methods:
        lines.append(
            "| " + METHOD_LABELS[method] + " | " + " | ".join(
                f"{fold_results[scenario.value][method]['high_confidence_error']:.3f}"
                for scenario in corruption_mechanisms
            ) + " |"
        )
    lines.extend(
        (
            "",
            "## Figures",
            "",
            f"- [Risk-coverage curve]({experiment_name}/risk_coverage.png)",
            f"- [Reliability diagram]({experiment_name}/reliability_diagram.png)",
            f"- [Mechanism-wise heatmap]({experiment_name}/mechanism_heatmap.png)",
            f"- [Provenance-noise curve]({experiment_name}/provenance_noise_curve.png)",
            f"- [Agent-count curve]({experiment_name}/agent_count_curve.png)",
            "",
            "## Interpretation boundary",
            "",
            (
                "These are controlled synthetic results. They validate whether the contract and routing "
                "signals behave under known interventions; they do not demonstrate S&P 500 "
                "predictability, LLM faithfulness, or investment performance."
            ),
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_synthetic_experiment(
    experiment_name: str,
    preregistration_path: str,
    profile: str,
    output_root: Path | str = "results",
    *,
    base_seeds: Sequence[int],
    bootstrap_repeats: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Run one frozen profile, materializing machine- and human-readable outputs."""

    if len(base_seeds) < 3:
        raise ValueError("at least three base seeds are required for clustered uncertainty")
    if bootstrap_repeats < 1:
        raise ValueError("bootstrap_repeats must be positive")
    root = Path(output_root)
    figure_root = root / experiment_name
    figure_root.mkdir(parents=True, exist_ok=True)
    methods = _profile_methods(profile)
    corruption_mechanisms = _profile_corruption_mechanisms(profile)
    generated_rows = generate_protocol_rows(base_seeds, profile=profile)
    partitions = mechanism_heldout_partitions(
        generated_rows, corruption_mechanisms=corruption_mechanisms
    )
    test_rows, thresholds = _annotate_test_rows(partitions, methods=methods)
    results = {method: _aggregate_threshold_metrics(test_rows, method) for method in methods}
    confidence_intervals = {
        method: _aggregate_bootstrap_ci(
            test_rows, method, bootstrap_repeats, methods=methods
        )
        for method in methods
    }
    fold_results = _fold_metrics(
        test_rows, methods=methods, corruption_mechanisms=corruption_mechanisms
    )
    _plot_risk_coverage(test_rows, figure_root / "risk_coverage.png", methods=methods)
    _plot_reliability(test_rows, figure_root / "reliability_diagram.png", methods=methods)
    _plot_heatmap(
        fold_results,
        figure_root / "mechanism_heatmap.png",
        methods=methods,
        corruption_mechanisms=corruption_mechanisms,
    )
    curve_methods = (
        ("provenance_v3", "source_overlap", "quality_only", "confidence")
        if profile == "v3"
        else ("provenance", "recent_performance", "confidence")
    )
    _plot_group_curve(
        test_rows,
        "source_quality_noise",
        figure_root / "provenance_noise_curve.png",
        "Noise sensitivity",
        methods=curve_methods,
    )
    _plot_group_curve(
        test_rows,
        "agent_count",
        figure_root / "agent_count_curve.png",
        "Agent-count sensitivity",
        methods=curve_methods,
    )
    report_path = root / f"{experiment_name}.md"
    profile_note = (
        "V1: confidence and recent performance both access source-quality information."
        if profile == "v1"
        else (
            "V2: confidence is independently miscalibrated and recent performance is agent-level; "
            "only provenance accesses the environment-held source-integrity audit."
            if profile == "v2"
            else "V3: quality, overlap, temporal, and conditional-provenance scores use the same "
            "environment-held audit; only conditional provenance includes the paired-intervention signal."
        )
    )
    _write_report(
        report_path,
        test_rows,
        thresholds,
        results,
        confidence_intervals,
        fold_results,
        experiment_name=experiment_name,
        preregistration_path=preregistration_path,
        base_seeds=base_seeds,
        profile_note=profile_note,
        methods=methods,
        corruption_mechanisms=corruption_mechanisms,
    )
    payload = {
        "protocol": {
            "base_seeds": list(base_seeds),
            "bootstrap_repeats": bootstrap_repeats,
            "threshold_quantile": TARGET_TRAIN_RISK_QUANTILE,
            "risk_at_coverage": RISK_AT_COVERAGE,
        },
        "thresholds": thresholds,
        "pooled_results": {method: asdict(result) for method, result in results.items()},
        "pooled_confidence_intervals": confidence_intervals,
        "held_out_results": fold_results,
        "row_count": len(test_rows),
    }
    (figure_root / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_synthetic_v1(
    output_root: Path | str = "results",
    *,
    base_seeds: Sequence[int] = BASE_SEEDS,
    bootstrap_repeats: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Run the original V1 protocol without changing its output location."""

    return run_synthetic_experiment(
        "synthetic_v1",
        "synthetic_v1_preregistration.md",
        "v1",
        output_root,
        base_seeds=base_seeds,
        bootstrap_repeats=bootstrap_repeats,
    )


def run_synthetic_v2(
    output_root: Path | str = "results",
    *,
    base_seeds: Sequence[int] = V2_BASE_SEEDS,
    bootstrap_repeats: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Run the decoupled-baseline V2 stress test in a separate result directory."""

    return run_synthetic_experiment(
        "synthetic_v2",
        "synthetic_v2_preregistration.md",
        "v2",
        output_root,
        base_seeds=base_seeds,
        bootstrap_repeats=bootstrap_repeats,
    )


def run_synthetic_v3(
    output_root: Path | str = "results",
    *,
    base_seeds: Sequence[int] = V3_BASE_SEEDS,
    bootstrap_repeats: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Run the separately preregistered conditional-provenance V3 benchmark."""

    return run_synthetic_experiment(
        "synthetic_v3",
        "synthetic_v3_preregistration.md",
        "v3",
        output_root,
        base_seeds=base_seeds,
        bootstrap_repeats=bootstrap_repeats,
    )


if __name__ == "__main__":
    run_synthetic_v1()
