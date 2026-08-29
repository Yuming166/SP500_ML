"""Metrics for calibrated and selective forecasts.

The functions in this module intentionally avoid a model dependency. This lets
the research compare logistic regression, tree models, and future models using
the same evaluation layer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, isfinite, log


@dataclass(frozen=True)
class SelectiveMetrics:
    """Summary of performance after retaining the most confident forecasts."""

    coverage: float
    accuracy: float
    mean_confidence: float
    selected_count: int
    total_count: int


def _validate_probabilities(probabilities: Sequence[float]) -> list[float]:
    values = [float(value) for value in probabilities]
    if not values:
        raise ValueError("probabilities must not be empty")
    if any(not isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise ValueError("probabilities must be finite values in [0, 1]")
    return values


def _validate_binary_inputs(
    y_true: Sequence[int], probabilities: Sequence[float]
) -> tuple[list[int], list[float]]:
    labels = [int(value) for value in y_true]
    probs = _validate_probabilities(probabilities)
    if len(labels) != len(probs):
        raise ValueError("y_true and probabilities must have the same length")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("y_true must contain only 0 and 1")
    return labels, probs


def binary_entropy(probabilities: Sequence[float]) -> list[float]:
    """Return Bernoulli entropy in nats for each probability."""

    values = _validate_probabilities(probabilities)
    result = []
    for probability in values:
        if probability in (0.0, 1.0):
            result.append(0.0)
            continue
        result.append(
            -probability * log(probability)
            - (1.0 - probability) * log(1.0 - probability)
        )
    return result


def brier_score(y_true: Sequence[int], probabilities: Sequence[float]) -> float:
    """Return the binary Brier score; lower is better."""

    labels, probs = _validate_binary_inputs(y_true, probabilities)
    return sum((probability - label) ** 2 for label, probability in zip(labels, probs)) / len(labels)


def expected_calibration_error(
    y_true: Sequence[int], probabilities: Sequence[float], n_bins: int = 10
) -> float:
    """Return equal-width expected calibration error for binary forecasts."""

    labels, probs = _validate_binary_inputs(y_true, probabilities)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")

    total_error = 0.0
    for bin_number in range(n_bins):
        lower = bin_number / n_bins
        upper = (bin_number + 1) / n_bins
        members = [
            index
            for index, probability in enumerate(probs)
            if (lower <= probability < upper)
            or (bin_number == n_bins - 1 and probability == upper)
        ]
        if not members:
            continue
        mean_probability = sum(probs[index] for index in members) / len(members)
        mean_label = sum(labels[index] for index in members) / len(members)
        total_error += len(members) / len(labels) * abs(mean_probability - mean_label)
    return total_error


def selective_metrics(
    y_true: Sequence[int], probabilities: Sequence[float], coverage: float
) -> SelectiveMetrics:
    """Evaluate the top-confidence forecasts at a requested coverage.

    ``coverage=0.4`` keeps the 40% of examples with the largest
    ``max(p_up, 1-p_up)``. Ties are resolved by original order for
    reproducibility.
    """

    labels, probs = _validate_binary_inputs(y_true, probabilities)
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")

    confidences = [max(probability, 1.0 - probability) for probability in probs]
    selected_count = max(1, ceil(len(labels) * coverage))
    ranked_indices = sorted(range(len(labels)), key=lambda index: (-confidences[index], index))
    selected = ranked_indices[:selected_count]
    correct = sum((probs[index] >= 0.5) == bool(labels[index]) for index in selected)
    return SelectiveMetrics(
        coverage=selected_count / len(labels),
        accuracy=correct / selected_count,
        mean_confidence=sum(confidences[index] for index in selected) / selected_count,
        selected_count=selected_count,
        total_count=len(labels),
    )


def interval_coverage(
    y_true: Sequence[float], lower: Sequence[float], upper: Sequence[float]
) -> float:
    """Return the fraction of observations contained in their intervals."""

    values = [float(value) for value in y_true]
    lowers = [float(value) for value in lower]
    uppers = [float(value) for value in upper]
    if not values or len(values) != len(lowers) or len(values) != len(uppers):
        raise ValueError("interval inputs must be non-empty and have the same length")
    if any(not isfinite(value) for value in values + lowers + uppers):
        raise ValueError("interval inputs must be finite")
    if any(low > high for low, high in zip(lowers, uppers)):
        raise ValueError("lower bounds cannot exceed upper bounds")
    return sum(low <= value <= high for value, low, high in zip(values, lowers, uppers)) / len(values)


def mean_interval_width(lower: Sequence[float], upper: Sequence[float]) -> float:
    """Return the mean width of a collection of finite prediction intervals."""

    lowers = [float(value) for value in lower]
    uppers = [float(value) for value in upper]
    if not lowers or len(lowers) != len(uppers):
        raise ValueError("lower and upper must be non-empty and have the same length")
    if any(not isfinite(value) for value in lowers + uppers):
        raise ValueError("interval bounds must be finite")
    if any(low > high for low, high in zip(lowers, uppers)):
        raise ValueError("lower bounds cannot exceed upper bounds")
    return sum(high - low for low, high in zip(lowers, uppers)) / len(lowers)

