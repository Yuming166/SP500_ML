"""Small, model-agnostic conformal prediction primitives."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil, isfinite


def conformal_radius(nonconformity_scores: Sequence[float], alpha: float = 0.1) -> float:
    """Return a finite-sample split-conformal radius.

    Scores are expected to be non-negative absolute residuals from a past
    calibration window. The quantile uses the conservative ``ceil((n+1)(1-a))``
    rank and never looks at the future test observations.
    """

    scores = [float(score) for score in nonconformity_scores]
    if not scores:
        raise ValueError("nonconformity_scores must not be empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if any(not isfinite(score) or score < 0.0 for score in scores):
        raise ValueError("scores must be finite and non-negative")

    ordered = sorted(scores)
    rank = ceil((len(ordered) + 1) * (1.0 - alpha))
    index = min(max(rank - 1, 0), len(ordered) - 1)
    return ordered[index]


def symmetric_interval(predictions: Sequence[float], radius: float) -> tuple[list[float], list[float]]:
    """Build symmetric prediction intervals around point predictions."""

    values = [float(value) for value in predictions]
    radius = float(radius)
    if any(not isfinite(value) for value in values) or not isfinite(radius) or radius < 0.0:
        raise ValueError("predictions and radius must be finite; radius must be non-negative")
    return (
        [value - radius for value in values],
        [value + radius for value in values],
    )

