from sp500_forecastability.conformal import conformal_radius, symmetric_interval
from sp500_forecastability.metrics import (
    brier_score,
    expected_calibration_error,
    interval_coverage,
    mean_interval_width,
    selective_metrics,
)


def test_selective_metrics_keeps_the_most_confident_examples() -> None:
    result = selective_metrics([1, 0, 0, 0], [0.9, 0.55, 0.1, 0.2], coverage=0.5)

    assert result.selected_count == 2
    assert result.coverage == 0.5
    assert result.accuracy == 1.0


def test_probability_metrics_have_expected_range() -> None:
    assert abs(brier_score([0, 1], [0.25, 0.75]) - 0.0625) < 1e-12
    assert abs(expected_calibration_error([0, 1], [0.25, 0.75], n_bins=2) - 0.25) < 1e-12


def test_conformal_interval_and_coverage() -> None:
    radius = conformal_radius([0.1, 0.2, 0.3], alpha=0.1)
    lower, upper = symmetric_interval([0.0, 1.0], radius)

    assert radius == 0.3
    assert interval_coverage([0.2, 1.25], lower, upper) == 1.0
    assert abs(mean_interval_width(lower, upper) - 0.6) < 1e-12
