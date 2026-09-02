from __future__ import annotations

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_11 as v311
from sp500_forecastability import recovery_v3_13_analysis as analysis


def test_analysis_adapter_matches_original_on_grouped_or_raw_records() -> None:
    records = [
        {"example_id": "x", "phase": "baseline", "action": "KEEP"},
        {"example_id": "y", "phase": "baseline", "action": "KEEP"},
    ]
    grouped = base._record_groups(records)
    original = analysis._ORIGINAL_POLICY_METRICS
    try:
        v311._policy_metrics = lambda examples, groups, selected: {
            "keys": sorted(groups),
            "selected": dict(selected),
        }
        analysis._ORIGINAL_POLICY_METRICS = v311._policy_metrics
        expected = {"keys": ["x", "y"], "selected": {"x": "KEEP"}}
        assert analysis._group_compatibility_adapter([], records, {"x": "KEEP"}) == expected
        assert analysis._group_compatibility_adapter([], grouped, {"x": "KEEP"}) == expected
    finally:
        analysis._ORIGINAL_POLICY_METRICS = original
        v311._policy_metrics = original
