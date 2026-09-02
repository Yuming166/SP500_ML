"""Analysis-only interface amendment for frozen Recovery V3.13 routes."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_11 as v311
from sp500_forecastability import recovery_v3_13 as v313

FROZEN_PROTOCOL_SHA256 = (
    "299dbc15999164bcb4cdbedcfde14bbcb8b384e2f2c2a7be6e67c572a854df55"
)


def _group_compatibility_adapter(
    examples: Sequence[Mapping[str, Any]],
    grouped_or_rows: Mapping[str, Sequence[Mapping[str, Any]]]
    | Sequence[Mapping[str, Any]],
    selected: Mapping[str, str],
) -> dict[str, Any]:
    """Apply the frozen metric to records after its required deterministic grouping."""
    grouped = (
        grouped_or_rows
        if isinstance(grouped_or_rows, Mapping)
        else base._record_groups(grouped_or_rows)
    )
    return _ORIGINAL_POLICY_METRICS(examples, grouped, selected)


_ORIGINAL_POLICY_METRICS = v311._policy_metrics


def evaluate() -> dict[str, Any]:
    if base._sha256_path(v313.PROTOCOL_MANIFEST) != FROZEN_PROTOCOL_SHA256:
        raise ValueError("the V3.13 frozen protocol manifest changed")
    v313.validate_protocol_manifest()
    original = v311._policy_metrics
    if original is not _ORIGINAL_POLICY_METRICS:
        raise ValueError("unexpected V3.13 metric monkeypatch")
    v311._policy_metrics = _group_compatibility_adapter
    try:
        return v313.evaluate()
    finally:
        v311._policy_metrics = original


def main() -> int:
    summary = evaluate()
    print(json.dumps(summary["primary_gates"], indent=2, sort_keys=True))
    print(f"verdict: {summary['verdict']}")
    return 0 if summary["passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
