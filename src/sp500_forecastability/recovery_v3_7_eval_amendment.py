"""Evaluation-only null handling for the V3.7.1 proof-only ablation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_4 as v34
from sp500_forecastability import recovery_v3_6_2 as v362
from sp500_forecastability import recovery_v3_7 as frozen

AMENDMENT_DOC = Path("docs/recovery_v3_7_1_evaluation_amendment.md")


def _safe_select_elar(
    examples: Sequence[Mapping[str, Any]],
    action_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    certificate_groups: Mapping[tuple[str, str], Mapping[str, Any]],
    ledger_groups: Mapping[tuple[str, str], Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], tuple[float, float]],
    *,
    confidence_threshold: float,
    lexical_threshold: float,
    unsupported_term_cap: int,
    require_ledger: bool = True,
) -> dict[str, str]:
    selected = {}
    for example in examples:
        example_id = str(example["example_id"])
        consensus, agreement, _baseline = base._baseline_state(action_groups[example_id])
        allowed = []
        if agreement >= base.HIGH_CONSENSUS:
            for action in v362.CERTIFICATE_ACTIONS:
                certificate_row = certificate_groups[(example_id, action)]
                if not v362._certificate_gate(certificate_row, consensus):
                    continue
                ledger_row = ledger_groups.get((example_id, action))
                if require_ledger and not frozen._ledger_gate(
                    ledger_row,
                    confidence_threshold=confidence_threshold,
                    lexical_threshold=lexical_threshold,
                    unsupported_term_cap=unsupported_term_cap,
                ):
                    continue
                # Evaluation-only amendment: a fail-closed row stores ledger=null.
                ledger = (ledger_row.get("ledger") or {}) if ledger_row else {}
                p_fix, p_harm = predictions[(example_id, action)]
                allowed.append(
                    (
                        float(ledger.get("min_confidence", 0.0)),
                        float(ledger.get("min_lexical_coverage", 0.0)),
                        p_fix - p_harm,
                        p_fix,
                        -p_harm,
                        action,
                    )
                )
        selected[example_id] = max(allowed)[-1] if allowed else "KEEP"
    return selected


def main() -> int:
    amendment_path = Path(__file__)
    original_writer = v34._write_or_validate_preoutcome

    def write_with_amendment(path: Path, payload: Mapping[str, Any]) -> None:
        enriched = dict(payload)
        enriched["evaluation_amendment"] = {
            "implementation_path": str(amendment_path),
            "implementation_sha256": base._sha256_path(amendment_path),
            "document_path": str(AMENDMENT_DOC),
            "document_sha256": base._sha256_path(AMENDMENT_DOC),
            "formal_outcomes_accessed_before_amendment": False,
        }
        original_writer(path, enriched)

    frozen._select_elar = _safe_select_elar
    v34._write_or_validate_preoutcome = write_with_amendment
    summary = frozen.evaluate(
        frozen.DEFAULT_ROOT / "selection_manifest.json",
        frozen.DEFAULT_ROOT / "router" / "manifest.json",
        frozen.DEFAULT_ROOT / "evaluation",
    )
    summary["evaluation_amendment"] = {
        "implementation_path": str(amendment_path),
        "implementation_sha256": base._sha256_path(amendment_path),
        "document_path": str(AMENDMENT_DOC),
        "document_sha256": base._sha256_path(AMENDMENT_DOC),
    }
    base._write_json(frozen.DEFAULT_ROOT / "evaluation" / "summary.json", summary)
    print(json.dumps(summary["primary_gates"], indent=2, sort_keys=True))
    print(f"verdict: {summary['verdict']}")
    return 0 if summary["passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
