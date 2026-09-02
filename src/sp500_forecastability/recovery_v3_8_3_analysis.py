"""Analysis-only None-ledger fix for frozen Recovery V3.8.3 outputs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_7 as frozen
from sp500_forecastability import recovery_v3_8_3 as v383

AMENDMENT_VERSION = "recovery-v3.8.3-analysis-none-ledger-2026-09-03"
ROOT = v383.DEFAULT_ROOT
ANALYSIS_MANIFEST = ROOT / "analysis_manifest.json"
DOCUMENT = Path("docs/recovery_v3_8_3_analysis_amendment.md")
ACTION_RECORDS = ROOT / "formal" / "actions" / "records.jsonl"
CERTIFICATE_RECORDS = ROOT / "formal" / "certificates" / "records.jsonl"
LEDGER_RECORDS = ROOT / "formal" / "ledgers" / "records.jsonl"

_ORIGINAL_SELECT_ELAR = frozen._select_elar


def _implementation_path() -> Path:
    return Path(__file__).resolve()


def _sanitize_ledger_groups(
    ledger_groups: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        key: ({**row, "ledger": {}} if row.get("ledger") is None else row)
        for key, row in ledger_groups.items()
    }


def _select_elar_none_safe(
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
    safe_groups = (
        ledger_groups if require_ledger else _sanitize_ledger_groups(ledger_groups)
    )
    return _ORIGINAL_SELECT_ELAR(
        examples,
        action_groups,
        certificate_groups,
        safe_groups,
        predictions,
        confidence_threshold=confidence_threshold,
        lexical_threshold=lexical_threshold,
        unsupported_term_cap=unsupported_term_cap,
        require_ledger=require_ledger,
    )


@contextmanager
def _patched_diagnostic() -> Any:
    old = frozen._select_elar
    frozen._select_elar = _select_elar_none_safe
    try:
        yield
    finally:
        frozen._select_elar = old


def build_analysis_manifest() -> dict[str, Any]:
    v383.validate_protocol_manifest()
    return {
        "amendment_version": AMENDMENT_VERSION,
        "status": "frozen_before_first_completed_outcome_evaluation",
        "scope": "atomic_proof_only_none_ledger_diagnostic_crash_fix",
        "primary_elar_changed": False,
        "target_model_calls": False,
        "outcomes_used_to_define_amendment": False,
        "document_path": str(DOCUMENT),
        "document_sha256": base._sha256_path(DOCUMENT),
        "implementation_path": str(_implementation_path()),
        "implementation_sha256": base._sha256_path(_implementation_path()),
        "protocol_manifest_sha256": base._sha256_path(
            ROOT / "protocol_manifest.json"
        ),
        "formal_records_sha256": {
            "actions": base._sha256_path(ACTION_RECORDS),
            "certificates": base._sha256_path(CERTIFICATE_RECORDS),
            "ledgers": base._sha256_path(LEDGER_RECORDS),
        },
    }


def prepare_analysis() -> dict[str, Any]:
    expected = build_analysis_manifest()
    if ANALYSIS_MANIFEST.exists():
        actual = json.loads(ANALYSIS_MANIFEST.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError("analysis amendment or frozen target outputs drifted")
        return expected
    if (ROOT / "evaluation" / "summary.json").exists():
        raise ValueError("cannot freeze analysis amendment after outcome summary exists")
    base._write_json(ANALYSIS_MANIFEST, expected)
    return expected


def validate_analysis_manifest() -> dict[str, Any]:
    actual = json.loads(ANALYSIS_MANIFEST.read_text(encoding="utf-8"))
    expected = build_analysis_manifest()
    if actual != expected:
        raise ValueError("analysis amendment or frozen target outputs drifted")
    return actual


def evaluate() -> dict[str, Any]:
    amendment = validate_analysis_manifest()
    with _patched_diagnostic():
        summary = v383.evaluate()
    summary["analysis_amendment"] = amendment
    base._write_json(ROOT / "evaluation" / "summary.json", summary)
    report_path = ROOT / "evaluation" / "report.md"
    report = report_path.read_text(encoding="utf-8")
    report += "\n".join(
        [
            "## Analysis-only amendment",
            "",
            (
                "The registered atomic-proof-only diagnostic treats a fail-closed "
                "None ledger as empty; primary ELAR is unchanged."
            ),
            f"Amendment: `{AMENDMENT_VERSION}`.",
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-analysis", "evaluate"))
    args = parser.parse_args(argv)
    if args.command == "prepare-analysis":
        print(json.dumps(prepare_analysis(), indent=2, sort_keys=True))
        return 0
    summary = evaluate()
    print(json.dumps(summary["primary_gates"], indent=2, sort_keys=True))
    print(f"verdict: {summary['verdict']}")
    return 0 if summary["passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
