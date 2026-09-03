"""Freeze and evaluate the V3.16 Ling transfer pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sp500_forecastability import detection_v3_16 as protocol
from sp500_forecastability import detection_v3_16_analysis as analysis
from sp500_forecastability import detection_v3_16_calls as calls

PILOT_VERSION = "detection-v3.16.5-ling-transfer-pilot-2026-09-03"
PREREGISTRATION = Path("docs/detection_v3_16_5_ling_transfer.md")
MISSINGNESS_AMENDMENT = Path("docs/detection_v3_16_2_ling_missingness.md")
TOKEN_AMENDMENT = Path("docs/detection_v3_16_3_ling_token_budget.md")
TOKEN_AMENDMENT_2 = Path("docs/detection_v3_16_4_ling_token_budget.md")
INTERFACE_AMENDMENT = Path("docs/detection_v3_16_5_common_interface.md")
ROOT = protocol.DEFAULT_ROOT / "calls_4" / "ling"
BINDING = ROOT / "ling_pilot_binding.json"
PARENT_ROOT = protocol.DEFAULT_ROOT / "calls_3" / "ling"
PARENT_BINDING = PARENT_ROOT / "ling_pilot_binding.json"
SUMMARY = ROOT / "analysis" / "development_summary.json"
REPORT = ROOT / "analysis" / "development_report.md"
MODEL_DIR = Path("/storage/lianjh/modelzoos/inclusionAI/Ling-3.0-tiny-int4")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _model_fingerprint() -> dict[str, Any]:
    small = [
        path
        for name in ("config.json", "generation_config.json", "tokenizer_config.json")
        for path in [MODEL_DIR / name]
        if path.is_file()
    ]
    shards = sorted(MODEL_DIR.glob("*.safetensors"))
    return {
        "path": str(MODEL_DIR),
        "small_file_sha256": {path.name: protocol.file_sha256(path) for path in small},
        "weight_sizes": {path.name: path.stat().st_size for path in shards},
    }


def build_binding() -> dict[str, Any]:
    model = calls.MODELS["ling"]
    base_protocol = ROOT / "protocol_manifest.json"
    return {
        "pilot_version": PILOT_VERSION,
        "status": "frozen_after_qwen_development_before_any_ling_v3_16_call",
        "preregistration_sha256": protocol.file_sha256(PREREGISTRATION),
        "missingness_amendment_sha256": protocol.file_sha256(MISSINGNESS_AMENDMENT),
        "token_amendment_sha256": protocol.file_sha256(TOKEN_AMENDMENT),
        "token_amendment_2_sha256": protocol.file_sha256(TOKEN_AMENDMENT_2),
        "interface_amendment_sha256": protocol.file_sha256(INTERFACE_AMENDMENT),
        "parent_smoke_binding_sha256": protocol.file_sha256(PARENT_BINDING),
        "parent_failed_smoke_records_sha256": protocol.file_sha256(
            PARENT_ROOT / "smoke" / "records.jsonl"
        ),
        "parent_failed_smoke_qualification_sha256": protocol.file_sha256(
            PARENT_ROOT / "smoke" / "qualification.json"
        ),
        "selection_manifest_sha256": protocol.file_sha256(protocol.SELECTION_MANIFEST),
        "risk_manifest_sha256": protocol.file_sha256(analysis.RISK_MANIFEST),
        "base_call_protocol_sha256": protocol.file_sha256(base_protocol),
        "analysis_implementation_sha256": protocol.file_sha256(Path(analysis.__file__)),
        "binding_implementation_sha256": protocol.file_sha256(Path(__file__)),
        "model": {"id": model.model, "endpoint": model.endpoint},
        "model_fingerprint": _model_fingerprint(),
        "weights": json.loads(analysis.RISK_MANIFEST.read_text(encoding="utf-8"))["weights"],
        "formal_calls_authorized": False,
    }


def freeze_binding() -> dict[str, Any]:
    model = calls.MODELS["ling"]
    calls.freeze_protocol(model)
    expected = build_binding()
    if BINDING.exists():
        actual = json.loads(BINDING.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError("Ling transfer binding drifted")
        return actual
    if list((ROOT / "development").glob("records*.jsonl")):
        raise ValueError("cannot freeze Ling development binding after development calls")
    _write_json(BINDING, expected)
    return expected


def validate_binding() -> dict[str, Any]:
    actual = json.loads(BINDING.read_text(encoding="utf-8"))
    if actual != build_binding():
        raise ValueError("Ling transfer binding drifted")
    return actual


def evaluate() -> dict[str, Any]:
    validate_binding()
    model = calls.MODELS["ling"]
    qualification = json.loads(
        (ROOT / "development" / "qualification.json").read_text(encoding="utf-8")
    )
    payload = analysis.freeze_preoutcome(model, "development")
    rows = [
        row
        for row in analysis.join_outcomes(payload)
        if float(row["agreement"]) >= analysis.HIGH_CONSENSUS
    ]
    risk_manifest = json.loads(analysis.RISK_MANIFEST.read_text(encoding="utf-8"))
    weights = tuple(float(risk_manifest["weights"][name]) for name in analysis.COORDINATES)
    metrics = analysis._metrics(rows, weights)
    intervals = analysis._bootstrap(rows, weights)
    by_label = metrics["by_label"]
    gates = {
        "valid_rate_at_least_098": qualification["valid_rate"] >= 0.98,
        "first_pass_valid_rate_at_least_095": qualification["first_pass_valid_rate"] >= 0.95,
        "high_consensus_at_least_20": len(rows) >= 20,
        "errors_per_label_at_least_4": all(by_label[label]["errors"] >= 4 for label in by_label),
        "overall_auroc_above_055": metrics["overall_auroc"] > 0.55,
        "macro_label_auroc_above_055": metrics["macro_label_auroc"] > 0.55,
        "worst_label_auroc_above_050": metrics["worst_label_auroc"] > 0.50,
        "risk80_reduction_nonnegative": metrics["risk_at_80"]["error_reduction"] >= 0.0,
    }
    summary = {
        "pilot_version": PILOT_VERSION,
        "status": "development_transfer_pilot",
        "model": model.model,
        "binding_sha256": protocol.file_sha256(BINDING),
        "records_sha256": payload["records_sha256"],
        "weights": risk_manifest["weights"],
        "metrics": metrics,
        "pair_bootstrap_ci": intervals,
        "gates": gates,
        "qualified_for_new_formal_preregistration": all(gates.values()),
        "formal_calls_authorized": False,
    }
    _write_json(SUMMARY, summary)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(_report(summary), encoding="utf-8")
    return summary


def _report(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    lines = [
        "# V3.16.1 Ling label-symmetric transfer pilot",
        "",
        "Status: **development transfer pilot; not a formal result**.",
        "",
        f"- High-consensus items: {metrics['n']}",
        f"- Errors: {metrics['errors']}",
        f"- Overall AUROC: {metrics['overall_auroc']:.3f}",
        f"- Macro-label AUROC: {metrics['macro_label_auroc']:.3f}",
        f"- Worst-label AUROC: {metrics['worst_label_auroc']:.3f}",
        (
            f"- Risk@80: {metrics['risk_at_80']['baseline_error']:.3f} -> "
            f"{metrics['risk_at_80']['retained_error']:.3f}"
        ),
        "",
        "## Frozen gates",
        "",
    ]
    lines.extend(
        f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in summary["gates"].items()
    )
    lines.extend(
        [
            "",
            "A pass permits a new formal preregistration. It does not convert these",
            "development items into paper evidence and does not authorize formal calls.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "validate", "evaluate"))
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze_binding()
    elif args.command == "validate":
        result = validate_binding()
    else:
        result = evaluate()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
