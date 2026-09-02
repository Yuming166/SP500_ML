"""Transport-only V3.8.1 amendment for zero-shot Qwen-to-Ling ELAR."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_6_2 as v362
from sp500_forecastability import recovery_v3_8 as v38

PROTOCOL_VERSION = "recovery-v3.8.1-qwen-to-ling-elar-2026-09-02"
DEFAULT_ROOT = Path("results/recovery_v3_8_1_ling")
PREREGISTRATION = Path("docs/recovery_v3_8_1_preregistration.md")
RUN_SCRIPT = Path("scripts/run_recovery_v3_8_1.sh")
V38_MANIFEST = Path("results/recovery_v3_8_ling/protocol_manifest.json")
V38_ABORT = Path("results/recovery_v3_8_ling/ABORTED.md")
V38_PARTIAL = Path("results/recovery_v3_8_ling/formal/actions/records.partial.jsonl")

_ORIGINAL_ACTION_PARSER = v362.parse_action_decision
_ORIGINAL_BUILD_MANIFEST = v38.build_protocol_manifest
_ORIGINAL_VALIDATE_MANIFEST = v38.validate_protocol_manifest
_ORIGINAL_ATTEMPT_STATS = v38._attempt_stats
_ORIGINAL_IMPLEMENTATION_PATH = v38._implementation_path


def parse_action_decision(content: str, allowed_ids: Sequence[str]) -> dict[str, Any]:
    """Apply only two semantic-preserving cross-model schema aliases."""
    payload = v362._extract_json_object(content)
    modes = []
    answer = payload.get("answer")
    if isinstance(answer, str):
        canonical = answer.strip().casefold()
        if canonical in {"yes", "no"} and canonical != answer:
            payload["answer"] = canonical
            modes.append("answer_casefold")
    if "evidence_ids" in payload and "cited_evidence_ids" not in payload:
        payload["cited_evidence_ids"] = payload.pop("evidence_ids")
        modes.append("evidence_ids_alias")
    decision = _ORIGINAL_ACTION_PARSER(json.dumps(payload), allowed_ids)
    if modes:
        decision["parse_mode"] = "v3_8_1_" + "_and_".join(modes)
    return decision


def _implementation_path() -> Path:
    return Path(__file__).resolve()


def _attempt_stats(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    stats = _ORIGINAL_ATTEMPT_STATS(records)
    stats["decision_parse_modes"] = dict(
        sorted(
            Counter(
                str(row["decision"].get("parse_mode", "unspecified"))
                for row in records
                if row.get("success") and isinstance(row.get("decision"), dict)
            ).items()
        )
    )
    return stats


def _audit_v38_abort() -> dict[str, Any]:
    """Bind the amendment to the outcome-blind facts that triggered the abort."""
    records = base._load_jsonl(V38_PARTIAL)
    failures = [row for row in records if not bool(row.get("success"))]
    expected_error = (
        "ValueError: response fields mismatch; unknown=['evidence_ids'], "
        "missing=['cited_evidence_ids']"
    )
    audit = {
        "rows": len(records),
        "complete_example_bundles": len({str(row["example_id"]) for row in records}),
        "terminal_failures": len(failures),
        "terminal_error_counts": dict(
            sorted(Counter(str(row.get("final_error")) for row in failures).items())
        ),
    }
    expected = {
        "rows": 640,
        "complete_example_bundles": 80,
        "terminal_failures": 3,
        "terminal_error_counts": {expected_error: 3},
    }
    if audit != expected:
        raise ValueError("V3.8 outcome-blind abort facts drifted")
    return {
        "partial_action_records": str(V38_PARTIAL),
        "partial_action_records_sha256": base._sha256_path(V38_PARTIAL),
        **audit,
    }


@contextmanager
def _configured_base() -> Iterator[None]:
    old = {
        "protocol_version": v38.PROTOCOL_VERSION,
        "default_root": v38.DEFAULT_ROOT,
        "preregistration": v38.PREREGISTRATION,
        "parser": v362.parse_action_decision,
        "validator": v38.validate_protocol_manifest,
        "attempt_stats": v38._attempt_stats,
        "implementation_path": v38._implementation_path,
    }
    v38.PROTOCOL_VERSION = PROTOCOL_VERSION
    v38.DEFAULT_ROOT = DEFAULT_ROOT
    v38.PREREGISTRATION = PREREGISTRATION
    v362.parse_action_decision = parse_action_decision
    v38.validate_protocol_manifest = validate_protocol_manifest
    v38._attempt_stats = _attempt_stats
    v38._implementation_path = _implementation_path
    try:
        yield
    finally:
        v38.PROTOCOL_VERSION = old["protocol_version"]
        v38.DEFAULT_ROOT = old["default_root"]
        v38.PREREGISTRATION = old["preregistration"]
        v362.parse_action_decision = old["parser"]
        v38.validate_protocol_manifest = old["validator"]
        v38._attempt_stats = old["attempt_stats"]
        v38._implementation_path = old["implementation_path"]


def _build_protocol_manifest() -> dict[str, Any]:
    manifest = _ORIGINAL_BUILD_MANIFEST()
    manifest["status"] = "frozen_before_any_v3_8_1_target_task_call"
    manifest["amendment"] = {
        "scope": "transport_only_action_parser_canonicalization",
        "runner_script": str(RUN_SCRIPT),
        "runner_script_sha256": base._sha256_path(RUN_SCRIPT),
        "prior_protocol_manifest": str(V38_MANIFEST),
        "prior_protocol_manifest_sha256": base._sha256_path(V38_MANIFEST),
        "prior_abort_record": str(V38_ABORT),
        "prior_abort_record_sha256": base._sha256_path(V38_ABORT),
        "prior_abort_audit": _audit_v38_abort(),
        "fresh_target_cache": True,
        "reruns_all_400_formal_examples": True,
        "uses_prior_target_responses_at_inference": False,
        "uses_target_correctness_or_action_outcomes": False,
        "canonicalizations": [
            "strip_and_casefold_answer_only_if_yes_or_no",
            "evidence_ids_to_cited_evidence_ids_only_if_unambiguous",
        ],
    }
    return manifest


def build_protocol_manifest() -> dict[str, Any]:
    with _configured_base():
        return _build_protocol_manifest()


def validate_protocol_manifest(
    path: Path = DEFAULT_ROOT / "protocol_manifest.json",
) -> None:
    with _configured_base():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != _build_protocol_manifest():
            raise ValueError("V3.8.1 manifest or a frozen dependency drifted")


def prepare(path: Path = DEFAULT_ROOT / "protocol_manifest.json") -> dict[str, Any]:
    with _configured_base():
        formal_root = DEFAULT_ROOT / "formal"
        if not path.exists() and any(formal_root.glob("**/records*.jsonl")):
            raise ValueError("cannot freeze V3.8.1 after target formal records exist")
        expected = _build_protocol_manifest()
        if path.exists():
            validate_protocol_manifest(path)
            return expected
        base._write_json(path, expected)
        return expected


def smoke(workers: int) -> None:
    with _configured_base():
        validate_protocol_manifest()
        v38.endpoint_model_ids()
        examples = v38._examples("development")[: v38.SMOKE_EXAMPLES]
        actions = v38.execute_bundles(
            examples,
            kind="actions",
            split="smoke",
            output_dir=DEFAULT_ROOT / "smoke" / "actions",
            cache_dir=DEFAULT_ROOT / "cache",
            workers=workers,
        )
        certificates = v38.execute_bundles(
            examples,
            kind="certificates",
            split="smoke",
            output_dir=DEFAULT_ROOT / "smoke" / "certificates",
            cache_dir=DEFAULT_ROOT / "cache",
            workers=workers,
        )
        v38.execute_ledgers(
            examples,
            actions,
            certificates,
            split="smoke",
            output_dir=DEFAULT_ROOT / "smoke" / "ledgers",
            cache_dir=DEFAULT_ROOT / "cache",
            workers=workers,
        )


def evaluate() -> dict[str, Any]:
    with _configured_base():
        summary = v38.evaluate(DEFAULT_ROOT / "evaluation")
        summary["verdict"] = (
            "PASS_ZERO_SHOT_CROSS_MODEL_ELAR_V3_8_1_TRANSPORT_AMENDMENT"
            if summary["passes"]
            else "NO_VERIFIED_CROSS_MODEL_ELAR_TRANSFER_V3_8_1"
        )
        summary["transport_amendment"] = _build_protocol_manifest()["amendment"]
        base._write_json(DEFAULT_ROOT / "evaluation" / "summary.json", summary)
        report_path = DEFAULT_ROOT / "evaluation" / "report.md"
        v38._write_report(summary, report_path)
        report = report_path.read_text(encoding="utf-8").replace(
            "# Recovery V3.8 result:", "# Recovery V3.8.1 result:", 1
        )
        parse_modes = summary["transport_and_schema"]["actions"].get(
            "decision_parse_modes", {}
        )
        report += "\n".join(
            [
                "## Transport-only amendment",
                "",
                (
                    "The fresh V3.8.1 run used only the preregistered yes/no "
                    "case-fold and unambiguous evidence-key alias."
                ),
                f"Accepted action parse modes: `{json.dumps(parse_modes, sort_keys=True)}`.",
                (
                    "The V3.8 abort record and its 640 partial action rows are "
                    "content-addressed in the V3.8.1 manifest and were not reused."
                ),
                "",
            ]
        )
        report_path.write_text(report, encoding="utf-8")
        return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("endpoint-check")
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--workers", type=int, default=2)
    for name in ("formal-actions", "formal-certificates", "formal-ledgers"):
        formal_parser = subparsers.add_parser(name)
        formal_parser.add_argument("--workers", type=int, default=8)
    subparsers.add_parser("evaluate")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(), indent=2, sort_keys=True))
        return 0
    if args.command == "endpoint-check":
        validate_protocol_manifest()
        print(json.dumps(sorted(v38.endpoint_model_ids())))
        return 0
    if args.command == "smoke":
        smoke(args.workers)
        return 0
    validate_protocol_manifest()
    with _configured_base():
        examples = v38._examples("formal")
        if args.command in {"formal-actions", "formal-certificates"}:
            kind = args.command.removeprefix("formal-")
            v38.execute_bundles(
                examples,
                kind=kind,
                split="formal",
                output_dir=DEFAULT_ROOT / "formal" / kind,
                cache_dir=DEFAULT_ROOT / "cache",
                workers=args.workers,
            )
            return 0
        if args.command == "formal-ledgers":
            v38.execute_ledgers(
                examples,
                base._load_jsonl(DEFAULT_ROOT / "formal" / "actions" / "records.jsonl"),
                base._load_jsonl(
                    DEFAULT_ROOT / "formal" / "certificates" / "records.jsonl"
                ),
                split="formal",
                output_dir=DEFAULT_ROOT / "formal" / "ledgers",
                cache_dir=DEFAULT_ROOT / "cache",
                workers=args.workers,
            )
            return 0
    if args.command == "evaluate":
        summary = evaluate()
        print(json.dumps(summary["primary_gates"], indent=2, sort_keys=True))
        print(f"verdict: {summary['verdict']}")
        return 0 if summary["passes"] else 2
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
