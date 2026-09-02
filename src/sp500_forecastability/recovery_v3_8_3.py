"""Final closed transport conformance for zero-shot Qwen-to-Ling ELAR."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_6_2 as v362
from sp500_forecastability import recovery_v3_8 as v38
from sp500_forecastability import recovery_v3_8_1 as v381

PROTOCOL_VERSION = "recovery-v3.8.3-qwen-to-ling-elar-2026-09-03"
DEFAULT_ROOT = Path("results/recovery_v3_8_3_ling")
PREREGISTRATION = Path("docs/recovery_v3_8_3_preregistration.md")
RUN_SCRIPT = Path("scripts/run_recovery_v3_8_3.sh")
V382_MANIFEST = Path("results/recovery_v3_8_2_ling/protocol_manifest.json")
V382_ABORT = Path("results/recovery_v3_8_2_ling/ABORTED.md")
V382_SMOKE_PARTIAL = Path(
    "results/recovery_v3_8_2_ling/smoke/actions/records.partial.jsonl"
)
V382_CACHE = Path("results/recovery_v3_8_2_ling/cache")

_STRICT_ACTION_PARSER = v381._ORIGINAL_ACTION_PARSER
_ORIGINAL_BUILD_MANIFEST = v381._ORIGINAL_BUILD_MANIFEST


def parse_action_decision(content: str, allowed_ids: Sequence[str]) -> dict[str, Any]:
    """Apply the final closed conformance layer, then validate strictly."""
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
    cited = payload.get("cited_evidence_ids")
    if cited == "":
        payload["cited_evidence_ids"] = []
        modes.append("empty_citation_string")
    elif isinstance(cited, str) and cited in allowed_ids:
        payload["cited_evidence_ids"] = [cited]
        modes.append("singleton_citation_string")
    confidence = payload.get("confidence")
    if (
        isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and math.isfinite(float(confidence))
        and 1 < float(confidence) <= 100
    ):
        payload["confidence"] = float(confidence) / 100
        modes.append("confidence_percent")
    decision = _STRICT_ACTION_PARSER(json.dumps(payload), allowed_ids)
    if modes:
        decision["parse_mode"] = "v3_8_3_" + "_and_".join(modes)
    return decision


def _implementation_path() -> Path:
    return Path(__file__).resolve()


def _attempt_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stats = v381._ORIGINAL_ATTEMPT_STATS(records)
    stats["decision_parse_modes"] = dict(
        sorted(
            Counter(
                str(row["decision"].get("parse_mode", "strict"))
                for row in records
                if row.get("success") and isinstance(row.get("decision"), dict)
            ).items()
        )
    )
    return stats


def _audit_v382_abort() -> dict[str, Any]:
    """Bind the final protocol to the outcome-blind development smoke failure."""
    records = base._load_jsonl(V382_SMOKE_PARTIAL)
    failures = [row for row in records if not bool(row.get("success"))]
    audit = {
        "rows": len(records),
        "development_examples": len({str(row["example_id"]) for row in records}),
        "terminal_failures": len(failures),
        "terminal_error_counts": dict(
            sorted(Counter(str(row.get("final_error")) for row in failures).items())
        ),
    }
    expected = {
        "rows": 16,
        "development_examples": 2,
        "terminal_failures": 1,
        "terminal_error_counts": {
            "ValueError: confidence must be finite and in [0, 1]": 1
        },
    }
    if audit != expected:
        raise ValueError("V3.8.2 development-smoke abort facts drifted")
    cache_artifacts = []
    for attempt in failures[0]["attempts"]:
        cache_key = str(attempt["cache_key"])
        cache_path = V382_CACHE / f"{cache_key}.json"
        response = json.loads(cache_path.read_text(encoding="utf-8"))
        payload = v362._extract_json_object(str(response["content"]))
        if payload.get("confidence") != 95:
            raise ValueError("V3.8.2 failed response does not use confidence 95")
        cache_artifacts.append(
            {
                "cache_key": cache_key,
                "sha256": base._sha256_path(cache_path),
                "confidence_type": "number",
                "confidence_value": 95,
            }
        )
    return {
        "smoke_action_records": str(V382_SMOKE_PARTIAL),
        "smoke_action_records_sha256": base._sha256_path(V382_SMOKE_PARTIAL),
        "failed_response_cache_artifacts": cache_artifacts,
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
    manifest["status"] = "frozen_before_any_v3_8_3_target_task_call"
    manifest["amendment"] = {
        "scope": "final_closed_semantic_json_conformance",
        "no_further_parser_extension_after_freeze": True,
        "runner_script": str(RUN_SCRIPT),
        "runner_script_sha256": base._sha256_path(RUN_SCRIPT),
        "prior_protocol_manifest": str(V382_MANIFEST),
        "prior_protocol_manifest_sha256": base._sha256_path(V382_MANIFEST),
        "prior_abort_record": str(V382_ABORT),
        "prior_abort_record_sha256": base._sha256_path(V382_ABORT),
        "prior_abort_audit": _audit_v382_abort(),
        "fresh_target_cache": True,
        "reruns_all_400_formal_examples": True,
        "uses_prior_target_responses_at_inference": False,
        "uses_target_correctness_or_action_outcomes": False,
        "canonicalizations": [
            "strip_and_casefold_answer_only_if_yes_or_no",
            "evidence_ids_to_cited_evidence_ids_only_if_unambiguous",
            "empty_citation_string_to_empty_list",
            "exact_allowed_citation_string_to_singleton_list",
            "finite_numeric_confidence_in_1_to_100_divided_by_100",
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
            raise ValueError("V3.8.3 manifest or a frozen dependency drifted")


def prepare(path: Path = DEFAULT_ROOT / "protocol_manifest.json") -> dict[str, Any]:
    with _configured_base():
        formal_root = DEFAULT_ROOT / "formal"
        if not path.exists() and any(formal_root.glob("**/records*.jsonl")):
            raise ValueError("cannot freeze V3.8.3 after target formal records exist")
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
            "PASS_ZERO_SHOT_CROSS_MODEL_ELAR_V3_8_3_FINAL_CONFORMANCE"
            if summary["passes"]
            else "NO_VERIFIED_CROSS_MODEL_ELAR_TRANSFER_V3_8_3"
        )
        summary["transport_amendment"] = _build_protocol_manifest()["amendment"]
        base._write_json(DEFAULT_ROOT / "evaluation" / "summary.json", summary)
        report_path = DEFAULT_ROOT / "evaluation" / "report.md"
        v38._write_report(summary, report_path)
        report = report_path.read_text(encoding="utf-8").replace(
            "# Recovery V3.8 result:", "# Recovery V3.8.3 result:", 1
        )
        parse_modes = summary["transport_and_schema"]["actions"].get(
            "decision_parse_modes", {}
        )
        report += "\n".join(
            [
                "## Final closed transport conformance",
                "",
                "The fresh run used only the five preregistered transformations.",
                f"Accepted action parse modes: `{json.dumps(parse_modes, sort_keys=True)}`.",
                (
                    "All prior schema aborts are preserved; none of their target "
                    "responses were reused at inference."
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
