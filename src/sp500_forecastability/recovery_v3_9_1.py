"""Trailing-prose transport amendment for frozen Qwen-to-Fin-R1 ELAR."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sp500_forecastability import recovery_v3 as base
from sp500_forecastability import recovery_v3_8 as v38
from sp500_forecastability import recovery_v3_8_3 as v383
from sp500_forecastability import recovery_v3_8_3_analysis as v383_analysis
from sp500_forecastability import recovery_v3_9 as v39

PROTOCOL_VERSION = "recovery-v3.9.1-qwen-to-finr1-elar-2026-09-03"
DEFAULT_ROOT = Path("results/recovery_v3_9_1_finr1")
PREREGISTRATION = Path("docs/recovery_v3_9_1_preregistration.md")
RUN_SCRIPT = Path("scripts/run_recovery_v3_9_1.sh")
V39_MANIFEST = Path("results/recovery_v3_9_finr1/protocol_manifest.json")
V39_ABORT = Path("results/recovery_v3_9_finr1/ABORTED.md")
V39_CACHE = Path("results/recovery_v3_9_finr1/cache")

_ORIGINAL_ACTION_PARSER = v383.parse_action_decision


def parse_action_decision(content: str, allowed_ids: Sequence[str]) -> dict[str, Any]:
    """Accept one leading JSON decision plus brace-free trailing prose."""
    try:
        return _ORIGINAL_ACTION_PARSER(content, allowed_ids)
    except (TypeError, ValueError) as original_error:
        text = content.strip()
        try:
            payload, end = json.JSONDecoder().raw_decode(text)
        except json.JSONDecodeError:
            raise original_error
        trailing = text[end:].strip()
        if (
            not isinstance(payload, Mapping)
            or not trailing
            or "{" in trailing
            or "}" in trailing
        ):
            raise
        decision = _ORIGINAL_ACTION_PARSER(json.dumps(payload), allowed_ids)
        prior_mode = str(decision.get("parse_mode", "strict"))
        suffix = "" if prior_mode == "strict" else f"_and_{prior_mode}"
        decision["parse_mode"] = "v3_9_1_leading_json_with_trailing_text" + suffix
        return decision


def _implementation_path() -> Path:
    return Path(__file__).resolve()


def _audit_v39_smoke_abort() -> dict[str, Any]:
    paths = sorted(V39_CACHE.glob("*.json"))
    if len(paths) != 16:
        raise ValueError("V3.9 smoke cache count drifted")
    strict = 0
    trailing = 0
    artifacts = []
    allowed = ["A00", "A01", "A02"]
    for path in paths:
        cached = json.loads(path.read_text(encoding="utf-8"))
        content = str(cached["content"])
        try:
            _ORIGINAL_ACTION_PARSER(content, allowed)
        except (TypeError, ValueError):
            parse_action_decision(content, allowed)
            trailing += 1
            mode = "leading_json_with_trailing_text"
        else:
            strict += 1
            mode = "strict"
        artifacts.append(
            {
                "path": str(path),
                "sha256": base._sha256_path(path),
                "transport_mode": mode,
            }
        )
    if (strict, trailing) != (4, 12):
        raise ValueError("V3.9 outcome-blind smoke transport counts drifted")
    return {
        "cache_artifacts": artifacts,
        "responses": len(paths),
        "strict_json": strict,
        "leading_json_with_trailing_text": trailing,
        "formal_target_calls": 0,
        "outcomes_accessed": False,
    }


@contextmanager
def _configured_base() -> Iterator[None]:
    old = {
        "protocol_version": v39.PROTOCOL_VERSION,
        "default_root": v39.DEFAULT_ROOT,
        "preregistration": v39.PREREGISTRATION,
        "run_script": v39.RUN_SCRIPT,
        "validator": v39.validate_protocol_manifest,
        "implementation_path": v39._implementation_path,
        "parser": v383.parse_action_decision,
    }
    v39.PROTOCOL_VERSION = PROTOCOL_VERSION
    v39.DEFAULT_ROOT = DEFAULT_ROOT
    v39.PREREGISTRATION = PREREGISTRATION
    v39.RUN_SCRIPT = RUN_SCRIPT
    v39.validate_protocol_manifest = validate_protocol_manifest
    v39._implementation_path = _implementation_path
    v383.parse_action_decision = parse_action_decision
    try:
        with v39._configured_base():
            yield
    finally:
        v39.PROTOCOL_VERSION = old["protocol_version"]
        v39.DEFAULT_ROOT = old["default_root"]
        v39.PREREGISTRATION = old["preregistration"]
        v39.RUN_SCRIPT = old["run_script"]
        v39.validate_protocol_manifest = old["validator"]
        v39._implementation_path = old["implementation_path"]
        v383.parse_action_decision = old["parser"]


def _build_protocol_manifest() -> dict[str, Any]:
    manifest = v39._build_protocol_manifest()
    manifest["status"] = "frozen_before_any_v3_9_1_target_task_call"
    manifest["transport_amendment"] = {
        "scope": "one_leading_json_object_plus_brace_free_trailing_prose",
        "no_further_parser_extension_after_freeze": True,
        "prior_protocol_manifest": str(V39_MANIFEST),
        "prior_protocol_manifest_sha256": base._sha256_path(V39_MANIFEST),
        "prior_abort_record": str(V39_ABORT),
        "prior_abort_record_sha256": base._sha256_path(V39_ABORT),
        "prior_outcome_blind_smoke_audit": _audit_v39_smoke_abort(),
        "fresh_target_cache": True,
        "reruns_all_400_formal_examples": True,
        "uses_prior_target_responses_at_inference": False,
        "uses_target_correctness_or_action_outcomes": False,
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
            raise ValueError("V3.9.1 manifest or a frozen dependency drifted")


def prepare(path: Path = DEFAULT_ROOT / "protocol_manifest.json") -> dict[str, Any]:
    with _configured_base():
        task_records_exist = any(
            (DEFAULT_ROOT / "formal").glob("**/records*.jsonl")
        ) or any((DEFAULT_ROOT / "smoke").glob("**/records*.jsonl"))
        if not path.exists() and task_records_exist:
            raise ValueError("cannot freeze V3.9.1 after target task records exist")
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
    with _configured_base(), v383_analysis._patched_diagnostic():
        summary = v38.evaluate(DEFAULT_ROOT / "evaluation")
        summary["verdict"] = (
            "PASS_ZERO_SHOT_QWEN_TO_FINR1_ELAR_V3_9_1"
            if summary["passes"]
            else "NO_VERIFIED_QWEN_TO_FINR1_ELAR_TRANSFER_V3_9_1"
        )
        summary["transport_amendment"] = _build_protocol_manifest()[
            "transport_amendment"
        ]
        summary["replication_context"] = {
            "registered_before_ling_result": True,
            "ling_outcomes_used_for_finr1_fit_or_selection": False,
            "different_organization_checkpoint": True,
            "broad_qwen_lineage_shared": True,
        }
        base._write_json(DEFAULT_ROOT / "evaluation" / "summary.json", summary)
        report_path = DEFAULT_ROOT / "evaluation" / "report.md"
        v38._write_report(summary, report_path)
        report = report_path.read_text(encoding="utf-8").replace(
            "# Recovery V3.8 result: zero-shot Qwen-to-Ling ELAR",
            "# Recovery V3.9.1 result: zero-shot Qwen-to-Fin-R1 ELAR",
            1,
        )
        modes = summary["transport_and_schema"]["actions"].get(
            "decision_parse_modes", {}
        )
        report += "\n".join(
            [
                "## Transport amendment and replication boundary",
                "",
                f"Accepted action parse modes: `{json.dumps(modes, sort_keys=True)}`.",
                (
                    "Leading JSON plus trailing prose was qualified on development "
                    "smoke only; prior responses were not reused."
                ),
                (
                    "Fin-R1 is a separately trained SUFE checkpoint that shares the "
                    "broad Qwen lineage with the source."
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
        with _configured_base():
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
